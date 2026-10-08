"""Budget-matched comparison of policies from saved evaluation trajectories.

    python -m tinyperf.eval.budget \\
        --runs agent_rl=artifacts/eval/rl_agent_test_seeds_trajectories.jsonl:multi \\
               nofb_rl=artifacts/eval/rl_nofb_test_seeds_trajectories.jsonl:multi \\
               rl1=artifacts/eval/rl1_test_seeds_trajectories.jsonl:oneshot \\
        --budgets 1 2 3 4 6 --reference agent_rl --out artifacts/eval/budget_test_seeds.json

*Budget* = number of env steps (sandbox executions) a method may spend on one task.

* ``multi``   - a multi-turn episode truncated after its first B actions, replaying the env's best-tracking
                rule (improvement must beat the current best by ``delta``).  A policy that STOPs earlier spends
                less; that is reported as ``edits_used``.
* ``oneshot`` - best of B independent one-shot samples (``env.horizon=1``), using the unbiased estimator of
                E[best of a random B-subset] from n >= B samples per task (the pass@k construction).

All numbers are *function-weighted*: tasks are averaged within each seed function, then over functions,
and paired differences against ``--reference`` get a bootstrap CI that resamples functions (renamed copies
of one function are not independent).  Only tasks present in every run are used.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from math import comb
from typing import Dict, List, Optional, Sequence, Tuple

COMPLETE_FRAC = 0.8     # 'complete fix' = speedup >= 0.8 x reference speedup
SUCCESS_THRESHOLD = 1.5   # default: well above timing noise (1.05 admits noise when real success is rare)


def load_records(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def truncated_best(rec: Dict, budget: int, delta: float) -> Tuple[float, int]:
    """(best ratio after at most `budget` actions, actions actually used) for one multi-turn episode."""
    best, used = 1.0, 0
    for s in rec["steps"]:
        if s["action_kind"] == "stop" or used >= budget:
            break
        used += 1
        r = s.get("ratio")
        if s.get("status") == "ok" and r is not None and float(r) < (1.0 - delta) * best:
            best = float(r)
    return best, used


def expected_best_of(values_sorted_by_speed: Sequence[float], k: int) -> float:
    """E[value of the fastest of a random k-subset] given n samples sorted ascending by speedup."""
    n = len(values_sorted_by_speed)
    if k > n:
        return float("nan")
    total = comb(n, k)
    return sum(v * comb(i, k - 1) for i, v in enumerate(values_sorted_by_speed) if i >= k - 1) / total


def per_task(records: List[Dict], kind: str, budget: int, delta: float, thr: float = SUCCESS_THRESHOLD) -> Dict[str, Dict[str, float]]:
    """task_id -> {success, log_speedup, edits_used, seed}"""
    by_task: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        by_task[r["task_id"]].append(r)
    out: Dict[str, Dict[str, float]] = {}
    for tid, rs in by_task.items():
        seed = rs[0].get("seed_name") or tid
        if kind == "multi":
            vals = [truncated_best(r, budget, delta) for r in rs]
            sp = [1.0 / max(b, 1e-9) for b, _ in vals]
            ref = rs[0].get("reference_speedup")
            out[tid] = {"complete": (sum(s >= COMPLETE_FRAC * ref for s in sp) / len(sp)) if ref else float("nan"),
                        "success": sum(s >= thr for s in sp) / len(sp),
                        "log_speedup": sum(math.log(s) for s in sp) / len(sp),
                        "edits_used": sum(u for _, u in vals) / len(vals), "seed": seed}
        else:
            sp = sorted(1.0 / max(float(r["best_ratio"]), 1e-9) for r in rs)
            ref = rs[0].get("reference_speedup")
            out[tid] = {"complete": expected_best_of([float(s >= COMPLETE_FRAC * ref) for s in sp], budget) if ref else float("nan"),
                        "success": expected_best_of([float(s >= thr) for s in sp], budget),
                        "log_speedup": expected_best_of([math.log(s) for s in sp], budget),
                        "edits_used": float(budget) if budget <= len(sp) else float("nan"), "seed": seed}
    return out


def per_seed(task_vals: Dict[str, Dict[str, float]], keep: set, key: str) -> Dict[str, float]:
    acc: Dict[str, List[float]] = defaultdict(list)
    for tid, v in task_vals.items():
        if tid in keep and not math.isnan(v[key]):
            acc[str(v["seed"])].append(v[key])
    return {s: sum(xs) / len(xs) for s, xs in acc.items()}


def paired_ci(a: Dict[str, float], b: Dict[str, float], n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Mean over functions of (a - b) with a function-level bootstrap CI."""
    common = sorted(set(a) & set(b))
    if not common:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_functions": 0}
    d = [a[s] - b[s] for s in common]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n_boot))
    return {"point": sum(d) / len(d), "lo": boots[int(0.025 * n_boot)], "hi": boots[int(0.975 * n_boot) - 1], "n_functions": len(common)}


def analyse(runs: Dict[str, Tuple[str, List[Dict]]], budgets: List[int], reference: Optional[str], delta: float,
            thr: float = SUCCESS_THRESHOLD) -> Dict:
    common_tasks = set.intersection(*[{r["task_id"] for r in recs} for _, recs in runs.values()])
    result: Dict = {"n_tasks": len(common_tasks), "success_threshold": thr, "budgets": {}}
    for B in budgets:
        tv = {name: per_task(recs, kind, B, delta, thr) for name, (kind, recs) in runs.items()}
        seeds = {name: {k: per_seed(v, common_tasks, k) for k in ("success", "complete", "log_speedup", "edits_used")} for name, v in tv.items()}
        row: Dict = {}
        for name, s in seeds.items():
            fns = s["success"]
            if not fns:
                row[name] = None
                continue
            comp = s["complete"]
            row[name] = {"success": sum(fns.values()) / len(fns),
                         "complete": (sum(comp.values()) / len(comp)) if comp else float("nan"),
                         "gm_speedup": math.exp(sum(s["log_speedup"].values()) / len(s["log_speedup"])),
                         "edits_used": sum(s["edits_used"].values()) / len(s["edits_used"]),
                         "n_functions": len(fns)}
            if reference and name != reference and reference in seeds:
                ref = seeds[reference]
                row[name]["vs_reference"] = {
                    "delta_success": paired_ci(ref["success"], s["success"]),
                    "delta_complete": paired_ci(ref["complete"], s["complete"]),
                    "delta_log_speedup": paired_ci(ref["log_speedup"], s["log_speedup"]),
                }
        result["budgets"][B] = row
    return result


def print_report(res: Dict, reference: Optional[str]) -> None:
    print(f"{res['n_tasks']} common tasks; success = speedup >= {res.get('success_threshold', SUCCESS_THRESHOLD)}x; function-weighted means; CIs resample functions")
    for B, row in res["budgets"].items():
        print(f"\n--- budget B={B} ---")
        for name, v in row.items():
            if v is None:
                print(f"  {name:14s} (not available at this budget)")
                continue
            line = f"  {name:16s} success={v['success']:.3f}  complete={v['complete']:.3f}  gm_speedup={v['gm_speedup']:7.2f}  edits_used={v['edits_used']:.2f}  [{v['n_functions']} fns]"
            vr = v.get("vs_reference")
            if vr:
                ds, dl = vr["delta_success"], vr["delta_log_speedup"]
                dc = vr["delta_complete"]
                line += (f"   | {reference} minus this: success {ds['point']:+.3f} [{ds['lo']:+.3f},{ds['hi']:+.3f}]"
                         f"  complete {dc['point']:+.3f} [{dc['lo']:+.3f},{dc['hi']:+.3f}]"
                         f"  speedup x{math.exp(dl['point']):.2f} [x{math.exp(dl['lo']):.2f},x{math.exp(dl['hi']):.2f}]")
            print(line)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="name=path/to/*_trajectories.jsonl:multi|oneshot")
    p.add_argument("--budgets", nargs="+", type=int, default=[1, 2, 3, 4, 6])
    p.add_argument("--reference", default=None)
    p.add_argument("--delta", type=float, default=0.03, help="env improvement threshold (env.improvement_delta)")
    p.add_argument("--success_threshold", type=float, default=SUCCESS_THRESHOLD)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    runs: Dict[str, Tuple[str, List[Dict]]] = {}
    for spec in a.runs:
        name, rest = spec.split("=", 1)
        path, kind = rest.rsplit(":", 1)
        if kind not in ("multi", "oneshot"):
            raise SystemExit(f"{spec}: kind must be multi or oneshot")
        runs[name] = (kind, load_records(path))
    res = analyse(runs, a.budgets, a.reference, a.delta, a.success_threshold)
    print_report(res, a.reference)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
