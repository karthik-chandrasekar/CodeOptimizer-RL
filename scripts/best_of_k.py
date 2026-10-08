"""Best-of-k results from n verified attempts per task (exact, without re-sampling).

    python scripts/best_of_k.py --k 1 4 8 16 --runs s2m_rl=artifacts/distill/eval/k16_s2m_rl_real_trajectories.jsonl ...

Each attempt is checked by the sandbox, so taking the best of k attempts is safe.  For a task with n >= k attempts:
* P(best of k reaches threshold t) = 1 - C(n - c, k) / C(n, k), where c attempts reach t (the pass@k estimator);
* E[log best-of-k speedup] = sum_i P(the i-th smallest is the largest of k) * log s_(i), with that probability
  C(i - 1, k - 1) / C(n, k) when k of the n attempts are chosen at random.
Reported per run and k: success (>=1.5x), no speedup (best < 1.05x), >=5x, and the geometric-mean best speedup -
both per file (raw) and function-weighted (each function cluster counts once, as in the budget tables).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from math import comb
from typing import Dict, List, Optional


def p_reach(speedups: List[float], k: int, t: float) -> float:
    n, c = len(speedups), sum(s >= t for s in speedups)
    return 1.0 - comb(n - c, k) / comb(n, k)


def p_below(speedups: List[float], k: int, t: float) -> float:
    return 1.0 - p_reach(speedups, k, t)


def expected_log_best(speedups: List[float], k: int) -> float:
    s = sorted(speedups)
    n = len(s)
    return sum(comb(i - 1, k - 1) / comb(n, k) * math.log(s[i - 1]) for i in range(k, n + 1))


def summarize(path: str, ks: List[int]) -> Dict[int, Dict[str, float]]:
    by_task: Dict[str, List[float]] = defaultdict(list)
    cluster: Dict[str, str] = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            by_task[r["task_id"]].append(1.0 / max(float(r["best_ratio"]), 1e-9))
            cluster[r["task_id"]] = r.get("seed_name") or r["task_id"]
    out = {}
    for k in ks:
        rows = {tid: sp for tid, sp in by_task.items() if len(sp) >= k}
        if not rows:
            continue
        per = {tid: (p_reach(sp, k, 1.5), p_below(sp, k, 1.05), p_reach(sp, k, 5.0), expected_log_best(sp, k)) for tid, sp in rows.items()}
        groups: Dict[str, List[tuple]] = defaultdict(list)
        for tid, v in per.items():
            groups[cluster[tid]].append(v)
        fw = [tuple(sum(x[j] for x in g) / len(g) for j in range(4)) for g in groups.values()]
        raw = list(per.values())
        mean = lambda vals, j: sum(v[j] for v in vals) / len(vals)
        out[k] = {"files": len(raw), "functions": len(fw), "min_attempts": min(len(sp) for sp in rows.values()),
                  "success": mean(raw, 0), "none": mean(raw, 1), "ge5": mean(raw, 2), "gm": math.exp(mean(raw, 3)),
                  "success_fw": mean(fw, 0), "none_fw": mean(fw, 1), "ge5_fw": mean(fw, 2), "gm_fw": math.exp(mean(fw, 3))}
    return out


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="name=trajectories.jsonl")
    p.add_argument("--k", nargs="+", type=int, default=[1, 4, 8, 16])
    a = p.parse_args(argv)
    print("best of k verified attempts | success = best >= 1.5x | none = best < 1.05x | GM = geometric-mean best speedup")
    print(f"{'run':16s} {'k':>3s} | {'files':>5s} {'success':>8s} {'none':>6s} {'>=5x':>6s} {'GM':>6s} | function-weighted: "
          f"{'success':>8s} {'none':>6s} {'>=5x':>6s} {'GM':>6s}")
    for spec in a.runs:
        name, path = spec.split("=", 1)
        for k, s in summarize(path, a.k).items():
            print(f"{name:16s} {k:3d} | {s['files']:5d} {s['success']:8.1%} {s['none']:6.1%} {s['ge5']:6.1%} {s['gm']:5.2f}x | "
                  f"{'':19s}{s['success_fw']:8.1%} {s['none_fw']:6.1%} {s['ge5_fw']:6.1%} {s['gm_fw']:5.2f}x")


if __name__ == "__main__":
    main()
