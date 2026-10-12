"""Whole-policy metrics, cluster-weighted estimates and paired cluster-bootstrap comparisons.

    python -m tinyperf_chain.eval.aggregate --runs artifacts/chain/runs/dev/episodes.jsonl --paired \\
        --cluster-bootstrap --reference B4 [--out report.json]

Primary endpoint (eq. 9), a cluster-macro success rate::

    p_b = 1/C sum_c 1/N_c sum_{j in c} 1/R_j sum_r 1{audit_pass_jr and S_jr >= 1.5}

The geometric-mean speedup uses the same hierarchical weights on log *effective* speedups (unchanged programs and
audit failures count as 1.0; correct but slower audited programs keep their ratio below 1).  File-level unweighted
numbers are reported separately.  Failed runs (infrastructure) are excluded from the denominators and reported;
they are never counted as candidate failures.  Paired comparisons resample clusters (all files and episodes of a
cluster move together) on per-cluster differences between two systems evaluated on the same tasks.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SUCCESS = 1.5


def load_records(paths: Sequence[str]) -> List[Dict[str, Any]]:
    out = []
    for p in paths:
        with open(p) as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
    return out


def _success(r: Dict[str, Any]) -> float:
    return float(bool(r.get("audit_pass")) and (r.get("raw_audit_speedup") or 0.0) >= SUCCESS)


def _log_eff(r: Dict[str, Any]) -> float:
    s = r.get("effective_speedup")
    if s is None or not r.get("audit_pass"):
        s = 1.0
    return math.log(max(float(s), 1e-9))


def hierarchical(records: Sequence[Dict[str, Any]], f) -> Tuple[float, Dict[str, float]]:
    """Cluster-macro mean of f with files weighted equally inside a cluster and episodes inside a file."""
    by_c: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_c[r["cluster_id"]][r["task_id"]].append(f(r))
    per_cluster = {c: sum(sum(v) / len(v) for v in files.values()) / len(files) for c, files in by_c.items()}
    if not per_cluster:
        return float("nan"), {}
    return sum(per_cluster.values()) / len(per_cluster), per_cluster


def _pct(xs: Sequence[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def system_metrics(records: Sequence[Dict[str, Any]], session: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ok = [r for r in records if not r.get("failed_run")]
    m: Dict[str, Any] = {"n_episodes": len(ok), "failed_runs": len(records) - len(ok),
                         "n_files": len({r["task_id"] for r in ok}), "n_clusters": len({r["cluster_id"] for r in ok})}
    if not ok:
        return m
    p, _ = hierarchical(ok, _success)
    lg, _ = hierarchical(ok, _log_eff)
    m["success_cluster_macro"] = p
    m["gm_speedup_cluster_macro"] = math.exp(lg)
    m["success_file_level"] = sum(_success(r) for r in ok) / len(ok)
    m["gm_speedup_file_level"] = math.exp(sum(_log_eff(r) for r in ok) / len(ok))
    changed = [r for r in ok if not r.get("audit_unchanged")]
    m["changed_rate"] = len(changed) / len(ok)
    m["audit_failure_rate"] = (sum(1 for r in changed if r.get("audit_pass") is False) / len(changed)) if changed else 0.0
    m["correct_but_slower_rate"] = (sum(1 for r in changed if r.get("audit_pass") and (r.get("raw_audit_speedup") or 1) < 1.0)
                                    / len(changed)) if changed else 0.0
    for k in ("accepted_edits", "patch_attempts", "role_calls", "input_tokens", "output_tokens", "extra_probe_calls",
              "repairs", "control_failures"):
        m[f"mean_{k}"] = sum(float(r.get(k) or 0) for r in ok) / len(ok)
    roles: Dict[str, float] = defaultdict(float)
    for r in ok:
        for k, v in (r.get("calls_by_role") or {}).items():
            roles[k] += v
    m["mean_calls_by_role"] = {k: v / len(ok) for k, v in sorted(roles.items())}
    sw = [r["search_wall_ns"] / 1e9 for r in ok]
    m["search_wall_s_median"] = _pct(sw, 0.5)
    m["search_wall_s_p95"] = _pct(sw, 0.95)
    m["audit_wall_s_mean"] = sum(r.get("audit_wall_ns", 0) for r in ok) / 1e9 / len(ok)
    m["model_latency_s_mean"] = sum(r.get("model_latency_ns", 0) for r in ok) / 1e9 / len(ok)
    m["verify_cpu_s_mean"] = sum(float(r.get("verify_cpu_s") or 0) for r in ok) / len(ok)
    term: Dict[str, int] = defaultdict(int)
    for r in ok:
        term[r.get("termination_reason") or "?"] += 1
    m["termination"] = dict(sorted(term.items()))
    if session and session.get("interval_s"):
        m["verified_successes_per_s"] = sum(_success(r) for r in ok) / session["interval_s"]
    return m


def paired(records_a: Sequence[Dict[str, Any]], records_b: Sequence[Dict[str, Any]], f, n_boot: int = 2000,
           seed: int = 0, level: float = 0.95) -> Dict[str, Any]:
    """Per-cluster difference (a - b) on shared (cluster, task) pairs; cluster bootstrap interval of the macro mean."""
    a = [r for r in records_a if not r.get("failed_run")]
    b = [r for r in records_b if not r.get("failed_run")]
    shared = {(r["cluster_id"], r["task_id"]) for r in a} & {(r["cluster_id"], r["task_id"]) for r in b}
    a = [r for r in a if (r["cluster_id"], r["task_id"]) in shared]
    b = [r for r in b if (r["cluster_id"], r["task_id"]) in shared]
    _, ca = hierarchical(a, f)
    _, cb = hierarchical(b, f)
    clusters = sorted(set(ca) & set(cb))
    if not clusters:
        return {"n_clusters": 0}
    diffs = [ca[c] - cb[c] for c in clusters]
    point = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    boots = sorted(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs) for _ in range(n_boot))
    lo = boots[int((1 - level) / 2 * n_boot)]
    hi = boots[min(n_boot - 1, int((1 + level) / 2 * n_boot))]
    return {"n_clusters": len(clusters), "n_files": len({t for _, t in shared}), "diff": point, "lo": lo, "hi": hi}


def aggregate(records: Sequence[Dict[str, Any]], reference: Optional[str] = None, do_paired: bool = True,
              n_boot: int = 2000, session: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    by_sys: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_sys[(r["system_id"], int(r.get("patch_slots_max") or 0))].append(r)
    report: Dict[str, Any] = {"systems": {}}
    for (s, slots), rs in sorted(by_sys.items()):
        report["systems"][f"{s}@{slots}"] = system_metrics(rs, session)
    if do_paired and reference:
        report["paired_vs_" + reference] = {}
        for (s, slots), rs in sorted(by_sys.items()):
            ref = by_sys.get((reference, slots))
            if s == reference or not ref:
                continue
            report["paired_vs_" + reference][f"{s}@{slots}"] = {
                "success": paired(rs, ref, _success, n_boot),
                "log_speedup": paired(rs, ref, _log_eff, n_boot),
            }
    return report


def format_table(report: Dict[str, Any]) -> str:
    keys = ["n_episodes", "n_clusters", "success_cluster_macro", "gm_speedup_cluster_macro", "audit_failure_rate",
            "mean_patch_attempts", "mean_role_calls", "mean_output_tokens", "search_wall_s_median"]
    head = f"{'system':16s}" + "".join(f"{k[:14]:>16s}" for k in keys)
    lines = [head, "-" * len(head)]
    for s, m in report["systems"].items():
        cells = []
        for k in keys:
            v = m.get(k, float("nan"))
            cells.append(f"{v:16.3f}" if isinstance(v, float) else f"{str(v):>16s}")
        lines.append(f"{s:16s}" + "".join(cells))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--reference", default=None)
    p.add_argument("--paired", action="store_true")
    p.add_argument("--cluster-bootstrap", action="store_true", help="(default) paired intervals resample clusters")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--session", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    recs = load_records(a.runs)
    session = None
    sp = a.session or os.path.join(os.path.dirname(a.runs[0]), "session.json")
    if os.path.exists(sp):
        with open(sp) as f:
            session = json.load(f)
    ref = a.reference
    if a.paired and not ref:
        ref = sorted({r["system_id"] for r in recs})[0]
    rep = aggregate(recs, ref, a.paired, a.n_boot, session)
    print(format_table(rep))
    if a.paired:
        print(json.dumps(rep.get("paired_vs_" + ref, {}), indent=1))
    out = a.out or os.path.join(os.path.dirname(a.runs[0]), "report.json")
    with open(out, "w") as f:
        json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
