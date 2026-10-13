"""Whole-policy metrics, cluster-weighted estimates and paired cluster-bootstrap comparisons.

    python -m tinyperf_chain.eval.aggregate --runs artifacts/chain/runs/dev/episodes.jsonl --paired \\
        --cluster-bootstrap --reference B4 [--out report.json] [--ablation-pairs B4:B3 B4:B1]

Besides the per-system table the report carries (15.2, 15.4, 15.5): the quality-versus-measured-cost frontier
(success and GM speedup against mean tokens, model latency, verification CPU and search wall); for every
``--ablation-pair full:ablated`` the probability that the ablated module changes the episode outcome; the useful
repair rate (accepted repairs per repair attempt); control-failure reasons by role; and, when several bundles are
replicates of one system (``replicate_of`` in the bundle, ``system_group`` in the records), the across-training-seed
interval of the group mean, reported separately from the checkpoint-conditional cluster bootstrap.

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
    sw = [float(r.get("search_wall_ns") or 0) / 1e9 for r in ok]
    m["search_wall_s_median"] = _pct(sw, 0.5)
    m["search_wall_s_p95"] = _pct(sw, 0.95)
    m["audit_wall_s_mean"] = sum(r.get("audit_wall_ns", 0) for r in ok) / 1e9 / len(ok)
    m["model_latency_s_mean"] = sum(r.get("model_latency_ns", 0) for r in ok) / 1e9 / len(ok)
    m["verify_cpu_s_mean"] = sum(float(r.get("verify_cpu_s") or 0) for r in ok) / len(ok)
    term: Dict[str, int] = defaultdict(int)
    for r in ok:
        term[r.get("termination_reason") or "?"] += 1
    m["termination"] = dict(sorted(term.items()))
    reasons: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in ok:
        for role, d in (r.get("control_failure_reasons") or {}).items():
            for k, v in d.items():
                reasons[role][k] += int(v)
    m["control_failure_reasons"] = {role: dict(sorted(d.items())) for role, d in sorted(reasons.items())}
    rep_att = sum(float(r.get("repairs") or 0) for r in ok)
    m["useful_repair_rate"] = (sum(float(r.get("accepted_repairs") or 0) for r in ok) / rep_att) if rep_att else None
    m["mean_prepare_s"] = sum(float(r.get("prepare_ns") or 0) for r in ok) / 1e9 / len(ok)
    m["mean_parse_s"] = sum(float(r.get("parse_ns") or 0) for r in ok) / 1e9 / len(ok)
    m["host_peak_rss_mb"] = max((float(r.get("host_peak_rss_mb") or 0) for r in ok), default=None)
    lat = [r.get("model_latency_ns", 0) / 1e9 / max(1, r.get("role_calls") or 1) for r in ok if r.get("role_calls")]
    m["model_latency_per_call_s_median"] = _pct(lat, 0.5)
    m["model_latency_per_call_s_p95"] = _pct(lat, 0.95)
    if session and session.get("interval_s"):
        n_sys = int(session.get("n_systems") or len(session.get("systems") or {}) or 1)
        rate = sum(_success(r) for r in ok) / session["interval_s"]
        if n_sys <= 1:
            m["verified_successes_per_s"] = rate      # saturated throughput: successes / full measurement interval
        else:
            # a paired session interleaves several systems: the shared interval is not one system's throughput
            m["successes_per_s_shared_interval"] = rate
            m["throughput_note"] = f"interval shared by {n_sys} systems; run one system per session for saturated throughput"
    return m


COST_AXES = ("mean_input_tokens", "mean_output_tokens", "model_latency_s_mean", "verify_cpu_s_mean",
             "search_wall_s_median", "mean_role_calls", "mean_patch_attempts")


def cost_frontier(systems: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Quality versus measured cost, one row per system@slots (15.2 'plot quality versus measured cost')."""
    rows = []
    for name, m in systems.items():
        if "success_cluster_macro" not in m:
            continue
        rows.append({"system": name, "success": m["success_cluster_macro"], "gm_speedup": m["gm_speedup_cluster_macro"],
                     **{k: m.get(k) for k in COST_AXES}})
    return rows


def outcome_change(full: Sequence[Dict[str, Any]], ablated: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Probability that replacing a module changes the episode outcome: paired by (task, seed, slots), an outcome is
    (audit success, effective speedup rounded to 2 decimals)."""
    def key(r):
        return (r["task_id"], r.get("seed"), r.get("patch_slots_max"))

    def out(r):
        return (bool(r.get("audit_pass")) and (r.get("raw_audit_speedup") or 0) >= SUCCESS, round(float(r.get("effective_speedup") or 1.0), 2))

    a = {key(r): out(r) for r in full if not r.get("failed_run")}
    b = {key(r): out(r) for r in ablated if not r.get("failed_run")}
    shared = sorted(set(a) & set(b))
    if not shared:
        return {"n": 0}
    changed = sum(1 for k in shared if a[k] != b[k])
    succ_changed = sum(1 for k in shared if a[k][0] != b[k][0])
    return {"n": len(shared), "p_outcome_changed": changed / len(shared), "p_success_changed": succ_changed / len(shared)}


def replicate_intervals(records: Sequence[Dict[str, Any]], f, level: float = 0.95) -> Dict[str, Any]:
    """Across-training-seed variability (15.5): per replicate group, the mean and a t-interval of the per-bundle
    cluster-macro values; needs >= 2 bundles in a group (>= 3 recommended)."""
    import math as _m

    groups: Dict[Tuple[str, int], Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.get("failed_run"):
            continue
        g = r.get("system_group") or r["system_id"]
        groups[(g, int(r.get("patch_slots_max") or 0))][r["system_id"]].append(r)
    out: Dict[str, Any] = {}
    for (g, slots), by_bundle in sorted(groups.items()):
        if len(by_bundle) < 2:
            continue
        vals = [hierarchical(rs, f)[0] for rs in by_bundle.values()]
        n = len(vals)
        mu = sum(vals) / n
        sd = _m.sqrt(sum((v - mu) ** 2 for v in vals) / (n - 1))
        # two-sided t quantiles for small n (df = n-1), 95%
        tq = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}.get(n - 1, 1.96)
        half = tq * sd / _m.sqrt(n)
        out[f"{g}@{slots}"] = {"n_bundles": n, "bundles": sorted(by_bundle), "mean": mu, "sd": sd, "lo": mu - half,
                               "hi": mu + half, "per_bundle": dict(zip(sorted(by_bundle), [hierarchical(by_bundle[k], f)[0] for k in sorted(by_bundle)]))}
    return out


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
              n_boot: int = 2000, session: Optional[Dict[str, Any]] = None,
              ablation_pairs: Sequence[Tuple[str, str]] = ()) -> Dict[str, Any]:
    by_sys: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_sys[(r["system_id"], int(r.get("patch_slots_max") or 0))].append(r)
    report: Dict[str, Any] = {"systems": {}}
    for (s, slots), rs in sorted(by_sys.items()):
        report["systems"][f"{s}@{slots}"] = system_metrics(rs, session)
    report["cost_frontier"] = cost_frontier(report["systems"])
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
    if ablation_pairs:
        report["ablations"] = {}
        for full, abl in ablation_pairs:
            for (s, slots), rs in sorted(by_sys.items()):
                if s != full:
                    continue
                other = by_sys.get((abl, slots))
                if other:
                    report["ablations"][f"{full}-vs-{abl}@{slots}"] = outcome_change(rs, other)
    rep_s = replicate_intervals(records, _success)
    rep_g = replicate_intervals(records, _log_eff)
    if rep_s:
        report["replicates"] = {"success": rep_s, "log_speedup": rep_g,
                                "note": "across-training-seed intervals (t, df=n_bundles-1); checkpoint-conditional intervals are in paired_vs_*"}
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
    p.add_argument("--runs", nargs="*", default=[], help="episodes.jsonl files (default: artifacts/chain/runs/development/episodes.jsonl)")
    p.add_argument("--reference", default=None)
    p.add_argument("--paired", action="store_true")
    p.add_argument("--cluster-bootstrap", action="store_true", help="(default) paired intervals resample clusters")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--session", default=None)
    p.add_argument("--ablation-pairs", nargs="*", default=[], help="full:ablated system-id pairs, e.g. B4:B3")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if not a.runs:
        default = os.path.join("artifacts", "chain", "runs", "development", "episodes.jsonl")
        if not os.path.exists(default):
            raise SystemExit(f"no --runs given and {default} does not exist")
        a.runs = [default]
    recs = load_records(a.runs)
    pairs = []
    for pr in a.ablation_pairs:
        if ":" not in pr:
            raise SystemExit(f"--ablation-pairs entries look like FULL:ABLATED, got {pr!r}")
        pairs.append(tuple(pr.split(":", 1)))
    session = None
    sp = a.session or os.path.join(os.path.dirname(a.runs[0]), "session.json")
    if os.path.exists(sp):
        with open(sp) as f:
            session = json.load(f)
    ref = a.reference
    if a.paired and not ref:
        ref = sorted({r["system_id"] for r in recs})[0]
    rep = aggregate(recs, ref, a.paired, a.n_boot, session, pairs)
    print(format_table(rep))
    if a.paired:
        print(json.dumps(rep.get("paired_vs_" + ref, {}), indent=1))
    out = a.out or os.path.join(os.path.dirname(a.runs[0]), "report.json")
    with open(out, "w") as f:
        json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
