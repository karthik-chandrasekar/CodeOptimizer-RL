"""T24 metrics: denominators, cluster weights, audit failures and below-one speedups."""
import math

from tinyperf_chain.eval.aggregate import aggregate, hierarchical, paired, system_metrics, _log_eff, _success
from tinyperf_chain.sandbox.timing import bootstrap_median_interval, timing_verdict


def rec(sys, cluster, task, audit_pass, raw, unchanged=False, failed=False):
    eff = (raw if audit_pass else 1.0) if raw is not None else 1.0
    return {"system_id": sys, "cluster_id": cluster, "task_id": task, "audit_pass": audit_pass, "raw_audit_speedup": raw,
            "effective_speedup": eff, "audit_unchanged": unchanged, "failed_run": failed, "patch_slots_max": 6,
            "search_wall_ns": 1e9, "audit_wall_ns": 0}


def test_t24_cluster_macro_and_effective_speedups():
    rs = [rec("A", "c1", "t1", True, 2.0), rec("A", "c1", "t1", True, 1.0, unchanged=True),
          rec("A", "c1", "t2", False, None), rec("A", "c2", "t3", True, 0.5),
          rec("A", "c2", "t3", True, 1.6), rec("A", "c3", "t4", True, 3.0, failed=True)]
    m = system_metrics(rs)
    assert m["failed_runs"] == 1 and m["n_episodes"] == 5
    # c1: files t1 (mean 0.5) and t2 (0) -> 0.25 ; c2: t3 (0.5) -> 0.5 ; macro = 0.375
    assert abs(m["success_cluster_macro"] - 0.375) < 1e-12
    # audit failure counts as 1.0, a correct but slower program keeps 0.5
    lg, per = hierarchical([r for r in rs if not r["failed_run"]], _log_eff)
    assert abs(per["c2"] - 0.5 * (math.log(0.5) + math.log(1.6))) < 1e-12
    assert abs(m["audit_failure_rate"] - 1 / 4) < 1e-12 and abs(m["correct_but_slower_rate"] - 1 / 4) < 1e-12


def test_paired_bootstrap_resamples_clusters():
    a = [rec("A", f"c{i}", f"t{i}", True, 2.0) for i in range(10)]
    b = [rec("B", f"c{i}", f"t{i}", True, 1.2) for i in range(10)]
    d = paired(a, b, _success, n_boot=500)
    assert d["n_clusters"] == 10 and d["diff"] == 1.0 and d["lo"] == d["hi"] == 1.0
    rep = aggregate(a + b, "B")
    assert "A@6" in rep["systems"] and rep["paired_vs_B"]["A@6"]["success"]["diff"] == 1.0


def test_timing_rule():
    v = timing_verdict([100.0] * 5, [80.0, 81.0, 79.0, 80.0, 82.0], threshold=1.03)
    assert v.decision == "accept" and v.lo >= 1.03
    v = timing_verdict([100.0] * 5, [99.0, 101.0, 100.0, 98.0, 102.0])
    assert v.decision in ("uncertain", "no_gain")
    v = timing_verdict([100.0] * 5, [130.0] * 5)
    assert v.decision == "slower" and v.speedup < 1
    v = timing_verdict([100.0] * 5, [96.0] * 5, rule="legacy_ratio")
    assert v.decision == "accept"
    lo, hi = bootstrap_median_interval([0.1, 0.2, 0.3], draws=200, seed=1)
    assert lo <= 0.2 <= hi
