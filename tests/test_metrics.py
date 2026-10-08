import math

from tinyperf.eval.metrics import compare, compute_metrics, per_family_metrics, steps_to_best


def rec(task_id, ratio, steps, families=("f",), ref=2.0, stopped=True, recovered=False):
    edits = [s for s in steps if s["action_kind"] == "edit"]
    return {"task_id": task_id, "families": list(families), "seed_name": "s", "reference_speedup": ref, "best_ratio": ratio,
            "n_edits": len(edits), "n_invalid": sum(1 for s in edits if not s["correct"]), "n_correct": sum(1 for s in edits if s["correct"]),
            "stopped": stopped, "recovered": recovered, "reward": math.log(1 / ratio) - 0.015 * len(edits), "horizon": 6, "steps": steps}


def e(step, status="ok", ratio=None, improved=False):
    return {"step": step, "action_kind": "edit", "status": status, "correct": status == "ok", "ratio": ratio, "improved": improved}


def test_compute_metrics_basic():
    recs = [
        rec("a", 0.5, [e(1, "ok", 0.5, True), {"step": 1, "action_kind": "stop", "status": "stop", "correct": True, "ratio": None, "improved": False}]),
        rec("b", 1.0, [e(1, "incorrect"), e(2, "ok", 1.2, False)], stopped=False),
        rec("c", 0.8, [e(1, "incorrect"), e(2, "ok", 0.8, True)], recovered=True),
    ]
    m = compute_metrics(recs)
    assert m["n_episodes"] == 3 and m["n_tasks"] == 3
    assert abs(m["success_rate"] - 2 / 3) < 1e-9
    assert abs(m["gm_speedup"] - math.exp((math.log(2) + 0 + math.log(1.25)) / 3)) < 1e-9
    assert m["p_speedup_ge_2.0"] == 1 / 3 and m["p_speedup_ge_1.05"] == 2 / 3
    assert m["regression_rate"] == 0.0
    assert abs(m["invalid_edit_rate"] - 2 / 5) < 1e-9
    assert abs(m["recovery_rate"] - 0.5) < 1e-9          # of the two episodes with a failure, one recovered
    assert abs(m["stop_rate"] - 2 / 3) < 1e-9
    assert abs(m["mean_edits_to_best"] - 1.5) < 1e-9     # successes reached best at edit 1 and edit 2
    assert 0 < m["reference_fraction"] < 1
    assert m["status_counts"]["incorrect"] == 2
    assert steps_to_best(recs[1]) is None


def test_per_family_and_compare():
    recs = [rec("a", 0.5, [e(1, "ok", 0.5, True)], families=("x",)), rec("b", 1.0, [e(1, "incorrect")], families=("y",))]
    fam = per_family_metrics(recs)
    assert fam["x"]["success_rate"] == 1.0 and fam["y"]["success_rate"] == 0.0
    d = compare(compute_metrics(recs[:1]), compute_metrics(recs[1:]))
    assert d["delta_success_rate"] == 1.0 and d["ratio_gm_speedup"] > 1.0
