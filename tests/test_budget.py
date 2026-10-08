"""Tests for the budget-matched comparison."""
import itertools
import math
import random

from tinyperf.eval.budget import analyse, expected_best_of, paired_ci, truncated_best


def test_expected_best_of_matches_enumeration():
    rng = random.Random(0)
    for n in range(1, 8):
        xs = sorted(rng.random() for _ in range(n))
        for k in range(1, n + 1):
            brute = sum(max(c) for c in itertools.combinations(xs, k)) / math.comb(n, k)
            assert abs(expected_best_of(xs, k) - brute) < 1e-12
    assert math.isnan(expected_best_of([1.0, 2.0], 3))


def _step(kind, status="ok", ratio=None):
    return {"step": 0, "action_kind": kind, "status": status, "correct": status == "ok", "ratio": ratio, "improved": False}


def test_truncation_replays_best_tracking():
    rec = {"steps": [_step("edit", "incorrect"), _step("edit", ratio=0.5), _step("edit", ratio=0.49), _step("edit", ratio=0.2), _step("stop")]}
    assert truncated_best(rec, 1, 0.03) == (1.0, 1)
    assert truncated_best(rec, 2, 0.03) == (0.5, 2)
    assert truncated_best(rec, 3, 0.03) == (0.5, 3)      # 0.49 is not a >=3% improvement on 0.5
    assert truncated_best(rec, 6, 0.03) == (0.2, 4)      # STOP after 4 actions: budget left unused


def test_analyse_agent_vs_best_of_k():
    # 3 functions x 2 renamed tasks. Agent: fails first edit, succeeds (2x) on the second.
    # One-shot: 4 samples per task, one of which succeeds (2x).
    recs_multi, recs_one = [], []
    for f in "abc":
        for t in range(2):
            tid = f"{f}{t}"
            recs_multi.append({"task_id": tid, "seed_name": f, "best_ratio": 0.5,
                               "steps": [_step("edit", "incorrect"), _step("edit", ratio=0.5), _step("stop")]})
            for i in range(4):
                recs_one.append({"task_id": tid, "seed_name": f, "best_ratio": 0.5 if i == 0 else 1.0, "steps": []})
    res = analyse({"agent": ("multi", recs_multi), "rl1": ("oneshot", recs_one)}, [1, 2, 4], "agent", 0.03, 1.05)
    b1, b2, b4 = (res["budgets"][b] for b in (1, 2, 4))
    assert b1["agent"]["success"] == 0.0 and abs(b1["rl1"]["success"] - 0.25) < 1e-12
    assert b2["agent"]["success"] == 1.0 and abs(b2["rl1"]["success"] - 0.5) < 1e-12   # 1 - C(3,2)/C(4,2)
    assert b4["rl1"]["success"] == 1.0 and b4["agent"]["edits_used"] == 2.0
    d = b2["rl1"]["vs_reference"]["delta_success"]
    assert abs(d["point"] - 0.5) < 1e-12 and d["n_functions"] == 3
    assert abs(math.exp(b2["agent"]["gm_speedup"] and math.log(b2["agent"]["gm_speedup"])) - 2.0) < 1e-9


def test_paired_ci_is_zero_width_for_constant_differences():
    ci = paired_ci({"a": 1.0, "b": 0.5}, {"a": 0.5, "b": 0.0})
    assert ci["point"] == ci["lo"] == ci["hi"] == 0.5
