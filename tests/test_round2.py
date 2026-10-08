"""Tests for the round-2 data tools: function split, difficulty filter, partial-fix filter."""
import json

from tinyperf.data.split_train import is_rl_function, split
from tinyperf.train.filter_tasks import filter_tasks
from tinyperf.train.sft_data import complete_enough


def _write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_split_keeps_functions_on_one_side(tmp_path):
    rows = [{"task_id": f"t{i}", "seed_name": f"fn{i % 20}", "families": ["hoist"]} for i in range(200)]
    _write(tmp_path / "train.jsonl", rows)
    res = split(str(tmp_path), rl_frac=0.4)
    assert res["sft"]["tasks"] + res["rl"]["tasks"] == 200
    sides = {}
    for side in ("sft", "rl"):
        for line in open(tmp_path / f"train_{side}.jsonl"):
            sides.setdefault(json.loads(line)["seed_name"], set()).add(side)
    assert all(len(v) == 1 for v in sides.values()) and len(sides) == 20
    assert 2 <= res["rl"]["functions"] <= 14                     # roughly 40% of 20
    assert is_rl_function("fn3", 0.4) == is_rl_function("fn3", 0.4)


def test_filter_keeps_middle_difficulty_and_unprobed(tmp_path):
    tasks = [{"task_id": f"{c}{i}", "seed_name": c, "families": ["x"]} for c in ("easy", "hard", "mid", "new") for i in range(3)]
    _write(tmp_path / "tasks.jsonl", tasks)
    probe = []
    for c, rate in (("easy", 1.0), ("hard", 0.0), ("mid", 0.5)):
        for i in range(10):
            ok = i < rate * 10
            probe.append({"task_id": f"{c}0", "seed_name": c, "families": ["x"], "best_ratio": 0.1 if ok else 1.0})
    _write(tmp_path / "probe.jsonl", probe)
    res = filter_tasks(str(tmp_path / "tasks.jsonl"), str(tmp_path / "probe.jsonl"), str(tmp_path / "out.jsonl"), 0.1, 0.9, 1.5, 8)
    kept = {json.loads(l)["seed_name"] for l in open(tmp_path / "out.jsonl")}
    assert kept == {"mid", "new"}
    assert res["chains_by_decision"] == {"too_easy": 1, "too_hard": 1, "keep": 1}
    assert res["kept_tasks"] == 6


class _T:
    def __init__(self, ref):
        self.reference_speedup = ref


def test_partial_fix_demonstrations_are_dropped():
    drops = {}
    assert complete_enough({"final_ratio": 0.1}, _T(10.0), 0.8, drops)          # 10x of a 10x reference
    assert not complete_enough({"final_ratio": 0.5}, _T(10.0), 0.8, drops)      # 2x of a 10x reference
    assert complete_enough({"final_ratio": 0.5}, _T(None), 0.8, drops)          # unknown reference: keep
    assert complete_enough({"final_ratio": 0.5}, _T(10.0), 0.0, drops)          # filter off
    assert drops == {"partial": 1}


def test_filter_by_task_can_drop_unprobed(tmp_path):
    tasks = [{"task_id": f"f{i}", "seed_name": "same", "families": ["x"]} for i in range(4)]
    _write(tmp_path / "tasks.jsonl", tasks)
    probe = []
    for tid, wins in (("f0", 0), ("f1", 2), ("f2", 4)):          # f3 is not probed
        probe += [{"task_id": tid, "seed_name": "same", "families": ["x"], "best_ratio": 0.5 if i < wins else 1.0} for i in range(4)]
    _write(tmp_path / "probe.jsonl", probe)
    res = filter_tasks(str(tmp_path / "tasks.jsonl"), str(tmp_path / "probe.jsonl"), str(tmp_path / "out.jsonl"),
                       0.2, 0.8, 1.05, 4, by="task", drop_unprobed=True)
    assert [json.loads(l)["task_id"] for l in open(tmp_path / "out.jsonl")] == ["f1"]
    assert res["tasks_by_decision"] == {"too_hard": 1, "keep": 1, "too_easy": 1, "unprobed": 1}
