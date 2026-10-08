"""Rejection fine-tuning data: cleaning successful episodes and replaying them."""
import json

from tinyperf.env.files import ENTRY, make_driver, with_driver
from tinyperf.env.task import Task
from tinyperf.train.rft_data import build, cleaned_edits

SRC = ("def a(xs):\n    return sum(xs)\n\n\n"
       "def b(xs):\n    out = []\n    for x in xs:\n        if x not in out:\n            out.append(x)\n    return out\n\n\n"
       "def c(xs):\n    out = []\n    for x in xs:\n        out = out + [x * 2]\n    return out\n")
B_FIX = ("def b(xs):\n    seen = set()\n    out = []\n    for x in xs:\n        if x not in seen:\n"
         "            seen.add(x)\n            out.append(x)\n    return out\n")
C_FIX = "def c(xs):\n    return [x * 2 for x in xs]\n"
A_EDIT = "def a(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total\n"


def _task(meta_extra=None, **kw):
    meta = {"kind": "file", "driver": make_driver(["a", "b", "c"]), "functions": ["a", "b", "c"],
            "slow_functions": ["b", "c"], "profile0": {"a": 0.05, "b": 0.25, "c": 0.70}}
    meta.update(meta_extra or {})
    return Task("f0", ENTRY, SRC, kw.get("correct", []), kw.get("perf", []), kw.get("expected", []), kw.get("ns", 1.0), meta=meta)


def _file(*defs):
    import ast
    from tinyperf.env.files import splice_functions
    out = SRC
    for d in defs:
        out = splice_functions(out, d)
    return ast.unparse(ast.parse(out))


def test_cleaning_keeps_changed_slow_functions_hottest_first():
    rec = {"best_code": _file(A_EDIT, B_FIX, C_FIX)}          # also rewrote the fast function `a`
    edits = cleaned_edits(_task(), rec)
    assert [e.split("(")[0] for e in edits] == ["def c", "def b"]   # `a` dropped; c (70%) before b (25%)
    assert cleaned_edits(_task(), {"best_code": SRC}) == []          # nothing changed
    assert cleaned_edits(_task(), {"best_code": "def (("}) == []     # unparseable


def test_build_replays_successes_and_rejects_failures(tmp_path):
    from tests.conftest import fast_env_cfg
    from tinyperf.common.config import load_config
    from tinyperf.env.executor import Executor
    from tinyperf.env.files import profile_shares
    from tinyperf.env.task import save_tasks

    cfg = load_config(None, [])
    cfg.env = fast_env_cfg()
    ex = Executor(cfg.env)
    drv = make_driver(["a", "b", "c"])
    correct = [(0, [1, 2]), (1, [1, 1, 2]), (2, [3]), (1, []), (2, [])]
    perf = [(0, list(range(2000)))] * 2 + [(1, list(range(500)))] * 2 + [(2, list(range(3000)))] * 2
    base = ex.baseline(with_driver(SRC, drv), ENTRY, correct, perf, group_by_first_arg=True)
    ex.close()
    prof = profile_shares(base["candidate_group_ns"], ["a", "b", "c"])
    task = _task({"profile0": prof}, correct=correct, perf=perf, expected=base["expected"], ns=base["candidate_median_ns"])
    save_tasks(str(tmp_path / "tasks.jsonl"), [task])
    fixed = _file(B_FIX, C_FIX)
    recs = [{"task_id": "f0", "best_ratio": 0.2, "best_code": fixed, "steps": []},     # success
            {"task_id": "f0", "best_ratio": 0.2, "best_code": fixed, "steps": []},     # duplicate success
            {"task_id": "f0", "best_ratio": 1.0, "best_code": SRC, "steps": []},       # failure
            {"task_id": "other", "best_ratio": 0.1, "best_code": fixed, "steps": []}]  # not a training task
    with open(tmp_path / "samples.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    stats = build(cfg, str(tmp_path / "tasks.jsonl"), str(tmp_path / "samples.jsonl"), str(tmp_path / "rft.jsonl"), 1.2, 2)
    rows = [json.loads(l) for l in open(tmp_path / "rft.jsonl")]
    assert stats["episodes"] == 3 and stats["successes"] == 2 and stats["jobs"] == 1 and len(rows) == 1
    row = rows[0]
    assert row["method"] == "rft" and [s["status"] for s in row["steps"]] == ["ok", "ok", "stop"]
    hottest = max(("b", "c"), key=lambda n: prof[n])
    assert row["steps"][0]["action"].split("def ")[1].startswith(hottest + "(")       # hottest slow function first
    assert 1 / row["final_ratio"] >= 1.2 and stats["first_edit_is_hottest_slow_function"] == 1.0
