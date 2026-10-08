"""Distillation: student sizes, SeqKD rows, and their use by the SFT trainer."""
import os
from types import SimpleNamespace

import pytest
import yaml

from tinyperf.env.protocol import Action, format_action
from tinyperf.train.seqkd_data import canonical_action, episode_to_row

torch = pytest.importorskip("torch")


def test_student_configs_hit_their_parameter_budgets():
    from tinyperf.common.config import ModelConfig
    from tinyperf.model.transformer import TinyPerfLM
    n = {}
    for name in ("student2m", "student200k"):
        cfg = ModelConfig(**yaml.safe_load(open(f"configs/model/{name}.yaml")))
        n[name] = (TinyPerfLM(cfg).n_params(), cfg.vocab_size, cfg.max_seq_len)
    assert 1.9e6 <= n["student2m"][0] <= 2.3e6 and n["student2m"][1:] == (8192, 2048)
    assert 1.8e5 <= n["student200k"][0] <= 2.2e5 and n["student200k"][1:] == (1024, 4096)


EDIT = "<EDIT>\ndef f(xs):\n    return sorted(set(xs))\n</EDIT>"


def _episode():
    task = SimpleNamespace(task_id="t0", source="def f(xs):\n    return sorted(list(set(xs)))", entry="f", families=["x"])
    turns = [SimpleNamespace(obs="<CODE>\nA\n</CODE>\n<STATE>\nstep=0\n</STATE>", action_text=EDIT + "\n", finished=True),
             SimpleNamespace(obs="<CODE>\nB\n</CODE>\n<STATE>\nstep=1\n</STATE>", action_text="<EDIT>\ndef f(", finished=False),
             SimpleNamespace(obs="<CODE>\nB\n</CODE>\n<STATE>\nstep=2\n</STATE>", action_text="<STOP>", finished=True)]
    steps = [SimpleNamespace(status="ok", ratio=0.5), SimpleNamespace(status="malformed", ratio=None), SimpleNamespace(status="stop", ratio=None)]
    summary = SimpleNamespace(best_ratio=0.5, best_code="def f(xs):\n    return sorted(set(xs))", n_edits=2, n_invalid=1, steps=steps)
    return SimpleNamespace(task=task, turns=turns, summary=summary, reward=0.6)


def test_seqkd_rows_keep_teacher_text_and_statuses():
    row = episode_to_row(_episode(), horizon=6)
    assert row["method"] == "seqkd" and row["task_id"] == "t0" and row["final_ratio"] == 0.5
    assert [s["status"] for s in row["steps"]] == ["ok", "malformed", "stop"]
    assert [s["finished"] for s in row["steps"]] == [True, False, True]
    assert row["steps"][0]["action"] == canonical_action(EDIT) and row["steps"][0]["action"].startswith("<EDIT>")
    assert row["steps"][2]["action"] == format_action(Action("stop"))
    assert [s["remaining"] for s in row["steps"]] == [6, 5, 4]


def test_sft_trains_only_on_the_teachers_valid_actions():
    path = os.environ.get("TINYPERF_TEST_TOKENIZER")
    if not path or not os.path.exists(path):
        pytest.skip("set TINYPERF_TEST_TOKENIZER to a trained tokenizer")
    from tinyperf.model.tokenizer import CodeTokenizer
    from tinyperf.train.sft import build_examples
    ex, stats = build_examples([episode_to_row(_episode(), horizon=6)], CodeTokenizer(path), 2048, "full", False, True)
    assert len(ex) == 2 and stats["failed_actions_skipped"] == 1     # the malformed step is masked


def test_compare_to_teacher_counts_wins_by_task(tmp_path):
    import json, math, sys
    sys.path.insert(0, "scripts")
    from compare_to_teacher import compare, load

    def write(name, speedups):
        p = tmp_path / name
        with open(p, "w") as f:
            for (tid, seed), sps in speedups.items():
                for s in sps:
                    f.write(json.dumps({"task_id": tid, "seed_name": seed, "best_ratio": 1 / s}) + "\n")
        return str(p)
    teacher = write("t.jsonl", {("a", "fa"): [2.0, 2.0], ("b", "fb"): [3.0, 3.0]})
    student = write("s.jsonl", {("a", "fa"): [4.0, 4.0], ("b", "fb"): [3.0, 3.05]})
    c = compare(load(teacher), load(student), margin=0.05, n_boot=200)
    assert c["tasks"] == 2 and c["win"] == 0.5 and c["loss"] == 0.0 and c["tie"] == 0.5
    assert abs(c["ratio"] - math.exp((math.log(2) + math.log(math.sqrt(3 * 3.05) / 3)) / 2)) < 1e-9
    assert abs(c["max_student"] - 4.0) < 1e-9 and abs(c["max_teacher"] - 3.0) < 1e-9 and c["top"][0][0] == "a"


def test_best_of_k_matches_hand_worked_example():
    import math, sys
    sys.path.insert(0, "scripts")
    from best_of_k import expected_log_best, p_below, p_reach
    sp = [1.0, 1.0, 2.0, 4.0]
    assert p_reach(sp, 1, 1.5) == 0.5 and p_reach(sp, 4, 1.5) == 1.0
    assert abs(p_reach(sp, 2, 1.5) - (1 - 1 / 6)) < 1e-12          # only the pair of 1.0x attempts fails
    assert abs(p_below(sp, 2, 1.05) - 1 / 6) < 1e-12
    assert abs(math.exp(expected_log_best(sp, 2)) - 2 ** (4 / 3)) < 1e-9   # max is 1x, 2x, 4x with prob 1/6, 2/6, 3/6
    assert abs(expected_log_best(sp, 1) - sum(map(math.log, sp)) / 4) < 1e-12
    assert abs(math.exp(expected_log_best(sp, 4)) - 4.0) < 1e-9
