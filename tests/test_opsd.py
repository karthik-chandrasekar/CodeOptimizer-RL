"""On-policy self-distillation: hints, reverse KL, item selection, warm-up data."""
from types import SimpleNamespace

import pytest

from tinyperf.train.hints import add_hint, build_hint, code_from_obs, localization_hint, outcome_hint, remaining_slow

SRC = ("def a(xs):\n    return sum(xs)\n\n\ndef b(xs):\n    out = []\n    for x in xs:\n        out = out + [x]\n    return out\n\n\n"
       "def c(xs):\n    out = []\n    for x in xs:\n        out = out + [x * 2]\n    return out")
TASK = SimpleNamespace(source=SRC, meta={"functions": ["a", "b", "c"], "slow_functions": ["b", "c"], "profile0": {"a": 0.05, "b": 0.25, "c": 0.70}})
OBS = f"<CODE>\n{SRC}\n</CODE>\n<STATE>\nbest_runtime=1.000\nstep=0\n</STATE>"


def test_hint_text_and_insertion():
    assert localization_hint(TASK, SRC) == "the slowest unfixed function is c"
    fixed_c = SRC.replace("out = out + [x * 2]", "out.append(x * 2)")
    assert remaining_slow(TASK, fixed_c) == ["b"] and localization_hint(TASK, fixed_c) == "the slowest unfixed function is b"
    both = fixed_c.replace("out = out + [x]", "out.append(x)")
    assert localization_hint(TASK, both).startswith("all slow functions are fixed")
    assert "changes its behaviour" in outcome_hint("edit", "incorrect", False, ["c"])
    assert outcome_hint("edit", "ok", False, ["a"]) == "editing a gives no speedup"
    assert outcome_hint("stop", "stop", False, [], ["b"]) == "stopping here is premature"
    h = add_hint(OBS, "x=1")
    assert h.endswith("hint=x=1\n</STATE>") and code_from_obs(h) == SRC and add_hint(OBS, None) == OBS
    hint = build_hint(TASK, OBS, "edit", "ok", False, "<EDIT>\ndef a(xs):\n    return sum(xs)\n</EDIT>", "error+loc")
    assert hint == "editing a gives no speedup; the slowest unfixed function is c"
    assert build_hint(TASK, OBS, "edit", "ok", False, "", "loc") == "the slowest unfixed function is c"


torch = pytest.importorskip("torch")


def _tiny():
    from tinyperf.common.config import ModelConfig
    from tinyperf.model.transformer import TinyPerfLM
    torch.manual_seed(0)
    return TinyPerfLM(ModelConfig(vocab_size=64, n_layer=2, n_embd=32, n_head=4, mlp_dim=64, max_seq_len=64))


def test_reverse_kl_is_zero_for_identical_prompts_and_trains_the_student_only():
    from tinyperf.train.grpo import OPSDItem, opsd_reverse_kl
    m = _tiny()
    same = [OPSDItem([1, 5, 6, 7], [1, 5, 6, 7], [9, 10, 11]), OPSDItem([1, 8], [1, 8], [12, 13])]
    kl0, n0 = opsd_reverse_kl(m, same, 0, "cpu")
    assert n0 == 5 and abs(float(kl0.detach())) < 1e-5
    diff = [OPSDItem([1, 5, 6, 7], [1, 30, 31, 32, 33, 5, 6, 7], [9, 10, 11])]
    kl, n = opsd_reverse_kl(m, diff, 0, "cpu")
    assert n == 3 and float(kl.detach()) > 0
    kl.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in m.parameters())


class _Tok:
    def encode(self, text, add_bos=False):
        return ([1] if add_bos else []) + [2 + (ord(ch) % 50) for ch in text]


def _episode(statuses):
    steps, turns = [], []
    for st, improved, kind, action in statuses:
        steps.append(SimpleNamespace(action_kind=kind, status=st, improved=improved))
        turns.append(SimpleNamespace(obs=OBS, prompt_ids=[1, 3, 4], response_ids=[5, 6], action_text=action, finished=True))
    return SimpleNamespace(task=TASK, turns=turns, summary=SimpleNamespace(steps=steps))


def test_opsd_items_cover_all_turns_or_only_problematic_ones():
    import random
    from tinyperf.train.grpo import build_opsd_items
    fix_c = "<EDIT>\ndef c(xs):\n    return [x * 2 for x in xs]\n</EDIT>"
    ep = _episode([("incorrect", False, "edit", fix_c), ("ok", True, "edit", fix_c), ("stop", False, "stop", "<STOP>")])
    all_items = build_opsd_items([ep], _Tok(), 4096, "error+loc", "all", 100, random.Random(0))
    bad_items = build_opsd_items([ep], _Tok(), 4096, "error+loc", "problematic", 100, random.Random(0))
    assert len(all_items) == 3
    assert len(bad_items) == 2          # the failed edit, and the premature STOP (b and c are still unfixed in OBS)
    assert all(it.teacher_ids != it.student_ids and it.response_ids == [5, 6] for it in all_items)
    assert len(build_opsd_items([ep] * 50, _Tok(), 4096, "error", "all", 10, random.Random(0))) == 10


def test_hinted_warmup_demo_replays_and_carries_hints():
    from tests.conftest import fast_env_cfg
    from tinyperf.common.config import load_config
    from tinyperf.env.executor import Executor
    from tinyperf.env.files import ENTRY, make_driver, profile_shares, with_driver
    from tinyperf.env.task import Task
    from tinyperf.train.hint_sft_data import hinted_demo
    import random

    cfg = load_config(None, [])
    cfg.env = fast_env_cfg()
    ex = Executor(cfg.env)
    drv = make_driver(["a", "b", "c"])
    fast = SRC.replace("out = out + [x]", "out.append(x)").replace("out = out + [x * 2]", "out.append(x * 2)")
    correct = [(0, [1, 2]), (1, [1, 2]), (2, [3]), (1, []), (2, [])]
    perf = [(0, list(range(2000)))] * 2 + [(1, list(range(1500)))] * 2 + [(2, list(range(3000)))] * 2
    base = ex.baseline(with_driver(SRC, drv), ENTRY, correct, perf, group_by_first_arg=True)
    prof = profile_shares(base["candidate_group_ns"], ["a", "b", "c"])
    task = Task("f0", ENTRY, SRC, correct, perf, base["expected"], base["candidate_median_ns"], fast_source=fast,
                meta={"kind": "file", "driver": drv, "functions": ["a", "b", "c"], "slow_functions": ["b", "c"], "profile0": prof})
    row = hinted_demo(ex, cfg, task, random.Random(0))
    ex.close()
    assert row is not None and [s["status"] for s in row["steps"]] == ["ok", "ok", "stop"]
    assert all("hint=" in s["obs"] for s in row["steps"])
    first = max(("b", "c"), key=lambda f: prof[f])
    assert f"def {first}(" in row["steps"][0]["action"]


def test_items_skip_truncated_responses_and_cap_the_distilled_prefix():
    import random
    from tinyperf.train.grpo import build_opsd_items
    ep = _episode([("ok", False, "edit", "<EDIT>\ndef a(xs):\n    return sum(xs)\n</EDIT>"), ("syntax_error", False, "edit", "<EDIT>\ndef c(")])
    ep.turns[0].response_ids = list(range(5, 40))
    ep.turns[1].finished = False                      # hit the token limit
    items = build_opsd_items([ep], _Tok(), 4096, "error+loc", "all", 100, random.Random(0), max_tokens=12)
    assert len(items) == 1 and items[0].response_ids == list(range(5, 17))
    assert len(build_opsd_items([ep], _Tok(), 4096, "error+loc", "all", 100, random.Random(0), skip_truncated=False)) == 2


def test_frozen_teacher_is_separate_and_gets_no_gradient():
    import copy
    from tinyperf.train.grpo import OPSDItem, opsd_reverse_kl
    student = _tiny()
    teacher = copy.deepcopy(student).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    items = [OPSDItem([1, 5, 6], [1, 5, 6], [9, 10, 11])]
    kl0, _ = opsd_reverse_kl(student, items, 0, "cpu", teacher=teacher)
    assert abs(float(kl0.detach())) < 1e-5                     # identical weights, identical prompts
    with torch.no_grad():
        for p in student.parameters():
            p.add_(0.05 * torch.randn_like(p))                  # the student moves, the frozen teacher does not
    kl, _ = opsd_reverse_kl(student, items, 0, "cpu", teacher=teacher)
    assert float(kl.detach()) > 0
    kl.backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert any(p.grad is not None for p in student.parameters())
