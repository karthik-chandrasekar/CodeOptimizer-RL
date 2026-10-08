"""File-task splicing and dispatcher."""
import pytest

from tinyperf.env.files import ENTRY, edited_functions, make_driver, splice_functions, with_driver

BASE = "def a(x):\n    return x + 1\n\n\ndef b(y):\n    return [v for v in y]\n"


def test_splice_replaces_by_name_and_keeps_order():
    out = splice_functions(BASE, "def b(y):\n    return list(y)\n")
    assert out.index("def a") < out.index("def b") and "list(y)" in out and "for v in y" not in out


def test_splice_adds_new_helpers_and_multiple_defs():
    out = splice_functions(BASE, "def a(x):\n    return 1 + x\n\ndef helper(z):\n    return z\n")
    assert "1 + x" in out and "def helper" in out and "def b" in out


def test_splice_rejects_edits_without_functions_and_ignores_the_dispatcher():
    with pytest.raises(SyntaxError):
        splice_functions(BASE, "x = 1\n")
    with pytest.raises(SyntaxError):
        splice_functions(BASE, "def oops(:\n  pass")
    with pytest.raises(SyntaxError):  # redefining the hidden entry point is not an edit
        splice_functions(BASE, f"def {ENTRY}(k, *a):\n    return 0\n")


def test_driver_dispatches_to_each_function():
    ns = {}
    exec(with_driver(BASE, make_driver(["a", "b"])), ns)
    assert ns[ENTRY](0, 41) == 42 and ns[ENTRY](1, (1, 2)) == [1, 2]
    assert edited_functions("def a(x):\n    return x\n") == ["a"] and edited_functions("def (") == []


def test_profile_shows_where_time_goes():
    from tests.conftest import fast_env_cfg
    from tinyperf.env.env import PerfEnv
    from tinyperf.env.executor import Executor
    from tinyperf.env.files import profile_shares
    from tinyperf.env.protocol import Action
    from tinyperf.env.task import Task

    cfg = fast_env_cfg()
    ex = Executor(cfg)
    src = ("def a(xs):\n    return sum(xs)\n\n\n"
           "def b(xs):\n    out = []\n    for x in xs:\n        if x not in out:\n            out.append(x)\n    return out\n")
    names, drv = ["a", "b"], make_driver(["a", "b"])
    correct = [(0, [1, 2]), (1, [1, 1, 2]), (1, [])]
    perf = [(0, list(range(2000)))] * 3 + [(1, list(range(600)))] * 3
    base = ex.baseline(with_driver(src, drv), ENTRY, correct, perf, group_by_first_arg=True)
    prof = profile_shares(base["candidate_group_ns"], names)
    assert abs(sum(prof.values()) - 1.0) < 1e-9 and prof["b"] > prof["a"]
    task = Task("f0", ENTRY, src, correct, perf, base["expected"], base["candidate_median_ns"],
                meta={"kind": "file", "driver": drv, "functions": names, "profile0": prof})
    env = PerfEnv(ex, cfg, feedback="profile", horizon=3)
    assert "time_share=a:" in env.reset(task)
    fixed = ("def b(xs):\n    seen = set()\n    out = []\n    for x in xs:\n        if x not in seen:\n"
             "            seen.add(x)\n            out.append(x)\n    return out\n")
    r = env.step_action(Action("edit", code=fixed))
    assert r.record.status == "ok" and env.best_profile["b"] < prof["b"]
    assert "time_share" not in PerfEnv(ex, cfg, feedback="full", horizon=3).reset(task)
    assert "time_share" not in PerfEnv(ex, cfg, feedback="none", horizon=3).reset(task)
    ex.close()
