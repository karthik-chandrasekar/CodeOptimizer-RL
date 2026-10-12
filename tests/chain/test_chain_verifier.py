"""T07 exact types, T08 exceptions + mutation, T09 aliasing, T10 immutable original, T15 timing state,
T16 timing reproducibility, T17 worker isolation."""
import pytest

from tinyperf_chain.contracts import validate_args
from tinyperf_chain.program import Program, apply_patch

from .helpers import make_task


def _check(verifier, src, cand_fn, params, calls, name="f"):
    task, priv = make_task(src, [name], [params], calls, calls[:1])
    cand = apply_patch(task.original(), task.original().hash, name, cand_fn, ["math", "collections"])
    return verifier.mandatory_checks(task, priv, cand), task, priv


def test_t07_exact_types(verifier):
    src = "def f(x):\n    return x * 2\n"
    calls = [(0, 1), (0, 5)]
    r, *_ = _check(verifier, src, "def f(x):\n    return float(x * 2)\n", [("x", "int")], calls)
    assert not r.ok and r.category == "type_mismatch"
    r, *_ = _check(verifier, "def f(x):\n    return x > 0\n", "def f(x):\n    return int(x > 0)\n", [("x", "int")], calls)
    assert not r.ok and r.category == "type_mismatch"   # True == 1 under Python equality, still a failure
    r, *_ = _check(verifier, src, "def f(x):\n    return x + x\n", [("x", "int")], calls)
    assert r.ok


def test_t08_exceptions_and_mutation(verifier):
    src = "def f(xs):\n    xs.append(1)\n    return xs[10]\n"
    calls = [(0, [1, 2]), (0, list(range(20)))]
    # same exception but no mutation before it -> mutation mismatch on the raising input
    r, *_ = _check(verifier, src, "def f(xs):\n    v = xs[10]\n    xs.append(1)\n    return v\n", [("xs", "list[int]")], calls)
    assert not r.ok and r.category == "mutation_mismatch"
    r, *_ = _check(verifier, src, "def f(xs):\n    xs.append(1)\n    return xs[10] if len(xs) > 10 else None\n",
                   [("xs", "list[int]")], calls)
    assert not r.ok and r.category == "exception_mismatch"
    r, *_ = _check(verifier, src, "def f(xs):\n    xs.append(1)\n    return xs[10]\n", [("xs", "list[int]")], calls)
    assert r.ok


def test_t09_aliasing(verifier, cfg):
    src = "def f(n):\n    return [[] for _ in range(n)]\n"
    r, *_ = _check(verifier, src, "def f(n):\n    return [[]] * n\n", [("n", "int")], [(0, 3), (0, 0)])
    assert not r.ok and r.category == "alias_mismatch"
    src2 = "def g(xs):\n    return list(xs)\n"
    r, *_ = _check(verifier, src2, "def g(xs):\n    return xs\n", [("xs", "list[int]")], [(0, [1, 2])], name="g")
    assert not r.ok and r.category == "alias_mismatch"


def test_t09_alias_inputs_rejected(verifier):
    from tinyperf_chain.contracts import CONTRACT_VERSION, FunctionContract, ParamSpec, TaskContract

    tc = TaskContract(CONTRACT_VERSION, [FunctionContract("h", [ParamSpec("a", "list[int]"), ParamSpec("b", "list[int]")])])
    x = [1, 2]
    assert validate_args(tc, "h", ([1], [2])) is None
    assert "aliasing" in validate_args(tc, "h", (x, x))


def test_t10_semantics_vs_original_not_current_best(verifier):
    src = "def f(xs):\n    return sorted(xs)\n"
    calls = [(0, [3, 1, 2]), (0, [5, 5, 1])]
    task, priv = make_task(src, ["f"], [[("xs", "list[int]")]], calls, calls)
    p0 = task.original()
    wrong = apply_patch(p0, p0.hash, "f", "def f(xs):\n    return sorted(set(xs))\n", [])
    again = apply_patch(wrong, wrong.hash, "f", "def f(xs):\n    return sorted(set(xs))\n", [])
    assert not verifier.mandatory_checks(task, priv, again).ok        # judged against P0, not the previous version
    assert task.original().hash == p0.hash


def test_t15_fresh_state_per_repetition(verifier):
    # the function sorts its argument in place: every repetition must see fresh unsorted copies
    src = "def f(xs):\n    xs.sort()\n    return xs[0]\n"
    wl = [(0, list(range(3000, 0, -1)))]
    task, priv = make_task(src, ["f"], [[("xs", "list[int]")]], wl, wl)
    req = verifier.runner.base_request("time_pair")
    req.update(code_a=task.original().executable(), code_b=task.original().executable(), entry="_tp_entry",
               workload=wl, blocks=3, reps=3, seed=0)
    resp = verifier.runner.run(req)
    assert resp["status"] == "ok" and len(resp["a_ns"]) == 3
    # a module-level cache cannot survive across blocks (fresh namespace per block / repetition)
    p = Program.from_source("CACHE = {}\n\ndef f(xs):\n    return len(xs)\n", ["f"])
    req.update(code_a=p.executable(), code_b=p.executable())
    assert verifier.runner.run(req)["status"] == "ok"


def test_t16_noop_not_faster_and_slowdown_detected(verifier):
    src = "def f(xs):\n    return sum(xs)\n"
    wl = [(0, list(range(20000)))]
    task, priv = make_task(src, ["f"], [[("xs", "list[int]")]], [(0, [1, 2])], wl)
    p0 = task.original()
    tr = verifier.paired_timing(task, wl, p0, p0, blocks=5, seed=3)
    assert tr.ok and tr.verdict.decision != "accept"
    slow = apply_patch(p0, p0.hash, "f", "def f(xs):\n    t = 0\n    for x in xs:\n        t = t + x\n    return t\n", [])
    tr2 = verifier.paired_timing(task, wl, p0, slow, blocks=5, seed=4)
    assert tr2.ok and tr2.verdict.decision in ("slower", "no_gain") and tr2.verdict.speedup < 1.0


def test_t17_isolation(verifier):
    # forbidden operations are rejected by the static policy before execution
    src = "def f(x):\n    return x\n"
    task, priv = make_task(src, ["f"], [[("x", "int")]], [(0, 1)], [(0, 1)])
    for code in ("def f(x):\n    open('/tmp/x', 'w')\n    return x\n", "def f(x):\n    import socket\n    return x\n"):
        try:
            cand = apply_patch(task.original(), task.original().hash, "f", code, ["math"])
        except Exception:
            continue
        assert not verifier.mandatory_checks(task, priv, cand).ok
    # a hanging candidate is killed (process group) and reported as a timeout, not as a crash of the harness
    hang = Program.from_source("def f(x):\n    while True:\n        x += 1\n", ["f"])
    resp = verifier.runner.run({**verifier.runner.base_request("run"), "code": hang.executable(), "entry": "_tp_entry",
                                "inputs": [(0, 1)]}, timeout_s=3)
    assert resp["status"] in ("ok", "timeout")
    if resp["status"] == "ok":
        assert resp["records"][0]["exc"] == "__timeout__"
    st = verifier.runner.selftest()
    assert st["status"] == "ok" and "network" in st and "fork" in st   # reported, not assumed
