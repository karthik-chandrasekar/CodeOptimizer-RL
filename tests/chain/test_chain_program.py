"""T06 patch scope; T10 (part) rejected patches never change the program."""
import pytest

from tinyperf_chain.program import PatchError, Program, apply_patch, dependency_closure

SRC = """import math

LIMIT = 10

def helper(x):
    return x * 2

def f(xs, k):
    out = []
    for x in xs:
        if x in k:
            out.append(helper(x))
    return out

def g(a):
    return a + 1
"""


@pytest.fixture()
def prog():
    return Program.from_source(SRC, ["f", "g"])


def ok(prog, code):
    return apply_patch(prog, prog.hash, "f", code, ["math", "collections"])


def bad(prog, code, kind, target="f", base=None):
    with pytest.raises(PatchError) as e:
        apply_patch(prog, base or prog.hash, target, code, ["math", "collections"])
    assert e.value.kind == kind, e.value


def test_t06_valid_patch_changes_only_target(prog):
    new = ok(prog, "def f(xs, k):\n    s = set(k)\n    return [helper(x) for x in xs if x in s]\n")
    assert new.function_source("g") == prog.function_source("g")
    assert new.function_source("helper") == prog.function_source("helper")
    assert new.hash != prog.hash and prog.function_source("f").startswith("def f(xs, k):\n    out = []")


def test_t06_scope_rules(prog):
    bad(prog, "def f(xs, k, extra=None):\n    return []\n", "scope_violation")            # signature
    bad(prog, "def f(xs, k):\n    return []\nprint(1)\n", "scope_violation")             # extra top-level statement
    bad(prog, "def f(xs, k):\n    return []\ndef g(a):\n    return a\n", "scope_violation")  # second function
    bad(prog, "def g(a):\n    return a\n", "scope_violation")                            # wrong function
    bad(prog, "import os\ndef f(xs, k):\n    return []\n", "scope_violation")             # module-level import
    bad(prog, "def f(xs, k):\n    import os\n    return []\n", "scope_violation")         # undeclared import
    bad(prog, "def f(xs, k):\n    return undefined_name(xs)\n", "scope_violation")      # escapes the closure
    bad(prog, "def f(xs, k):\n    global LIMIT\n    return []\n", "policy_violation")
    bad(prog, "def f(xs, k):\n    helper.cache = {}\n    return []\n", "policy_violation")  # module state write
    bad(prog, "def f(xs, k):\n    yield 1\n", "policy_violation")
    bad(prog, "@staticmethod\ndef f(xs, k):\n    return []\n", "scope_violation")
    bad(prog, "def f(xs, k):\n    return [\n", "syntax_error")
    bad(prog, "def f(xs, k):\n    return []\n", "stale_base", base="0" * 64)
    bad(prog, "def _tp_entry(k, *args):\n    return 1\n", "scope_violation", target="_tp_entry")
    ok(prog, "def f(xs, k):\n    import math\n    return [helper(x) for x in xs if x in k and math.isfinite(LIMIT)]\n")


def test_dependency_closure(prog):
    cl = dependency_closure(prog, "f")
    assert cl.functions == ["helper"] and not cl.unsupported
    p2 = Program.from_source("def f(x):\n    return getattr(x, name)\n", ["f"])
    assert dependency_closure(p2, "f").unsupported
