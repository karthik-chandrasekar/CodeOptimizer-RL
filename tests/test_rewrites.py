"""The inverse rewrites must be able to *undo* the degradation operators (execution-verified)."""
import ast
import random

import pytest

from tinyperf.data.astutil import function_name, normalize
from tinyperf.data.degradations import applicable
from tinyperf.data.rewrites import propose
from tinyperf.data.seeds import SEEDS, make_inputs
from tinyperf.env.task import Task

# degradation family -> the rewrite family that should undo it
INVERSE = {
    "set_to_list_membership": "set_membership",
    "unhoist_call": "hoist", "unhoist_sort": "hoist", "unhoist_convert": "hoist", "unhoist_arith": "hoist", "unhoist_alloc": "hoist",
    "join_to_concat": "join", "worse_builtins": "minmax", "materialize": "dematerialize",
    "comprehension_to_loop": "comprehension", "append_to_concat": "append", "dict_build": "setdefault",
    "repeated_scan": "counter", "dict_linear_search": "dict_lookup",
}


def alpha_canon(src: str) -> str:
    """Canonical form up to local-variable names (function name, args and stores renamed by first appearance)."""
    tree = ast.parse(src)
    fn = tree.body[-1]
    m = {}
    for n in ast.walk(fn):
        if isinstance(n, ast.FunctionDef) and n.name not in m:
            m[n.name] = f"v{len(m)}"
        elif isinstance(n, ast.arg) and n.arg not in m:
            m[n.arg] = f"v{len(m)}"
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id not in m:
            m[n.id] = f"v{len(m)}"

    class R(ast.NodeTransformer):
        def visit_Name(self, n):
            n.id = m.get(n.id, n.id)
            return n

        def visit_arg(self, n):
            n.arg = m.get(n.arg, n.arg)
            return n

        def visit_FunctionDef(self, n):
            n.name = m.get(n.name, n.name)
            self.generic_visit(n)
            return n

    return ast.unparse(R().visit(tree))


def test_proposals_are_valid_python_and_not_noops():
    for s in SEEDS[:20]:
        src = normalize(s.source)
        for fam, cand in propose(src):
            assert cand != src
            compile(cand, "<c>", "exec")


def test_rewrites_undo_degradations_syntactically():
    """Every degradation family must be undone by its inverse rewrite family; most variants must be
    recovered exactly (up to variable names - hoisting may place the invariant right before the loop)."""
    attempted, by_inverse, exact = {}, {}, {}
    for s in SEEDS:
        src = normalize(s.source)
        for fam, variants in applicable(src).items():
            inv = INVERSE.get(fam)
            if inv is None:
                continue
            for v in variants:
                attempted[fam] = attempted.get(fam, 0) + 1
                props = propose(v)
                if any(f == inv for f, _ in props):
                    by_inverse[fam] = by_inverse.get(fam, 0) + 1
                if any(alpha_canon(normalize(c)) == alpha_canon(src) for _, c in props):
                    exact[fam] = exact.get(fam, 0) + 1
    assert attempted, "no degradations applicable"
    never = [f for f in attempted if by_inverse.get(f, 0) == 0]
    assert not never, f"inverse rewrite never fires for: {never}"
    rate = sum(exact.values()) / sum(attempted.values())
    assert rate >= 0.5, {f: exact.get(f, 0) / n for f, n in attempted.items()}


@pytest.mark.parametrize("seed_name", ["count_common", "count_above_mean"])
def test_search_proposer_finds_verified_speedup(executor, seed_name):
    """End-to-end: on a degraded seed, at least one proposal is correct and faster in the sandbox."""
    s = next(x for x in SEEDS if x.name == seed_name)
    src = normalize(s.source)
    apps = applicable(src)
    if not apps:
        pytest.skip("seed has no AST degradations")
    fam, degraded = next(iter(apps.items()))
    degraded = degraded[0]
    rng = random.Random(0)
    correct, perf = make_inputs(s, rng, 400, n_perf=3)
    b = executor.baseline(degraded, function_name(degraded), correct, perf)
    assert b["status"] == "ok"
    task = Task("t", function_name(degraded), degraded, correct, perf, b["expected"], float(b["candidate_median_ns"]))
    ok = []
    for _, cand in propose(degraded):
        r = executor.evaluate(cand, task, baseline_code=degraded)
        if r.get("status") == "ok" and r.get("ratio") is not None:
            ok.append(r["ratio"])
    assert ok and min(ok) < 1.0, f"no verified proposal for family {fam}"
