"""Argument types for mined functions.

Real functions do not come with inputs.  Each parameter gets a *spec* (``list_int``, ``str``, ``int_size``, ...)
guessed from how the function uses it, then confirmed by running the function (see :mod:`tinyperf.data.mine`).
A spec is a generator of size-``n`` values, so mined functions plug into the same workload machinery as the
hand-written seeds (``gen(rng, n) -> args``).
"""
from __future__ import annotations

import ast
import itertools
import random
import string
from collections import defaultdict
from typing import Callable, Dict, List, Sequence, Tuple

_VOCAB = ["".join(random.Random(i).choice(string.ascii_lowercase) for _ in range(random.Random(i + 7).randint(2, 8)))
          for i in range(4000)]


def _words(rng: random.Random, n: int) -> List[str]:
    v = max(3, n // 2 + 2)
    return [_VOCAB[rng.randrange(v)] for _ in range(n)]


def _ints(rng: random.Random, n: int) -> List[int]:
    return [rng.randint(-n - 5, n + 5) for _ in range(n)]


SPECS: Dict[str, Callable[[random.Random, int], object]] = {
    "list_int": _ints,
    "list_str": _words,
    "str": lambda rng, n: "".join(rng.choice(string.ascii_lowercase + "  ") for _ in range(n)),
    "int_size": lambda rng, n: rng.randint(n // 2, n) if n > 0 else rng.randint(0, 1),
    "int_small": lambda rng, n: rng.randint(-3, 12),
    "list_float": lambda rng, n: [round(rng.uniform(-100, 100), 3) for _ in range(n)],
    "float": lambda rng, n: round(rng.uniform(-100, 100), 3),
    "list_pair_int": lambda rng, n: [(rng.randint(0, n + 3), rng.randint(0, n + 3)) for _ in range(n)],
    "list_list_int": lambda rng, n: [_ints(rng, rng.randint(0, 6)) for _ in range(n)],
    "dict_str_int": lambda rng, n: {w: rng.randint(-50, 50) for w in _words(rng, n)},
    "dict_int_int": lambda rng, n: {k: rng.randint(-50, 50) for k in _ints(rng, n)},
    "set_int": lambda rng, n: set(_ints(rng, n)),
    "bool": lambda rng, n: rng.random() < 0.5,
}
SPEC_NAMES = list(SPECS)


def gen_value(spec: str, rng: random.Random, n: int):
    return SPECS[spec](rng, n)


def make_gen(specs: Sequence[str]) -> Callable[[random.Random, int], Tuple]:
    specs = list(specs)
    return lambda rng, n: tuple(gen_value(s, rng, n) for s in specs)


# --------------------------------------------------------------------------- #
# usage-based guesses
# --------------------------------------------------------------------------- #
LIST_M = {"append", "extend", "pop", "insert", "sort", "reverse", "index", "remove", "copy", "clear"}
STR_M = {"split", "strip", "lstrip", "rstrip", "lower", "upper", "startswith", "endswith", "replace", "find", "rfind",
         "isdigit", "isalpha", "isspace", "isupper", "islower", "title", "capitalize", "encode", "splitlines", "partition"}
DICT_M = {"items", "keys", "values", "get", "setdefault", "update"}
SET_M = {"add", "discard", "union", "intersection", "difference", "issubset", "issuperset"}
SEQ_AGG = {"sum", "max", "min", "sorted", "set", "list", "tuple", "any", "all", "enumerate", "reversed"}


def _is(n: ast.AST, name: str) -> bool:
    return isinstance(n, ast.Name) and n.id == name


def _num_const(n: ast.AST) -> bool:
    return isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool)


def _element_usage(body: List[ast.AST], v: str, score: Dict[str, float]) -> None:
    """How is loop variable `v` (an element of the parameter) used?"""
    for node in (x for b in body for x in ast.walk(b)):
        if isinstance(node, (ast.BinOp, ast.AugAssign)):
            ops = [node.left, node.right] if isinstance(node, ast.BinOp) else [node.target, node.value]
            if any(_is(o, v) for o in ops):
                score["list_int"] += 2; score["list_float"] += 1
        elif isinstance(node, ast.Compare) and (_is(node.left, v) or any(_is(c, v) for c in node.comparators)):
            if any(_num_const(c) for c in [node.left] + node.comparators):
                score["list_int"] += 1.5
            if any(isinstance(c, ast.Constant) and isinstance(c.value, str) for c in [node.left] + node.comparators):
                score["str"] += 2.5; score["list_str"] += 1   # iterating a string yields characters
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and _is(node.func.value, v):
            if node.func.attr in STR_M:
                score["list_str"] += 3
        elif isinstance(node, ast.For) and _is(node.iter, v):
            score["list_list_int"] += 3
        elif isinstance(node, ast.comprehension) and _is(node.iter, v):
            score["list_list_int"] += 3
        elif isinstance(node, ast.Subscript) and _is(node.value, v):
            score["list_list_int"] += 1.5; score["list_pair_int"] += 1.5
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "len" and node.args and _is(node.args[0], v):
            score["list_str"] += 1; score["list_list_int"] += 1


def guess_specs(fn: ast.FunctionDef, max_per_param: int = 4) -> List[List[str]]:
    params = [a.arg for a in fn.args.posonlyargs + fn.args.args]
    out = []
    for p in params:
        s: Dict[str, float] = defaultdict(float)
        for d, w in (("list_int", 0.3), ("int_size", 0.2), ("str", 0.2), ("list_str", 0.1)):
            s[d] += w
        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Attribute) and _is(f.value, p):
                    m = f.attr
                    if m in LIST_M:
                        s["list_int"] += 2; s["list_str"] += 1.5
                    if m in STR_M:
                        s["str"] += 3
                    if m in DICT_M:
                        s["dict_str_int"] += 3; s["dict_int_int"] += 2
                    if m in SET_M:
                        s["set_int"] += 3
                    if m == "count":
                        s["list_int"] += 1; s["str"] += 1
                if isinstance(f, ast.Name) and f.id == "range" and any(_is(a, p) for a in node.args):
                    s["int_size"] += 4
                if isinstance(f, ast.Name) and f.id == "len" and node.args and _is(node.args[0], p):
                    s["list_int"] += 1; s["str"] += 1; s["list_str"] += 0.5
                if isinstance(f, ast.Name) and f.id in SEQ_AGG and node.args and _is(node.args[0], p):
                    s["list_int"] += 1.5; s["list_float"] += 0.5
                if isinstance(f, ast.Attribute) and f.attr == "join" and isinstance(f.value, ast.Constant) and node.args and _is(node.args[0], p):
                    s["list_str"] += 4
            elif isinstance(node, ast.Subscript) and _is(node.value, p):
                sl = node.slice
                if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                    s["dict_str_int"] += 3
                else:
                    s["list_int"] += 1; s["str"] += 0.5; s["list_list_int"] += 0.3
            elif isinstance(node, (ast.For, ast.comprehension)) and _is(node.iter, p):
                s["list_int"] += 1; s["list_str"] += 0.8; s["str"] += 0.5
                tgt = node.target
                if isinstance(tgt, ast.Tuple) and len(tgt.elts) == 2:
                    s["list_pair_int"] += 4
                elif isinstance(tgt, ast.Name):
                    body = node.body + node.orelse if isinstance(node, ast.For) else []
                    _element_usage(body, tgt.id, s)
            elif isinstance(node, ast.BinOp) and (_is(node.left, p) or _is(node.right, p)):
                other = node.right if _is(node.left, p) else node.left
                if _num_const(other) or isinstance(node.op, (ast.Mod, ast.FloorDiv)):
                    s["int_small"] += 2; s["float"] += 1
                elif isinstance(other, ast.Constant) and isinstance(other.value, str):
                    s["str"] += 2
            elif isinstance(node, ast.Compare):
                if _is(node.left, p) and any(_num_const(c) for c in node.comparators):
                    s["int_small"] += 1.5; s["float"] += 0.5
                if any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops) and any(_is(c, p) for c in node.comparators):
                    s["list_int"] += 1; s["set_int"] += 1; s["str"] += 0.5; s["dict_str_int"] += 0.5
            elif isinstance(node, (ast.If, ast.While)) and _is(node.test, p):
                s["bool"] += 1
        out.append([k for k, _ in sorted(s.items(), key=lambda kv: -kv[1])][:max_per_param])
    return out


def ranked_combos(per_param: List[List[str]], cap: int = 16) -> List[Tuple[str, ...]]:
    """Cartesian product of per-parameter guesses, best-ranked combinations first."""
    combos = list(itertools.product(*[list(enumerate(c)) for c in per_param])) if per_param else [()]
    combos.sort(key=lambda c: sum(i for i, _ in c))
    return [tuple(s for _, s in c) for c in combos[:cap]]
