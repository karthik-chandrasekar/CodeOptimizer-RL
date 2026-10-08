"""Identifier renaming (variety) and bug mutations (failure injection)."""
from __future__ import annotations

import ast
import copy
import random
from typing import Dict, List, Optional, Set

from tinyperf.data.astutil import all_names, get_function, parse_module, unparse

FUNC_WORDS_A = ["compute", "find", "collect", "count", "build", "make", "select", "score", "reduce", "scan",
                "merge", "resolve", "gather", "check", "measure", "extract", "filter", "rank", "map", "fold"]
FUNC_WORDS_B = ["items", "values", "records", "entries", "tokens", "elements", "pairs", "scores", "keys",
                "groups", "chunks", "spans", "matches", "results", "counts", "points", "rows", "cells"]
VAR_NAMES = ["a", "b", "c", "d", "e", "g", "h", "m", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
             "i", "j", "k", "n", "idx", "pos", "item", "elem", "val", "key", "cur", "prev", "acc", "tmp", "buf",
             "res", "out", "ret", "seq", "arr", "lst", "data", "vals", "keys", "xs", "ys", "zs", "items", "elems",
             "entries", "rows", "cols", "left", "right", "lo", "hi", "start", "end", "total", "count", "num",
             "size", "limit", "bound", "cache", "table", "found", "flag", "first", "last", "head", "tail",
             "word", "token", "text", "chars", "line", "parts", "piece", "chunk", "group", "bucket", "node"]

PROTECTED = {"_", "math", "itertools", "functools", "collections", "heapq", "bisect", "operator", "string", "re",
             "array", "statistics", "fractions", "decimal", "self"}


def local_names(src: str) -> Set[str]:
    """Function name, parameters, locals, nested function names, comprehension targets."""
    tree = parse_module(src)
    fn = get_function(tree)
    names: Set[str] = {fn.name}
    imported: Set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.update(a.asname or a.name for a in node.names)
    for n in ast.walk(fn):
        if isinstance(n, ast.arg):
            names.add(n.arg)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            names.add(n.id)
        elif isinstance(n, ast.FunctionDef):
            names.add(n.name)
    return {n for n in names if n not in PROTECTED and n not in imported}


class _Renamer(ast.NodeTransformer):
    def __init__(self, mapping: Dict[str, str]):
        self.m = mapping

    def visit_Name(self, node):
        if node.id in self.m:
            node.id = self.m[node.id]
        return node

    def visit_arg(self, node):
        if node.arg in self.m:
            node.arg = self.m[node.arg]
        return node

    def visit_FunctionDef(self, node):
        if node.name in self.m:
            node.name = self.m[node.name]
        self.generic_visit(node)
        return node

    def visit_keyword(self, node):
        # keyword argument names are attribute-like; never rename
        self.generic_visit(node)
        return node


def make_rename_map(sources: List[str], rng: random.Random, func_name: str) -> Dict[str, str]:
    names: Set[str] = set()
    for s in sources:
        names |= local_names(s)
    taken = set().union(*(all_names(parse_module(s)) for s in sources))
    mapping: Dict[str, str] = {}
    new_fn = f"{rng.choice(FUNC_WORDS_A)}_{rng.choice(FUNC_WORDS_B)}"
    if rng.random() < 0.3:
        new_fn += str(rng.randint(2, 9))
    mapping[func_name] = new_fn
    pool = [v for v in VAR_NAMES if v not in taken]
    rng.shuffle(pool)
    for n in sorted(names - {func_name}):
        if n.startswith("_"):
            continue
        if rng.random() < 0.15 or not pool:
            continue  # keep some names untouched
        mapping[n] = pool.pop()
    return mapping


def apply_rename(src: str, mapping: Dict[str, str]) -> str:
    tree = parse_module(src)
    _Renamer(mapping).visit(tree)
    return unparse(tree)


# --------------------------------------------------------------------------- #
# Bug mutations (for failure-injection in SFT trajectories)
# --------------------------------------------------------------------------- #
_CMP_SWAP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
             ast.In: ast.NotIn, ast.NotIn: ast.In}
_BIN_SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Add}


def mutate_bug(src: str, rng: random.Random) -> Optional[str]:
    """Return a plausibly-wrong variant of src (must be *verified* incorrect by execution)."""
    tree = parse_module(src)
    fn = get_function(tree)
    sites = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in _CMP_SWAP:
            sites.append(("cmp", n))
        elif isinstance(n, ast.BinOp) and type(n.op) in _BIN_SWAP:
            sites.append(("bin", n))
        elif isinstance(n, ast.Constant) and isinstance(n.value, int) and not isinstance(n.value, bool):
            sites.append(("const", n))
        elif isinstance(n, ast.Return) and n.value is not None:
            sites.append(("ret", n))
        elif isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Slice):
            sites.append(("slice", n))
    if not sites:
        return None
    kind, node = rng.choice(sites)
    if kind == "cmp":
        node.ops[0] = _CMP_SWAP[type(node.ops[0])]()
    elif kind == "bin":
        node.op = _BIN_SWAP[type(node.op)]()
    elif kind == "const":
        node.value = node.value + rng.choice([-1, 1])
    elif kind == "ret":
        v = node.value
        if isinstance(v, (ast.Name, ast.Subscript, ast.Call, ast.BinOp, ast.ListComp)):
            node.value = ast.Constant(None) if rng.random() < 0.3 else ast.UnaryOp(ast.Not(), v)
    elif kind == "slice":
        s = node.slice
        if s.upper is not None:
            s.upper = ast.BinOp(s.upper, ast.Sub(), ast.Constant(1))
        elif s.lower is not None:
            s.lower = ast.BinOp(s.lower, ast.Add(), ast.Constant(1))
        else:
            s.lower = ast.Constant(1)
    out = unparse(tree)
    return None if out == src else out


def mutate_syntax(src: str, rng: random.Random) -> str:
    """Produce a syntactically broken variant (unbalanced paren / missing colon)."""
    lines = src.rstrip("\n").split("\n")
    i = rng.randrange(len(lines))
    line = lines[i]
    if line.rstrip().endswith(":") and rng.random() < 0.5:
        lines[i] = line.rstrip()[:-1]
    elif "(" in line:
        lines[i] = line.replace("(", "", 1)
    else:
        lines[i] = line + " )"
    return "\n".join(lines) + "\n"
