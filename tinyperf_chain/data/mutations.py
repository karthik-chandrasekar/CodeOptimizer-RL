"""Controlled near-miss mutations with an explicit *family* label (8.2, 10.5).

The family of a mutation is privileged provenance: it labels M4 training data and lets evaluation hold out whole
bug families (``roles.m4 --holdout-families``), but it never enters a model prompt.  Every mutation must still be
*verified* incorrect by execution before it is used; a mutation that happens to preserve behaviour is discarded.
"""
from __future__ import annotations

import ast
import random
from typing import List, Optional, Tuple

_CMP_SWAP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
             ast.In: ast.NotIn, ast.NotIn: ast.In}
_BIN_SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult, ast.Mod: ast.FloorDiv}
_CONTAINER_SWAP = {"set": "list", "list": "set", "sorted": "reversed", "frozenset": "tuple", "dict": "list"}

FAMILIES: Tuple[str, ...] = ("cmp_swap", "bin_swap", "const_shift", "return_change", "slice_shift", "container_swap",
                             "drop_condition", "off_by_one_range")


def _function(tree: ast.Module) -> ast.FunctionDef:
    for n in tree.body:
        if isinstance(n, ast.FunctionDef):
            return n
    raise ValueError("no function")


def _sites(fn: ast.FunctionDef) -> List[Tuple[str, ast.AST]]:
    sites: List[Tuple[str, ast.AST]] = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in _CMP_SWAP:
            sites.append(("cmp_swap", n))
        elif isinstance(n, ast.BinOp) and type(n.op) in _BIN_SWAP:
            sites.append(("bin_swap", n))
        elif isinstance(n, ast.Constant) and isinstance(n.value, int) and not isinstance(n.value, bool):
            sites.append(("const_shift", n))
        elif isinstance(n, ast.Return) and n.value is not None:
            sites.append(("return_change", n))
        elif isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Slice):
            sites.append(("slice_shift", n))
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _CONTAINER_SWAP:
            sites.append(("container_swap", n))
        elif isinstance(n, ast.If) and not n.orelse and len(n.body) >= 1:
            sites.append(("drop_condition", n))
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "range" and n.args:
            sites.append(("off_by_one_range", n))
    return sites


def mutate_family(src: str, rng: random.Random, families: Optional[List[str]] = None) -> Optional[Tuple[str, str]]:
    """Return ``(mutated_source, family)`` or None.  ``families`` restricts the choice (hold-out experiments pick
    the complement)."""
    try:
        tree = ast.parse(src)
        fn = _function(tree)
    except (SyntaxError, ValueError):
        return None
    sites = _sites(fn)
    if families is not None:
        sites = [s for s in sites if s[0] in families]
    if not sites:
        return None
    family, node = rng.choice(sites)
    if family == "cmp_swap":
        node.ops[0] = _CMP_SWAP[type(node.ops[0])]()
    elif family == "bin_swap":
        node.op = _BIN_SWAP[type(node.op)]()
    elif family == "const_shift":
        node.value = node.value + rng.choice([-1, 1])
    elif family == "return_change":
        v = node.value
        if isinstance(v, (ast.Name, ast.Subscript, ast.Call, ast.BinOp, ast.ListComp)):
            node.value = ast.Constant(None) if rng.random() < 0.3 else ast.UnaryOp(ast.Not(), v)
        else:
            return None
    elif family == "slice_shift":
        s = node.slice
        if s.upper is not None:
            s.upper = ast.BinOp(s.upper, ast.Sub(), ast.Constant(1))
        elif s.lower is not None:
            s.lower = ast.BinOp(s.lower, ast.Add(), ast.Constant(1))
        else:
            s.lower = ast.Constant(1)
    elif family == "container_swap":
        node.func.id = _CONTAINER_SWAP[node.func.id]
    elif family == "drop_condition":
        node.test = ast.Constant(True)
    elif family == "off_by_one_range":
        a = node.args[-1]
        node.args[-1] = ast.BinOp(a, ast.Sub() if rng.random() < 0.5 else ast.Add(), ast.Constant(1))
    ast.fix_missing_locations(tree)
    try:
        out = ast.unparse(tree)
    except Exception:  # noqa: BLE001
        return None
    if out.strip() == src.strip():
        return None
    return out.rstrip("\n") + "\n", family
