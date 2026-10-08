"""Small AST helpers used by the degradation / rewrite operators."""
from __future__ import annotations

import ast
import copy
from typing import Iterator, List, Optional, Set, Tuple

PURE_BUILTIN_CALLS = {
    "len", "sum", "max", "min", "abs", "sorted", "set", "list", "tuple", "dict", "frozenset",
    "str", "int", "float", "round", "bool", "any", "all", "divmod", "pow", "ord", "chr",
}
PURE_MATH = {"sqrt", "floor", "ceil", "log", "exp", "fabs", "hypot", "isqrt", "gcd"}
ITERATOR_CALLS = {"zip", "enumerate", "map", "filter", "reversed", "iter", "range"}
MUTATING_METHODS = {
    "append", "extend", "insert", "pop", "remove", "clear", "update", "add", "sort",
    "reverse", "setdefault", "discard", "popitem",
}


def parse_module(src: str) -> ast.Module:
    return ast.parse(src)


def unparse(tree: ast.AST) -> str:
    ast.fix_missing_locations(tree)
    return ast.unparse(tree).rstrip("\n") + "\n"


def normalize(src: str) -> str:
    """Canonical formatting (so fast/slow variants share one style)."""
    return unparse(parse_module(src))


def get_function(tree: ast.Module, name: Optional[str] = None) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and (name is None or node.name == name):
            return node
    raise ValueError(f"no function {name!r} found")


def function_name(src: str) -> str:
    return get_function(parse_module(src)).name


def param_names(fn: ast.FunctionDef) -> Set[str]:
    a = fn.args
    names = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
    if a.vararg:
        names.append(a.vararg.arg)
    if a.kwarg:
        names.append(a.kwarg.arg)
    return set(names)


def stored_names(node: ast.AST) -> List[str]:
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.append(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.Lambda)):
            pass
        elif isinstance(n, ast.arg):
            out.append(n.arg)
    return out


def all_names(node: ast.AST) -> Set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | {n.arg for n in ast.walk(node) if isinstance(n, ast.arg)}


def loaded_names(node: ast.AST) -> Set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def mutated_names(node: ast.AST) -> Set[str]:
    """Names that receive an in-place mutating method call or AugAssign/Subscript store."""
    out: Set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in MUTATING_METHODS:
            base = n.func.value
            while isinstance(base, (ast.Subscript, ast.Attribute)):
                base = base.value
            if isinstance(base, ast.Name):
                out.add(base.id)
        elif isinstance(n, ast.AugAssign):
            t = n.target
            while isinstance(t, (ast.Subscript, ast.Attribute)):
                t = t.value
            if isinstance(t, ast.Name):
                out.add(t.id)
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in targets:
                for sub in ast.walk(t):
                    if isinstance(sub, ast.Subscript):
                        b = sub.value
                        while isinstance(b, (ast.Subscript, ast.Attribute)):
                            b = b.value
                        if isinstance(b, ast.Name):
                            out.add(b.id)
    return out


def fresh_name(base: str, taken: Set[str]) -> str:
    if base not in taken:
        taken.add(base)
        return base
    i = 2
    while f"{base}{i}" in taken:
        i += 1
    taken.add(f"{base}{i}")
    return f"{base}{i}"


def is_pure_expr(expr: ast.AST, invariant_names: Set[str]) -> bool:
    """True if `expr` is side-effect free, re-evaluable, and depends only on invariant names."""
    for n in ast.walk(expr):
        if isinstance(n, ast.Name):
            if isinstance(n.ctx, ast.Load) and n.id not in invariant_names and n.id not in PURE_BUILTIN_CALLS:
                return False
        elif isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                if f.id not in PURE_BUILTIN_CALLS:
                    return False
            elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "math" and f.attr in PURE_MATH:
                pass
            else:
                return False
        elif isinstance(n, (ast.Lambda, ast.Yield, ast.YieldFrom, ast.Await, ast.NamedExpr, ast.GeneratorExp)):
            return False
        elif isinstance(n, ast.comprehension):
            return False  # comprehensions bind names; keep hoisting simple
    return True


def iter_stmt_lists(node: ast.AST) -> Iterator[Tuple[ast.AST, str, List[ast.stmt]]]:
    """Yield (parent, field, list) for every statement list under `node`."""
    for parent in ast.walk(node):
        for field in ("body", "orelse", "finalbody"):
            lst = getattr(parent, field, None)
            if isinstance(lst, list) and lst and all(isinstance(s, ast.stmt) for s in lst):
                yield parent, field, lst
        for h in getattr(parent, "handlers", []) or []:
            yield h, "body", h.body


def replace_stmt(fn: ast.FunctionDef, old: ast.stmt, new: List[ast.stmt]) -> bool:
    for _, _, lst in iter_stmt_lists(fn):
        for i, s in enumerate(lst):
            if s is old:
                lst[i : i + 1] = new
                return True
    return False


class NameReplacer(ast.NodeTransformer):
    def __init__(self, name: str, expr: ast.AST):
        self.name, self.expr, self.count = name, expr, 0

    def visit_Name(self, node: ast.Name):
        if node.id == self.name and isinstance(node.ctx, ast.Load):
            self.count += 1
            return copy.deepcopy(self.expr)
        return node


def hot_regions(fn: ast.FunctionDef) -> List[ast.AST]:
    """Subtrees evaluated repeatedly: loop bodies/tests, comprehension elt/ifs."""
    regions: List[ast.AST] = []
    for n in ast.walk(fn):
        if isinstance(n, ast.For):
            regions.extend(n.body)
            regions.extend(n.orelse)
        elif isinstance(n, ast.While):
            regions.append(n.test)
            regions.extend(n.body)
        elif isinstance(n, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            regions.append(n.elt)
            for g in n.generators:
                regions.extend(g.ifs)
            for g in n.generators[1:]:
                regions.append(g.iter)
        elif isinstance(n, ast.DictComp):
            regions.extend([n.key, n.value])
            for g in n.generators:
                regions.extend(g.ifs)
    return regions


def used_in_hot_region(fn: ast.FunctionDef, name: str) -> bool:
    for r in hot_regions(fn):
        if any(isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load) for n in ast.walk(r)):
            return True
    return False


def count_loads(node: ast.AST, name: str) -> int:
    return sum(1 for n in ast.walk(node) if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load))


def stmt_index(fn: ast.FunctionDef, stmt: ast.stmt) -> Optional[int]:
    for i, s in enumerate(fn.body):
        if s is stmt:
            return i
    return None


def names_stored_in(stmts: List[ast.stmt]) -> Set[str]:
    out: Set[str] = set()
    for s in stmts:
        out.update(stored_names(s))
        out.update(mutated_names(s))
    return out


def parents_map(root: ast.AST):
    m = {}
    for p in ast.walk(root):
        for c in ast.iter_child_nodes(p):
            m[c] = p
    return m
