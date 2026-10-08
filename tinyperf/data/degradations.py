"""Degradation operators: start from efficient code and make it slower.

Each operator exposes ``variants(src) -> List[str]``: every *single-site*
application of the operator to the function in ``src``.  Operators are written
to be behaviour-preserving, but nothing here is trusted — the dataset builder
runs differential tests and benchmarks and discards anything that fails.

Families (used for held-out-family evaluation splits)::

    set_to_list_membership   set lookup            -> repeated list lookup
    unhoist_call             hoisted len/sum/...   -> recomputed inside the loop
    unhoist_sort             single sort           -> repeated sort
    unhoist_convert          one conversion        -> conversion inside loop
    unhoist_arith            precomputed expr      -> repeated calculation
    unhoist_alloc            buffer reuse          -> repeated allocation
    dict_linear_search       hash map              -> linear search over pairs
    join_to_concat           str.join              -> repeated concatenation
    worse_builtins           max/min               -> sort-based selection
    materialize              lazy any/sum/next     -> materialized lists
    comprehension_to_loop    list comprehension    -> append loop
    append_to_concat         list.append           -> list = list + [x]
    dict_build               setdefault(...).append -> copy-on-append
    repeated_scan            counts.get(t,0)+1     -> tokens.count(t)
    algorithmic              hand-written complexity regressions (from seeds)
"""
from __future__ import annotations

import ast
import copy
from typing import Callable, Dict, List, Optional, Set

from tinyperf.data.astutil import (
    NameReplacer, all_names, count_loads, fresh_name, get_function, hot_regions, is_pure_expr,
    iter_stmt_lists, mutated_names, names_stored_in, param_names, parse_module, replace_stmt,
    stmt_index, stored_names, unparse, used_in_hot_region,
)


class Degradation:
    family: str = ""

    def variants(self, src: str) -> List[str]:
        raise NotImplementedError

    # helpers --------------------------------------------------------------
    @staticmethod
    def _load(src: str):
        tree = parse_module(src)
        fn = get_function(tree)
        return tree, fn


def _is_name(node, name: Optional[str] = None) -> bool:
    return isinstance(node, ast.Name) and (name is None or node.id == name)


def _is_call_to(node, name: str) -> bool:
    return isinstance(node, ast.Call) and _is_name(node.func, name)


def _membership_uses_only(fn: ast.FunctionDef, name: str, assign: ast.stmt) -> bool:
    """All Load references to `name` (outside `assign`) are `x in name` / `x not in name`."""
    membership_ids = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], (ast.In, ast.NotIn)):
            if _is_name(n.comparators[0], name):
                membership_ids.add(id(n.comparators[0]))
    for n in ast.walk(fn):
        if _is_name(n, name):
            if isinstance(n.ctx, ast.Load) and id(n) not in membership_ids:
                return False
    if stored_names(fn).count(name) != 1:
        return False
    return True


# --------------------------------------------------------------------------- #
class SetMembershipToList(Degradation):
    family = "set_to_list_membership"

    def variants(self, src: str) -> List[str]:
        out = []
        tree, fn = self._load(src)
        for _, _, lst in list(iter_stmt_lists(fn)):
            for stmt in list(lst):
                if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and _is_name(stmt.targets[0])):
                    continue
                name = stmt.targets[0].id
                val = stmt.value
                # (a) name = set(<expr>)
                if _is_call_to(val, "set") and len(val.args) == 1 and not val.keywords:
                    arg = val.args[0]
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        repl = ast.Call(ast.Name("list", ast.Load()), [copy.deepcopy(arg)], [])
                    elif isinstance(arg, ast.GeneratorExp):
                        repl = ast.ListComp(arg.elt, arg.generators)
                    elif isinstance(arg, (ast.Name, ast.ListComp, ast.Call, ast.Subscript, ast.Attribute)):
                        repl = arg
                    else:
                        continue
                    if not _membership_uses_only(fn, name, stmt):
                        continue
                    if isinstance(repl, ast.Name) and repl.id in mutated_names(fn):
                        continue
                    t2 = copy.deepcopy(tree)
                    fn2 = get_function(t2)
                    st2 = fn2.body if False else None  # noqa
                    self._apply_replace(fn2, name, repl)
                    out.append(unparse(t2))
                # (b) name = set()  with name.add(x) and membership
                elif _is_call_to(val, "set") and not val.args:
                    if not self._incremental_ok(fn, name):
                        continue
                    t2 = copy.deepcopy(tree)
                    fn2 = get_function(t2)
                    for n in ast.walk(fn2):
                        if isinstance(n, ast.Assign) and len(n.targets) == 1 and _is_name(n.targets[0], name) and _is_call_to(n.value, "set"):
                            n.value = ast.List([], ast.Load())
                        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and _is_name(n.func.value, name) and n.func.attr == "add":
                            n.func.attr = "append"
                    out.append(unparse(t2))
        return out

    @staticmethod
    def _apply_replace(fn: ast.FunctionDef, name: str, repl: ast.AST) -> None:
        for _, _, lst in list(iter_stmt_lists(fn)):
            for i, s in enumerate(list(lst)):
                if isinstance(s, ast.Assign) and len(s.targets) == 1 and _is_name(s.targets[0], name):
                    lst.remove(s)
        NameReplacer(name, repl).visit(fn)

    @staticmethod
    def _incremental_ok(fn: ast.FunctionDef, name: str) -> bool:
        if stored_names(fn).count(name) != 1:
            return False
        ok_ids = set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], (ast.In, ast.NotIn)) and _is_name(n.comparators[0], name):
                ok_ids.add(id(n.comparators[0]))
            if (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                    and _is_name(n.value.func.value, name) and n.value.func.attr == "add" and len(n.value.args) == 1):
                ok_ids.add(id(n.value.func.value))
        n_adds = 0
        for n in ast.walk(fn):
            if _is_name(n, name) and isinstance(n.ctx, ast.Load) and id(n) not in ok_ids:
                return False
            if isinstance(n, ast.Attribute) and _is_name(n.value, name) and n.attr == "add":
                n_adds += 1
        return n_adds >= 1


# --------------------------------------------------------------------------- #
class InlineInvariant(Degradation):
    """Un-hoist: `name = <pure expr>` before a loop -> recompute at every use."""

    family = "unhoist"

    def __init__(self, kind: Optional[str] = None):
        self.kind = kind

    @staticmethod
    def classify(expr: ast.AST) -> Optional[str]:
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
            f = expr.func.id
            if f == "sorted":
                return "unhoist_sort"
            if f in {"set", "list", "tuple", "dict", "frozenset", "str", "int", "float"}:
                return "unhoist_convert"
            if f in {"len", "sum", "max", "min", "abs", "round"}:
                return "unhoist_call"
            return None
        if isinstance(expr, (ast.List, ast.Tuple, ast.Dict)):
            return "unhoist_alloc"
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Mult) and isinstance(expr.left, ast.List):
            return "unhoist_alloc"
        if isinstance(expr, (ast.BinOp, ast.UnaryOp, ast.Subscript, ast.Compare, ast.BoolOp)):
            return "unhoist_arith"
        return None

    def variants(self, src: str) -> List[str]:
        out = []
        tree, fn = self._load(src)
        params = param_names(fn)
        for i, stmt in enumerate(fn.body):
            if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and _is_name(stmt.targets[0])):
                continue
            name = stmt.targets[0].id
            kind = self.classify(stmt.value)
            if kind is None or (self.kind and kind != self.kind):
                continue
            if name in params or stored_names(fn).count(name) != 1 or name in mutated_names(fn):
                continue
            later = fn.body[i + 1:]
            later_stored = names_stored_in(later)
            free = {n.id for n in ast.walk(stmt.value) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
            invariant = (params | names_stored_in(fn.body[:i])) - later_stored
            if not is_pure_expr(stmt.value, invariant | {"math"}):
                continue
            if free & later_stored:
                continue
            if not used_in_hot_region(fn, name) or count_loads(fn, name) == 0:
                continue
            t2 = copy.deepcopy(tree)
            fn2 = get_function(t2)
            target = fn2.body[i]
            fn2.body.pop(i)
            NameReplacer(name, stmt.value).visit(fn2)
            out.append((kind, unparse(t2)))
        return out  # type: ignore[return-value]


class InlineInvariantKind(Degradation):
    def __init__(self, kind: str):
        self.family = kind
        self.kind = kind
        self._inner = InlineInvariant(kind)

    def variants(self, src: str) -> List[str]:
        return [s for k, s in self._inner.variants(src) if k == self.kind]  # type: ignore[misc]


# --------------------------------------------------------------------------- #
class DictGetToLinearSearch(Degradation):
    family = "dict_linear_search"

    def variants(self, src: str) -> List[str]:
        out = []
        tree, fn = self._load(src)
        for i, stmt in enumerate(fn.body):
            if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and _is_name(stmt.targets[0])):
                continue
            name = stmt.targets[0].id
            pairs = self._pairs_source(stmt.value)
            if pairs is None or stored_names(fn).count(name) != 1 or name in mutated_names(fn):
                continue
            if pairs in names_stored_in(fn.body[i + 1:]) or pairs in mutated_names(fn):
                continue
            if not self._uses_ok(fn, name):
                continue
            t2 = copy.deepcopy(tree)
            fn2 = get_function(t2)
            fn2.body.pop(i)
            taken = all_names(fn2)
            kv = (fresh_name("key", taken), fresh_name("value", taken))
            self._rewrite_uses(fn2, name, pairs, kv)
            out.append(unparse(t2))
        return out

    @staticmethod
    def _pairs_source(val: ast.AST) -> Optional[str]:
        if _is_call_to(val, "dict") and len(val.args) == 1 and not val.keywords and _is_name(val.args[0]):
            return val.args[0].id
        if isinstance(val, ast.DictComp) and len(val.generators) == 1 and not val.generators[0].ifs:
            g = val.generators[0]
            if (isinstance(g.target, ast.Tuple) and len(g.target.elts) == 2 and all(_is_name(e) for e in g.target.elts)
                    and _is_name(g.iter) and _is_name(val.key, g.target.elts[0].id) and _is_name(val.value, g.target.elts[1].id)):
                return g.iter.id
        return None

    @staticmethod
    def _uses_ok(fn: ast.FunctionDef, name: str) -> bool:
        ok = set()
        for n in ast.walk(fn):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and _is_name(n.func.value, name)
                    and n.func.attr == "get" and 1 <= len(n.args) <= 2 and not n.keywords):
                ok.add(id(n.func.value))
            if isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], (ast.In, ast.NotIn)) and _is_name(n.comparators[0], name):
                ok.add(id(n.comparators[0]))
        n_uses = 0
        for n in ast.walk(fn):
            if _is_name(n, name) and isinstance(n.ctx, ast.Load):
                if id(n) not in ok:
                    return False
                n_uses += 1
        return n_uses >= 1

    @staticmethod
    def _rewrite_uses(fn: ast.FunctionDef, name: str, pairs: str, kv) -> None:
        k, v = kv

        class T(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                if isinstance(node.func, ast.Attribute) and _is_name(node.func.value, name) and node.func.attr == "get":
                    key = node.args[0]
                    default = node.args[1] if len(node.args) > 1 else ast.Constant(None)
                    gen = ast.GeneratorExp(
                        elt=ast.Name(v, ast.Load()),
                        generators=[ast.comprehension(
                            target=ast.Tuple([ast.Name(k, ast.Store()), ast.Name(v, ast.Store())], ast.Store()),
                            iter=ast.Call(ast.Name("reversed", ast.Load()), [ast.Name(pairs, ast.Load())], []),
                            ifs=[ast.Compare(ast.Name(k, ast.Load()), [ast.Eq()], [key])], is_async=0)])
                    return ast.Call(ast.Name("next", ast.Load()), [gen, default], [])
                return node

            def visit_Compare(self, node):
                self.generic_visit(node)
                if len(node.ops) == 1 and isinstance(node.ops[0], (ast.In, ast.NotIn)) and _is_name(node.comparators[0], name):
                    gen = ast.GeneratorExp(
                        elt=ast.Compare(ast.Name(k, ast.Load()), [ast.Eq()], [node.left]),
                        generators=[ast.comprehension(
                            target=ast.Tuple([ast.Name(k, ast.Store()), ast.Name("_", ast.Store())], ast.Store()),
                            iter=ast.Name(pairs, ast.Load()), ifs=[], is_async=0)])
                    call = ast.Call(ast.Name("any", ast.Load()), [gen], [])
                    return call if isinstance(node.ops[0], ast.In) else ast.UnaryOp(ast.Not(), call)
                return node

        T().visit(fn)


# --------------------------------------------------------------------------- #
class JoinToConcat(Degradation):
    family = "join_to_concat"

    def variants(self, src: str) -> List[str]:
        out = []
        tree, fn = self._load(src)
        for _, _, lst in list(iter_stmt_lists(fn)):
            for stmt in list(lst):
                call = None
                if isinstance(stmt, ast.Return) and stmt.value is not None:
                    call = stmt.value
                elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and _is_name(stmt.targets[0]):
                    call = stmt.value
                if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "join"
                        and isinstance(call.func.value, ast.Constant) and isinstance(call.func.value.value, str)
                        and len(call.args) == 1 and not call.keywords):
                    continue
                sep = call.func.value.value
                t2 = copy.deepcopy(tree)
                fn2 = get_function(t2)
                # locate the corresponding statement in the copy
                target = None
                for _, _, lst2 in iter_stmt_lists(fn2):
                    for s2 in lst2:
                        if ast.dump(s2) == ast.dump(stmt):
                            target = s2
                            break
                    if target is not None:
                        break
                if target is None:
                    continue
                taken = all_names(fn2)
                acc, piece, first = fresh_name("result", taken), fresh_name("part", taken), fresh_name("first", taken)
                iterable = copy.deepcopy(call.args[0])
                body: List[ast.stmt] = []
                if sep:
                    body.append(ast.If(ast.Name(first, ast.Load()),
                                       [ast.Assign([ast.Name(first, ast.Store())], ast.Constant(False))],
                                       [ast.AugAssign(ast.Name(acc, ast.Store()), ast.Add(), ast.Constant(sep))]))
                body.append(ast.AugAssign(ast.Name(acc, ast.Store()), ast.Add(), ast.Name(piece, ast.Load())))
                new: List[ast.stmt] = [ast.Assign([ast.Name(acc, ast.Store())], ast.Constant(""))]
                if sep:
                    new.append(ast.Assign([ast.Name(first, ast.Store())], ast.Constant(True)))
                new.append(ast.For(ast.Name(piece, ast.Store()), iterable, body, [], None))
                if isinstance(target, ast.Return):
                    new.append(ast.Return(ast.Name(acc, ast.Load())))
                else:
                    new.append(ast.Assign([copy.deepcopy(target.targets[0])], ast.Name(acc, ast.Load())))
                replace_stmt(fn2, target, new)
                out.append(unparse(t2))
        return out


# --------------------------------------------------------------------------- #
class _SingleSiteExprRewrite(Degradation):
    """Apply a single-expression rewrite at exactly one matching site per variant."""

    def match(self, node: ast.AST) -> Optional[ast.AST]:
        raise NotImplementedError

    def variants(self, src: str) -> List[str]:
        tree, fn = self._load(src)
        sites = [n for n in ast.walk(fn) if self.match(n) is not None]
        out = []
        for idx in range(len(sites)):
            t2 = copy.deepcopy(tree)
            fn2 = get_function(t2)
            counter = {"i": 0}
            op = self

            class T(ast.NodeTransformer):
                def generic_visit(self, node):
                    repl = op.match(node)
                    if repl is not None:
                        if counter["i"] == idx:
                            counter["i"] += 1
                            return repl
                        counter["i"] += 1
                    return super().generic_visit(node)

            T().visit(fn2)
            out.append(unparse(t2))
        return out


class WorseBuiltins(_SingleSiteExprRewrite):
    family = "worse_builtins"

    def match(self, node):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("max", "min") and len(node.args) == 1):
            return None
        kws = {k.arg: k.value for k in node.keywords}
        if set(kws) - {"key"}:
            return None
        arg = copy.deepcopy(node.args[0])
        if isinstance(arg, ast.GeneratorExp):
            arg = ast.ListComp(arg.elt, arg.generators)
        keywords = [ast.keyword("key", copy.deepcopy(kws["key"]))] if "key" in kws else []
        if node.func.id == "max":
            if "key" in kws:
                keywords.append(ast.keyword("reverse", ast.Constant(True)))
                return ast.Subscript(ast.Call(ast.Name("sorted", ast.Load()), [arg], keywords), ast.Constant(0), ast.Load())
            return ast.Subscript(ast.Call(ast.Name("sorted", ast.Load()), [arg], []), ast.UnaryOp(ast.USub(), ast.Constant(1)), ast.Load())
        return ast.Subscript(ast.Call(ast.Name("sorted", ast.Load()), [arg], keywords), ast.Constant(0), ast.Load())


class Materialize(_SingleSiteExprRewrite):
    family = "materialize"

    def match(self, node):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            return None
        f = node.func.id
        if f == "any" and len(node.args) == 1 and isinstance(node.args[0], ast.GeneratorExp):
            g = node.args[0]
            gens = copy.deepcopy(g.generators)
            gens[-1].ifs.append(copy.deepcopy(g.elt))  # any(c for ..) -> len([1 for .. if c]) > 0
            lc = ast.ListComp(ast.Constant(1), gens)
            return ast.Compare(ast.Call(ast.Name("len", ast.Load()), [lc], []), [ast.Gt()], [ast.Constant(0)])
        if f == "all" and len(node.args) == 1 and isinstance(node.args[0], ast.GeneratorExp):
            g = node.args[0]
            gens = copy.deepcopy(g.generators)
            gens[-1].ifs.append(ast.UnaryOp(ast.Not(), copy.deepcopy(g.elt)))
            lc = ast.ListComp(ast.Constant(1), gens)
            return ast.Compare(ast.Call(ast.Name("len", ast.Load()), [lc], []), [ast.Eq()], [ast.Constant(0)])
        if f == "sum" and len(node.args) == 1 and isinstance(node.args[0], ast.GeneratorExp) and not node.keywords:
            g = node.args[0]
            if isinstance(g.elt, ast.Constant) and g.elt.value == 1:
                return ast.Call(ast.Name("len", ast.Load()), [ast.ListComp(ast.Constant(1), copy.deepcopy(g.generators))], [])
            return ast.Call(ast.Name("sum", ast.Load()), [ast.ListComp(copy.deepcopy(g.elt), copy.deepcopy(g.generators))], [])
        if f == "next" and len(node.args) == 2 and isinstance(node.args[0], ast.GeneratorExp):
            g = node.args[0]
            lc = ast.ListComp(copy.deepcopy(g.elt), copy.deepcopy(g.generators))
            return ast.Subscript(ast.BinOp(lc, ast.Add(), ast.List([copy.deepcopy(node.args[1])], ast.Load())), ast.Constant(0), ast.Load())
        return None


# --------------------------------------------------------------------------- #
class ComprehensionToLoop(Degradation):
    family = "comprehension_to_loop"

    def variants(self, src: str) -> List[str]:
        out = []
        tree, fn = self._load(src)
        for _, _, lst in list(iter_stmt_lists(fn)):
            for stmt in list(lst):
                if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.ListComp):
                    comp, target_name = stmt.value, None
                elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and _is_name(stmt.targets[0]) and isinstance(stmt.value, ast.ListComp):
                    comp, target_name = stmt.value, stmt.targets[0].id
                else:
                    continue
                if len(comp.generators) != 1 or comp.generators[0].is_async:
                    continue
                g = comp.generators[0]
                loop_vars = {n.id for n in ast.walk(g.target) if isinstance(n, ast.Name)}
                outside = all_names(fn) - {n.id for n in ast.walk(comp) if isinstance(n, ast.Name)}
                if loop_vars & outside:
                    continue  # loop var would leak onto an existing name
                t2 = copy.deepcopy(tree)
                fn2 = get_function(t2)
                target = None
                for _, _, lst2 in iter_stmt_lists(fn2):
                    for s2 in lst2:
                        if ast.dump(s2) == ast.dump(stmt):
                            target = s2
                            break
                    if target is not None:
                        break
                if target is None:
                    continue
                taken = all_names(fn2)
                acc = target_name or fresh_name("out", taken)
                app = ast.Expr(ast.Call(ast.Attribute(ast.Name(acc, ast.Load()), "append", ast.Load()), [copy.deepcopy(comp.elt)], []))
                body: List[ast.stmt] = [app]
                for cond in reversed(g.ifs):
                    body = [ast.If(copy.deepcopy(cond), body, [])]
                new: List[ast.stmt] = [ast.Assign([ast.Name(acc, ast.Store())], ast.List([], ast.Load())),
                                       ast.For(copy.deepcopy(g.target), copy.deepcopy(g.iter), body, [], None)]
                if isinstance(target, ast.Return):
                    new.append(ast.Return(ast.Name(acc, ast.Load())))
                replace_stmt(fn2, target, new)
                out.append(unparse(t2))
        return out


# --------------------------------------------------------------------------- #
class AppendToConcat(Degradation):
    family = "append_to_concat"

    def variants(self, src: str) -> List[str]:
        out = []
        tree, fn = self._load(src)
        for stmt in list(ast.walk(fn)):
            if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and _is_name(stmt.targets[0])
                    and isinstance(stmt.value, ast.List) and not stmt.value.elts):
                continue
            name = stmt.targets[0].id
            if stored_names(fn).count(name) != 1 or not self._uses_ok(fn, name):
                continue
            t2 = copy.deepcopy(tree)
            fn2 = get_function(t2)
            changed = False
            for _, _, lst in list(iter_stmt_lists(fn2)):
                for i, s in enumerate(list(lst)):
                    if (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and isinstance(s.value.func, ast.Attribute)
                            and _is_name(s.value.func.value, name) and s.value.func.attr == "append" and len(s.value.args) == 1):
                        lst[i] = ast.Assign([ast.Name(name, ast.Store())],
                                            ast.BinOp(ast.Name(name, ast.Load()), ast.Add(), ast.List([s.value.args[0]], ast.Load())))
                        changed = True
            if changed:
                out.append(unparse(t2))
        return out

    @staticmethod
    def _uses_ok(fn: ast.FunctionDef, name: str) -> bool:
        parents = {}
        for p in ast.walk(fn):
            for c in ast.iter_child_nodes(p):
                parents[c] = p
        n_app = 0
        for n in ast.walk(fn):
            if not (_is_name(n, name) and isinstance(n.ctx, ast.Load)):
                continue
            p = parents.get(n)
            if isinstance(p, ast.Attribute) and p.attr == "append":
                n_app += 1
                continue
            if isinstance(p, ast.Return):
                continue
            if isinstance(p, ast.Call) and isinstance(p.func, ast.Name) and p.func.id == "len":
                continue
            if isinstance(p, (ast.Subscript, ast.UnaryOp, ast.BoolOp, ast.If, ast.While)):
                continue
            return False
        return n_app >= 1


# --------------------------------------------------------------------------- #
class SetdefaultToCopy(Degradation):
    family = "dict_build"

    def variants(self, src: str) -> List[str]:
        out = []
        tree, fn = self._load(src)
        for _, _, lst in list(iter_stmt_lists(fn)):
            for stmt in list(lst):
                m = self._match(stmt)
                if m is None:
                    continue
                d, key, val = m
                t2 = copy.deepcopy(tree)
                fn2 = get_function(t2)
                target = None
                for _, _, lst2 in iter_stmt_lists(fn2):
                    for s2 in lst2:
                        if ast.dump(s2) == ast.dump(stmt):
                            target = s2
                            break
                    if target is not None:
                        break
                if target is None:
                    continue
                sub = lambda: ast.Subscript(ast.Name(d, ast.Load()), copy.deepcopy(key), ast.Load())  # noqa: E731
                store = lambda: ast.Subscript(ast.Name(d, ast.Load()), copy.deepcopy(key), ast.Store())  # noqa: E731
                new = ast.If(
                    ast.Compare(copy.deepcopy(key), [ast.In()], [ast.Name(d, ast.Load())]),
                    [ast.Assign([store()], ast.BinOp(sub(), ast.Add(), ast.List([copy.deepcopy(val)], ast.Load())))],
                    [ast.Assign([store()], ast.List([copy.deepcopy(val)], ast.Load()))],
                )
                replace_stmt(fn2, target, [new])
                out.append(unparse(t2))
        return out

    @staticmethod
    def _match(stmt):
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
            return None
        outer = stmt.value
        if not (isinstance(outer.func, ast.Attribute) and outer.func.attr == "append" and len(outer.args) == 1):
            return None
        inner = outer.func.value
        if not (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) and inner.func.attr == "setdefault"
                and _is_name(inner.func.value) and len(inner.args) == 2 and isinstance(inner.args[1], ast.List) and not inner.args[1].elts):
            return None
        return inner.func.value.id, inner.args[0], outer.args[0]


# --------------------------------------------------------------------------- #
class CounterToCount(Degradation):
    family = "repeated_scan"

    def variants(self, src: str) -> List[str]:
        out = []
        tree, fn = self._load(src)
        for loop in ast.walk(fn):
            if not (isinstance(loop, ast.For) and _is_name(loop.target) and _is_name(loop.iter)):
                continue
            t, seq = loop.target.id, loop.iter.id
            for stmt in loop.body:
                # counts[t] = counts.get(t, 0) + 1
                if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Subscript)
                        and _is_name(stmt.targets[0].slice, t) and _is_name(stmt.targets[0].value)):
                    continue
                d = stmt.targets[0].value.id
                v = stmt.value
                if not (isinstance(v, ast.BinOp) and isinstance(v.op, ast.Add) and isinstance(v.right, ast.Constant) and v.right.value == 1
                        and isinstance(v.left, ast.Call) and isinstance(v.left.func, ast.Attribute) and _is_name(v.left.func.value, d)
                        and v.left.func.attr == "get" and len(v.left.args) == 2 and _is_name(v.left.args[0], t)
                        and isinstance(v.left.args[1], ast.Constant) and v.left.args[1].value == 0):
                    continue
                if seq in mutated_names(fn):
                    continue
                t2 = copy.deepcopy(tree)
                fn2 = get_function(t2)
                target = None
                for _, _, lst2 in iter_stmt_lists(fn2):
                    for s2 in lst2:
                        if ast.dump(s2) == ast.dump(stmt):
                            target = s2
                            break
                    if target is not None:
                        break
                if target is None:
                    continue
                target.value = ast.Call(ast.Attribute(ast.Name(seq, ast.Load()), "count", ast.Load()), [ast.Name(t, ast.Load())], [])
                out.append(unparse(t2))
        return out


# --------------------------------------------------------------------------- #
class Algorithmic(Degradation):
    """Hand-written slow variants attached to a seed (looked up by the builder)."""

    family = "algorithmic"

    def __init__(self, variants_by_fast: Optional[Dict[str, List[str]]] = None):
        self.table = variants_by_fast or {}

    def variants(self, src: str) -> List[str]:
        from tinyperf.data.astutil import normalize

        return [normalize(v) for v in self.table.get(normalize(src), [])]


ALL_OPERATORS: List[Degradation] = [
    SetMembershipToList(),
    InlineInvariantKind("unhoist_call"),
    InlineInvariantKind("unhoist_sort"),
    InlineInvariantKind("unhoist_convert"),
    InlineInvariantKind("unhoist_arith"),
    InlineInvariantKind("unhoist_alloc"),
    DictGetToLinearSearch(),
    JoinToConcat(),
    WorseBuiltins(),
    Materialize(),
    ComprehensionToLoop(),
    AppendToConcat(),
    SetdefaultToCopy(),
    CounterToCount(),
]

FAMILIES = [op.family for op in ALL_OPERATORS] + ["algorithmic"]


def applicable(src: str, operators: Optional[List[Degradation]] = None) -> Dict[str, List[str]]:
    """Map family -> list of degraded variants applicable to `src`."""
    res: Dict[str, List[str]] = {}
    for op in operators or ALL_OPERATORS:
        try:
            vs = op.variants(src)
        except Exception:  # noqa: BLE001 - an operator bug must never kill the build
            vs = []
        vs = [v for v in vs if isinstance(v, str) and v.strip() and v != src]
        if vs:
            res[op.family] = vs
    return res
