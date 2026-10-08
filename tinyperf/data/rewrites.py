"""Inverse (speed-up) rewrites used as the *proposer* for search-generated SFT
trajectories (Method C) and as a sanity oracle in tests.

Nothing here is trusted: every proposal is executed by the sandbox and only
candidates that are correct *and* measurably faster are ever used.

    propose(src) -> [(family, new_src), ...]

Families mirror :mod:`tinyperf.data.degradations` (the proposer is roughly the
inverse of the degradation operators plus a few extra peepholes)::

    hoist               loop-invariant code motion  (unhoist_*)
    set_membership      list membership -> set       (set_to_list_membership)
    minmax              sorted(x)[0|-1] -> min/max   (worse_builtins)
    dematerialize       len([..])>0 -> any(..) etc.  (materialize)
    comprehension       append-loop -> comprehension (comprehension_to_loop)
    append              n = n + [x] -> n.append(x)   (append_to_concat)
    setdefault          copy-on-append -> setdefault (dict_build)
    counter             seq.count(t) -> get(t,0)+1   (repeated_scan)
    join                concat loop -> str.join      (join_to_concat)
    dict_lookup         linear search -> dict        (dict_linear_search)
"""
from __future__ import annotations

import ast
import copy
from typing import Callable, Dict, List, Optional, Set, Tuple

from tinyperf.data.astutil import (
    PURE_BUILTIN_CALLS, all_names, fresh_name, get_function, is_pure_expr, iter_stmt_lists, mutated_names,
    param_names, parents_map, parse_module, replace_stmt, stored_names, unparse,
)

Proposal = Tuple[str, str]  # (family, source)


def _is_name(n, name: Optional[str] = None) -> bool:
    return isinstance(n, ast.Name) and (name is None or n.id == name)


def _call_to(n, *names: str) -> bool:
    return isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in names


def _load(src: str):
    tree = parse_module(src)
    return tree, get_function(tree)


def _dump_eq(a: ast.AST, b: ast.AST) -> bool:
    return ast.dump(a) == ast.dump(b)


def _find_copy(fn2: ast.FunctionDef, stmt: ast.stmt) -> Optional[ast.stmt]:
    """Locate the statement in a deep-copied function that corresponds to `stmt`."""
    for _, _, lst in iter_stmt_lists(fn2):
        for s in lst:
            if _dump_eq(s, stmt):
                return s
    return None


def _stmt_list_of(fn: ast.FunctionDef, stmt: ast.stmt) -> Optional[Tuple[List[ast.stmt], int]]:
    for _, _, lst in iter_stmt_lists(fn):
        for i, s in enumerate(lst):
            if s is stmt:
                return lst, i
    return None


def _stores_in(node: ast.AST) -> Set[str]:
    return set(stored_names(node)) | mutated_names(node)


def _negate(e: ast.expr) -> ast.expr:
    if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.Not):
        return copy.deepcopy(e.operand)
    return ast.UnaryOp(ast.Not(), copy.deepcopy(e))


def _gen_from_listcomp(lc: ast.ListComp) -> ast.GeneratorExp:
    return ast.GeneratorExp(copy.deepcopy(lc.elt), copy.deepcopy(lc.generators))


# --------------------------------------------------------------------------- #
# 1. Loop-invariant code motion
# --------------------------------------------------------------------------- #
_HOIST_CALLS = {"sorted", "set", "list", "tuple", "dict", "frozenset", "len", "sum", "max", "min", "abs", "round", "str", "int", "float"}


def _loop_like(n: ast.AST) -> bool:
    return isinstance(n, (ast.For, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))


def _hot_children(n: ast.AST) -> List[ast.AST]:
    if isinstance(n, ast.For):
        return list(n.body) + list(n.orelse)
    if isinstance(n, ast.While):
        return [n.test] + list(n.body) + list(n.orelse)
    if isinstance(n, ast.DictComp):
        return [n.key, n.value] + [i for g in n.generators for i in g.ifs] + [g.iter for g in n.generators[1:]]
    return [n.elt] + [i for g in n.generators for i in g.ifs] + [g.iter for g in n.generators[1:]]  # type: ignore[attr-defined]


def _hoistable(e: ast.AST, invariant: Set[str]) -> bool:
    if isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and e.func.id in _HOIST_CALLS:
        pass
    elif isinstance(e, ast.Call) and isinstance(e.func, ast.Attribute) and _is_name(e.func.value, "math"):
        pass
    elif isinstance(e, (ast.BinOp, ast.UnaryOp, ast.Subscript, ast.List, ast.Tuple, ast.Dict)):
        pass
    else:
        return False
    if not any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) for n in ast.walk(e)):
        return False  # constant folding is not our business
    if isinstance(e, ast.UnaryOp) and isinstance(e.operand, (ast.Constant, ast.Name)):
        return False
    return is_pure_expr(e, invariant)


def _maximal_hoistable(root: ast.AST, invariant: Set[str]) -> List[ast.AST]:
    out: List[ast.AST] = []

    def visit(n: ast.AST) -> None:
        if isinstance(n, ast.expr) and _hoistable(n, invariant):
            out.append(n)
            return
        for c in ast.iter_child_nodes(n):
            visit(c)

    visit(root)
    return out


def rewrite_hoist(src: str) -> List[str]:
    tree, fn = _load(src)
    params = param_names(fn)
    parents = parents_map(fn)
    out, seen = [], set()
    for loop in [n for n in ast.walk(fn) if _loop_like(n)]:
        # the outermost loop-like ancestor chain (inner-most first)
        chain = [loop]
        p = parents.get(loop)
        while p is not None and p is not fn:
            if _loop_like(p):
                chain.append(p)
            p = parents.get(p)
        stores_here = _stores_in(loop) | ({loop.target.id} if isinstance(loop, ast.For) and _is_name(loop.target) else set())
        invariant = (params | set(stored_names(fn))) - stores_here
        for e in [x for child in _hot_children(loop) for x in _maximal_hoistable(child, invariant)]:
            key = ast.dump(e)
            if key in seen:
                continue
            seen.add(key)
            # hoist out of the outermost enclosing loop that keeps e invariant
            target_loop = loop
            for outer in chain[1:]:
                st = _stores_in(outer) | ({outer.target.id} if isinstance(outer, ast.For) and _is_name(outer.target) else set())
                if is_pure_expr(e, (params | set(stored_names(fn))) - st):
                    target_loop = outer
                else:
                    break
            # the statement before which the hoisted assignment goes
            anchor = target_loop
            while not isinstance(anchor, ast.stmt):
                anchor = parents[anchor]
            if isinstance(anchor, ast.FunctionDef):
                continue
            t2 = copy.deepcopy(tree)
            fn2 = get_function(t2)
            anchor2 = _find_copy(fn2, anchor)
            if anchor2 is None:
                continue
            loc = _stmt_list_of(fn2, anchor2)
            if loc is None:
                continue
            lst, idx = loc
            taken = all_names(fn2)
            if isinstance(e, ast.Call) and isinstance(e.func, ast.Name):
                arg0 = e.args[0] if e.args else None
                base = f"{e.func.id}_{arg0.id}" if _is_name(arg0) else f"{e.func.id}_val"
            else:
                base = "tmp"
            name = fresh_name(base, taken)
            n_repl = _replace_expr_in(anchor2, e, ast.Name(name, ast.Load()))
            if n_repl == 0:
                continue
            lst.insert(idx, ast.Assign([ast.Name(name, ast.Store())], copy.deepcopy(e)))
            out.append(unparse(t2))
    return out


def _replace_expr_in(root: ast.AST, target: ast.AST, repl: ast.AST) -> int:
    key = ast.dump(target)
    count = {"n": 0}

    class T(ast.NodeTransformer):
        def generic_visit(self, node):
            if isinstance(node, ast.expr) and ast.dump(node) == key:
                count["n"] += 1
                return copy.deepcopy(repl)
            return super().generic_visit(node)

    T().visit(root)
    return count["n"]


# --------------------------------------------------------------------------- #
# 2. list membership -> set
# --------------------------------------------------------------------------- #
def _membership_sites(fn: ast.FunctionDef, name: str) -> List[ast.Compare]:
    return [n for n in ast.walk(fn) if isinstance(n, ast.Compare) and len(n.ops) == 1
            and isinstance(n.ops[0], (ast.In, ast.NotIn)) and _is_name(n.comparators[0], name)]


def _only_membership_uses(fn: ast.FunctionDef, name: str, extra_ok: Callable[[ast.AST, ast.AST], bool] = lambda n, p: False) -> bool:
    parents = parents_map(fn)
    ok_ids = {id(c.comparators[0]) for c in _membership_sites(fn, name)}
    n_uses = 0
    for n in ast.walk(fn):
        if _is_name(n, name) and isinstance(n.ctx, ast.Load):
            if id(n) in ok_ids:
                n_uses += 1
                continue
            if extra_ok(n, parents.get(n)):
                continue
            return False
    return n_uses >= 1


def _in_hot_region(fn: ast.FunctionDef, node: ast.AST) -> bool:
    parents = parents_map(fn)
    p = parents.get(node)
    while p is not None and p is not fn:
        if _loop_like(p):
            return True
        p = parents.get(p)
    return False


def rewrite_set_membership(src: str) -> List[str]:
    tree, fn = _load(src)
    params = param_names(fn)
    out = []
    stores = stored_names(fn)
    # candidate container names: anything appearing as the RHS of `in`
    names = {c.comparators[0].id for c in _membership_sites(fn, "") if False} | {
        n.comparators[0].id for n in ast.walk(fn)
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], (ast.In, ast.NotIn)) and _is_name(n.comparators[0])}
    for name in sorted(names):
        sites = _membership_sites(fn, name)
        if not any(_in_hot_region(fn, s) for s in sites):
            continue
        assign = None
        for _, _, lst in iter_stmt_lists(fn):
            for s in lst:
                if isinstance(s, ast.Assign) and len(s.targets) == 1 and _is_name(s.targets[0], name):
                    assign = s
        # (c) incremental: name = [] ; name.append(x) ; x in name
        if assign is not None and isinstance(assign.value, ast.List) and not assign.value.elts and stores.count(name) == 1:
            def app_ok(n, p):
                return isinstance(p, ast.Attribute) and p.attr == "append"
            if _only_membership_uses(fn, name, app_ok):
                t2 = copy.deepcopy(tree)
                fn2 = get_function(t2)
                for n in ast.walk(fn2):
                    if isinstance(n, ast.Assign) and len(n.targets) == 1 and _is_name(n.targets[0], name) and isinstance(n.value, ast.List) and not n.value.elts:
                        n.value = ast.Call(ast.Name("set", ast.Load()), [], [])
                    if isinstance(n, ast.Attribute) and _is_name(n.value, name) and n.attr == "append":
                        n.attr = "add"
                out.append(unparse(t2))
            continue
        # (a) name = <list-ish expr>, used only in membership tests -> make it a set
        if assign is not None and stores.count(name) == 1 and name not in mutated_names(fn) and _only_membership_uses(fn, name):
            v = assign.value
            repl: Optional[ast.expr] = None
            if isinstance(v, ast.ListComp):
                repl = ast.SetComp(copy.deepcopy(v.elt), copy.deepcopy(v.generators))
            elif _call_to(v, "list", "sorted", "tuple") and len(v.args) == 1 and not v.keywords:
                repl = ast.Call(ast.Name("set", ast.Load()), [copy.deepcopy(v.args[0])], [])
            elif isinstance(v, (ast.List, ast.Tuple)) and v.elts:
                repl = ast.Set([copy.deepcopy(e) for e in v.elts])
            elif isinstance(v, (ast.Name, ast.Call, ast.Subscript, ast.Attribute, ast.BinOp)) and not _call_to(v, "set", "frozenset", "dict"):
                repl = ast.Call(ast.Name("set", ast.Load()), [copy.deepcopy(v)], [])
            if repl is not None:
                t2 = copy.deepcopy(tree)
                fn2 = get_function(t2)
                a2 = _find_copy(fn2, assign)
                if a2 is not None:
                    a2.value = repl  # type: ignore[attr-defined]
                    out.append(unparse(t2))
            continue
        # (b) parameter / other sequence tested in a loop -> build a set once, before the loop
        if name in mutated_names(fn) or stores.count(name) > 1:  # params count once (the ast.arg)
            continue
        parents = parents_map(fn)
        # outermost loop statement containing all the hot membership sites
        anchors = []
        for s in sites:
            anc = None
            p = parents.get(s)
            while p is not None and p is not fn:
                if isinstance(p, ast.stmt) and parents.get(p) is fn and _loop_like(p):
                    anc = p
                if _loop_like(p):
                    stmt_p = p
                    while not isinstance(stmt_p, ast.stmt):
                        stmt_p = parents[stmt_p]
                    if parents.get(stmt_p) is fn:
                        anc = stmt_p
                p = parents.get(p)
            anchors.append(anc)
        anchors = [a for a in anchors if a is not None]
        if not anchors:
            continue
        anchor = anchors[0]
        if name not in params:
            # the (single) assignment must precede the anchor at top level
            idx_assign = next((i for i, s in enumerate(fn.body) if isinstance(s, ast.Assign) and any(_is_name(t, name) for t in s.targets)), None)
            idx_anchor = next((i for i, s in enumerate(fn.body) if s is anchor), None)
            if idx_assign is None or idx_anchor is None or idx_assign >= idx_anchor:
                continue
        t2 = copy.deepcopy(tree)
        fn2 = get_function(t2)
        anchor2 = _find_copy(fn2, anchor)
        if anchor2 is None:
            continue
        taken = all_names(fn2)
        sname = fresh_name(f"{name}_set", taken)
        changed = 0
        for n in ast.walk(anchor2):
            if isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], (ast.In, ast.NotIn)) and _is_name(n.comparators[0], name):
                n.comparators[0] = ast.Name(sname, ast.Load())
                changed += 1
        if not changed:
            continue
        lst, idx = _stmt_list_of(fn2, anchor2)  # type: ignore[misc]
        lst.insert(idx, ast.Assign([ast.Name(sname, ast.Store())], ast.Call(ast.Name("set", ast.Load()), [ast.Name(name, ast.Load())], [])))
        out.append(unparse(t2))
    return out


# --------------------------------------------------------------------------- #
# 3. sorted(x)[0] / [-1] -> min / max
# --------------------------------------------------------------------------- #
def _minmax_match(node: ast.AST) -> Optional[ast.AST]:
    if not (isinstance(node, ast.Subscript) and _call_to(node.value, "sorted") and len(node.value.args) == 1):
        return None
    call = node.value
    kws = {k.arg: k.value for k in call.keywords}
    if set(kws) - {"key", "reverse"}:
        return None
    rev = isinstance(kws.get("reverse"), ast.Constant) and kws["reverse"].value is True
    if "reverse" in kws and not rev:
        return None
    sl = node.slice
    if isinstance(sl, ast.Constant) and sl.value == 0:
        first = True
    elif isinstance(sl, ast.UnaryOp) and isinstance(sl.op, ast.USub) and isinstance(sl.operand, ast.Constant) and sl.operand.value == 1:
        first = False
    else:
        return None
    want_min = first != rev
    arg = copy.deepcopy(call.args[0])
    if isinstance(arg, ast.ListComp):
        arg = _gen_from_listcomp(arg)
    keywords = [ast.keyword("key", copy.deepcopy(kws["key"]))] if "key" in kws else []
    return ast.Call(ast.Name("min" if want_min else "max", ast.Load()), [arg], keywords)


# --------------------------------------------------------------------------- #
# 4. dematerialize
# --------------------------------------------------------------------------- #
def _demat_match(node: ast.AST) -> Optional[ast.AST]:
    # len([...]) > 0  /  != 0  ->  any(...)   ;   len([1 for .. if not c]) == 0 -> all(c for ..)
    if (isinstance(node, ast.Compare) and len(node.ops) == 1 and _call_to(node.left, "len") and len(node.left.args) == 1
            and isinstance(node.left.args[0], ast.ListComp) and isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value == 0):
        lc = node.left.args[0]
        op = node.ops[0]
        if isinstance(op, (ast.Gt, ast.NotEq)):
            elt = copy.deepcopy(lc.elt)
            gens = copy.deepcopy(lc.generators)
            if isinstance(elt, ast.Constant) and gens[-1].ifs:
                elt = gens[-1].ifs.pop()
            return ast.Call(ast.Name("any", ast.Load()), [ast.GeneratorExp(elt, gens)], [])
        if isinstance(op, ast.Eq):
            gens = copy.deepcopy(lc.generators)
            if isinstance(lc.elt, ast.Constant) and gens[-1].ifs:
                cond = gens[-1].ifs.pop()
                return ast.Call(ast.Name("all", ast.Load()), [ast.GeneratorExp(_negate(cond), gens)], [])
            return ast.UnaryOp(ast.Not(), ast.Call(ast.Name("any", ast.Load()), [ast.GeneratorExp(copy.deepcopy(lc.elt), gens)], []))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and len(node.args) == 1 and isinstance(node.args[0], ast.ListComp):
        f, lc = node.func.id, node.args[0]
        if f == "len" and isinstance(lc.elt, ast.Constant) and lc.elt.value == 1:
            return ast.Call(ast.Name("sum", ast.Load()), [_gen_from_listcomp(lc)], [])
        if f in ("sum", "max", "min", "any", "all") :
            return ast.Call(ast.Name(f, ast.Load()), [_gen_from_listcomp(lc)], [copy.deepcopy(k) for k in node.keywords])
    # ([...] + [d])[0] -> next((...), d)
    if (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value == 0 and isinstance(node.value, ast.BinOp)
            and isinstance(node.value.op, ast.Add) and isinstance(node.value.left, ast.ListComp) and isinstance(node.value.right, ast.List)
            and len(node.value.right.elts) == 1):
        return ast.Call(ast.Name("next", ast.Load()), [_gen_from_listcomp(node.value.left), copy.deepcopy(node.value.right.elts[0])], [])
    return None


def _single_site(src: str, match: Callable[[ast.AST], Optional[ast.AST]]) -> List[str]:
    tree, fn = _load(src)
    n_sites = sum(1 for n in ast.walk(fn) if match(n) is not None)
    out = []
    for idx in range(n_sites):
        t2 = copy.deepcopy(tree)
        fn2 = get_function(t2)
        counter = {"i": 0}

        class T(ast.NodeTransformer):
            def generic_visit(self, node):
                repl = match(node)
                if repl is not None:
                    if counter["i"] == idx:
                        counter["i"] += 1
                        return repl
                    counter["i"] += 1
                return super().generic_visit(node)

        T().visit(fn2)
        out.append(unparse(t2))
    return out


def rewrite_minmax(src: str) -> List[str]:
    return _single_site(src, _minmax_match)


def rewrite_dematerialize(src: str) -> List[str]:
    return _single_site(src, _demat_match)


# --------------------------------------------------------------------------- #
# 5. append loop -> list comprehension
# --------------------------------------------------------------------------- #
def _append_body(body: List[ast.stmt], acc: str) -> Optional[Tuple[ast.expr, List[ast.expr]]]:
    """Body of the form  [if c: [if d: ...]] acc.append(e)  ->  (e, [c, d, ...])."""
    conds: List[ast.expr] = []
    while True:
        if len(body) != 1:
            return None
        s = body[0]
        if isinstance(s, ast.If) and not s.orelse:
            conds.append(s.test)
            body = s.body
            continue
        if (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and isinstance(s.value.func, ast.Attribute)
                and _is_name(s.value.func.value, acc) and s.value.func.attr == "append" and len(s.value.args) == 1 and not s.value.keywords):
            return s.value.args[0], conds
        return None


def rewrite_comprehension(src: str) -> List[str]:
    tree, fn = _load(src)
    out = []
    for _, _, lst in list(iter_stmt_lists(fn)):
        for i in range(len(lst) - 1):
            init, loop = lst[i], lst[i + 1]
            if not (isinstance(init, ast.Assign) and len(init.targets) == 1 and _is_name(init.targets[0])
                    and isinstance(init.value, ast.List) and not init.value.elts and isinstance(loop, ast.For) and not loop.orelse):
                continue
            acc = init.targets[0].id
            m = _append_body(loop.body, acc)
            if m is None:
                continue
            elt, conds = m
            if any(_is_name(n, acc) for n in ast.walk(elt)) or any(_is_name(n, acc) for c in conds for n in ast.walk(c)):
                continue
            loop_vars = {n.id for n in ast.walk(loop.target) if isinstance(n, ast.Name)}
            after = lst[i + 2:]
            if loop_vars & (set(all_names(ast.Module(after, [])) ) if after else set()):
                continue  # loop var used after the loop (would no longer leak)
            t2 = copy.deepcopy(tree)
            fn2 = get_function(t2)
            init2, loop2 = _find_copy(fn2, init), _find_copy(fn2, loop)
            if init2 is None or loop2 is None:
                continue
            comp = ast.ListComp(copy.deepcopy(elt), [ast.comprehension(copy.deepcopy(loop.target), copy.deepcopy(loop.iter), [copy.deepcopy(c) for c in conds], 0)])
            loc = _stmt_list_of(fn2, loop2)
            if loc is None:
                continue
            l2, j = loc
            # fold into `return acc` when it immediately follows and acc is otherwise unused
            nxt = l2[j + 1] if j + 1 < len(l2) else None
            uses = sum(1 for n in ast.walk(fn2) if _is_name(n, acc) and isinstance(n.ctx, ast.Load))
            if isinstance(nxt, ast.Return) and _is_name(nxt.value, acc) and uses == 2:  # append target + return
                del l2[j + 1]
                l2[j] = ast.Return(comp)
                l2.remove(init2)
            else:
                l2[j] = ast.Assign([ast.Name(acc, ast.Store())], comp)
                l2.remove(init2)
            out.append(unparse(t2))
    return out


# --------------------------------------------------------------------------- #
# 6. n = n + [x] -> n.append(x)
# --------------------------------------------------------------------------- #
def rewrite_append(src: str) -> List[str]:
    tree, fn = _load(src)
    fresh_lists = {s.targets[0].id for s in ast.walk(fn) if isinstance(s, ast.Assign) and len(s.targets) == 1 and _is_name(s.targets[0])
                   and (isinstance(s.value, (ast.List, ast.ListComp)) or _call_to(s.value, "list", "sorted"))}
    out = []
    for name in sorted(fresh_lists):
        t2 = copy.deepcopy(tree)
        fn2 = get_function(t2)
        changed = 0
        for _, _, lst in list(iter_stmt_lists(fn2)):
            for i, s in enumerate(list(lst)):
                if (isinstance(s, ast.Assign) and len(s.targets) == 1 and _is_name(s.targets[0], name) and isinstance(s.value, ast.BinOp)
                        and isinstance(s.value.op, ast.Add) and _is_name(s.value.left, name) and isinstance(s.value.right, ast.List)
                        and len(s.value.right.elts) == 1):
                    lst[i] = ast.Expr(ast.Call(ast.Attribute(ast.Name(name, ast.Load()), "append", ast.Load()), [s.value.right.elts[0]], []))
                    changed += 1
                elif (isinstance(s, ast.AugAssign) and _is_name(s.target, name) and isinstance(s.op, ast.Add) and isinstance(s.value, ast.List)
                        and len(s.value.elts) == 1):
                    lst[i] = ast.Expr(ast.Call(ast.Attribute(ast.Name(name, ast.Load()), "append", ast.Load()), [s.value.elts[0]], []))
                    changed += 1
        if changed:
            out.append(unparse(t2))
    return out


# --------------------------------------------------------------------------- #
# 7. copy-on-append -> setdefault
# --------------------------------------------------------------------------- #
def _setdefault_match(stmt: ast.stmt) -> Optional[Tuple[str, ast.expr, ast.expr]]:
    if not (isinstance(stmt, ast.If) and isinstance(stmt.test, ast.Compare) and len(stmt.test.ops) == 1 and len(stmt.body) == 1 and len(stmt.orelse) == 1):
        return None
    op = stmt.test.ops[0]
    if not (isinstance(op, (ast.In, ast.NotIn)) and _is_name(stmt.test.comparators[0])):
        return None
    d, key = stmt.test.comparators[0].id, stmt.test.left
    present, absent = (stmt.body[0], stmt.orelse[0]) if isinstance(op, ast.In) else (stmt.orelse[0], stmt.body[0])

    def is_store(s, d_, key_):
        return (isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Subscript)
                and _is_name(s.targets[0].value, d_) and _dump_eq(s.targets[0].slice, key_))

    if not (is_store(present, d, key) and is_store(absent, d, key)):
        return None
    pv, av = present.value, absent.value  # type: ignore[union-attr]
    if not (isinstance(av, ast.List) and len(av.elts) == 1):
        return None
    v = av.elts[0]
    ok = (isinstance(pv, ast.BinOp) and isinstance(pv.op, ast.Add) and isinstance(pv.left, ast.Subscript) and _is_name(pv.left.value, d)
          and _dump_eq(pv.left.slice, key) and isinstance(pv.right, ast.List) and len(pv.right.elts) == 1 and _dump_eq(pv.right.elts[0], v))
    return (d, key, v) if ok else None


def rewrite_setdefault(src: str) -> List[str]:
    tree, fn = _load(src)
    out = []
    for _, _, lst in list(iter_stmt_lists(fn)):
        for stmt in list(lst):
            m = _setdefault_match(stmt)
            if m is None:
                continue
            d, key, v = m
            t2 = copy.deepcopy(tree)
            fn2 = get_function(t2)
            s2 = _find_copy(fn2, stmt)
            if s2 is None:
                continue
            new = ast.Expr(ast.Call(
                ast.Attribute(ast.Call(ast.Attribute(ast.Name(d, ast.Load()), "setdefault", ast.Load()), [copy.deepcopy(key), ast.List([], ast.Load())], []),
                              "append", ast.Load()), [copy.deepcopy(v)], []))
            replace_stmt(fn2, s2, [new])
            out.append(unparse(t2))
    return out


# --------------------------------------------------------------------------- #
# 8. counts[t] = seq.count(t)  ->  counts[t] = counts.get(t, 0) + 1
# --------------------------------------------------------------------------- #
def rewrite_counter(src: str) -> List[str]:
    tree, fn = _load(src)
    out = []
    for loop in ast.walk(fn):
        if not (isinstance(loop, ast.For) and _is_name(loop.target) and _is_name(loop.iter)):
            continue
        t, seq = loop.target.id, loop.iter.id
        for stmt in loop.body:
            if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Subscript)
                    and _is_name(stmt.targets[0].value) and _is_name(stmt.targets[0].slice, t)):
                continue
            v = stmt.value
            if not (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and _is_name(v.func.value, seq) and v.func.attr == "count"
                    and len(v.args) == 1 and _is_name(v.args[0], t)):
                continue
            d = stmt.targets[0].value.id
            t2 = copy.deepcopy(tree)
            fn2 = get_function(t2)
            s2 = _find_copy(fn2, stmt)
            if s2 is None:
                continue
            s2.value = ast.BinOp(ast.Call(ast.Attribute(ast.Name(d, ast.Load()), "get", ast.Load()), [ast.Name(t, ast.Load()), ast.Constant(0)], []),  # type: ignore[attr-defined]
                                 ast.Add(), ast.Constant(1))
            out.append(unparse(t2))
    return out


# --------------------------------------------------------------------------- #
# 9. concatenation loop -> str.join
# --------------------------------------------------------------------------- #
def _concat_loop_match(lst: List[ast.stmt], i: int) -> Optional[Tuple[int, int, str, ast.expr, ast.expr, str, ast.expr]]:
    """Match  acc = ''  [first = True]  for p in it: [if first: ... else: acc += sep]  acc += <expr>.

    Returns (start, end_exclusive, acc, iterable, loop_target, sep, piece_expr)."""
    init = lst[i]
    if not (isinstance(init, ast.Assign) and len(init.targets) == 1 and _is_name(init.targets[0])
            and isinstance(init.value, ast.Constant) and init.value.value == ""):
        return None
    acc = init.targets[0].id
    j = i + 1
    first_flag = None
    if j < len(lst) and isinstance(lst[j], ast.Assign) and len(lst[j].targets) == 1 and _is_name(lst[j].targets[0]) \
            and isinstance(lst[j].value, ast.Constant) and lst[j].value.value is True:
        first_flag = lst[j].targets[0].id
        j += 1
    if not (j < len(lst) and isinstance(lst[j], ast.For) and not lst[j].orelse):
        return None
    loop = lst[j]
    body = list(loop.body)
    sep = ""
    if first_flag is not None:
        if not (body and isinstance(body[0], ast.If) and _is_name(body[0].test, first_flag) and len(body[0].body) == 1 and len(body[0].orelse) == 1):
            return None
        el = body[0].orelse[0]
        if not (isinstance(el, ast.AugAssign) and _is_name(el.target, acc) and isinstance(el.op, ast.Add) and isinstance(el.value, ast.Constant)):
            return None
        sep = el.value.value
        body = body[1:]
    if len(body) != 1:
        return None
    s = body[0]
    if isinstance(s, ast.AugAssign) and _is_name(s.target, acc) and isinstance(s.op, ast.Add):
        piece = s.value
    elif (isinstance(s, ast.Assign) and len(s.targets) == 1 and _is_name(s.targets[0], acc) and isinstance(s.value, ast.BinOp)
          and isinstance(s.value.op, ast.Add) and _is_name(s.value.left, acc)):
        piece = s.value.right
    else:
        return None
    if any(_is_name(n, acc) for n in ast.walk(piece)):
        return None
    return i, j + 1, acc, loop.iter, loop.target, sep, piece


def rewrite_join(src: str) -> List[str]:
    tree, fn = _load(src)
    out = []
    for _, _, lst in list(iter_stmt_lists(fn)):
        for i in range(len(lst)):
            m = _concat_loop_match(lst, i)
            if m is None:
                continue
            start, end, acc, it, tgt, sep, piece = m
            if _is_name(piece) and _is_name(tgt, piece.id):
                arg: ast.expr = copy.deepcopy(it)
            else:
                arg = ast.GeneratorExp(copy.deepcopy(piece), [ast.comprehension(copy.deepcopy(tgt), copy.deepcopy(it), [], 0)])
            joined = ast.Call(ast.Attribute(ast.Constant(sep), "join", ast.Load()), [arg], [])
            t2 = copy.deepcopy(tree)
            fn2 = get_function(t2)
            first2 = _find_copy(fn2, lst[start])
            loc = _stmt_list_of(fn2, first2) if first2 is not None else None
            if loc is None:
                continue
            l2, k = loc
            n = end - start
            follow = l2[k + n] if k + n < len(l2) else None
            if isinstance(follow, ast.Return) and _is_name(follow.value, acc):
                l2[k:k + n + 1] = [ast.Return(joined)]
            else:
                l2[k:k + n] = [ast.Assign([ast.Name(acc, ast.Store())], joined)]
            out.append(unparse(t2))
    return out


# --------------------------------------------------------------------------- #
# 10. linear search over (key, value) pairs -> dict
# --------------------------------------------------------------------------- #
def _linear_get(node: ast.AST) -> Optional[Tuple[str, ast.expr, ast.expr]]:
    """next((v for k, v in reversed(pairs) if k == key), default) -> (pairs, key, default)"""
    if not (_call_to(node, "next") and len(node.args) == 2 and isinstance(node.args[0], ast.GeneratorExp)):
        return None
    g = node.args[0]
    if len(g.generators) != 1 or len(g.generators[0].ifs) != 1:
        return None
    gen = g.generators[0]
    it = gen.iter
    if _call_to(it, "reversed") and len(it.args) == 1:
        it = it.args[0]
    if not (_is_name(it) and isinstance(gen.target, ast.Tuple) and len(gen.target.elts) == 2 and all(_is_name(e) for e in gen.target.elts)):
        return None
    k, v = gen.target.elts[0].id, gen.target.elts[1].id
    cond = gen.ifs[0]
    if not (isinstance(cond, ast.Compare) and len(cond.ops) == 1 and isinstance(cond.ops[0], ast.Eq) and _is_name(cond.left, k) and _is_name(g.elt, v)):
        return None
    return it.id, cond.comparators[0], node.args[1]


def _linear_in(node: ast.AST) -> Optional[Tuple[str, ast.expr, bool]]:
    """any(k == x for k, _ in pairs) -> (pairs, x, True);  not any(...) -> (pairs, x, False)"""
    neg = False
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        node, neg = node.operand, True
    if not (_call_to(node, "any") and len(node.args) == 1 and isinstance(node.args[0], ast.GeneratorExp)):
        return None
    g = node.args[0]
    if len(g.generators) != 1 or g.generators[0].ifs:
        return None
    gen = g.generators[0]
    if not (_is_name(gen.iter) and isinstance(gen.target, ast.Tuple) and len(gen.target.elts) == 2 and _is_name(gen.target.elts[0])):
        return None
    k = gen.target.elts[0].id
    e = g.elt
    if not (isinstance(e, ast.Compare) and len(e.ops) == 1 and isinstance(e.ops[0], ast.Eq) and _is_name(e.left, k)):
        return None
    return gen.iter.id, e.comparators[0], not neg


def rewrite_dict_lookup(src: str) -> List[str]:
    tree, fn = _load(src)
    params = param_names(fn)
    pairs_names = set()
    for n in ast.walk(fn):
        m = _linear_get(n)
        if m:
            pairs_names.add(m[0])
        m2 = _linear_in(n)
        if m2:
            pairs_names.add(m2[0])
    out = []
    for pairs in sorted(pairs_names):
        if pairs in mutated_names(fn):
            continue
        t2 = copy.deepcopy(tree)
        fn2 = get_function(t2)
        taken = all_names(fn2)
        dname = fresh_name(f"{pairs}_map", taken)

        class T(ast.NodeTransformer):
            def generic_visit(self, node):
                m = _linear_get(node)
                if m and m[0] == pairs:
                    return ast.Call(ast.Attribute(ast.Name(dname, ast.Load()), "get", ast.Load()), [copy.deepcopy(m[1]), copy.deepcopy(m[2])], [])
                m2 = _linear_in(node)
                if m2 and m2[0] == pairs:
                    return ast.Compare(copy.deepcopy(m2[1]), [ast.In() if m2[2] else ast.NotIn()], [ast.Name(dname, ast.Load())])
                return super().generic_visit(node)

        T().visit(fn2)
        build = ast.Assign([ast.Name(dname, ast.Store())], ast.Call(ast.Name("dict", ast.Load()), [ast.Name(pairs, ast.Load())], []))
        if pairs in params:
            fn2.body.insert(0, build)
        else:
            idx = next((i for i, s in enumerate(fn2.body) if pairs in stored_names(s)), None)
            if idx is None:
                continue
            fn2.body.insert(idx + 1, build)
        out.append(unparse(t2))
    return out


# --------------------------------------------------------------------------- #
REWRITES: Dict[str, Callable[[str], List[str]]] = {
    "hoist": rewrite_hoist,
    "set_membership": rewrite_set_membership,
    "minmax": rewrite_minmax,
    "dematerialize": rewrite_dematerialize,
    "comprehension": rewrite_comprehension,
    "append": rewrite_append,
    "setdefault": rewrite_setdefault,
    "counter": rewrite_counter,
    "join": rewrite_join,
    "dict_lookup": rewrite_dict_lookup,
}


def propose(src: str, families: Optional[List[str]] = None, max_per_family: int = 6) -> List[Proposal]:
    """All single-application rewrites of `src` (deduplicated, excluding no-ops)."""
    out: List[Proposal] = []
    seen = {src}
    for fam, fn in REWRITES.items():
        if families and fam not in families:
            continue
        try:
            cands = fn(src)
        except Exception:  # noqa: BLE001 - a proposer bug must never kill data generation
            cands = []
        n = 0
        for c in cands:
            if not isinstance(c, str) or c in seen:
                continue
            try:
                ast.parse(c)
            except SyntaxError:
                continue
            seen.add(c)
            out.append((fam, c))
            n += 1
            if n >= max_per_family:
                break
    return out
