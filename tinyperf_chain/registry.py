"""Trusted registries: transformation mechanisms (M2's action space) and correctness-probe recipes (M4's menu).

Transformation registry
-----------------------
Eight mechanisms.  Each entry declares legal operands, permissive *static applicability hints* (sites), declared
semantic preconditions checked against the task contract, a programmatic proposer for training-data construction
(the legacy inverse rewrites of :mod:`tinyperf.data.rewrites` plus two reduction rewrites defined here), and a
trusted list of likely risk probes.  A proposer creates candidates; it is **not** an equivalence oracle - every
candidate is executed by the verifier.

Operands use the function's *anchors*: loops and comprehensions numbered ``n00, n01, ...`` in preorder, shown to
the planner together with their relative line numbers.

Probe registry
--------------
Stable recipe ids ``p01..p14``.  A recipe builds one contract-valid argument tuple for a function from its declared
types and bounds (deterministically from ``(task, function, recipe)``); a recipe that cannot produce a valid input
under the contract never appears on the menu.  The original program supplies reference behaviour; the trusted
comparator decides whether a discrepancy exists.
"""
from __future__ import annotations

import ast
import copy
import random
import string
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from tinyperf.data.astutil import PURE_BUILTIN_CALLS, fresh_name, parents_map, stored_names
from tinyperf.data import rewrites as legacy_rewrites
from tinyperf_chain.contracts import (FunctionContract, ParamSpec, TaskContract, TypeExpr, element_hashable,
                                      parse_type, validate_args)
from tinyperf_chain.hashing import hash_json, sha256_text
from tinyperf_chain.schemas import Plan

REGISTRY_VERSION = "transforms_v1"
PROBE_REGISTRY_VERSION = "probes_v1"
PRESERVE_VALUES = ("order", "duplicates", "ties", "types", "exceptions", "mutation")

_LOOP_KINDS = {ast.For: "for", ast.While: "while", ast.ListComp: "listcomp", ast.SetComp: "setcomp",
               ast.DictComp: "dictcomp", ast.GeneratorExp: "genexp"}


# --------------------------------------------------------------------------- #
# Anchors
# --------------------------------------------------------------------------- #
@dataclass
class Anchor:
    id: str
    kind: str
    line: int          # relative to the def line (1-based)
    node: ast.AST


def _preorder(node: ast.AST) -> Iterable[ast.AST]:
    yield node
    for child in ast.iter_child_nodes(node):
        yield from _preorder(child)


def anchors(fn: ast.FunctionDef) -> List[Anchor]:
    out = []
    for n in _preorder(fn):
        k = _LOOP_KINDS.get(type(n))
        if k:
            out.append(Anchor(f"n{len(out):02d}", k, getattr(n, "lineno", fn.lineno) - fn.lineno + 1, n))
    return out


def render_anchors(fn: ast.FunctionDef) -> str:
    return ",".join(f"{a.id}:L{a.line}:{a.kind}" for a in anchors(fn)) or "none"


def operand_names(fn: ast.FunctionDef) -> List[str]:
    a = fn.args
    params = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
    locs = []
    for n in _preorder(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id not in params and n.id not in locs:
            locs.append(n.id)
    return params + locs


def _contains(outer: ast.AST, inner: ast.AST) -> bool:
    return any(x is inner for x in ast.walk(outer))


# --------------------------------------------------------------------------- #
# Sites (static applicability hints)
# --------------------------------------------------------------------------- #
@dataclass
class Site:
    src: Optional[str]
    loop: Optional[str]
    node: ast.AST
    detail: str = ""


class _FnInfo:
    def __init__(self, fn: ast.FunctionDef):
        self.fn = fn
        self.parents = parents_map(fn)
        self.anchors = anchors(fn)
        self.anchor_of = {id(a.node): a.id for a in self.anchors}

    def enclosing_loop(self, node: ast.AST) -> Optional[str]:
        p = self.parents.get(node)
        while p is not None and p is not self.fn:
            if id(p) in self.anchor_of:
                return self.anchor_of[id(p)]
            p = self.parents.get(p)
        return None

    def anchor_node(self, aid: str) -> Optional[ast.AST]:
        for a in self.anchors:
            if a.id == aid:
                return a.node
        return None

    def assigned_const_before(self, name: str, pred: Callable[[ast.AST], bool]) -> bool:
        for n in ast.walk(self.fn):
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) \
                    and n.targets[0].id == name and pred(n.value):
                return True
        return False


def _name(n: ast.AST) -> Optional[str]:
    return n.id if isinstance(n, ast.Name) else None


def _is_str_const(n: ast.AST) -> bool:
    return isinstance(n, ast.Constant) and isinstance(n.value, str)


def _is_zero(n: ast.AST) -> bool:
    return isinstance(n, ast.Constant) and type(n.value) is int and n.value == 0


_HASHED_CTORS = ("set", "frozenset", "dict", "Counter", "defaultdict")


def _hashed_locals(fn: ast.FunctionDef) -> Set[str]:
    """Locals that are only ever bound to set/dict-like values (membership on them is already O(1))."""
    kinds: Dict[str, Set[bool]] = {}
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            v = n.value
            hashed = isinstance(v, (ast.Set, ast.SetComp, ast.Dict, ast.DictComp)) or (
                isinstance(v, ast.Call) and isinstance(v.func, (ast.Name, ast.Attribute))
                and (getattr(v.func, "id", None) or getattr(v.func, "attr", None)) in _HASHED_CTORS)
            kinds.setdefault(n.targets[0].id, set()).add(hashed)
    return {k for k, v in kinds.items() if v == {True}}


def _sites_membership(info: _FnInfo) -> List[Site]:
    out = []
    hashed = _hashed_locals(info.fn)
    for n in ast.walk(info.fn):
        if isinstance(n, ast.Compare) and any(isinstance(op, (ast.In, ast.NotIn)) for op in n.ops):
            loop = info.enclosing_loop(n)
            for c in n.comparators:
                nm = _name(c)
                if nm and loop is not None and nm not in hashed:
                    out.append(Site(nm, loop, n, "membership"))
        if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple) and len(n.target.elts) == 2 and _name(n.iter):
            k = _name(n.target.elts[0])
            for b in ast.walk(n):
                if isinstance(b, ast.Compare) and any(isinstance(op, ast.Eq) for op in b.ops) and k and \
                        any(_name(x) == k for x in [b.left] + b.comparators):
                    loop = info.enclosing_loop(n) or info.anchor_of.get(id(n))
                    out.append(Site(_name(n.iter), loop, n, "linear_lookup"))
                    break
    return out


def _sites_sequence_builder(info: _FnInfo) -> List[Site]:
    out = []
    for n in ast.walk(info.fn):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and _name(n.targets[0]) and isinstance(n.value, ast.BinOp) \
                and isinstance(n.value.op, ast.Add) and _name(n.value.left) == _name(n.targets[0]) \
                and isinstance(n.value.right, (ast.List, ast.ListComp)):
            loop = info.enclosing_loop(n)
            if loop:
                out.append(Site(_name(n.targets[0]), loop, n, "concat_append"))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "append" and _name(n.func.value):
            loop = info.enclosing_loop(n)
            if loop:
                out.append(Site(_name(n.func.value), loop, n, "append_loop"))
    return out


def _sites_string_join(info: _FnInfo) -> List[Site]:
    out = []
    for n in ast.walk(info.fn):
        tgt = None
        if isinstance(n, ast.AugAssign) and isinstance(n.op, ast.Add) and _name(n.target):
            tgt = _name(n.target)
        elif isinstance(n, ast.Assign) and len(n.targets) == 1 and _name(n.targets[0]) and isinstance(n.value, ast.BinOp) \
                and isinstance(n.value.op, ast.Add) and _name(n.value.left) == _name(n.targets[0]):
            tgt = _name(n.targets[0])
        if tgt and info.enclosing_loop(n) and info.assigned_const_before(tgt, _is_str_const):
            out.append(Site(tgt, info.enclosing_loop(n), n, "concat_loop"))
    return out


def _sites_reduction(info: _FnInfo) -> List[Site]:
    out = []
    for n in ast.walk(info.fn):
        if isinstance(n, ast.For) and len(n.body) == 1 and not n.orelse:
            b = n.body[0]
            if isinstance(b, ast.AugAssign) and isinstance(b.op, ast.Add) and _name(b.target) \
                    and info.assigned_const_before(_name(b.target), _is_zero):
                out.append(Site(_name(b.target), info.anchor_of.get(id(n)), n, "sum_loop"))
            if isinstance(b, ast.If) and not b.orelse and len(b.body) == 2 and isinstance(b.body[1], ast.Break) \
                    and isinstance(b.body[0], ast.Assign) and len(b.body[0].targets) == 1 and _name(b.body[0].targets[0]) \
                    and isinstance(b.body[0].value, ast.Constant) and b.body[0].value.value is True:
                out.append(Site(_name(b.body[0].targets[0]), info.anchor_of.get(id(n)), n, "any_loop"))
    return out


def _sites_hoist(info: _FnInfo) -> List[Site]:
    out = []
    for a in info.anchors:
        if a.kind not in ("for", "while", "listcomp", "setcomp", "dictcomp", "genexp"):
            continue
        stores = set(stored_names(a.node))
        for n in ast.walk(a.node):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in PURE_BUILTIN_CALLS and n.args \
                    and all(isinstance(x, ast.Name) and x.id not in stores for x in n.args):
                if info.enclosing_loop(n) == a.id or _contains(a.node, n):
                    out.append(Site(n.args[0].id, a.id, n, f"invariant_{n.func.id}"))
    return out


def _sites_scan(info: _FnInfo) -> List[Site]:
    out = []
    for n in ast.walk(info.fn):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "count" and _name(n.func.value):
            loop = info.enclosing_loop(n)
            if loop:
                out.append(Site(_name(n.func.value), loop, n, "repeated_count"))
    return out


def _sites_selection(info: _FnInfo) -> List[Site]:
    out = []
    for n in ast.walk(info.fn):
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) \
                and n.value.func.id == "sorted" and n.value.args and _name(n.value.args[0]):
            sl = n.slice
            ok = (isinstance(sl, ast.Constant) and sl.value == 0) or (
                isinstance(sl, ast.UnaryOp) and isinstance(sl.op, ast.USub) and isinstance(sl.operand, ast.Constant) and sl.operand.value == 1)
            if ok:
                out.append(Site(_name(n.value.args[0]), info.enclosing_loop(n), n, "sorted_extreme"))
    return out


def _sites_allocation(info: _FnInfo) -> List[Site]:
    out = []
    for n in ast.walk(info.fn):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("len", "any", "all", "sum", "min", "max", "next") \
                and n.args and isinstance(n.args[0], ast.ListComp):
            gen = n.args[0].generators[0]
            out.append(Site(_name(gen.iter), info.enclosing_loop(n), n, f"materialized_{n.func.id}"))
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Subscript) \
                and _name(n.targets[0].value) and isinstance(n.value, ast.BinOp) and isinstance(n.value.op, ast.Add):
            out.append(Site(_name(n.targets[0].value), info.enclosing_loop(n), n, "copy_on_append"))
    return out


# --------------------------------------------------------------------------- #
# Additional programmatic proposers (builtin reductions)
# --------------------------------------------------------------------------- #
def _stmt_lists(fn: ast.FunctionDef):
    for node in ast.walk(fn):
        for fld in ("body", "orelse", "finalbody"):
            lst = getattr(node, fld, None)
            if isinstance(lst, list) and lst and isinstance(lst[0], ast.stmt):
                yield lst


def _rewrite_reduction(src: str) -> List[str]:
    out = []
    tree = ast.parse(src)
    fn0 = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    for li, lst in enumerate(list(_stmt_lists(fn0))):
        for i, st in enumerate(lst):
            if not (isinstance(st, ast.For) and len(st.body) == 1 and not st.orelse):
                continue
            b = st.body[0]
            t2 = copy.deepcopy(tree)
            fn2 = next(n for n in t2.body if isinstance(n, ast.FunctionDef))
            lst2 = list(_stmt_lists(fn2))[li]
            # sum loop:  acc = 0 ... for T in I: acc += E   ->   acc += sum(E for T in I)
            if isinstance(b, ast.AugAssign) and isinstance(b.op, ast.Add) and _name(b.target):
                acc = _name(b.target)
                if acc in {x.id for x in ast.walk(b.value) if isinstance(x, ast.Name)}:
                    continue
                if not any(isinstance(p, ast.Assign) and len(p.targets) == 1 and _name(p.targets[0]) == acc and _is_zero(p.value)
                           for p in lst[:i]):
                    continue
                e = copy.deepcopy(b.value)
                tgt = copy.deepcopy(st.target)
                it = copy.deepcopy(st.iter)
                if isinstance(e, ast.Name) and isinstance(tgt, ast.Name) and e.id == tgt.id:
                    arg = it
                else:
                    arg = ast.GeneratorExp(e, [ast.comprehension(tgt, it, [], 0)])
                new = ast.AugAssign(ast.Name(acc, ast.Store()), ast.Add(), ast.Call(ast.Name("sum", ast.Load()), [arg], []))
                lst2[i] = new
                ast.fix_missing_locations(t2)
                out.append(ast.unparse(t2).rstrip("\n") + "\n")
            # any loop:  f = False ... for T in I: if C: f = True; break   ->   f = any(C for T in I)
            elif isinstance(b, ast.If) and not b.orelse and len(b.body) == 2 and isinstance(b.body[1], ast.Break) \
                    and isinstance(b.body[0], ast.Assign) and len(b.body[0].targets) == 1 and _name(b.body[0].targets[0]) \
                    and isinstance(b.body[0].value, ast.Constant) and b.body[0].value.value is True:
                flag = _name(b.body[0].targets[0])
                prev = [p for p in lst[:i] if flag in stored_names(p)]
                if not prev or not (isinstance(prev[-1], ast.Assign) and isinstance(prev[-1].value, ast.Constant)
                                    and prev[-1].value.value is False):
                    continue
                gen = ast.GeneratorExp(copy.deepcopy(b.test), [ast.comprehension(copy.deepcopy(st.target), copy.deepcopy(st.iter), [], 0)])
                lst2[i] = ast.Assign([ast.Name(flag, ast.Store())], ast.Call(ast.Name("any", ast.Load()), [gen], []))
                ast.fix_missing_locations(t2)
                out.append(ast.unparse(t2).rstrip("\n") + "\n")
    return out


# --------------------------------------------------------------------------- #
# Mechanisms
# --------------------------------------------------------------------------- #
@dataclass
class Mechanism:
    id: str
    description: str
    operands: Tuple[str, ...]
    preserve: Tuple[str, ...]
    sites: Callable[[_FnInfo], List[Site]]
    backends: Tuple[str, ...]                  # legacy rewrite families
    extra_proposers: Tuple[Callable[[str], List[str]], ...] = ()
    preconditions: Tuple[str, ...] = ()
    risk_probes: Tuple[str, ...] = ()
    legacy_families: Tuple[str, ...] = ()      # degradation families whose inverse this mechanism is
    priority: int = 0


MECHANISMS: Dict[str, Mechanism] = {}


def _register(m: Mechanism) -> None:
    MECHANISMS[m.id] = m


_register(Mechanism(
    "membership_index", "avoid repeated linear membership scans / linear lookups by building a set or dict once",
    ("src", "loop"), ("order", "duplicates"), _sites_membership, ("set_membership", "dict_lookup"),
    preconditions=("elements of src are hashable", "equality semantics unchanged by hashing", "order and duplicates preserved"),
    risk_probes=("p03", "p01", "p08", "p02"), legacy_families=("set_to_list_membership", "dict_linear_search"), priority=0))
_register(Mechanism(
    "sequence_builder", "avoid repeated sequence concatenation; build with append or a comprehension",
    ("src", "loop"), ("order",), _sites_sequence_builder, ("append", "comprehension"),
    preconditions=("return type unchanged", "evaluation order unchanged", "no aliasing of the accumulator"),
    risk_probes=("p01", "p03", "p06", "p02"), legacy_families=("append_to_concat", "comprehension_to_loop"), priority=3))
_register(Mechanism(
    "string_join", "assemble many strings with str.join instead of repeated concatenation",
    ("src", "loop"), ("order",), _sites_string_join, ("join",),
    preconditions=("all pieces are str", "order and empty-input result preserved"),
    risk_probes=("p01", "p02", "p11"), legacy_families=("join_to_concat",), priority=2))
_register(Mechanism(
    "builtin_reduction", "replace an eligible manual reduction loop with sum/any",
    ("src", "loop"), ("order",), _sites_reduction, (), (_rewrite_reduction,),
    preconditions=("integer (non-float) numeric semantics", "empty-input default preserved", "early termination preserved"),
    risk_probes=("p01", "p07", "p13"), legacy_families=(), priority=5))
_register(Mechanism(
    "loop_invariant_hoist", "compute loop-invariant work once, before the loop",
    ("src", "loop"), (), _sites_hoist, ("hoist",),
    preconditions=("no side effects", "operands not mutated in the loop", "exception timing preserved"),
    risk_probes=("p01", "p02", "p09"), legacy_families=("unhoist", "unhoist_call", "unhoist_sort", "unhoist_convert",
                                                        "unhoist_arith", "unhoist_alloc"), priority=1))
_register(Mechanism(
    "scan_fusion", "replace repeated scans (e.g. seq.count in a loop) with one counting pass",
    ("src", "loop"), ("order",), _sites_scan, ("counter",),
    preconditions=("elements hashable", "short-circuiting and order preserved"),
    risk_probes=("p03", "p04", "p01"), legacy_families=("repeated_scan",), priority=4))
_register(Mechanism(
    "selection_vs_sorting", "select an extreme with min/max instead of a full sort",
    ("src", "loop"), ("ties", "order"), _sites_selection, ("minmax",),
    preconditions=("ties resolve to the same element", "empty input raises the same exception"),
    risk_probes=("p13", "p01", "p02"), legacy_families=("worse_builtins",), priority=6))
_register(Mechanism(
    "allocation_reduction", "remove intermediate materialisation (lists built only to be consumed once, copy-on-append)",
    ("src", "loop"), ("order",), _sites_allocation, ("dematerialize", "setdefault"),
    preconditions=("iterator consumed once", "no aliasing of intermediates", "evaluation order preserved"),
    risk_probes=("p01", "p08", "p03"), legacy_families=("materialize", "dict_build"), priority=7))


def registry_hash() -> str:
    return hash_json([REGISTRY_VERSION] + [[m.id, list(m.operands), list(m.preserve), list(m.backends), list(m.risk_probes)]
                                           for m in MECHANISMS.values()])


def mechanism_for_family(family: str) -> Optional[str]:
    for m in MECHANISMS.values():
        if family in m.legacy_families:
            return m.id
    return None


# --------------------------------------------------------------------------- #
# Planning support
# --------------------------------------------------------------------------- #
def _fn_from_source(fn_src: str) -> ast.FunctionDef:
    tree = ast.parse(fn_src)
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not fns:
        raise ValueError("no function in source")
    return fns[0]


def sites_for(fn: ast.FunctionDef, mechanism: str) -> List[Site]:
    try:
        return MECHANISMS[mechanism].sites(_FnInfo(fn))
    except Exception:  # noqa: BLE001 - a hint bug must never break the controller
        return []


def applicable_mechanisms(fn: ast.FunctionDef) -> List[str]:
    """Permissive static hints, in registry priority order (the M2 strategy menu)."""
    ms = sorted(MECHANISMS.values(), key=lambda m: m.priority)
    return [m.id for m in ms if sites_for(fn, m.id)]


def _param_type(contract: Optional[FunctionContract], name: Optional[str]) -> Optional[TypeExpr]:
    if contract is None or name is None:
        return None
    for p in contract.params:
        if p.name == name:
            try:
                return parse_type(p.type)
            except ValueError:
                return None
    return None


def check_preconditions(mechanism: str, src: Optional[str], fcontract: Optional[FunctionContract]) -> Optional[str]:
    """Trusted precondition checks that are representable from the contract; None = no violation found."""
    t = _param_type(fcontract, src)
    if t is None:
        return None
    if mechanism in ("membership_index", "scan_fusion"):
        if t.head in ("list", "set", "frozenset") or (t.head == "tuple" and t.variadic):
            el = t.args[0]
            if mechanism == "membership_index" and el.head == "tuple" and len(el.args) == 2:
                if not element_hashable(el.args[0]):
                    return "lookup keys are not hashable under the contract"
            elif not element_hashable(el):
                return "elements are not hashable under the contract"
        if t.head == "dict" and not element_hashable(t.args[0]):
            return "keys are not hashable"
    if mechanism == "builtin_reduction" and t.mentions("float"):
        return "float reductions change numeric semantics (sum uses compensated summation)"
    if mechanism == "string_join" and t.head in ("list", "tuple") and t.args and t.args[0].head not in ("str", "?"):
        return "pieces are not all str under the contract"
    return None


@dataclass
class PlanCheck:
    ok: bool
    reason: str = ""
    sites: List[Site] = field(default_factory=list)
    kind: str = "ok"      # unknown_transform | wrong_operand | unknown_preserve | no_site | precondition_rejected


def validate_plan(plan: Plan, fn: ast.FunctionDef, fcontract: Optional[FunctionContract]) -> PlanCheck:
    """Trusted applicability check of a plan against the current function and its contract (6.2: a plan cannot
    certify its own preconditions).  ``kind`` classifies rejections so wrong-operand and rejected-precondition rates
    can be reported separately (6.3)."""
    if plan.transform_id not in MECHANISMS:
        return PlanCheck(False, f"unknown transform {plan.transform_id!r}", kind="unknown_transform")
    m = MECHANISMS[plan.transform_id]
    names = set(operand_names(fn))
    anchor_ids = {a.id for a in anchors(fn)}
    for o in plan.operands:
        k, _, v = o.partition("=")
        if k not in m.operands:
            return PlanCheck(False, f"operand {k!r} is not legal for {m.id}", kind="wrong_operand")
        if k == "src" and v not in names:
            return PlanCheck(False, f"src={v} is not a name in the function", kind="wrong_operand")
        if k == "loop" and v not in anchor_ids:
            return PlanCheck(False, f"loop={v} is not an anchor of the function", kind="wrong_operand")
    for p in plan.preserve:
        if p not in PRESERVE_VALUES:
            return PlanCheck(False, f"unknown preserve value {p!r}", kind="unknown_preserve")
    sites = sites_for(fn, m.id)
    src, loop = plan.operand("src"), plan.operand("loop")
    matching = [s for s in sites if (src is None or s.src == src) and (loop is None or s.loop == loop)]
    if not matching:
        return PlanCheck(False, "no applicable site matches the operands", kind="no_site" if sites else "inapplicable")
    pre = check_preconditions(m.id, src or matching[0].src, fcontract)
    if pre:
        return PlanCheck(False, f"precondition: {pre}", kind="precondition_rejected")
    if m.id == "loop_invariant_hoist":
        # representable static precondition (6.2): the hoisted operand is not rebound inside the loop
        info = _FnInfo(fn)
        for s in matching:
            node = info.anchor_node(s.loop) if s.loop else None
            if node is not None and s.src in set(stored_names(node)):
                return PlanCheck(False, f"precondition: {s.src} is assigned inside {s.loop}", kind="precondition_rejected")
    return PlanCheck(True, "", matching)


def static_plans(fn: ast.FunctionDef, fcontract: Optional[FunctionContract], limit: int = 8) -> List[Plan]:
    """Rule planner support: one plan per (mechanism, site), priority order, preconditions checked."""
    out: List[Plan] = []
    seen: Set[str] = set()
    for mid in applicable_mechanisms(fn):
        m = MECHANISMS[mid]
        for s in sites_for(fn, mid):
            ops = []
            if s.src:
                ops.append(f"src={s.src}")
            if s.loop:
                ops.append(f"loop={s.loop}")
            p = Plan("", mid, tuple(ops), tuple(m.preserve))
            if p.key() in seen or check_preconditions(mid, s.src, fcontract):
                continue
            seen.add(p.key())
            out.append(p)
            if len(out) >= limit:
                return out
    return out


# --------------------------------------------------------------------------- #
# Programmatic proposals and operand extraction (training construction)
# --------------------------------------------------------------------------- #
def _stmts(fn: ast.FunctionDef) -> List[ast.stmt]:
    return [n for n in _preorder(fn) if isinstance(n, ast.stmt) and n is not fn]


def changed_statements(old_fn: ast.FunctionDef, new_fn: ast.FunctionDef) -> List[ast.stmt]:
    """Innermost statements of ``old_fn`` that do not survive (structurally) in ``new_fn``."""
    new_dumps: Dict[str, int] = {}
    for s in _stmts(new_fn):
        d = ast.dump(s)
        new_dumps[d] = new_dumps.get(d, 0) + 1
    changed = []
    for s in _stmts(old_fn):
        d = ast.dump(s)
        if new_dumps.get(d, 0) > 0:
            new_dumps[d] -= 1
        else:
            changed.append(s)
    innermost = [s for s in changed if not any(o is not s and _contains(s, o) for o in changed)]
    return innermost or changed


def site_in_change(site: Site, info: _FnInfo, changed: Sequence[ast.stmt]) -> bool:
    loop_node = info.anchor_node(site.loop) if site.loop else None
    for st in changed:
        if _contains(st, site.node) or _contains(site.node, st):
            return True
        if loop_node is not None and (_contains(loop_node, st) or _contains(st, loop_node)):
            return True
    return False


def _calls(fn: ast.AST, names: Sequence[str]) -> List[ast.Call]:
    return [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in names]


def _method_calls(fn: ast.AST, attr: str) -> List[ast.Call]:
    return [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == attr]


def _in_loop(info: _FnInfo, node: ast.AST) -> bool:
    return info.enclosing_loop(node) is not None


def _hashed_ctor_count(fn: ast.AST) -> int:
    n = 0
    for x in ast.walk(fn):
        if isinstance(x, (ast.Set, ast.SetComp, ast.Dict, ast.DictComp)):
            n += 1
        elif isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id in _HASHED_CTORS:
            n += 1
    return n


def plan_adherence(mechanism: str, old_fn: ast.FunctionDef, new_fn: ast.FunctionDef, site: Optional[Site] = None) -> bool:
    """Does ``new_fn`` structurally *realise* ``mechanism`` relative to ``old_fn``?  Site overlap alone would label
    any algorithmic rewrite that touches a loop with whatever mechanism has a site there (e.g. Kadane's algorithm
    as a 'hoist'); each mechanism therefore needs its signature construct to appear (6.2, 10.3 'correct operands')."""
    oi, ni = _FnInfo(old_fn), _FnInfo(new_fn)
    if mechanism == "membership_index":
        if _hashed_ctor_count(new_fn) <= _hashed_ctor_count(old_fn):
            return False
        new_hashed = _hashed_locals(new_fn)
        for n in ast.walk(new_fn):
            if isinstance(n, ast.Compare) and any(isinstance(op, (ast.In, ast.NotIn)) for op in n.ops) and \
                    any(_name(c) in new_hashed for c in n.comparators):
                return True
            if isinstance(n, ast.Subscript) and _name(n.value) in new_hashed:
                return True
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get" and _name(n.func.value) in new_hashed:
                return True
        return False
    if mechanism == "sequence_builder":
        old_concat = sum(1 for s in _sites_sequence_builder(oi) if s.detail == "concat_append")
        new_concat = sum(1 for s in _sites_sequence_builder(ni) if s.detail == "concat_append")
        old_app = sum(1 for c in _method_calls(old_fn, "append") if _in_loop(oi, c))
        new_app = sum(1 for c in _method_calls(new_fn, "append") if _in_loop(ni, c))
        old_comp = sum(1 for n in ast.walk(old_fn) if isinstance(n, (ast.ListComp, ast.GeneratorExp)))
        new_comp = sum(1 for n in ast.walk(new_fn) if isinstance(n, (ast.ListComp, ast.GeneratorExp)))
        if new_concat < old_concat:
            return True                                    # concatenation replaced by append / comprehension
        return new_app < old_app and new_comp > old_comp   # append loop replaced by a comprehension
    if mechanism == "string_join":
        return len(_method_calls(new_fn, "join")) > len(_method_calls(old_fn, "join"))
    if mechanism == "builtin_reduction":
        red = ("sum", "any", "all", "min", "max")
        return len(_calls(new_fn, red)) > len(_calls(old_fn, red)) and \
            sum(1 for n in ast.walk(new_fn) if isinstance(n, ast.For)) < sum(1 for n in ast.walk(old_fn) if isinstance(n, ast.For))
    if mechanism == "loop_invariant_hoist":
        def inv_calls(fn: ast.FunctionDef, info: _FnInfo):
            inside, outside = [], []
            for c in _calls(fn, list(PURE_BUILTIN_CALLS)):
                (inside if _in_loop(info, c) else outside).append(ast.dump(c))
            return inside, outside
        oi_in, oi_out = inv_calls(old_fn, oi)
        ni_in, ni_out = inv_calls(new_fn, ni)
        moved = [d for d in oi_in if d not in ni_in and d in ni_out]
        return bool(moved) or (len(ni_in) < len(oi_in) and len(ni_out) > len(oi_out))
    if mechanism == "scan_fusion":
        old_counts = sum(1 for c in _method_calls(old_fn, "count") if _in_loop(oi, c))
        new_counts = sum(1 for c in _method_calls(new_fn, "count") if _in_loop(ni, c))
        return new_counts < old_counts
    if mechanism == "selection_vs_sorting":
        old_sorted = len(_sites_selection(oi))
        new_sorted = len(_sites_selection(ni))
        return new_sorted < old_sorted and len(_calls(new_fn, ("min", "max"))) > len(_calls(old_fn, ("min", "max")))
    if mechanism == "allocation_reduction":
        old_mat = sum(1 for s in _sites_allocation(oi))
        new_mat = sum(1 for s in _sites_allocation(ni))
        new_gen = sum(1 for n in ast.walk(new_fn) if isinstance(n, ast.GeneratorExp))
        old_gen = sum(1 for n in ast.walk(old_fn) if isinstance(n, ast.GeneratorExp))
        return new_mat < old_mat and (new_gen > old_gen or len(_method_calls(new_fn, "append")) > len(_method_calls(old_fn, "append"))
                                      or len(_method_calls(new_fn, "setdefault")) > len(_method_calls(old_fn, "setdefault")))
    return False


def extract_plan(old_fn_src: str, new_fn_src: str, mechanism: str, function_id: str = "", strict: bool = True) -> Optional[Plan]:
    """Operands of the plan a verified rewrite instantiates (training labels for M2/M3).  With ``strict`` (the
    default) the rewrite must also pass :func:`plan_adherence`; a rewrite that merely touches the site of a mechanism
    without realising it is *unlabelled*, never attributed to that mechanism."""
    try:
        old_fn, new_fn = _fn_from_source(old_fn_src), _fn_from_source(new_fn_src)
    except (SyntaxError, ValueError):
        return None
    info = _FnInfo(old_fn)
    sites = MECHANISMS[mechanism].sites(info) if mechanism in MECHANISMS else []
    changed = changed_statements(old_fn, new_fn)
    if not changed:
        return None
    for s in sites:
        if site_in_change(s, info, changed):
            if strict and not plan_adherence(mechanism, old_fn, new_fn, s):
                continue
            ops = []
            if s.src:
                ops.append(f"src={s.src}")
            if s.loop:
                ops.append(f"loop={s.loop}")
            return Plan(function_id, mechanism, tuple(ops), tuple(MECHANISMS[mechanism].preserve))
    return None


def label_rewrite(old_fn_src: str, new_fn_src: str, function_id: str = "") -> Optional[Plan]:
    """The plan a rewrite instantiates, over all mechanisms in priority order, or None (unlabelled rewrite)."""
    for m in sorted(MECHANISMS.values(), key=lambda x: x.priority):
        p = extract_plan(old_fn_src, new_fn_src, m.id, function_id)
        if p is not None:
            return p
    return None


def propose(fn_src: str, mechanism: str, plan: Optional[Plan] = None, max_per_backend: int = 6) -> List[str]:
    """Candidate replacement functions for ``mechanism`` (filtered to the plan's site when given)."""
    m = MECHANISMS[mechanism]
    cands: List[str] = []
    if m.backends:
        for _fam, c in legacy_rewrites.propose(fn_src, families=list(m.backends), max_per_family=max_per_backend):
            cands.append(c)
    for extra in m.extra_proposers:
        try:
            cands.extend(extra(fn_src)[:max_per_backend])
        except Exception:  # noqa: BLE001
            pass
    out, seen = [], set()
    for c in cands:
        try:
            norm = ast.unparse(ast.parse(c)).rstrip("\n") + "\n"
        except SyntaxError:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        if plan is not None and (plan.operands):
            got = extract_plan(fn_src, norm, mechanism)
            if got is None:
                continue
            want_src, want_loop = plan.operand("src"), plan.operand("loop")
            if (want_src and got.operand("src") != want_src) or (want_loop and got.operand("loop") != want_loop):
                continue
        out.append(norm)
    return out


# --------------------------------------------------------------------------- #
# Probe recipes
# --------------------------------------------------------------------------- #
PROBE_RECIPES: Dict[str, str] = {
    "p01": "empty_minimal", "p02": "single_element", "p03": "duplicates_and_order", "p04": "all_equal",
    "p05": "sorted_ascending", "p06": "sorted_descending", "p07": "mixed_sign", "p08": "missing_keys",
    "p09": "boundary_values", "p10": "large_magnitude", "p11": "whitespace_runs", "p12": "nested_empty",
    "p13": "ties_for_extremes", "p14": "random_small", "p15": "aliased_inputs",
}
# p15 (aliasing where supported, 8.1): two arguments share one mutable object.  Contract v1 declares
# ``aliasing: none`` for every task, so the recipe never reaches a menu until a contract version supports it.
FIXED_EDGE_SUITE = ("p01", "p02", "p03")


class _Inapplicable(Exception):
    pass


def probe_registry_hash() -> str:
    return hash_json([PROBE_REGISTRY_VERSION, PROBE_RECIPES])


class _Gen:
    def __init__(self, spec: ParamSpec, rng: random.Random):
        self.spec = spec
        self.rng = rng
        self.alpha = spec.alphabet if spec.alphabet else string.ascii_lowercase
        self.letters = "".join(c for c in self.alpha if not c.isspace()) or self.alpha
        lo = spec.int_min if spec.int_min is not None else -1000
        hi = spec.int_max if spec.int_max is not None else 1000
        self.lo, self.hi = lo, hi

    def cap(self, n: int, depth: int) -> int:
        lim = self.spec.max_len if depth == 0 else self.spec.max_inner_len
        if lim is None:
            lim = 64
        return max(0, min(n, lim))

    def int_small(self) -> int:
        a, b = max(self.lo, -9), min(self.hi, 9)
        if a > b:
            a, b = self.lo, self.hi
        return self.rng.randint(a, b)

    def distinct_ints(self, k: int) -> List[int]:
        a, b = max(self.lo, -9), min(self.hi, 9)
        if b - a + 1 < k:
            a, b = self.lo, self.hi
        if b - a + 1 < k:
            raise _Inapplicable()
        return sorted(self.rng.sample(range(a, b + 1), k))

    def word(self, n: Optional[int] = None) -> str:
        msl = self.spec.max_str_len if self.spec.max_str_len is not None else 8
        n = min(msl, n if n is not None else self.rng.randint(1, 3))
        return "".join(self.rng.choice(self.letters) for _ in range(max(0, n)))

    def distinct_words(self, k: int) -> List[str]:
        out: Set[str] = set()
        tries = 0
        while len(out) < k and tries < 200:
            out.add(self.word(self.rng.randint(1, 3)))
            tries += 1
        if len(out) < k:
            raise _Inapplicable()
        return sorted(out)

    def scalar(self, t: TypeExpr, mode: str) -> object:
        h = t.head
        if h == "none":
            return None
        if h == "bool":
            if mode not in ("empty", "small", "single"):
                raise _Inapplicable()
            return mode != "empty" and self.rng.random() < 0.5
        if h == "int":
            if mode == "empty":
                return 0 if self.lo <= 0 <= self.hi else self.lo
            if mode == "boundary":
                return self.rng.choice([self.lo, self.hi, max(self.lo, min(self.hi, 0)), max(self.lo, min(self.hi, 1))])
            if mode == "large":
                return self.rng.choice([self.lo, self.hi])
            if mode == "mixed_sign":
                if not (self.lo < 0 < self.hi):
                    raise _Inapplicable()
                return self.rng.choice([-1, 0, 1])
            if mode in ("small", "single"):
                return self.int_small()
            raise _Inapplicable()
        if h == "str":
            if mode == "empty":
                return ""
            if mode == "single":
                return self.word(1)
            if mode == "whitespace":
                if " " not in self.alpha:
                    raise _Inapplicable()
                w = self.word(2)
                return self._cap_str(f"  {w}  {w} ")
            if mode in ("dupes", "all_equal"):
                c = self.rng.choice(self.letters)
                return self._cap_str(c * 4 if mode == "all_equal" else (c + self.rng.choice(self.letters)) * 2)
            if mode == "small":
                return self.word()
            raise _Inapplicable()
        if h == "bytes":
            if mode == "empty":
                return b""
            if mode == "small":
                return bytes(self.rng.randrange(256) for _ in range(3))
            raise _Inapplicable()
        raise _Inapplicable()

    def _cap_str(self, s: str) -> str:
        lim = self.spec.max_str_len
        if lim is None:
            lim = self.spec.max_len
        return s if lim is None else s[:lim]

    def elems(self, t: TypeExpr, mode: str, depth: int) -> List[object]:
        if t.head == "?":
            if mode == "empty":
                return []
            raise _Inapplicable()
        if mode == "empty":
            return []
        if mode == "single":
            return [self.value(t, "small", depth)]
        if mode == "nested_empty":
            if t.head not in ("list", "tuple", "set", "frozenset", "dict"):
                raise _Inapplicable()
            return [self.value(t, "empty", depth), self.value(t, "small", depth), self.value(t, "empty", depth)]
        base = self._distinct(t, 3, depth)
        if mode == "dupes":
            return [base[1], base[0], base[1], base[2], base[0]]
        if mode == "all_equal":
            return [base[0]] * 4
        if mode == "sorted_asc":
            return list(base)
        if mode == "sorted_desc":
            return list(reversed(base))
        if mode == "ties":
            return [base[2], base[0], base[2], base[1], base[0]]
        if mode == "mixed_sign":
            if t.head != "int":
                raise _Inapplicable()
            vals = [v for v in (-3, 0, 2, -1, 5) if self.lo <= v <= self.hi]
            if not any(v < 0 for v in vals) or not any(v > 0 for v in vals):
                raise _Inapplicable()
            return vals
        if mode == "large":
            if t.head != "int":
                raise _Inapplicable()
            return [self.hi, self.lo, self.hi]
        if mode == "boundary":
            return [self.value(t, "boundary", depth) for _ in range(2)]
        if mode == "whitespace":
            if t.head != "str":
                raise _Inapplicable()
            return [self.value(t, "whitespace", depth), self.value(t, "small", depth)]
        n = self.rng.randint(2, 6)
        return [self.value(t, "small", depth) for _ in range(n)]

    def _distinct(self, t: TypeExpr, k: int, depth: int) -> List[object]:
        if t.head == "int":
            return self.distinct_ints(k)
        if t.head == "str":
            return self.distinct_words(k)
        if t.head == "bool":
            raise _Inapplicable()
        vals: List[object] = []
        tries = 0
        while len(vals) < k and tries < 50:
            v = self.value(t, "small", depth)
            if v not in vals:
                vals.append(v)
            tries += 1
        if len(vals) < k:
            raise _Inapplicable()
        try:
            return sorted(vals)  # type: ignore[type-var]
        except TypeError:
            return vals

    def value(self, t: TypeExpr, mode: str, depth: int = 0) -> object:
        h = t.head
        if h in ("optional", "union"):
            members = [TypeExpr("none"), t.args[0]] if h == "optional" else list(t.args)
            if mode == "empty" and any(m.head == "none" for m in members):
                return None
            non_none = [m for m in members if m.head != "none"]
            return self.value(self.rng.choice(non_none or members), mode, depth)
        if h in ("none", "bool", "int", "str", "bytes"):
            return self.scalar(t, mode)
        if h == "float":
            raise _Inapplicable()
        if h == "tuple" and not t.variadic:
            sub = "small" if mode in ("dupes", "all_equal", "sorted_asc", "sorted_desc", "ties", "nested_empty") else mode
            return tuple(self.value(a, sub, depth + 1) for a in t.args)
        if h in ("list", "set", "frozenset") or (h == "tuple" and t.variadic):
            el = t.args[0]
            items = self.elems(el, mode, depth + 1)
            items = items[: self.cap(len(items), depth)]
            if mode not in ("empty",) and not items and self.cap(1, depth) == 0:
                raise _Inapplicable()
            if h == "list":
                return list(items)
            if h == "tuple":
                return tuple(items)
            try:
                return set(items) if h == "set" else frozenset(items)
            except TypeError:
                raise _Inapplicable() from None
        if h == "dict":
            kt, vt = t.args
            if mode == "empty":
                return {}
            ks = self.elems(kt, "sorted_asc" if mode not in ("single",) else "single", depth + 1)
            out = {}
            for k in ks[: self.cap(len(ks), depth)]:
                try:
                    out[k] = self.value(vt, "small" if mode not in ("ties", "all_equal") else "small", depth + 1)
                except TypeError:
                    raise _Inapplicable() from None
            return out
        raise _Inapplicable()


_MODE = {"p01": "empty", "p02": "single", "p03": "dupes", "p04": "all_equal", "p05": "sorted_asc", "p06": "sorted_desc",
         "p07": "mixed_sign", "p09": "boundary", "p10": "large", "p11": "whitespace", "p12": "nested_empty",
         "p13": "ties", "p14": "small"}


def _probe_rng(task_id: str, function: str, pid: str) -> random.Random:
    return random.Random(int(sha256_text(f"probe|{task_id}|{function}|{pid}")[:16], 16))


def build_probe(contract: TaskContract, task_id: str, function: str, pid: str) -> Optional[Tuple]:
    """One contract-valid argument tuple for recipe ``pid`` (None when inapplicable under the contract)."""
    if pid not in PROBE_RECIPES:
        return None
    fc = contract.function(function)
    rng = _probe_rng(task_id, function, pid)
    try:
        types = [p.texpr() for p in fc.params]
    except ValueError:
        return None
    try:
        if pid == "p08":
            args = _build_missing(fc, types, rng)
        elif pid == "p15":
            if contract.aliasing == "none":
                return None
            args = _build_aliased(fc, types, rng)
        else:
            mode = _MODE[pid]
            everywhere = mode in ("empty", "boundary", "large", "mixed_sign", "small")
            containers = [i for i, t in enumerate(types) if t.head in ("list", "set", "frozenset", "dict", "tuple", "str")]
            applied = mode == "small"
            out = []
            for i, (p, t) in enumerate(zip(fc.params, types)):
                g = _Gen(p, rng)
                targeted = everywhere or (containers and i == containers[0])
                if targeted and mode != "small":
                    try:
                        out.append(g.value(t, mode))
                        if t.head != "none":
                            applied = True
                        continue
                    except _Inapplicable:
                        if mode == "empty":
                            raise
                out.append(g.value(t, "small"))
            if not applied:
                return None
            args = tuple(out)
    except _Inapplicable:
        return None
    if validate_args(contract, function, args) is not None:
        return None
    return tuple(args)


def _build_missing(fc: FunctionContract, types: List[TypeExpr], rng: random.Random) -> Tuple:
    """Two collections of the same element type where the queried elements are absent from the lookup one."""
    idx = [i for i, t in enumerate(types) if t.head in ("list", "set", "frozenset", "tuple") and t.args]
    dicts = [i for i, t in enumerate(types) if t.head == "dict"]
    gens = [_Gen(p, rng) for p in fc.params]
    args: List[object] = [g.value(t, "small") for g, t in zip(gens, types)]
    if len(idx) >= 2 or (idx and dicts):
        q = idx[0]
        lk = idx[1] if len(idx) >= 2 else dicts[0]
        el_q = types[q].args[0]
        el_l = types[lk].args[0]
        if str(el_q) != str(el_l) and el_l.head != "?":
            raise _Inapplicable()
        vals = gens[q]._distinct(el_q, 6, 1)
        a, b = vals[:3], vals[3:]
        args[q] = type(args[q])(a) if not isinstance(args[q], dict) else args[q]
        if types[lk].head == "dict":
            args[lk] = {k: gens[lk].value(types[lk].args[1], "small", 1) for k in b}
        else:
            args[lk] = type(args[lk])(b)
        return tuple(args)
    raise _Inapplicable()


def _build_aliased(fc: FunctionContract, types: List[TypeExpr], rng: random.Random) -> Tuple:
    """Two parameters of the same mutable container type bound to the *same* object."""
    idx = [i for i, t in enumerate(types) if t.head in ("list", "set", "dict")]
    gens = [_Gen(p, rng) for p in fc.params]
    args: List[object] = [g.value(t, "small") for g, t in zip(gens, types)]
    pairs = [(a, b) for a in idx for b in idx if a < b and str(types[a]) == str(types[b])]
    if not pairs:
        raise _Inapplicable()
    a, b = pairs[0]
    args[b] = args[a]
    return tuple(args)


@dataclass
class ProbeMenu:
    function: str
    entries: List[Tuple[str, str]]          # (pid, recipe name)
    inputs: Dict[str, Tuple]                # trusted: pid -> argument tuple (never rendered into prompts)

    def ids(self) -> List[str]:
        return [p for p, _ in self.entries]

    def render(self) -> str:
        return ",".join(f"{p}:{n}" for p, n in self.entries) or "none"


def probe_menu(contract: TaskContract, task_id: str, function: str, menu_size: int = 12) -> ProbeMenu:
    entries, inputs, seen = [], {}, set()
    for pid, name in PROBE_RECIPES.items():
        args = build_probe(contract, task_id, function, pid)
        if args is None:
            continue
        key = repr(args)
        if key in seen:
            continue
        seen.add(key)
        entries.append((pid, name))
        inputs[pid] = args
        if len(entries) >= menu_size:
            break
    return ProbeMenu(function, entries, inputs)
