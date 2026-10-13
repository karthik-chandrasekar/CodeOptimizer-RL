"""Task contracts: what inputs a task admits and how behaviour is compared.

A contract is authoritative metadata written by the dataset constructor (or by a user).  It is frozen into the
task manifest and versioned.  ``python_plain_v1`` supports deterministic, synchronous functions over
``None, bool, int, str, bytes`` and ``list / tuple / dict / set / frozenset`` of those, with bounded sizes.
Floats are part of the type grammar but **deferred** in v1: a contract that mentions ``float`` is marked
unsupported (the task is excluded and the reason recorded), rather than silently adopting a tolerance.

Contracts for tasks built from TinyPerf seed generators are *generator-derived*: the dataset constructor runs its
own input generator many times at construction time (never online, never from a handful of observed calls) and
records the resulting type/bound declaration.  ``origin`` says which kind a contract is, so audits can report it.

Type grammar::

    T := none | bool | int | float | str | bytes | ?            (? = element of an always-empty container)
       | list[T] | set[T] | frozenset[T] | dict[T,T]
       | tuple[T,T,...]            (fixed arity)     | tuple[T,...]  (homogeneous, any length)
       | optional[T] | union[T,T,...]
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from tinyperf_chain.sandbox.codec import CodecError, encode, has_aliasing

CONTRACT_VERSION = "python_plain_v1"
SCALARS = ("none", "bool", "int", "float", "str", "bytes", "?")
CONTAINERS = ("list", "set", "frozenset", "dict", "tuple", "optional", "union")


# --------------------------------------------------------------------------- #
# Type expressions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TypeExpr:
    head: str
    args: Tuple["TypeExpr", ...] = ()
    variadic: bool = False      # tuple[T,...]

    def __str__(self) -> str:
        if not self.args:
            return self.head
        inner = ",".join(str(a) for a in self.args)
        if self.variadic:
            inner += ",..."
        return f"{self.head}[{inner}]"

    def mentions(self, head: str) -> bool:
        return self.head == head or any(a.mentions(head) for a in self.args)


class TypeParseError(ValueError):
    pass


def parse_type(s: str) -> TypeExpr:
    s = s.replace(" ", "")
    pos = 0

    def ident() -> str:
        nonlocal pos
        start = pos
        while pos < len(s) and (s[pos].isalnum() or s[pos] in "_?"):
            pos += 1
        if start == pos:
            raise TypeParseError(f"expected a type name at {pos} in {s!r}")
        return s[start:pos]

    def expr() -> TypeExpr:
        nonlocal pos
        head = ident()
        if head in SCALARS:
            return TypeExpr(head)
        if head not in CONTAINERS:
            raise TypeParseError(f"unknown type {head!r}")
        if pos >= len(s) or s[pos] != "[":
            raise TypeParseError(f"{head} needs [..] arguments")
        pos += 1
        args: List[TypeExpr] = []
        variadic = False
        while True:
            if s.startswith("...", pos):
                variadic = True
                pos += 3
            else:
                args.append(expr())
            if pos < len(s) and s[pos] == ",":
                pos += 1
                continue
            if pos < len(s) and s[pos] == "]":
                pos += 1
                break
            raise TypeParseError(f"bad type syntax at {pos} in {s!r}")
        arity = {"list": 1, "set": 1, "frozenset": 1, "dict": 2, "optional": 1}
        if head in arity and (len(args) != arity[head] or variadic):
            raise TypeParseError(f"{head} takes {arity[head]} argument(s)")
        if variadic and (head != "tuple" or len(args) != 1):
            raise TypeParseError("only tuple[T,...] may be variadic")
        if head in ("union",) and len(args) < 2:
            raise TypeParseError("union needs at least two members")
        return TypeExpr(head, tuple(args), variadic)

    t = expr()
    if pos != len(s):
        raise TypeParseError(f"trailing characters in {s!r}")
    return t


def describe(v: Any) -> TypeExpr:
    """Type of one concrete value (exact types; bool is not int)."""
    t = type(v)
    if v is None:
        return TypeExpr("none")
    if t is bool:
        return TypeExpr("bool")
    if t is int:
        return TypeExpr("int")
    if t is float:
        return TypeExpr("float")
    if t is str:
        return TypeExpr("str")
    if t is bytes:
        return TypeExpr("bytes")
    if t in (list, set, frozenset):
        el = unify_all([describe(x) for x in v])
        return TypeExpr(t.__name__, (el,))
    if t is dict:
        return TypeExpr("dict", (unify_all([describe(k) for k in v]), unify_all([describe(x) for x in v.values()])))
    if t is tuple:
        return TypeExpr("tuple", tuple(describe(x) for x in v))
    raise TypeParseError(f"unsupported value type {t.__name__}")


def unify_all(ts: Iterable[TypeExpr]) -> TypeExpr:
    out: Optional[TypeExpr] = None
    for t in ts:
        out = t if out is None else unify(out, t)
    return out if out is not None else TypeExpr("?")


def _union_members(t: TypeExpr) -> List[TypeExpr]:
    if t.head == "union":
        return list(t.args)
    if t.head == "optional":
        return [TypeExpr("none"), t.args[0]]
    return [t]


def _make_union(members: List[TypeExpr]) -> TypeExpr:
    uniq: List[TypeExpr] = []
    for m in members:
        merged = False
        for i, u in enumerate(uniq):
            if u == m:
                merged = True
                break
            if u.head == m.head and u.head in ("list", "set", "frozenset", "dict", "tuple"):
                try:
                    uniq[i] = unify(u, m)
                    merged = True
                    break
                except TypeParseError:
                    pass
        if not merged:
            uniq.append(m)
    uniq.sort(key=str)
    if len(uniq) == 1:
        return uniq[0]
    non_none = [u for u in uniq if u.head != "none"]
    if len(uniq) == 2 and len(non_none) == 1:
        return TypeExpr("optional", (non_none[0],))
    return TypeExpr("union", tuple(uniq))


def unify(a: TypeExpr, b: TypeExpr) -> TypeExpr:
    if a == b:
        return a
    if a.head == "?":
        return b
    if b.head == "?":
        return a
    if a.head == b.head and a.head in ("list", "set", "frozenset"):
        return TypeExpr(a.head, (unify(a.args[0], b.args[0]),))
    if a.head == b.head == "dict":
        return TypeExpr("dict", (unify(a.args[0], b.args[0]), unify(a.args[1], b.args[1])))
    if a.head == b.head == "tuple":
        if not a.variadic and not b.variadic and len(a.args) == len(b.args):
            return TypeExpr("tuple", tuple(unify(x, y) for x, y in zip(a.args, b.args)))
        el = unify_all(list(a.args) + list(b.args))
        return TypeExpr("tuple", (el,), variadic=True)
    return _make_union(_union_members(a) + _union_members(b))


def type_matches(v: Any, t: TypeExpr) -> bool:
    h = t.head
    if h == "?":
        return False   # an always-empty element type admits no element
    if h in ("optional", "union"):
        return any(type_matches(v, m) for m in _union_members(t))
    tv = type(v)
    if h == "none":
        return v is None
    if h in ("bool", "int", "float", "str", "bytes"):
        return tv.__name__ == h
    if h in ("list", "set", "frozenset"):
        return tv.__name__ == h and all(type_matches(x, t.args[0]) for x in v)
    if h == "dict":
        return tv is dict and all(type_matches(k, t.args[0]) and type_matches(x, t.args[1]) for k, x in v.items())
    if h == "tuple":
        if tv is not tuple:
            return False
        if t.variadic:
            return all(type_matches(x, t.args[0]) for x in v)
        return len(v) == len(t.args) and all(type_matches(x, a) for x, a in zip(v, t.args))
    return False


def element_hashable(t: TypeExpr) -> bool:
    """Whether every value of type ``t`` is hashable (needed by set/dict-based rewrites)."""
    if t.head in ("none", "bool", "int", "float", "str", "bytes", "?"):
        return True
    if t.head in ("list", "set", "dict"):
        return False
    if t.head == "frozenset":
        return True
    if t.head == "tuple":
        return all(element_hashable(a) for a in t.args)
    if t.head in ("optional", "union"):
        return all(element_hashable(m) for m in _union_members(t))
    return False


# --------------------------------------------------------------------------- #
# Contracts
# --------------------------------------------------------------------------- #
@dataclass
class ParamSpec:
    name: str
    type: str
    max_len: Optional[int] = None        # top-level length bound for containers / strings
    max_inner_len: Optional[int] = None  # bound on nested containers' lengths
    int_min: Optional[int] = None        # bounds on every int anywhere in the value
    int_max: Optional[int] = None
    max_str_len: Optional[int] = None    # bound on every str anywhere in the value
    alphabet: Optional[str] = None       # characters allowed in strings (None = unrestricted)

    def texpr(self) -> TypeExpr:
        return parse_type(self.type)


@dataclass
class FunctionContract:
    name: str
    params: List[ParamSpec]
    returns: str = ""                     # declared return type (informational; comparison is exact anyway)
    mutates_args: bool = False            # observed during construction; mutation is always compared

    def arity(self) -> int:
        return len(self.params)


@dataclass
class TaskContract:
    contract_version: str
    functions: List[FunctionContract]
    entrypoint: str = "_tp_entry"
    aliasing: str = "none"                # v1: inputs never share mutable objects
    external_state: str = "forbidden"
    return_policy: str = "exact_type"
    exception_policy: str = "type_only"
    mutation_policy: str = "observe_all_args"
    dict_order: str = "ignore"            # dict equality ignores insertion order (Python ==), like the legacy verifier
    float_policy: str = "deferred"        # v1: float tasks are excluded
    permitted_imports: List[str] = field(default_factory=list)
    origin: str = "declared"              # "declared" | "generator_derived_v1"
    notes: str = ""

    # -------------------------------------------------------------- #
    def function(self, name: str) -> FunctionContract:
        for f in self.functions:
            if f.name == name:
                return f
        raise KeyError(name)

    def index_of(self, name: str) -> int:
        for i, f in enumerate(self.functions):
            if f.name == name:
                return i
        raise KeyError(name)

    def unsupported_reasons(self) -> List[str]:
        out = []
        if self.contract_version != CONTRACT_VERSION:
            out.append(f"contract_version_{self.contract_version}")
        for f in self.functions:
            for p in f.params:
                try:
                    t = p.texpr()
                except TypeParseError as e:
                    out.append(f"bad_type:{f.name}.{p.name}:{e}")
                    continue
                if t.mentions("float") and self.float_policy == "deferred":
                    out.append(f"float_deferred:{f.name}.{p.name}")
                if t.mentions("?") and t.head == "?":
                    out.append(f"untyped_param:{f.name}.{p.name}")
        if self.aliasing != "none":
            out.append("aliasing_not_supported_v1")
        # The verifier implements exactly one policy per field in v1.  A contract that declares another is a
        # task/contract mismatch and is rejected at construction, never silently run under the default (3.1, 13.1).
        for fld, supported in (("external_state", ("forbidden",)), ("return_policy", ("exact_type",)),
                               ("exception_policy", ("type_only",)), ("mutation_policy", ("observe_all_args",)),
                               ("dict_order", ("ignore",)), ("float_policy", ("deferred",))):
            if getattr(self, fld) not in supported:
                out.append(f"{fld}_{getattr(self, fld)}_not_supported_v1")
        return out

    SUPPORTED_POLICIES = {"external_state": "forbidden", "return_policy": "exact_type", "exception_policy": "type_only",
                          "mutation_policy": "observe_all_args", "dict_order": "ignore", "float_policy": "deferred",
                          "aliasing": "none"}

    def supported(self) -> bool:
        return not self.unsupported_reasons()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskContract":
        d = dict(d)
        funcs = [FunctionContract(name=f["name"], params=[ParamSpec(**p) for p in f["params"]],
                                  returns=f.get("returns", ""), mutates_args=f.get("mutates_args", False))
                 for f in d.pop("functions")]
        return cls(functions=funcs, **d)

    def render(self, function: str) -> str:
        """Compact public rendering for prompts, e.g. ``items:list[str]<=4096;allowed:list[str]<=4096``."""
        f = self.function(function)
        parts = []
        for p in f.params:
            s = f"{p.name}:{p.type}"
            if p.max_len is not None:
                s += f"<={p.max_len}"
            parts.append(s)
        return ";".join(parts) if parts else "()"


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _walk_bounds(v: Any, spec: ParamSpec, depth: int) -> Optional[str]:
    t = type(v)
    if t is int:
        if spec.int_min is not None and v < spec.int_min:
            return f"int {v} < {spec.int_min}"
        if spec.int_max is not None and v > spec.int_max:
            return f"int {v} > {spec.int_max}"
        return None
    if t is str:
        if spec.max_str_len is not None and len(v) > spec.max_str_len:
            return f"str length {len(v)} > {spec.max_str_len}"
        if depth == 0 and spec.max_len is not None and len(v) > spec.max_len:
            return f"length {len(v)} > {spec.max_len}"
        if spec.alphabet is not None:
            bad = set(v) - set(spec.alphabet)
            if bad:
                return f"characters outside the declared alphabet: {sorted(bad)[:5]}"
        return None
    if t in (list, tuple, set, frozenset, dict):
        n = len(v)
        if depth == 0 and spec.max_len is not None and n > spec.max_len:
            return f"length {n} > {spec.max_len}"
        if depth > 0 and spec.max_inner_len is not None and n > spec.max_inner_len:
            return f"inner length {n} > {spec.max_inner_len}"
        items = list(v.items()) if t is dict else list(v)
        for x in items:
            if t is dict:
                r = _walk_bounds(x[0], spec, depth + 1) or _walk_bounds(x[1], spec, depth + 1)
            else:
                r = _walk_bounds(x, spec, depth + 1)
            if r:
                return r
    return None


def validate_args(contract: TaskContract, function: str, args: Sequence[Any]) -> Optional[str]:
    """None when ``args`` is a valid input for ``function`` under the contract; otherwise the reason."""
    try:
        f = contract.function(function)
    except KeyError:
        return f"unknown function {function!r}"
    if len(args) != f.arity():
        return f"arity {len(args)} != {f.arity()}"
    for a, p in zip(args, f.params):
        try:
            t = p.texpr()
        except TypeParseError as e:
            return f"bad declared type: {e}"
        if not type_matches(a, t):
            return f"{p.name}: value does not match {p.type}"
        r = _walk_bounds(a, p, 0)
        if r:
            return f"{p.name}: {r}"
    if contract.aliasing == "none":
        try:
            if has_aliasing(encode(tuple(args))):
                return "inputs share mutable objects (aliasing is not supported by this contract)"
        except CodecError as e:
            return f"unencodable input: {e}"
    return None


# --------------------------------------------------------------------------- #
# Construction-time derivation from a generator
# --------------------------------------------------------------------------- #
def _collect_stats(v: Any, st: Dict[str, Any], depth: int) -> None:
    t = type(v)
    if t is int:
        st["int_min"] = v if st["int_min"] is None else min(st["int_min"], v)
        st["int_max"] = v if st["int_max"] is None else max(st["int_max"], v)
    elif t is str:
        st["max_str_len"] = max(st["max_str_len"] or 0, len(v))
        st["chars"].update(v)
        if depth == 0:
            st["max_len"] = max(st["max_len"] or 0, len(v))
    elif t in (list, tuple, set, frozenset, dict):
        if depth == 0:
            st["max_len"] = max(st["max_len"] or 0, len(v))
        else:
            st["max_inner_len"] = max(st["max_inner_len"] or 0, len(v))
        items = list(v.items()) if t is dict else list(v)
        for x in items:
            if t is dict:
                _collect_stats(x[0], st, depth + 1)
                _collect_stats(x[1], st, depth + 1)
            else:
                _collect_stats(x, st, depth + 1)


def derive_function_contract(name: str, params: List[str], draws: Iterable[Sequence[Any]],
                             headroom: float = 1.0) -> FunctionContract:
    """Declare a function contract from the dataset constructor's own input draws (construction time only).

    ``headroom`` scales length bounds (1.0 = the constructor's observed maxima).  Integer bounds are taken as
    observed; the alphabet is the set of characters the constructor emits.
    """
    stats = [{"t": None, "int_min": None, "int_max": None, "max_str_len": None, "chars": set(), "max_len": None,
              "max_inner_len": None} for _ in params]
    n = 0
    for args in draws:
        n += 1
        if len(args) != len(params):
            raise ValueError(f"{name}: generator produced {len(args)} args for {len(params)} params")
        for st, a in zip(stats, args):
            t = describe(a)
            st["t"] = t if st["t"] is None else unify(st["t"], t)
            _collect_stats(a, st, 0)
    if n == 0:
        raise ValueError(f"{name}: no draws")
    specs = []
    for pname, st in zip(params, stats):
        scale = (lambda x: None if x is None else int(x * headroom + 0.999))
        specs.append(ParamSpec(
            name=pname, type=str(st["t"] or TypeExpr("?")), max_len=scale(st["max_len"]),
            max_inner_len=scale(st["max_inner_len"]), int_min=st["int_min"], int_max=st["int_max"],
            max_str_len=scale(st["max_str_len"]), alphabet="".join(sorted(st["chars"])) if st["chars"] else None,
        ))
    return FunctionContract(name=name, params=specs)


def sample_draws(gen: Callable[[random.Random, int], Tuple], sizes: Sequence[int], rng: random.Random,
                 extra: Sequence[Tuple] = ()) -> List[Tuple]:
    out = list(extra)
    for n in sizes:
        out.append(tuple(gen(rng, n)))
    return out
