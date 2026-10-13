"""Program versions and scope-checked function patches.

A :class:`Program` is one immutable version of a task's file (normalized source, without the hidden dispatcher).
The episode keeps the immutable original ``P0`` and the current best ``Pb`` separately; candidates are separate
objects until accepted.  :func:`apply_patch` replaces exactly one top-level function and enforces the patch scope:

* the payload is exactly one ``def`` of the selected function (no other top-level statements, no imports at
  module level, no decorators, no async);
* the signature (arguments, defaults, annotations, return annotation) is unchanged;
* imports inside the function are limited to the task's permitted modules;
* every free name resolves to a parameter/local, a sandbox builtin, or a module-level name of the same program;
* no ``global`` / ``nonlocal`` / class definitions / generators, and no writes to module-level state (attribute or
  subscript stores, augmented assignment, mutating method calls on module-level objects);
* the patch applies only to the program version it was produced for (``base_hash``), otherwise it is stale.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List, Optional, Sequence, Set, Tuple

from tinyperf.data.astutil import MUTATING_METHODS
from tinyperf.env.files import ENTRY, make_driver, with_driver
from tinyperf.env.worker import SAFE_BUILTIN_NAMES
from tinyperf_chain.hashing import sha256_text

SAFE_BUILTINS = frozenset(SAFE_BUILTIN_NAMES)


class PatchError(Exception):
    """kind: stale_base | syntax_error | scope_violation | policy_violation"""

    def __init__(self, kind: str, message: str):
        super().__init__(f"{kind}: {message}")
        self.kind = kind
        self.message = message


def normalize_source(src: str) -> str:
    tree = ast.parse(src)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree).rstrip("\n") + "\n"


@dataclass(frozen=True)
class Program:
    source: str
    functions: Tuple[str, ...]          # dispatch order of the task's entry functions (fixed for the task)

    @cached_property
    def hash(self) -> str:
        return sha256_text(self.source)

    @cached_property
    def tree(self) -> ast.Module:
        return ast.parse(self.source)

    def driver(self) -> str:
        return make_driver(list(self.functions))

    def executable(self) -> str:
        """Source with the hidden dispatcher appended (what the sandbox runs)."""
        return with_driver(self.source, self.driver())

    def function_node(self, name: str) -> ast.FunctionDef:
        for n in self.tree.body:
            if isinstance(n, ast.FunctionDef) and n.name == name:
                return n
        raise KeyError(name)

    def function_source(self, name: str) -> str:
        return ast.unparse(self.function_node(name)).rstrip("\n") + "\n"

    def top_level_functions(self) -> List[str]:
        return [n.name for n in self.tree.body if isinstance(n, ast.FunctionDef)]

    @cached_property
    def module_names(self) -> Dict[str, str]:
        """Module-level bound names -> kind ('function' | 'constant' | 'import')."""
        out: Dict[str, str] = {}
        for n in self.tree.body:
            if isinstance(n, ast.FunctionDef):
                out[n.name] = "function"
            elif isinstance(n, (ast.Assign, ast.AnnAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in targets:
                    for x in ast.walk(t):
                        if isinstance(x, ast.Name):
                            out[x.id] = "constant"
            elif isinstance(n, ast.Import):
                for a in n.names:
                    out[(a.asname or a.name).split(".")[0]] = "import"
            elif isinstance(n, ast.ImportFrom):
                for a in n.names:
                    out[a.asname or a.name] = "import"
        return out

    @classmethod
    def from_source(cls, src: str, functions: Sequence[str]) -> "Program":
        return cls(normalize_source(src), tuple(functions))


# --------------------------------------------------------------------------- #
# Name binding analysis
# --------------------------------------------------------------------------- #
def locally_bound(fn: ast.FunctionDef) -> Set[str]:
    """Names bound anywhere inside the function (over-approximation: nested scopes included)."""
    out: Set[str] = set()
    for n in ast.walk(fn):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = n.args
            for x in a.posonlyargs + a.args + a.kwonlyargs:
                out.add(x.arg)
            if a.vararg:
                out.add(a.vararg.arg)
            if a.kwarg:
                out.add(a.kwarg.arg)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not fn:
                out.add(n.name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.arg):
            out.add(n.arg)
    return out


def free_names(fn: ast.FunctionDef) -> Set[str]:
    bound = locally_bound(fn)
    return {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in bound}


def _base_name(node: ast.AST) -> Optional[str]:
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Starred, ast.NamedExpr)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


# Builtins that never mutate their arguments: a module-level object may be passed to these.
_PURE_BUILTINS = frozenset({
    "len", "sorted", "list", "tuple", "set", "frozenset", "dict", "iter", "enumerate", "zip", "reversed", "sum", "min",
    "max", "any", "all", "isinstance", "issubclass", "repr", "str", "bool", "int", "float", "complex", "map", "filter",
    "hash", "id", "abs", "round", "range", "format", "ascii", "bin", "hex", "oct", "chr", "ord", "divmod", "pow",
    "callable", "bytes", "slice", "hasattr",
})
# Functions of permitted modules that mutate their (first) argument, or whose module is treated as opaque: a
# module-level object may not be handed to them.  Modules not listed in _PURE_MODULES are opaque too.
_MODULE_MUTATORS = {
    "heapq": {"heappush", "heappop", "heapify", "heappushpop", "heapreplace", "merge"},
    "bisect": {"insort", "insort_left", "insort_right"},
    "operator": {"setitem", "delitem", "setattr", "delattr", "iadd", "isub", "imul", "ior", "iand", "ixor", "ilshift",
                 "irshift", "imod", "ipow", "itruediv", "ifloordiv", "imatmul", "iconcat", "methodcaller", "attrgetter",
                 "itemgetter"},
    "random": {"shuffle"},
    "array": set(),
}
_PURE_MODULES = frozenset({"math", "itertools", "functools", "string", "re", "statistics", "fractions", "decimal", "collections",
                           "copy", "json"})
_DUNDER_MUTATORS = frozenset({
    "__setitem__", "__delitem__", "__setattr__", "__delattr__", "__iadd__", "__isub__", "__imul__", "__ior__",
    "__iand__", "__ixor__", "__ilshift__", "__irshift__", "__imod__", "__ipow__", "__itruediv__", "__ifloordiv__",
    "__imatmul__", "__enter__", "__exit__",
})
_MUTATORS = frozenset(MUTATING_METHODS) | _DUNDER_MUTATORS
_BUILTIN_TYPES = frozenset({"dict", "list", "set", "bytearray", "object", "type", "frozenset", "str", "tuple"})


def _targets(node: ast.AST) -> List[ast.AST]:
    if isinstance(node, (ast.Tuple, ast.List)):
        out: List[ast.AST] = []
        for e in node.elts:
            out += _targets(e)
        return out
    if isinstance(node, ast.Starred):
        return _targets(node.value)
    return [node]


_CONTAINER_BUILTINS = frozenset({"list", "tuple", "set", "frozenset", "dict", "sorted", "reversed", "iter", "enumerate",
                                 "zip", "map", "filter"})
_ELEMENT_BUILTINS = frozenset({"min", "max", "next"})
_INTERIOR_METHODS = frozenset({"get", "setdefault", "pop", "popitem", "values", "items", "keys", "copy", "__getitem__"})


def module_aliases(fn: ast.FunctionDef, module: Dict[str, str], bound: Set[str]) -> Tuple[Set[str], Dict[str, int]]:
    """Local names that may reach a module-level object.

    Returns ``(aliases, holders)``.  An *alias* (level 0) may be a module object or an interior part of one
    (``c = CACHE``, ``row = TABLE[i]``, ``for k in TABLE``, ``v = CACHE.get(k)``, ``m = min(TABLE)``).  A *holder*
    of level ``k >= 1`` is a fresh container with ``k`` layers of fresh containers above module-interior objects
    (``rows = list(TABLE)`` is level 1: its elements are aliases; ``rows = [list(r) for r in TABLE]`` is level 2).
    Mutating a holder's own layers is fine; reaching below them (``rows[i].append``, ``rows[i][j] = v`` at level 1)
    is a write into module state.  A fixed point over assignments, loop/comprehension targets, with-items and
    walrus expressions; augmented assignment does not propagate (``total += TABLE[i]`` is a read); reading through
    an alias is always allowed."""
    protected = {n for n, k in module.items() if k == "constant"} - bound
    aliases: Set[str] = set()
    holders: Dict[str, int] = {}

    def lvl(value: Optional[ast.AST]) -> Optional[int]:
        """None (unrelated), 0 (alias) or k >= 1 (holder level) for the object a value expression produces."""
        if value is None:
            return None
        if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
            ks = [k for k in (lvl(e) for e in value.elts) if k is not None]
            return (max(ks) + 1) if ks else None      # a literal container one layer above its elements
        if isinstance(value, ast.Dict):
            ks = [k for k in (lvl(e) for e in list(value.keys) + list(value.values) if e is not None) if k is not None]
            return (max(ks) + 1) if ks else None
        if isinstance(value, (ast.IfExp, ast.BoolOp)):
            parts = [value.body, value.orelse] if isinstance(value, ast.IfExp) else list(value.values)
            ks = [k for k in (lvl(e) for e in parts) if k is not None]
            return min(ks) if ks else None            # the most exposed branch decides
        if isinstance(value, (ast.NamedExpr, ast.Starred)):
            return lvl(value.value)
        if isinstance(value, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            k = lvl(value.elt)
            return None if k is None else k + 1
        if isinstance(value, ast.DictComp):
            ks = [k for k in (lvl(value.key), lvl(value.value)) if k is not None]
            return (min(ks) + 1) if ks else None
        if isinstance(value, ast.Call):
            if isinstance(value.func, ast.Attribute):
                recv = lvl(value.func.value)
                if recv is None:
                    return None
                if value.func.attr == "copy":
                    return max(1, recv)                # shallow copy: same elements, one fresh layer
                if value.func.attr in _INTERIOR_METHODS:
                    return max(0, recv - 1)            # takes an element out
                return None                            # other methods (split, count, lower, ...) return fresh scalars/containers
            if isinstance(value.func, ast.Name):
                ks = [k for k in (lvl(a) for a in value.args) if k is not None]
                if not ks:
                    return None
                if value.func.id == "getattr":
                    return 0                           # a bound method / attribute of a module object is the object
                if value.func.id in _CONTAINER_BUILTINS:
                    return max(1, min(ks))             # same elements in a fresh container
                if value.func.id in _ELEMENT_BUILTINS:
                    return max(0, min(ks) - 1)         # one element out
                if value.func.id in _PURE_BUILTINS:
                    return None                        # scalar-producing builtins
                return min(ks)                         # unknown callee: assume it may return its argument
            return None
        if isinstance(value, ast.Subscript):
            base = lvl(value.value)
            if base is None:
                return None
            if isinstance(value.slice, ast.Slice):
                return max(1, base)                    # a slice is a fresh container of the same elements
            return max(0, base - 1)
        if isinstance(value, ast.Attribute):
            return lvl(value.value)
        if isinstance(value, ast.Name):
            if value.id in aliases or (value.id in protected and value.id not in bound):
                return 0
            if value.id in holders:
                return holders[value.id]
            return None
        return None

    def element_lvl(value: Optional[ast.AST]) -> Optional[int]:
        """What iterating over ``value`` yields."""
        k = lvl(value)
        return None if k is None else max(0, k - 1)

    changed = True
    while changed:
        changed = False
        for n in ast.walk(fn):
            pairs: List[Tuple[ast.AST, Optional[int]]] = []
            if isinstance(n, ast.Assign):
                pairs = [(t, lvl(n.value)) for t in n.targets]
            elif isinstance(n, ast.AnnAssign):
                pairs = [(n.target, lvl(n.value))]
            elif isinstance(n, ast.NamedExpr):
                pairs = [(n.target, lvl(n.value))]
            elif isinstance(n, (ast.For, ast.comprehension)):
                pairs = [(n.target, element_lvl(n.iter))]
            elif isinstance(n, ast.withitem) and n.optional_vars is not None:
                pairs = [(n.optional_vars, lvl(n.context_expr))]
            for target, k in pairs:
                if k is None:
                    continue
                if isinstance(target, (ast.Tuple, ast.List)):
                    k = max(0, k - 1)                  # unpacking takes elements out
                for t in _targets(target):
                    if not isinstance(t, ast.Name):
                        continue
                    if k == 0:
                        if t.id not in aliases:
                            aliases.add(t.id)
                            holders.pop(t.id, None)
                            changed = True
                    elif t.id not in aliases and holders.get(t.id, 1 << 30) > k:
                        holders[t.id] = k              # keep the most exposed level seen for the name
                        changed = True
    return aliases, holders


def _base_and_depth(node: ast.AST) -> Tuple[Optional[str], int]:
    """Base name of an attribute/subscript chain and the number of non-slice hops (``rows[0].x`` -> ('rows', 2))."""
    depth = 0
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Starred, ast.NamedExpr)):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
            depth += 0            # a slice is a fresh container; it does not reach an interior object
        elif isinstance(node, (ast.Attribute, ast.Subscript)):
            depth += 1
        node = node.value
    return (node.id if isinstance(node, ast.Name) else None), depth


def check_function_scope(fn: ast.FunctionDef, program: Program, permitted_imports: Sequence[str]) -> None:
    """Raise PatchError if ``fn`` violates the patch policy relative to ``program``'s module scope.

    Module-level objects (constants, functions, imported modules) are read-only for a patch: no stores, deletes,
    augmented assignments or mutating method calls on them *or on any local alias of them*, and no handing them to
    callables other than pure builtins or the program's own module-level functions.  Together with the sandbox's
    fresh namespace per trial this closes the memoisation / hidden-state route to a fake speedup."""
    module = program.module_names
    bound = locally_bound(fn)
    permitted = set(permitted_imports)
    aliases, holders = module_aliases(fn, module, bound)
    protected_consts = {n for n, k in module.items() if k == "constant"} - bound

    def guarded(name: Optional[str]) -> bool:
        """Is ``name`` a module-level object or a local alias of one?"""
        if name is None:
            return False
        if name in aliases:
            return True
        return name not in bound and name in module

    def interior(node: ast.AST, extra: int) -> Optional[str]:
        """Name of a holder whose module-interior objects ``node`` reaches: ``extra`` is 0 for a mutating-method
        receiver (``rows[i].append`` at level 1) and 1 for a store target (``rows[i][j] = v`` at level 1)."""
        b, d = _base_and_depth(node)
        return b if (b in holders and d >= holders[b] + extra) else None

    for n in ast.walk(fn):
        if isinstance(n, (ast.Global, ast.Nonlocal)):
            raise PatchError("policy_violation", "global/nonlocal statements are not allowed")
        if isinstance(n, ast.ClassDef):
            raise PatchError("policy_violation", "class definitions are not allowed")
        if isinstance(n, (ast.Yield, ast.YieldFrom)):
            raise PatchError("policy_violation", "generators change the return type and are not allowed")
        if isinstance(n, (ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith)):
            raise PatchError("policy_violation", "async constructs are not allowed")
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] not in permitted:
                    raise PatchError("scope_violation", f"undeclared import {a.name!r}")
        if isinstance(n, ast.ImportFrom):
            if n.level or (n.module or "").split(".")[0] not in permitted:
                raise PatchError("scope_violation", f"undeclared import from {n.module!r}")
        # writes to module-level state, directly or through an alias
        if isinstance(n, (ast.Attribute, ast.Subscript)) and isinstance(n.ctx, (ast.Store, ast.Del)):
            b = _base_name(n)
            if guarded(b):
                raise PatchError("policy_violation", f"writes to module-level object {b!r}")
            h = interior(n, 1)                        # rows[i][j] = v writes into a module-interior object
            if h is not None:
                raise PatchError("policy_violation", f"writes into a module-level object held by {h!r}")
        if isinstance(n, ast.AugAssign):
            b = _base_name(n.target)
            if guarded(b) or (isinstance(n.target, ast.Name) and n.target.id in aliases):
                raise PatchError("policy_violation", f"writes to module-level object {b!r}")
            h = interior(n.target, 1)
            if h is not None:
                raise PatchError("policy_violation", f"writes into a module-level object held by {h!r}")
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Attribute):
                recv = _base_name(n.func.value)
                if n.func.attr in _MUTATORS and guarded(recv):
                    raise PatchError("policy_violation", f"mutates module-level object {recv!r}")
                if n.func.attr in _MUTATORS and interior(n.func.value, 0) is not None:
                    raise PatchError("policy_violation", f"mutates a module-level object held by {recv!r}")
                # unbound-method form: dict.update(CACHE, ...), list.append(TABLE, x)
                if recv in _BUILTIN_TYPES and recv not in bound and n.func.attr in _MUTATORS and n.args:
                    first = _base_name(n.args[0])
                    if guarded(first):
                        raise PatchError("policy_violation", f"mutates module-level object {first!r} via {recv}.{n.func.attr}")
            callee = n.func.id if isinstance(n.func, ast.Name) else None
            if callee == "getattr" and n.args:
                b = _base_name(n.args[0])
                if guarded(b) or interior(n.args[0], 0) is not None:
                    attr = n.args[1].value if len(n.args) > 1 and isinstance(n.args[1], ast.Constant) else None
                    if attr is None or attr in _MUTATORS:
                        raise PatchError("policy_violation", f"getattr on module-level object {b!r} with a mutating attribute")
            elif callee is not None and callee not in _PURE_BUILTINS and not (module.get(callee) == "function" and callee not in bound):
                for a in list(n.args) + [k.value for k in n.keywords]:
                    b = _base_name(a)
                    if b is not None and (b in protected_consts or b in aliases):
                        raise PatchError("policy_violation", f"module-level object {b!r} may not be passed to {callee}()")
            if isinstance(n.func, ast.Attribute):
                # module function: heapq.heappush(TABLE, x), operator.setitem(CACHE, k, v), unknown.module.f(CACHE)
                mod_name = n.func.value.id if isinstance(n.func.value, ast.Name) else None
                if mod_name is not None and mod_name not in bound and module.get(mod_name) == "import":
                    fname = n.func.attr
                    opaque = mod_name not in _PURE_MODULES
                    mutating = fname in _MODULE_MUTATORS.get(mod_name, set())
                    if opaque or mutating:
                        for a in list(n.args) + [k.value for k in n.keywords]:
                            b = _base_name(a)
                            if guarded(b) or interior(a, 0) is not None:
                                raise PatchError("policy_violation", f"module-level object {b!r} may not be passed to {mod_name}.{fname}()")
    for name in free_names(fn):
        if name in SAFE_BUILTINS or name in module:
            continue
        raise PatchError("scope_violation", f"unresolved name {name!r} escapes the dependency closure")


def parse_replacement(code: str, target: str) -> ast.FunctionDef:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError) as e:
        msg = getattr(e, "msg", str(e))
        line = getattr(e, "lineno", None)
        raise PatchError("syntax_error", f"SyntaxError: {msg}" + (f" (line {line})" if line else "")) from None
    except (MemoryError, RecursionError):
        raise PatchError("syntax_error", "source too deeply nested") from None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        kinds = [type(n).__name__ for n in tree.body]
        raise PatchError("scope_violation", f"payload must be exactly one function definition, got {kinds}")
    fn = tree.body[0]
    if fn.name != target:
        raise PatchError("scope_violation", f"payload defines {fn.name!r}, expected {target!r}")
    return fn


def _sig(fn: ast.FunctionDef) -> str:
    parts = [ast.dump(fn.args), ast.dump(fn.returns) if fn.returns else "",
             "|".join(ast.dump(d) for d in fn.decorator_list)]
    tp = getattr(fn, "type_params", None)
    if tp:
        parts.append("|".join(ast.dump(t) for t in tp))
    return "##".join(parts)


def apply_patch(program: Program, base_hash: str, target: str, code: str, permitted_imports: Sequence[str]) -> Program:
    """Return the candidate program (``program`` itself is never modified)."""
    if program.hash != base_hash:
        raise PatchError("stale_base", "patch was produced for a different program version")
    if target == ENTRY:
        raise PatchError("scope_violation", "the dispatcher is not editable")
    try:
        old = program.function_node(target)
    except KeyError:
        raise PatchError("scope_violation", f"no function {target!r} in the program") from None
    new = parse_replacement(code, target)
    if new.decorator_list:
        raise PatchError("scope_violation", "decorators are not allowed")
    if _sig(new) != _sig(old):
        raise PatchError("scope_violation", "the function signature (arguments, defaults, annotations) changed")
    check_function_scope(new, program, permitted_imports)
    tree = ast.parse(program.source)
    for i, n in enumerate(tree.body):
        if isinstance(n, ast.FunctionDef) and n.name == target:
            tree.body[i] = new
            break
    ast.fix_missing_locations(tree)
    src = ast.unparse(tree).rstrip("\n") + "\n"
    return Program(src, program.functions)


# --------------------------------------------------------------------------- #
# Dependency closure
# --------------------------------------------------------------------------- #
@dataclass
class Closure:
    target: str
    functions: List[str] = field(default_factory=list)    # module-level functions reachable from the target (excl.)
    constants: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    unsupported: List[str] = field(default_factory=list)  # reasons the closure is unsupported (dynamic lookup, ...)

    def fingerprint(self, program: Program) -> str:
        parts = [program.function_source(self.target)]
        for f in sorted(self.functions):
            parts.append(program.function_source(f))
        for n in program.tree.body:
            if isinstance(n, (ast.Assign, ast.AnnAssign, ast.Import, ast.ImportFrom)):
                parts.append(ast.unparse(n))
        return sha256_text("\n".join(parts))


def dependency_sources(program: Program, closure: "Closure") -> List[str]:
    """Source text of everything in the closure a writer needs to see (4.4): the imports and module-level constants
    the target (transitively) references, then the referenced module-level functions, in program order."""
    out: List[str] = []
    for n in program.tree.body:
        if isinstance(n, ast.Import) and any((a.asname or a.name).split(".")[0] in closure.imports for a in n.names):
            out.append(ast.unparse(n))
        elif isinstance(n, ast.ImportFrom) and any((a.asname or a.name) in closure.imports for a in n.names):
            out.append(ast.unparse(n))
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            names = {x.id for t in targets for x in ast.walk(t) if isinstance(x, ast.Name)}
            if names & set(closure.constants):
                out.append(ast.unparse(n))
    for f in closure.functions:
        out.append(program.function_source(f).rstrip("\n"))
    return out


_DYNAMIC = {"getattr", "hasattr", "setattr", "delattr", "eval", "exec", "globals", "locals", "vars", "__import__"}


def dependency_closure(program: Program, target: str) -> Closure:
    module = program.module_names
    cl = Closure(target)
    seen: Set[str] = set()
    stack = [target]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        fn = program.function_node(name)
        for n in ast.walk(fn):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _DYNAMIC:
                if n.func.id in ("getattr", "hasattr") and len(n.args) >= 2 and isinstance(n.args[1], ast.Constant):
                    continue
                cl.unsupported.append(f"dynamic lookup {n.func.id}() in {name}")
        for free in sorted(free_names(fn)):
            kind = module.get(free)
            if kind == "function" and free != ENTRY:
                if free != target and free not in cl.functions:
                    cl.functions.append(free)
                stack.append(free)
            elif kind == "constant" and free not in cl.constants:
                cl.constants.append(free)
            elif kind == "import" and free not in cl.imports:
                cl.imports.append(free)
    return cl
