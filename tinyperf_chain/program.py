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
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def check_function_scope(fn: ast.FunctionDef, program: Program, permitted_imports: Sequence[str]) -> None:
    """Raise PatchError if ``fn`` violates the patch policy relative to ``program``'s module scope."""
    module = program.module_names
    bound = locally_bound(fn)
    permitted = set(permitted_imports)
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
        # writes to module-level state
        if isinstance(n, (ast.Attribute, ast.Subscript)) and isinstance(n.ctx, (ast.Store, ast.Del)):
            b = _base_name(n)
            if b is not None and b not in bound and b in module:
                raise PatchError("policy_violation", f"writes to module-level object {b!r}")
        if isinstance(n, ast.AugAssign):
            b = _base_name(n.target)
            if b is not None and b not in bound and b in module:
                raise PatchError("policy_violation", f"writes to module-level object {b!r}")
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in MUTATING_METHODS:
            b = _base_name(n.func.value)
            if b is not None and b not in bound and module.get(b) == "constant":
                raise PatchError("policy_violation", f"mutates module-level object {b!r}")
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
