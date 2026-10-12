"""Content hashes and source fingerprints.

* ``sha256_*`` hash exact bytes (artifacts, checkpoints, tokenizers, programs).
* :func:`canonical_json` gives one byte string per JSON-compatible value (sorted keys, no whitespace), so a
  hash of a record does not depend on dict insertion order.
* :func:`ast_fingerprint` hashes the normalized AST (formatting and comments removed).
* :func:`alpha_fingerprint` additionally renames every bound identifier to a positional placeholder and drops
  docstrings / annotations.  It is used **only for leakage detection** between splits; it is not assumed to be
  behaviour-preserving (it is never executed).
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

_BUILTIN_NAMES = set(dir(builtins))


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def hash_json(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def short(h: str, n: int = 12) -> str:
    return h[:n]


def combine(parts: Iterable[str]) -> str:
    """Order-sensitive hash of a sequence of hex digests / strings."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Source fingerprints
# --------------------------------------------------------------------------- #
def ast_fingerprint(src: str) -> Optional[str]:
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None
    return sha256_text(ast.dump(tree, annotate_fields=True, include_attributes=False))


class _AlphaNormalizer(ast.NodeTransformer):
    """Rename every bound identifier (functions, params, locals, loop/comprehension targets) to vN."""

    def __init__(self, bound: Dict[str, str]):
        self.bound = bound

    def _name(self, n: str) -> str:
        return self.bound.get(n, n)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        node.name = self._name(node.name)
        node.returns = None
        node.decorator_list = []
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                and isinstance(body[0].value.value, str):
            body = body[1:] or [ast.Pass()]
        node.body = body
        self.generic_visit(node)
        return node

    def visit_arg(self, node: ast.arg):
        node.arg = self._name(node.arg)
        node.annotation = None
        return node

    def visit_Name(self, node: ast.Name):
        node.id = self._name(node.id)
        return node


def _bound_names(tree: ast.AST) -> Dict[str, str]:
    order: Dict[str, str] = {}

    def add(n: str) -> None:
        if n not in order and n not in _BUILTIN_NAMES:
            order[n] = f"v{len(order)}"

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            add(node.name)
            for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                add(a.arg)
            if node.args.vararg:
                add(node.args.vararg.arg)
            if node.args.kwarg:
                add(node.args.kwarg.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            add(node.id)
    return order


def alpha_fingerprint(src: str) -> Optional[str]:
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None
    tree = _AlphaNormalizer(_bound_names(tree)).visit(tree)
    body = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant))]
    tree.body = body
    return sha256_text(ast.dump(tree, annotate_fields=False, include_attributes=False))


def fingerprints(src: str) -> Dict[str, Optional[str]]:
    return {"exact": sha256_text(src), "ast": ast_fingerprint(src), "alpha": alpha_fingerprint(src)}
