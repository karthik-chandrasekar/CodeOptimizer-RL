"""Multi-function file tasks: hidden dispatcher + function-level edits.

A file task's ``source`` is a module with several top-level functions.  The sandbox measures the
whole file through a hidden dispatcher appended at evaluation time (``task.meta["driver"]``)::

    def _tp_entry(k, *args):
        return (f0, f1, f2)[k](*args)

so every hidden input is ``(k, *args)`` for function ``k`` and the paired timing covers all of
them.  The policy never sees the dispatcher.  An edit is one or more function definitions, spliced
into the current best file by name (unknown names are added as new helpers).
"""
from __future__ import annotations

import ast
from typing import List

ENTRY = "_tp_entry"


def make_driver(names: List[str]) -> str:
    return f"def {ENTRY}(k, *args):\n    return ({', '.join(names)},)[k](*args)\n"


def with_driver(code: str, driver: str) -> str:
    return code.rstrip() + "\n\n\n" + driver


def splice_functions(base: str, edit: str) -> str:
    """Replace the functions defined in `edit` inside `base` (by name); raise SyntaxError if `edit` defines none."""
    base_tree, edit_tree = ast.parse(base), ast.parse(edit)
    defs = [n for n in edit_tree.body if isinstance(n, ast.FunctionDef) and n.name != ENTRY]
    if not defs:
        raise SyntaxError("edit defines no function")
    index = {n.name: i for i, n in enumerate(base_tree.body) if isinstance(n, ast.FunctionDef)}
    for d in defs:
        if d.name in index:
            base_tree.body[index[d.name]] = d
        else:
            base_tree.body.append(d)
    imports = [n for n in edit_tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    base_tree.body[0:0] = imports
    return ast.unparse(base_tree)


def edited_functions(edit: str) -> List[str]:
    try:
        return [n.name for n in ast.parse(edit).body if isinstance(n, ast.FunctionDef)]
    except (SyntaxError, MemoryError, RecursionError, ValueError):
        return []


def profile_shares(group_ns, names: List[str]):
    """{function name: share of the file's runtime} from the sandbox's per-group timings (keys = function index)."""
    if not group_ns:
        return None
    total = float(sum(group_ns.values())) or 1.0
    return {names[int(k)]: float(v) / total for k, v in sorted(group_ns.items()) if 0 <= int(k) < len(names)}
