"""Hints for on-policy self-distillation (OPSD).

The *teacher* is the policy itself, shown the observation plus a ``hint=...`` line in its ``<STATE>`` block;
the *student* is the same policy without it.  Hints come from what the environment knows about a step:

* outcome hints - the sandbox's verdict on the action that follows the observation (tool-call errors such as
  "changes its behaviour" / "does not compile" / "times out", and negative feedback such as "gives no speedup");
* localization hints - privileged task information the student never sees: the slowest function that is still
  unfixed in the current code (from the task's slow functions and step-0 profile).

Hints are built in hindsight for training only; evaluation never shows them.
"""
from __future__ import annotations

import ast
from typing import Dict, List, Optional

from tinyperf.env.files import edited_functions
from tinyperf.env.protocol import parse_action

HINT_MODES = ("error", "loc", "error+loc")


def add_hint(obs: str, hint: Optional[str]) -> str:
    """Insert ``hint=...`` as the last line of the observation's <STATE> block."""
    if not hint:
        return obs
    i = obs.rfind("</STATE>")
    if i < 0:
        return obs + f"\nhint={hint}"
    return obs[:i] + f"hint={hint}\n" + obs[i:]


def code_from_obs(obs: str) -> str:
    a, b = obs.find("<CODE>"), obs.find("</CODE>")
    return obs[a + len("<CODE>"):b].strip("\n") if a >= 0 and b > a else ""


def _functions(src: str) -> Dict[str, str]:
    try:
        return {n.name: ast.unparse(n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return {}


def remaining_slow(task, code: str) -> List[str]:
    """Slow functions whose code is still identical to the original (i.e. not yet rewritten)."""
    slow = list((task.meta or {}).get("slow_functions") or [])
    if not slow:
        return []
    orig, cur = _functions(task.source), _functions(code)
    return [f for f in slow if f in cur and cur[f] == orig.get(f)]


def localization_hint(task, code: str) -> Optional[str]:
    if not (task.meta or {}).get("slow_functions"):
        return None                                   # single-function task: nothing to localize
    rem = remaining_slow(task, code)
    if not rem:
        return "all slow functions are fixed; STOP is right"
    prof = (task.meta or {}).get("profile0") or {}
    g = max(rem, key=lambda f: prof.get(f, 0.0))
    return f"the slowest unfixed function is {g}"


def outcome_hint(action_kind: str, status: str, improved: bool, fnames: List[str], remaining: Optional[List[str]] = None) -> str:
    f = ", ".join(fnames) if fnames else "the code"
    if action_kind == "stop":
        return "stopping here is premature" if remaining else "stopping here is right"
    if status == "malformed":
        return "the next action is malformed; answer with a complete <EDIT> or <STOP>"
    if status == "incorrect":
        return f"the next edit to {f} changes its behaviour; it must preserve it exactly"
    if status in ("syntax_error", "static_error", "load_error"):
        return f"the next edit to {f} is not valid; it must be plain valid Python"
    if status == "runtime_error":
        return f"the next edit to {f} raises an error"
    if status in ("timeout", "resource"):
        return f"the next edit to {f} is too slow or hangs"
    if status == "ok" and not improved:
        return f"editing {f} gives no speedup"
    if status == "ok" and improved:
        return f"editing {f} speeds the code up"
    return f"the next edit to {f} fails ({status})"


def build_hint(task, obs: str, action_kind: str, status: str, improved: bool, raw_action: str, mode: str) -> Optional[str]:
    """Hint for the teacher at the state `obs`, about the action that was taken there (hindsight)."""
    parts = []
    code = code_from_obs(obs)
    rem = remaining_slow(task, code)
    if "error" in mode:
        try:
            a = parse_action(raw_action or "")
            fnames = edited_functions(a.code) if a.kind == "edit" and a.code else []
        except Exception:  # noqa: BLE001
            fnames = []
        parts.append(outcome_hint(action_kind, status, improved, fnames, rem))
    if "loc" in mode:
        h = localization_hint(task, code)
        if h:
            parts.append(h)
    return "; ".join(parts) or None
