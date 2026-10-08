"""The text protocol between environment and policy.

Observation::

    <CODE>
    def f(xs):
        ...
    </CODE>
    <STATE>
    best_runtime=0.61
    last_runtime=0.74
    last_correct=1
    last_status=slower
    step=3
    remaining=3
    </STATE>

Action::

    <EDIT>
    def f(xs):
        ...
    </EDIT>

or::

    <STOP>

Runtimes are normalized: 1.00 = original, 0.50 = 2x faster.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

CODE_OPEN, CODE_CLOSE = "<CODE>", "</CODE>"
STATE_OPEN, STATE_CLOSE = "<STATE>", "</STATE>"
EDIT_OPEN, EDIT_CLOSE = "<EDIT>", "</EDIT>"
STOP_TOKEN = "<STOP>"
BOS, EOS, PAD = "<BOS>", "<EOS>", "<PAD>"

SPECIAL_TOKENS = [PAD, BOS, EOS, CODE_OPEN, CODE_CLOSE, STATE_OPEN, STATE_CLOSE, EDIT_OPEN, EDIT_CLOSE, STOP_TOKEN]
STOP_STRINGS = [EDIT_CLOSE, STOP_TOKEN, EOS]


@dataclass
class Action:
    kind: str                   # "edit" | "stop" | "malformed"
    code: Optional[str] = None
    raw: str = ""


_EDIT_RE = re.compile(re.escape(EDIT_OPEN) + r"\s*\n?(.*?)(?:" + re.escape(EDIT_CLOSE) + r"|$)", re.S)


def parse_action(text: str) -> Action:
    """Parse policy output. First protocol token wins; anything else is malformed."""
    s = text.strip()
    i_stop = s.find(STOP_TOKEN)
    i_edit = s.find(EDIT_OPEN)
    if i_stop != -1 and (i_edit == -1 or i_stop < i_edit):
        return Action("stop", raw=text)
    if i_edit != -1:
        m = _EDIT_RE.search(s)
        code = (m.group(1) if m else "").rstrip()
        if code.strip():
            return Action("edit", code=code, raw=text)
    return Action("malformed", raw=text)


def format_action(action: Action) -> str:
    if action.kind == "stop":
        return STOP_TOKEN
    assert action.code is not None
    return f"{EDIT_OPEN}\n{action.code.rstrip()}\n{EDIT_CLOSE}"


def format_observation(code: str, state: dict, feedback: str = "full") -> str:
    lines = [CODE_OPEN, code.rstrip(), CODE_CLOSE, STATE_OPEN]
    if feedback == "full":
        for k in ("best_runtime", "last_runtime", "last_correct", "last_status", "step", "remaining"):
            v = state[k]
            if isinstance(v, float):
                v = f"{v:.3f}"
            lines.append(f"{k}={v}")
        if state.get("profile"):  # profiler feedback: each function's share of the current best file's runtime
            lines.append("time_share=" + ",".join(f"{n}:{v:.2f}" for n, v in state["profile"].items()))
    else:  # no-feedback ablation: only the step counter is exposed
        lines.append(f"step={state['step']}")
        lines.append(f"remaining={state['remaining']}")
    lines.append(STATE_CLOSE)
    return "\n".join(lines) + "\n"
