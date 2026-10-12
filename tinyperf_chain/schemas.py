"""Typed role payloads, strict parsers/serializers, request envelopes and the source-field escaper.

Wire format (what a role *generates*; the harness attaches all trusted metadata in :class:`RequestEnvelope`)::

    M1   TARGET f02                     | NO_TARGET
    M2   PLAN membership_index; src=allowed; loop=n04; preserve=order,duplicates  | NO_PLAN
    M3   <EDIT>\\n<one function>\\n</EDIT>   | ABANDON
    M4   PROBES p01,p04,p09             | PROBES none
    M5   same as M3

Parsing is strict: the only thing removed before matching is the protocol terminal marker (``<EOS>``) and a
single trailing newline.  Anything else that does not match the grammar is a ``malformed`` control output
(logged, budget-charged, never silently cleaned up or retried).  Semantic validity (does ``f02`` exist, is the
plan applicable, are probe ids on the menu) is decided later by trusted code.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

from tinyperf.env.protocol import EDIT_CLOSE, EDIT_OPEN, EOS, SPECIAL_TOKENS

Role = Literal["M1", "M2", "M3", "M4", "M5"]
ROLES: Tuple[str, ...] = ("M1", "M2", "M3", "M4", "M5")
PATCH_ROLES = ("M3", "M5")
SCHEMA_VERSIONS = {"M1": "m1_target_v1", "M2": "m2_plan_v1", "M3": "m3_patch_v1", "M4": "m4_probes_v1", "M5": "m5_repair_v1",
                   "prompt": "prompt_v1", "envelope": "envelope_v1"}
PARSER_VERSION = "strict_v1"

NO_TARGET = "NO_TARGET"
NO_PLAN = "NO_PLAN"
ABANDON = "ABANDON"
PROBES_NONE = "none"


# --------------------------------------------------------------------------- #
# Escaping of source / free-text fields
# --------------------------------------------------------------------------- #
_TAGS = sorted(SPECIAL_TOKENS, key=len, reverse=True)


def escape_field(s: str) -> str:
    """Reversibly neutralise protocol tag strings inside source text (``<CODE>`` -> ``<~CODE>``)."""
    s = s.replace("<~", "<~~")
    for t in _TAGS:
        s = s.replace(t, "<~" + t[1:])
    return s


def unescape_field(s: str) -> str:
    for t in _TAGS:
        s = s.replace("<~" + t[1:], t)
    return s.replace("<~~", "<~")


def state_value(s: str) -> str:
    """One-line value for a ``key=value`` state line (backslash / newline escaped, tags neutralised)."""
    return escape_field(s.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r"))


# --------------------------------------------------------------------------- #
# Envelope
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RequestEnvelope:
    role: str
    task_id: str
    request_id: str
    program_hash: str
    bundle_hash: str
    schema_version: str
    max_new_tokens: int
    deadline_monotonic: float
    parser_version: str = PARSER_VERSION
    decoding: Tuple[Tuple[str, Any], ...] = ()
    allowed_actions: Tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Payloads
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TargetChoice:
    function_id: Optional[str]          # None == NO_TARGET

    @property
    def is_no_target(self) -> bool:
        return self.function_id is None


@dataclass(frozen=True)
class Plan:
    function_id: str
    transform_id: Optional[str]         # None == NO_PLAN
    operands: Tuple[str, ...] = ()      # ("src=allowed", "loop=n04") in emitted order
    preserve: Tuple[str, ...] = ()
    # Preconditions are resolved by the trusted registry; they are never claims made authoritative by the model.

    @property
    def is_no_plan(self) -> bool:
        return self.transform_id is None

    def operand(self, key: str) -> Optional[str]:
        for o in self.operands:
            k, _, v = o.partition("=")
            if k == key:
                return v
        return None

    def key(self) -> str:
        """Identity used for duplicate-plan detection (operand order does not matter)."""
        return f"{self.transform_id}|{';'.join(sorted(self.operands))}|{','.join(sorted(self.preserve))}"


@dataclass(frozen=True)
class PatchAction:
    kind: str                           # "edit" | "abandon"
    code: Optional[str] = None

    @property
    def is_abandon(self) -> bool:
        return self.kind == "abandon"


@dataclass(frozen=True)
class ProbeSelection:
    probe_ids: Tuple[str, ...] = ()


@dataclass
class ParseResult:
    ok: bool
    payload: Any = None
    error: str = ""
    raw: str = ""


# --------------------------------------------------------------------------- #
# Serializers (canonical text the models are trained to emit)
# --------------------------------------------------------------------------- #
def format_target(c: TargetChoice) -> str:
    return NO_TARGET if c.function_id is None else f"TARGET {c.function_id}"


def format_plan(p: Plan) -> str:
    if p.transform_id is None:
        return NO_PLAN
    parts = [f"PLAN {p.transform_id}"] + list(p.operands)
    if p.preserve:
        parts.append("preserve=" + ",".join(p.preserve))
    return "; ".join(parts)


def format_patch(a: PatchAction) -> str:
    if a.is_abandon:
        return ABANDON
    assert a.code is not None
    return f"{EDIT_OPEN}\n{escape_field(a.code.rstrip())}\n{EDIT_CLOSE}"


def format_probes(s: ProbeSelection) -> str:
    return "PROBES " + (",".join(s.probe_ids) if s.probe_ids else PROBES_NONE)


def format_payload(role: str, payload: Any) -> str:
    if role == "M1":
        return format_target(payload)
    if role == "M2":
        return format_plan(payload)
    if role in PATCH_ROLES:
        return format_patch(payload)
    if role == "M4":
        return format_probes(payload)
    raise ValueError(role)


# --------------------------------------------------------------------------- #
# Strict parsers
# --------------------------------------------------------------------------- #
_ID = r"f\d{2,3}"
_TARGET_RE = re.compile(rf"^TARGET ({_ID})$")
_TID = r"[a-z][a-z0-9_]{1,40}"
_OPERAND_RE = re.compile(r"^([a-z][a-z_]{0,15})=([A-Za-z_][A-Za-z0-9_]{0,40})$")
_PRESERVE_RE = re.compile(r"^preserve=([a-z_]+(?:,[a-z_]+){0,7})$")
_PROBES_RE = re.compile(r"^PROBES (none|p\d{2}(?:,p\d{2}){0,11})$")


def _strip_terminal(text: str) -> str:
    s = text
    if s.endswith(EOS):
        s = s[: -len(EOS)]
    if s.endswith("\n"):
        s = s[:-1]
    return s


def parse_target(text: str) -> ParseResult:
    s = _strip_terminal(text)
    if s == NO_TARGET:
        return ParseResult(True, TargetChoice(None), raw=text)
    m = _TARGET_RE.fullmatch(s)
    if not m:
        return ParseResult(False, error="malformed M1 output", raw=text)
    return ParseResult(True, TargetChoice(m.group(1)), raw=text)


def parse_plan(text: str, function_id: str) -> ParseResult:
    s = _strip_terminal(text)
    if s == NO_PLAN:
        return ParseResult(True, Plan(function_id, None), raw=text)
    if not s.startswith("PLAN "):
        return ParseResult(False, error="malformed M2 output", raw=text)
    fields = s[len("PLAN "):].split("; ")
    tid = fields[0]
    if not re.fullmatch(_TID, tid):
        return ParseResult(False, error="malformed transform id", raw=text)
    operands: List[str] = []
    preserve: Tuple[str, ...] = ()
    seen = set()
    for f in fields[1:]:
        pm = _PRESERVE_RE.fullmatch(f)
        if pm:
            if preserve:
                return ParseResult(False, error="duplicate preserve field", raw=text)
            preserve = tuple(pm.group(1).split(","))
            continue
        om = _OPERAND_RE.fullmatch(f)
        if not om or om.group(1) == "preserve":
            return ParseResult(False, error=f"malformed plan field {f!r}", raw=text)
        if om.group(1) in seen:
            return ParseResult(False, error=f"duplicate operand {om.group(1)!r}", raw=text)
        seen.add(om.group(1))
        operands.append(f)
    return ParseResult(True, Plan(function_id, tid, tuple(operands), preserve), raw=text)


def parse_patch(text: str) -> ParseResult:
    s = _strip_terminal(text)
    if s == ABANDON:
        return ParseResult(True, PatchAction("abandon"), raw=text)
    head, tail = EDIT_OPEN + "\n", "\n" + EDIT_CLOSE
    if not (s.startswith(head) and s.endswith(tail)) or len(s) < len(head) + len(tail):
        return ParseResult(False, error="malformed patch output", raw=text)
    body = s[len(head): len(s) - len(tail)]
    if EDIT_OPEN in body or EDIT_CLOSE in body or not body.strip():
        return ParseResult(False, error="malformed patch body", raw=text)
    return ParseResult(True, PatchAction("edit", unescape_field(body)), raw=text)


def parse_probes(text: str) -> ParseResult:
    s = _strip_terminal(text)
    m = _PROBES_RE.fullmatch(s)
    if not m:
        return ParseResult(False, error="malformed M4 output", raw=text)
    ids = () if m.group(1) == PROBES_NONE else tuple(m.group(1).split(","))
    return ParseResult(True, ProbeSelection(ids), raw=text)


def parse_role_output(role: str, text: str, function_id: str = "") -> ParseResult:
    if role == "M1":
        return parse_target(text)
    if role == "M2":
        return parse_plan(text, function_id)
    if role in PATCH_ROLES:
        return parse_patch(text)
    if role == "M4":
        return parse_probes(text)
    raise ValueError(f"unknown role {role}")


# --------------------------------------------------------------------------- #
# Role responses (what an adapter hands back to the controller)
# --------------------------------------------------------------------------- #
@dataclass
class RoleResponse:
    role: str
    request_id: str
    text: str
    parsed: ParseResult
    adapter_kind: str                     # "neural" | "rule" | "scripted" | "legacy"
    prompt_tokens: int = 0
    output_tokens: int = 0
    finished: bool = True                 # ended on a stop token (False = hit the output cap)
    token_ids: Optional[List[int]] = None
    prompt_ids: Optional[List[int]] = None
    logprobs: Optional[List[float]] = None
    latency_ns: int = 0
    error: Optional[str] = None           # "context_overflow" | "timeout" | "deadline" | adapter error text
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and self.parsed.ok
