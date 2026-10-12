"""Role prompts: public views, deterministic serialization and tokenizer-aware context budgets.

Every prompt has the same skeleton (protocol tags are the tokenizer's single-token specials; role names and field
names are ordinary BPE text, so no embedding resize is needed)::

    role=M3 v=chain_v1
    <CODE>
    <escaped source of the target function>
    </CODE>
    deps
    <CODE>
    <escaped dependency sources>
    </CODE>
    <STATE>
    key=value
    ...
    </STATE>

Sections are *required* (target code, dependencies, contract, plan, failure kind, probe menu, catalog) or
*optional* (attempt history, anchors listing, diffs).  Preflight drops optional sections in priority order and
re-tokenizes; if the required content still does not fit the role's input cap the call is refused with
:class:`ContextOverflow` (logged as an unsupported/abandon outcome).  Nothing is ever tail-truncated.

Builders only accept *public views* (no private suites, seeds, references or privileged provenance).
"""
from __future__ import annotations

import difflib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from tinyperf.env.protocol import CODE_CLOSE, CODE_OPEN, STATE_CLOSE, STATE_OPEN
from tinyperf_chain import SCHEMA_VERSION
from tinyperf_chain.config import RolesCfg
from tinyperf_chain.schemas import escape_field, state_value


class ContextOverflow(Exception):
    def __init__(self, role: str, n_tokens: int, cap: int):
        super().__init__(f"{role}: required context needs {n_tokens} tokens > input cap {cap}")
        self.role, self.n_tokens, self.cap = role, n_tokens, cap


class TokenCounter(Protocol):
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]: ...


# --------------------------------------------------------------------------- #
# Public views
# --------------------------------------------------------------------------- #
@dataclass
class CatalogEntry:
    display_id: str
    name: str
    source: str
    self_bucket: int = 0          # 0..20  (exclusive time share)
    cum_bucket: int = 0           # 0..20  (inclusive time share)
    calls_bucket: int = 0         # floor(log2(ncalls + 1))
    lines: int = 0
    loops: int = 0
    tried: int = 0
    summary: str = ""


@dataclass
class M1View:
    catalog: List[CatalogEntry]
    history: List[str] = field(default_factory=list)       # "f01:membership_index:probe_mismatch"
    remaining_patches: int = 0


@dataclass
class M2View:
    function_id: str
    target_source: str
    dep_sources: List[str]
    contract: str
    anchors: str
    names: List[str]
    menu: List[str]
    history: List[str] = field(default_factory=list)       # "membership_index(src=a;loop=n00):private_mismatch"


@dataclass
class M3View:
    function_id: str
    target_source: str
    dep_sources: List[str]
    contract: str
    plan: str
    anchors: str = ""


@dataclass
class M4View:
    function_id: str
    current_source: str
    candidate_source: str
    contract: str
    plan: str
    menu: str
    diff: str = ""


@dataclass
class M5View:
    function_id: str
    current_source: str
    failed_source: str
    dep_sources: List[str]
    contract: str
    plan: str
    feedback: Dict[str, Any] = field(default_factory=dict)  # failure_kind, probe_id, input, expected, got (public only)


@dataclass
class BuiltPrompt:
    role: str
    text: str
    ids: List[int]
    dropped: List[str]
    compact: bool = False

    @property
    def n_tokens(self) -> int:
        return len(self.ids)


# --------------------------------------------------------------------------- #
# Helpers for views
# --------------------------------------------------------------------------- #
def share_bucket(x: float) -> int:
    return int(max(0, min(20, round(20 * float(x)))))


def calls_bucket(n: int) -> int:
    return int(min(15, math.floor(math.log2(max(0, int(n)) + 1))))


def compact_diff(a: str, b: str, max_lines: int = 40) -> str:
    lines = list(difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0))[2:]
    lines = [l for l in lines if not l.startswith("@@")]
    return "\n".join(lines[:max_lines])


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
@dataclass
class _Section:
    name: str
    text: str
    required: bool
    drop_priority: int = 0        # higher = dropped first


class PromptBuilder:
    def __init__(self, tok: TokenCounter, roles: RolesCfg, context_tokens: int = 2048):
        self.tok = tok
        self.roles = roles
        self.context_tokens = context_tokens

    # -------------------------------------------------------------- #
    @staticmethod
    def _code(src: str) -> str:
        return f"{CODE_OPEN}\n{escape_field(src.rstrip())}\n{CODE_CLOSE}\n"

    @staticmethod
    def _kv(key: str, value: Any) -> str:
        return f"{key}={state_value(str(value))}\n"

    def _assemble(self, role: str, sections: List[_Section]) -> str:
        head = f"role={role} v={SCHEMA_VERSION}\n"
        code = [s for s in sections if s.name.startswith("code")]
        state = [s for s in sections if not s.name.startswith("code")]
        body = "".join(s.text for s in code)
        st = "".join(s.text for s in state)
        return head + body + f"{STATE_OPEN}\n" + st + f"{STATE_CLOSE}\n"

    def _fit(self, role: str, sections: List[_Section]) -> Tuple[str, List[int], List[str]]:
        cap = self.roles.get(role).max_input
        out_cap = self.roles.get(role).max_output
        cap = min(cap, self.context_tokens - out_cap)
        dropped: List[str] = []
        live = list(sections)
        while True:
            text = self._assemble(role, live)
            ids = self.tok.encode(text, add_bos=True)
            if len(ids) <= cap:
                return text, ids, dropped
            optional = [s for s in live if not s.required]
            if not optional:
                raise ContextOverflow(role, len(ids), cap)
            victim = max(optional, key=lambda s: s.drop_priority)
            live.remove(victim)
            dropped.append(victim.name)

    # -------------------------------------------------------------- #
    def build(self, role: str, view: Any) -> BuiltPrompt:
        if role == "M1":
            return self._m1(view)
        sections = getattr(self, f"_sections_{role.lower()}")(view)
        text, ids, dropped = self._fit(role, sections)
        return BuiltPrompt(role, text, ids, dropped)

    def _m1(self, v: M1View) -> BuiltPrompt:
        full = "".join(f"# {e.display_id}\n{e.source.rstrip()}\n\n" for e in v.catalog)
        try:
            text, ids, dropped = self._fit("M1", self._sections_m1(v, full))
            return BuiltPrompt("M1", text, ids, dropped)
        except ContextOverflow:
            summary = "".join(f"# {e.display_id}\n{e.summary.rstrip()}\n\n" for e in v.catalog)
            text, ids, dropped = self._fit("M1", self._sections_m1(v, summary))
            return BuiltPrompt("M1", text, ids, dropped + ["sources->summaries"], compact=True)

    def _sections_m1(self, v: M1View, code: str) -> List[_Section]:
        cat = "".join(f"{e.display_id} self={e.self_bucket} cum={e.cum_bucket} calls=b{e.calls_bucket} "
                      f"lines={e.lines} loops={e.loops} tried={e.tried}\n" for e in v.catalog)
        return [
            _Section("code_catalog", self._code(code), True),
            _Section("catalog_ids", self._kv("catalog", ",".join(e.display_id for e in v.catalog)), True),
            _Section("profile", cat, True),
            _Section("remaining", self._kv("remaining_patches", v.remaining_patches), True),
            _Section("history", "".join(self._kv("tried", h) for h in v.history[-12:]), False, 10),
        ]

    def _sections_m2(self, v: M2View) -> List[_Section]:
        secs = [_Section("code_target", self._code(v.target_source), True)]
        if v.dep_sources:
            secs.append(_Section("code_deps", "deps\n" + self._code("\n\n".join(v.dep_sources)), True))
        secs += [
            _Section("function", self._kv("function", v.function_id), True),
            _Section("contract", self._kv("contract", v.contract), True),
            _Section("anchors", self._kv("anchors", v.anchors), True),
            _Section("names", self._kv("names", ",".join(v.names)), False, 2),
            _Section("menu", self._kv("menu", ",".join(v.menu) or "none"), True),
            _Section("history", "".join(self._kv("tried", h) for h in v.history[-8:]), False, 10),
        ]
        return secs

    def _sections_m3(self, v: M3View) -> List[_Section]:
        secs = [_Section("code_target", self._code(v.target_source), True)]
        if v.dep_sources:
            secs.append(_Section("code_deps", "deps\n" + self._code("\n\n".join(v.dep_sources)), True))
        secs += [
            _Section("function", self._kv("function", v.function_id), True),
            _Section("plan", self._kv("plan", v.plan), True),
            _Section("contract", self._kv("contract", v.contract), True),
            _Section("anchors", self._kv("anchors", v.anchors or "none"), False, 2),
        ]
        return secs

    def _sections_m4(self, v: M4View) -> List[_Section]:
        return [
            _Section("code_current", self._code(v.current_source), True),
            _Section("code_candidate", "candidate\n" + self._code(v.candidate_source), True),
            _Section("function", self._kv("function", v.function_id), True),
            _Section("plan", self._kv("plan", v.plan), True),
            _Section("contract", self._kv("contract", v.contract), True),
            _Section("menu", self._kv("menu", v.menu), True),
            _Section("diff", self._kv("diff", v.diff) if v.diff else "", False, 5),
        ]

    def _sections_m5(self, v: M5View) -> List[_Section]:
        secs = [
            _Section("code_current", self._code(v.current_source), True),
            _Section("code_failed", "failed\n" + self._code(v.failed_source), True),
        ]
        if v.dep_sources:
            secs.append(_Section("code_deps", "deps\n" + self._code("\n\n".join(v.dep_sources)), True))
        fb = v.feedback or {}
        secs += [
            _Section("function", self._kv("function", v.function_id), True),
            _Section("plan", self._kv("plan", v.plan), True),
            _Section("contract", self._kv("contract", v.contract), True),
            _Section("failure", self._kv("failure", fb.get("failure_kind", "unknown")), True),
        ]
        detail = ""
        if fb.get("probe_id"):
            detail += self._kv("probe", fb["probe_id"])
        for k in ("input", "expected", "got", "note", "message"):
            if fb.get(k):
                detail += self._kv(k, fb[k])
        if detail:
            secs.append(_Section("feedback_detail", detail, False, 3))
        return secs
