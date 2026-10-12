"""Adapter interface shared by neural, rule-based, scripted and legacy role implementations.

A role call has two phases so the controller can reserve budget *before* any generation:

1. ``prepare(role, view, ctx)`` builds (and size-checks) the prompt and returns its token count and the maximum
   response length (rule adapters return zeros); it may raise :class:`~tinyperf_chain.context.ContextOverflow`;
2. ``execute(prepared, envelope)`` produces the response text, which the controller parses with the same strict
   parser for every adapter kind.

``view`` is the public prompt view (:mod:`tinyperf_chain.context`); ``ctx`` is the same public information in
structured form (rule adapters use it).  Neither ever contains private suites or privileged provenance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tinyperf_chain.schemas import RequestEnvelope, RoleResponse


@dataclass
class Prepared:
    role: str
    prompt_role: str
    prompt_ids: List[int] = field(default_factory=list)
    prompt_text: str = ""
    max_new_tokens: int = 0
    view: Any = None
    ctx: Dict[str, Any] = field(default_factory=dict)
    dropped_sections: List[str] = field(default_factory=list)

    @property
    def prompt_tokens(self) -> int:
        return len(self.prompt_ids)


class RoleAdapter:
    kind = "abstract"

    def prepare(self, role: str, view: Any, ctx: Dict[str, Any]) -> Prepared:
        return Prepared(role, role, view=view, ctx=ctx)

    def execute(self, prep: Prepared, env: RequestEnvelope) -> RoleResponse:  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {"kind": self.kind}


class RoleRouter:
    """role -> adapter mapping for one frozen bundle."""

    def __init__(self, adapters: Dict[str, RoleAdapter], bundle_hash: str = "unbundled", system_id: str = "custom"):
        missing = [r for r in ("M1", "M2", "M3", "M4", "M5") if r not in adapters]
        if missing:
            raise ValueError(f"router is missing roles {missing}")
        self.adapters = adapters
        self.bundle_hash = bundle_hash
        self.system_id = system_id

    def __getitem__(self, role: str) -> RoleAdapter:
        return self.adapters[role]

    def describe(self) -> Dict[str, Any]:
        return {r: a.describe() for r, a in self.adapters.items()}
