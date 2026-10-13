"""Atomic, thread-safe budget ledger for one optimization episode.

Every resource is *reserved before use* and committed afterwards:

* patch slots (M3 and M5 calls), charged at reservation and never refunded - abandonment, malformed output,
  model timeouts and failed repairs all consume their slot;
* role calls (M1-M5, rule-based or neural), charged at reservation, never refunded;
* input+output tokens: the prompt tokens plus the *maximum* response length are reserved before dispatch; the
  unused part of the response reservation is released when the call returns;
* extra public probe calls (at most ``max_probes_per_candidate`` per candidate, from one global pool);
* profile runs;
* wall time: a monotonic deadline for the whole search; a call that cannot start before the deadline (minus
  ``min_call_seconds``) is refused.

A refused reservation raises :class:`BudgetExhausted` / :class:`DeadlineExceeded`; the episode wrapper catches
them, keeps the last committed program and only runs the final audit.
"""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

from tinyperf_chain.config import BudgetCfg


class SearchStop(Exception):
    """Base class: the search must stop (the current best is kept)."""

    reason = "stopped"


class BudgetExhausted(SearchStop):
    def __init__(self, resource: str):
        super().__init__(f"budget exhausted: {resource}")
        self.resource = resource
        self.reason = f"{resource}_budget"


class DeadlineExceeded(SearchStop):
    reason = "wall_time"

    def __init__(self, what: str = ""):
        super().__init__(f"search deadline exceeded {what}".strip())


@dataclass
class Usage:
    patch_slots: int = 0
    role_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reserved_tokens: int = 0
    extra_probe_calls: int = 0
    profiles: int = 0
    calls_by_role: Dict[str, int] = field(default_factory=dict)


@dataclass
class Reservation:
    role: str
    prompt_tokens: int
    max_new_tokens: int
    patch_slot: bool
    committed: bool = False


class BudgetLedger:
    def __init__(self, cfg: BudgetCfg, *, start: Optional[float] = None, clock=time.monotonic,
                 patch_slots: Optional[int] = None):
        self.cfg = cfg
        self.clock = clock
        self.t0 = clock() if start is None else start
        self.deadline: Optional[float] = None if cfg.search_wall_seconds is None else self.t0 + float(cfg.search_wall_seconds)
        self.max_patch_slots = cfg.patch_slots if patch_slots is None else patch_slots
        self.used = Usage()
        self._lock = threading.Lock()

    # -------------------------------------------------------------- #
    def remaining_wall(self) -> Optional[float]:
        return None if self.deadline is None else self.deadline - self.clock()

    def check_deadline(self, what: str = "") -> None:
        r = self.remaining_wall()
        if r is not None and r <= max(0.0, self.cfg.min_call_seconds):
            raise DeadlineExceeded(what)

    def tokens_left(self) -> int:
        return self.cfg.total_input_output_tokens - (self.used.input_tokens + self.used.output_tokens + self.used.reserved_tokens)

    def can_request_patch(self, min_prompt_tokens: int = 1, max_new_tokens: int = 1) -> bool:
        with self._lock:
            if self.used.patch_slots >= self.max_patch_slots:
                return False
            if self.used.role_calls >= self.cfg.role_calls:
                return False
            if self.tokens_left() < min_prompt_tokens + max_new_tokens:
                return False
        r = self.remaining_wall()
        return r is None or r > max(0.0, self.cfg.min_call_seconds)

    def can_call(self, role: str, prompt_tokens: int = 0, max_new_tokens: int = 0) -> bool:
        with self._lock:
            if self.used.role_calls >= self.cfg.role_calls:
                return False
            if role in ("M3", "M5") and self.used.patch_slots >= self.max_patch_slots:
                return False
            if self.tokens_left() < prompt_tokens + max_new_tokens:
                return False
        r = self.remaining_wall()
        return r is None or r > max(0.0, self.cfg.min_call_seconds)

    # -------------------------------------------------------------- #
    def reserve_call(self, role: str, prompt_tokens: int, max_new_tokens: int, label: Optional[str] = None) -> Reservation:
        """Atomically charge a role call (+ a patch slot for M3/M5) and reserve tokens. Raises on refusal.
        ``label`` is the per-role bookkeeping key when it differs from the slot role (B0's direct edits are charged
        like M3 writes but attributed to ``legacy``)."""
        self.check_deadline(f"before {role} call")
        patch = role in ("M3", "M5")
        with self._lock:
            if patch and self.used.patch_slots >= self.max_patch_slots:
                raise BudgetExhausted("patch")
            if self.used.role_calls >= self.cfg.role_calls:
                raise BudgetExhausted("role_call")
            need = prompt_tokens + max_new_tokens
            if self.tokens_left() < need:
                raise BudgetExhausted("token")
            self.used.role_calls += 1
            key = label or role
            self.used.calls_by_role[key] = self.used.calls_by_role.get(key, 0) + 1
            if patch:
                self.used.patch_slots += 1
            self.used.reserved_tokens += need
        return Reservation(role, prompt_tokens, max_new_tokens, patch)

    def commit_call(self, res: Reservation, output_tokens: int, prompt_tokens: Optional[int] = None) -> None:
        """Charge actual tokens and release the unused part of the reservation (idempotent)."""
        with self._lock:
            if res.committed:
                return
            res.committed = True
            self.used.reserved_tokens -= res.prompt_tokens + res.max_new_tokens
            p = res.prompt_tokens if prompt_tokens is None else prompt_tokens
            self.used.input_tokens += p
            self.used.output_tokens += min(max(0, output_tokens), res.max_new_tokens)

    def charge_failed_call(self, res: Reservation) -> None:
        """A call that failed before producing tokens (overflow / timeout): the prompt still counts if sent."""
        self.commit_call(res, 0)

    def reserve_probes(self, n: int) -> int:
        """Grant up to ``n`` extra probe calls (respecting the per-candidate cap); returns the granted count."""
        n = min(n, self.cfg.max_probes_per_candidate)
        with self._lock:
            left = self.cfg.extra_probe_calls - self.used.extra_probe_calls
            g = max(0, min(n, left))
            self.used.extra_probe_calls += g
            return g

    def probes_left(self) -> int:
        with self._lock:
            return self.cfg.extra_probe_calls - self.used.extra_probe_calls

    def reserve_profile(self) -> bool:
        with self._lock:
            if self.used.profiles >= self.cfg.max_profiles:
                return False
            self.used.profiles += 1
            return True

    def snapshot(self) -> Dict:
        with self._lock:
            d = asdict(self.used)
        d["elapsed_s"] = self.clock() - self.t0
        d["patch_slots_max"] = self.max_patch_slots
        return d
