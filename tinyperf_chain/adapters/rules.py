"""Rule-based and scripted role implementations (baselines, ablation replacements, tests).

Rule roles use the same public information as the learned roles (passed in structured form) and emit the same
canonical text, which the controller parses with the same strict parser.

=====  =====================================================================================
M1     ``hottest_self`` | ``hottest_cumulative`` | ``random`` | ``opportunity`` (has an applicable registry
       plan, ranked by exclusive time)
M2     ``first_applicable``: registry sites in priority order, preconditions checked, attempted plans skipped
M3     ``programmatic``: the registry's programmatic proposer for the plan (deterministic; ABANDON if none)
M4     ``fixed_edge`` (p01,p02,p03 when on the menu) | ``random`` | ``transform_specific`` (registry risk probes)
M5     ``programmatic_repair``: another registry proposal for the same plan, different from the failed one
=====  =====================================================================================
"""
from __future__ import annotations

import ast
import random
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from tinyperf_chain import registry as R
from tinyperf_chain.adapters.base import Prepared, RoleAdapter
from tinyperf_chain.schemas import (PatchAction, Plan, ProbeSelection, RequestEnvelope, RoleResponse, TargetChoice,
                                    format_patch, format_plan, format_probes, format_target, parse_role_output)


def _respond(prep: Prepared, env: RequestEnvelope, text: str, kind: str, t0: int, meta: Optional[Dict] = None) -> RoleResponse:
    parsed = parse_role_output(prep.role, text, prep.ctx.get("function_id", ""))
    return RoleResponse(prep.role, env.request_id, text, parsed, kind, 0, 0, True,
                        latency_ns=time.perf_counter_ns() - t0, meta=meta or {})


class RuleTargetSelector(RoleAdapter):
    kind = "rule"

    def __init__(self, strategy: str = "hottest_self", seed: int = 0):
        if strategy not in ("hottest_self", "hottest_cumulative", "random", "opportunity"):
            raise ValueError(strategy)
        self.strategy = strategy
        self.seed = seed

    def describe(self) -> Dict[str, Any]:
        return {"kind": self.kind, "rule": f"M1:{self.strategy}"}

    def execute(self, prep: Prepared, env: RequestEnvelope) -> RoleResponse:
        t0 = time.perf_counter_ns()
        cat: List[Dict[str, Any]] = prep.ctx.get("catalog", [])
        if self.strategy == "opportunity":
            cat = [c for c in cat if c.get("has_plan")]
        if not cat:
            return _respond(prep, env, format_target(TargetChoice(None)), self.kind, t0)
        if self.strategy == "random":
            rng = random.Random(f"{self.seed}|{env.request_id}")
            pick = rng.choice(cat)
        else:
            key = "cum_share" if self.strategy == "hottest_cumulative" else "self_share"
            pick = max(cat, key=lambda c: (c.get(key, 0.0), -int(c["display_id"][1:])))
        return _respond(prep, env, format_target(TargetChoice(pick["display_id"])), self.kind, t0)


class RulePlanner(RoleAdapter):
    kind = "rule"

    def describe(self) -> Dict[str, Any]:
        return {"kind": self.kind, "rule": "M2:first_applicable"}

    def execute(self, prep: Prepared, env: RequestEnvelope) -> RoleResponse:
        t0 = time.perf_counter_ns()
        fid = prep.ctx["function_id"]
        fn = ast.parse(prep.ctx["fn_source"]).body[0]
        attempted = set(prep.ctx.get("attempted", ()))
        for p in R.static_plans(fn, prep.ctx.get("fcontract")):
            p = Plan(fid, p.transform_id, p.operands, p.preserve)
            if p.key() not in attempted:
                return _respond(prep, env, format_plan(p), self.kind, t0)
        return _respond(prep, env, format_plan(Plan(fid, None)), self.kind, t0)


class ProgrammaticWriter(RoleAdapter):
    """Deterministic plan executor built on the registry proposer (a rule M3, not a learned one)."""

    kind = "rule"

    def __init__(self, pick: int = 0):
        self.pick = pick

    def describe(self) -> Dict[str, Any]:
        return {"kind": self.kind, "rule": "M3:programmatic"}

    def execute(self, prep: Prepared, env: RequestEnvelope) -> RoleResponse:
        t0 = time.perf_counter_ns()
        plan: Plan = prep.ctx["plan"]
        cands = R.propose(prep.ctx["fn_source"], plan.transform_id, plan) if plan.transform_id in R.MECHANISMS else []
        if len(cands) <= self.pick:
            return _respond(prep, env, format_patch(PatchAction("abandon")), self.kind, t0)
        return _respond(prep, env, format_patch(PatchAction("edit", cands[self.pick])), self.kind, t0)


class RuleProbeSelector(RoleAdapter):
    kind = "rule"

    def __init__(self, strategy: str = "fixed_edge", k: int = 3, seed: int = 0):
        if strategy not in ("fixed_edge", "random", "transform_specific"):
            raise ValueError(strategy)
        self.strategy, self.k, self.seed = strategy, k, seed

    def describe(self) -> Dict[str, Any]:
        return {"kind": self.kind, "rule": f"M4:{self.strategy}"}

    def execute(self, prep: Prepared, env: RequestEnvelope) -> RoleResponse:
        t0 = time.perf_counter_ns()
        menu: List[str] = list(prep.ctx.get("menu_ids", []))
        if self.strategy == "fixed_edge":
            ids = [p for p in R.FIXED_EDGE_SUITE if p in menu]
        elif self.strategy == "random":
            rng = random.Random(f"{self.seed}|{env.request_id}")
            ids = sorted(rng.sample(menu, min(self.k, len(menu))))
        else:
            plan: Optional[Plan] = prep.ctx.get("plan")
            risk = R.MECHANISMS[plan.transform_id].risk_probes if plan and plan.transform_id in R.MECHANISMS else R.FIXED_EDGE_SUITE
            ids = [p for p in risk if p in menu]
        return _respond(prep, env, format_probes(ProbeSelection(tuple(ids[: self.k]))), self.kind, t0)


class RuleRepairer(RoleAdapter):
    kind = "rule"

    def describe(self) -> Dict[str, Any]:
        return {"kind": self.kind, "rule": "M5:programmatic_repair"}

    def execute(self, prep: Prepared, env: RequestEnvelope) -> RoleResponse:
        t0 = time.perf_counter_ns()
        plan: Plan = prep.ctx["plan"]
        failed = prep.ctx.get("failed_source", "")
        try:
            failed_norm = ast.unparse(ast.parse(failed)).rstrip("\n") + "\n"
        except SyntaxError:
            failed_norm = failed
        cands = R.propose(prep.ctx["fn_source"], plan.transform_id, plan) if plan.transform_id in R.MECHANISMS else []
        for c in cands:
            if c != failed_norm:
                return _respond(prep, env, format_patch(PatchAction("edit", c)), self.kind, t0)
        return _respond(prep, env, format_patch(PatchAction("abandon")), self.kind, t0)


class ScriptedAdapter(RoleAdapter):
    """Returns pre-scripted outputs (texts, or callables of the prepared request), in order. For tests."""

    kind = "scripted"

    def __init__(self, outputs: Sequence[Any], error: Optional[str] = None):
        self.outputs = list(outputs)
        self.calls: List[Prepared] = []
        self.error = error

    def execute(self, prep: Prepared, env: RequestEnvelope) -> RoleResponse:
        t0 = time.perf_counter_ns()
        self.calls.append(prep)
        if not self.outputs:
            text = {"M1": "NO_TARGET", "M2": "NO_PLAN", "M3": "ABANDON", "M4": "PROBES none", "M5": "ABANDON"}[prep.role]
        else:
            o = self.outputs.pop(0)
            text = o(prep) if callable(o) else o
        if text == "__timeout__":
            r = _respond(prep, env, "", self.kind, t0)
            r.error = "timeout"
            return r
        return _respond(prep, env, text, self.kind, t0)


def rule_router_adapters(target: str = "hottest_self", probes: str = "transform_specific", seed: int = 0) -> Dict[str, RoleAdapter]:
    return {"M1": RuleTargetSelector(target, seed), "M2": RulePlanner(), "M3": ProgrammaticWriter(),
            "M4": RuleProbeSelector(probes, seed=seed), "M5": RuleRepairer()}
