"""T10 immutable original / rollback, T12 repair accounting, T13 concurrent reservations, T14 budget exhaustion,
T18 no private information in prompts; plus an end-to-end rule episode."""
import ast
import threading

import pytest

from tinyperf_chain import registry as R
from tinyperf_chain.adapters.base import Prepared, RoleRouter
from tinyperf_chain.adapters.rules import ScriptedAdapter, rule_router_adapters
from tinyperf_chain.budget import BudgetExhausted, BudgetLedger
from tinyperf_chain.config import BudgetCfg
from tinyperf_chain.context import PromptBuilder
from tinyperf_chain.controller import ChainController, EpisodeSpec
from tinyperf_chain.schemas import Plan, format_plan


def _first_target(prep):
    return f"TARGET {prep.ctx['catalog'][0]['display_id']}"


def _first_plan(prep):
    fn = ast.parse(prep.ctx["fn_source"]).body[0]
    p = R.static_plans(fn, prep.ctx["fcontract"])[0]
    return format_plan(Plan(prep.ctx["function_id"], p.transform_id, p.operands, p.preserve))


def _router(m1, m2, m3, m4, m5):
    return RoleRouter({"M1": m1, "M2": m2, "M3": m3, "M4": m4, "M5": m5}, "scripted", "S")


def test_end_to_end_rule_episode(cfg, verifier, unit_count_common):
    u = unit_count_common
    ctrl = ChainController(cfg, verifier, RoleRouter(rule_router_adapters(), "rules", "R"))
    rec = ctrl.run_episode(u.public, u.private, EpisodeSpec("e2e", seed=1, mode="dev"))
    assert rec["accepted_edits"] >= 1, rec["termination_reason"]
    assert rec["audit_pass"] is True and rec["raw_audit_speedup"] > 1.0
    kinds = [e["kind"] for e in rec["events"]]
    assert kinds.index("final_audit") > kinds.index("search_end")
    assert all(e["kind"] != "role_call" for e in rec["events"][kinds.index("search_end"):])


def _next_plan(prep):
    fn = ast.parse(prep.ctx["fn_source"]).body[0]
    for p in R.static_plans(fn, prep.ctx["fcontract"]):
        q = Plan(prep.ctx["function_id"], p.transform_id, p.operands, p.preserve)
        if q.key() not in prep.ctx["attempted"]:
            return format_plan(q)
    return "NO_PLAN"


def test_t12_repair_accounting(cfg, verifier, file_task):
    pub, priv = file_task
    syntax_err = lambda prep: "<EDIT>\ndef " + prep.ctx["function_name"] + "(x):\n    return (\n</EDIT>"
    m3 = ScriptedAdapter([syntax_err, "garbage", "__timeout__", "ABANDON"] * 3)
    m5 = ScriptedAdapter([syntax_err] * 5)
    ctrl = ChainController(cfg, verifier, _router(ScriptedAdapter([_first_target] * 30), ScriptedAdapter([_next_plan] * 30),
                                                  m3, ScriptedAdapter([]), m5))
    rec = ctrl.run_episode(pub, priv, EpisodeSpec("t12", seed=2, mode="dev", patch_slots=6))
    # every fresh write and every repair consumed exactly one slot, whatever its outcome
    assert rec["patch_attempts"] == len(m3.calls) + len(m5.calls) <= 6
    # repairs only follow repairable failures (syntax), never malformed / timeout / abandon; one per lineage
    texts = [e for e in rec["events"] if e["kind"] == "patch_outcome"]
    for i, e in enumerate(texts):
        if e["role"] == "M5":
            assert texts[i - 1]["role"] == "M3" and texts[i - 1]["outcome"] == "syntax_error"
            assert i + 1 >= len(texts) or texts[i + 1]["role"] == "M3"
    assert len(m5.calls) >= 1
    outcomes = {e["outcome"] for e in texts if e["role"] == "M3"}
    assert {"syntax_error", "malformed", "timeout"} <= outcomes
    assert rec["selected_sha256"] == rec["original_sha256"]     # nothing accepted, best unchanged


def test_t10_rejected_edit_keeps_best(cfg, verifier, unit_count_common):
    u = unit_count_common
    wrong = "<EDIT>\ndef count_common(queries, values):\n    return 0\n</EDIT>"
    m3 = ScriptedAdapter([wrong])
    ctrl = ChainController(cfg, verifier, _router(ScriptedAdapter([_first_target] * 3), ScriptedAdapter([_first_plan] * 3),
                                                  m3, ScriptedAdapter(["PROBES none"]), ScriptedAdapter(["ABANDON"])))
    rec = ctrl.run_episode(u.public, u.private, EpisodeSpec("t10", seed=3, mode="dev", patch_slots=2))
    assert rec["accepted_edits"] == 0 and rec["selected_sha256"] == rec["original_sha256"]
    outcomes = [e for e in rec["events"] if e["kind"] == "patch_outcome"]
    assert outcomes and outcomes[0]["stage"] in ("smoke", "probe", "mandatory") and outcomes[0]["outcome"] != "accepted"


def test_t13_concurrent_reservations_never_overbook():
    b = BudgetLedger(BudgetCfg(patch_slots=6, role_calls=32, total_input_output_tokens=20000))
    granted = []
    lock = threading.Lock()

    def worker():
        for _ in range(20):
            try:
                r = b.reserve_call("M3", 1000, 640)
            except BudgetExhausted:
                return
            with lock:
                granted.append(r)
            b.commit_call(r, 100)

    ts = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(granted) == 6 and b.used.patch_slots == 6
    assert b.used.reserved_tokens == 0 and b.used.input_tokens == 6000 and b.used.output_tokens == 600
    b2 = BudgetLedger(BudgetCfg(patch_slots=6, role_calls=32, total_input_output_tokens=3000))
    r = b2.reserve_call("M1", 1000, 1000)
    with pytest.raises(BudgetExhausted):
        b2.reserve_call("M2", 1000, 1000)                       # reservation, not usage, blocks overbooking
    b2.commit_call(r, 10)
    b2.reserve_call("M2", 1000, 900)                           # unused reservation was released
    assert b2.reserve_probes(5) == 3 and b2.probes_left() == 15


def test_t14_exhaustion_terminates_cleanly(cfg, verifier, unit_count_common):
    u = unit_count_common
    ctrl = ChainController(cfg, verifier, RoleRouter(rule_router_adapters(), "rules", "R"))
    rec = ctrl.run_episode(u.public, u.private, EpisodeSpec("t14", seed=4, mode="dev", patch_slots=1))
    assert rec["patch_attempts"] <= 1
    assert rec["audit_pass"] is not None and not rec["failed_run"]


class _CapturingRule(ScriptedAdapter):
    def __init__(self, inner, builder, sink):
        super().__init__([])
        self.inner, self.builder, self.sink = inner, builder, sink

    def prepare(self, role, view, ctx):
        self.sink.append((self.builder.build(role, view).text, ctx))
        return self.inner.prepare(role, view, ctx)

    def execute(self, prep, env):
        return self.inner.execute(prep, env)


def test_t18_no_private_information_in_prompts(cfg, verifier, tok, file_task):
    pub, priv = file_task
    builder = PromptBuilder(tok, cfg.roles, 2048)
    sink = []
    rules = rule_router_adapters()
    ctrl = ChainController(cfg, verifier, RoleRouter({r: _CapturingRule(a, builder, sink) for r, a in rules.items()}, "cap", "C"))
    ctrl.run_episode(pub, priv, EpisodeSpec("t18", seed=5, mode="dev"))
    assert sink
    secrets = [repr(c[1:]) for c in priv.online_correctness + priv.audit_correctness if len(repr(c[1:])) > 12]
    fast = priv.privileged["fast_source"]
    for text, ctx in sink:
        for s in secrets[:40]:
            assert s not in text
        for k, v in ctx.items():
            assert type(v).__name__ not in ("PrivateSuite",)
        assert str(priv.seeds.get("base")) not in text
    slow = set(priv.privileged["slow_functions"])
    fast_fns = [n for n in pub.functions if n in slow]
    from tinyperf_chain.program import Program
    first_prompt = sink[0][0]                       # before any accepted edit nothing derived from the reference exists
    for n in fast_fns:
        ref = Program(fast, tuple(pub.functions)).function_source(n)
        assert ref.strip() not in first_prompt
