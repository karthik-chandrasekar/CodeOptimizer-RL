"""Resident-role scheduler (batching, request-level seeds) and a neural bundle driving the controller."""
import json
import os
from concurrent.futures import ThreadPoolExecutor

from tinyperf_chain.serve.engine import RoleEngine, SamplingSpec
from tinyperf_chain.serve.scheduler import RoleScheduler


def test_scheduler_batches_without_changing_results(tiny_model):
    eng = RoleEngine(tiny_model)
    spec = SamplingSpec(temperature=0.8)
    prompts = [[1, 5, 6, 7], [1, 8], [1, 9, 10, 11, 12], [1, 13, 14]]
    direct = [eng.generate([p], 8, spec, [999999], pad_id=0, seeds=[100 + i])[0].ids for i, p in enumerate(prompts)]
    sched = RoleScheduler({"m": eng}, window_ms=20, max_batch=8, max_pending=2)
    try:
        with ThreadPoolExecutor(4) as ex:
            futs = [ex.submit(sched.generate, "m", p, 8, spec, [999999], 100 + i, 0) for i, p in enumerate(prompts)]
            got = [f.result().ids for f in futs]
        assert got == direct
        assert sched.stats["m"]["rows"] == 4 and sched.stats["m"]["batches"] <= 4
    finally:
        sched.close()


def test_neural_bundle_runs_through_controller(tmp_path, cfg, verifier, tok_path, tiny_model, unit_count_common):
    from tinyperf_chain.bundle import build_router, load_bundle_json
    from tinyperf_chain.controller import ChainController, EpisodeSpec

    ck = str(tmp_path / "tiny.pt")
    tiny_model.save(ck)
    roles = {r: {"adapter": "neural", "checkpoint": ck} for r in ("M1", "M2", "M3", "M4", "M5")}
    roles["M2"] = {"adapter": "rule"}
    b = {"schema_version": "bundle_v1", "system_id": "T", "tokenizer": tok_path, "roles": roles}
    bp = str(tmp_path / "b.json")
    json.dump(b, open(bp, "w"))
    import copy
    c = copy.deepcopy(cfg)
    c.roles.M3.max_output = 48
    c.roles.M5.max_output = 48
    router, res = build_router(load_bundle_json(bp), c, device="cpu", use_scheduler=True)
    try:
        assert len(res["engines"]) == 1                     # one checkpoint shared by several roles -> one engine
        u = unit_count_common
        rec = ChainController(c, verifier, router).run_episode(u.public, u.private, EpisodeSpec("neural", seed=1, mode="dev"))
        calls = [e for e in rec["events"] if e["kind"] == "role_call"]
        assert calls and calls[0]["role"] == "M1" and calls[0]["prompt_tokens"] > 0
        assert rec["input_tokens"] >= calls[0]["prompt_tokens"] and rec["audit_pass"] is not None
    finally:
        if res["scheduler"]:
            res["scheduler"].close()
