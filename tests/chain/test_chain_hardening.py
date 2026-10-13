"""Strengthened acceptance tests and tests for the audit fixes.

T06+  alias-aware patch policy (module state cannot be written through aliases, holders or unbound methods);
T11   invalid probe selections consume only the role call; T15 fresh namespace per independent trial;
T16   repeated no-op timing never consistently accepts; T17 process-group kill, limits in the selftest, canary;
T18   public traces (events) contain no private inputs and no failing-index leak; T22/T23 early-EOS parity, empty
prompts, per-row deadlines, wasted-step accounting; plan adherence labelling; replay / curriculum weights;
scheduler per-request deadlines; spec 17.1 config; contract policy rejection; promotion gate; RL reward on reverts.
"""
from __future__ import annotations

import json
import os
import sys
import textwrap
import time

import pytest
import torch

from tinyperf_chain.program import PatchError, Program, apply_patch

from .helpers import make_task

BASE = textwrap.dedent('''
    CACHE = {}
    TABLE = [[1, 2], [3, 4]]
    ALLOWED = {"a", "b"}
    def helper(x):
        return x + 1
    def f(xs):
        return [x for x in xs if x in ALLOWED]
''')
RET = "    return [x for x in xs if x in ALLOWED]\n"


def _apply(code: str):
    p = Program.from_source(BASE, ["f"])
    return apply_patch(p, p.hash, "f", code, [])


@pytest.mark.parametrize("code", [
    "def f(xs):\n    CACHE['k'] = 1\n" + RET,
    "def f(xs):\n    c = CACHE\n    c['k'] = 1\n" + RET,
    "def f(xs):\n    key = tuple(xs)\n    c = CACHE\n    if key in c:\n        return c[key]\n    r = [x for x in xs if x in ALLOWED]\n    c[key] = r\n    return r\n",
    "def f(xs):\n    dict.update(CACHE, {'k': 1})\n" + RET,
    "def f(xs):\n    CACHE.__setitem__('k', 1)\n" + RET,
    "def f(xs):\n    row = TABLE[0]\n    row.append(9)\n" + RET,
    "def f(xs):\n    for row in TABLE:\n        row.append(9)\n" + RET,
    "def f(xs):\n    rows = list(TABLE)\n    rows[0].append(9)\n" + RET,
    "def f(xs):\n    rows = TABLE.copy()\n    for r in rows:\n        r.append(1)\n" + RET,
    "def f(xs):\n    m = min(TABLE)\n    m.append(1)\n" + RET,
    "def f(xs):\n    c = TABLE\n    c += [[5]]\n" + RET,
    "def f(xs):\n    helper.cache = {}\n" + RET,
    "def f(xs):\n    a, b = TABLE\n    a.append(1)\n" + RET,
    "def f(xs):\n    rows = [[r] for r in TABLE]\n    rows[0][0].append(5)\n" + RET,
])
def test_t06_alias_policy_rejects_hidden_state(code):
    with pytest.raises(PatchError) as e:
        _apply(code)
    assert e.value.kind == "policy_violation"


@pytest.mark.parametrize("code", [
    "def f(xs):\n" + RET,
    "def f(xs):\n    s = set(ALLOWED)\n    s.add('c')\n    return [x for x in xs if x in s]\n",
    "def f(xs):\n    rows = list(TABLE)\n    rows.sort()\n" + RET,
    "def f(xs):\n    total = 0\n    for r in TABLE:\n        total += len(r)\n    return [x for x in xs if x in ALLOWED][:total]\n",
    "def f(xs):\n    k = helper(len(TABLE))\n    return [x for x in xs if x in ALLOWED][:k]\n",
    "def f(xs):\n    rows = [list(r) for r in TABLE]\n    rows[1].append(1)\n" + RET,
    "def f(xs):\n    d = dict(CACHE.items())\n    d['x'] = 1\n" + RET,
    "def f(xs):\n    v = CACHE.get('k', 0)\n    return [x for x in xs if x in ALLOWED][v:]\n",
])
def test_t06_alias_policy_accepts_reads_and_fresh_containers(code):
    assert _apply(code).function_source("f")


# --------------------------------------------------------------------------- #
def test_t15_fresh_namespace_per_trial(verifier):
    src = "CACHE = {}\n\ndef f(xs):\n    if CACHE:\n        raise RuntimeError('persisted')\n    CACHE['seen'] = True\n    return sum(xs)\n"
    p = Program.from_source(src, ["f"])
    r = verifier.runner
    req = r.base_request("run")
    req.update(code=p.executable(), entry="_tp_entry", inputs=[(0, [1, 2]), (0, [3]), (0, [4, 5])], output="full")
    resp = r.run(req)
    assert resp["status"] == "ok" and [x["exc"] for x in resp["records"]] == [None, None, None]
    assert all(isinstance(x.get("elapsed_ns"), int) for x in resp["records"])
    req = r.base_request("time_pair")
    req.update(code_a=p.executable(), code_b=p.executable(), entry="_tp_entry", workload=[(0, [1, 2, 3])], blocks=2, reps=4, seed=1)
    resp = r.run(req)
    assert resp["status"] == "ok", resp.get("message")
    assert resp["fresh_namespace_per_rep"] and resp["warmup"] == 1 and "cpu_model" in resp["environment"]
    assert resp["_meta"]["uid"] == r.uid and "taskset_applied" in resp["_meta"]


def test_t16_noop_never_consistently_faster(verifier):
    src = "def f(xs):\n    return sum(xs)\n"
    wl = [(0, list(range(5000)))]
    task, priv = make_task(src, ["f"], [[("xs", "list[int]")]], [(0, [1, 2])], wl)
    p0 = task.original()
    vs = [verifier.paired_timing(task, wl, p0, p0, blocks=5, seed=s).verdict for s in range(8)]
    # "does not consistently appear faster" (T16): a rare noise acceptance with 3 ms blocks on a loaded host is
    # possible, a pattern is not; the point estimates must straddle 1.0
    assert sum(v.accepted for v in vs) <= 1
    med = sorted(v.speedup for v in vs)[4]
    assert 0.9 < med < 1.1


def test_t17_hang_killed_as_group_and_limits_reported(verifier):
    r = verifier.runner
    hang = Program.from_source("def f(x):\n    while True:\n        x += 1\n", ["f"])
    req = r.base_request("run")
    req["limits"]["call_timeout_s"] = 30.0            # the in-process alarm would fire too late: the group kill must
    req.update(code=hang.executable(), entry="_tp_entry", inputs=[(0, 1)])
    t0 = time.monotonic()
    resp = r.run(req, timeout_s=2)
    assert resp["status"] == "timeout" and resp["_meta"]["timed_out"] and time.monotonic() - t0 < 10
    st = r.selftest()
    assert st["status"] == "ok" and st["limits_applied"] and st["home_listable"] is False and "network" in st
    if verifier.cfg.sandbox.isolation in ("unshare", "bwrap"):
        assert st["network"] is False                 # only an OS namespace blocks the network; rlimits mode reports it
    assert st["rlimit_nproc"] == 0 and st["rlimit_as_mb"] == verifier.cfg.sandbox.memory_mb
    if r.uid is not None:                             # root dropped to an unprivileged uid: NPROC=0 is enforced
        assert st["fork"] is False and st["uid"] == r.uid


def test_t17_canary_distinguishes_infrastructure_from_candidate(tmp_path, cfg):
    """A worker that dies without a reply triggers a canary; if the canary also fails the result is an
    infrastructure error, otherwise a candidate crash."""
    from tinyperf_chain.config import SandboxCfg
    from tinyperf_chain.sandbox.runner import IsolationUnavailable, SandboxRunner

    import stat
    import tempfile

    # the worker may run as an unprivileged uid: the wrapper must live somewhere world-readable
    d = tempfile.mkdtemp(prefix="tpchain_wrap_")
    os.chmod(d, 0o755)
    wrapper = os.path.join(d, "flaky_python.py")
    with open(wrapper, "w") as f:
        f.write(textwrap.dedent(f'''
            import subprocess, sys
            data = sys.stdin.buffer.read()
            if b"CRASHME" in data:
                sys.exit(2)
            p = subprocess.run([{sys.executable!r}] + sys.argv[1:], input=data, capture_output=True)
            sys.stdout.buffer.write(p.stdout); sys.stderr.buffer.write(p.stderr); sys.exit(p.returncode)
        '''))
    os.chmod(wrapper, 0o644)
    sc = SandboxCfg(python=sys.executable)
    runner = SandboxRunner(sc, selfcheck=False)
    # the runner's command is [python, flags, worker]: route it through the wrapper
    runner._cmd = lambda cpus, workdir, _orig=runner._cmd: [sys.executable, wrapper] + _orig(cpus, workdir)[1:]  # type: ignore[method-assign]
    good = Program.from_source("def f(x):\n    return x\n", ["f"])
    req = runner.base_request("run")
    req.update(code=good.executable(), entry="_tp_entry", inputs=[(0, 1)])
    assert runner.run(req)["status"] == "ok"
    bad = Program.from_source("def f(x):\n    return 'CRASHME'\n", ["f"])      # the marker makes the wrapper die
    req.update(code=bad.executable())
    resp = runner.run(req)
    assert resp["status"] == "worker_error" and resp["_meta"]["canary_ok"] is True   # candidate crash, canary healthy
    # a dead interpreter: the canary fails too -> infrastructure error, and the selfcheck refuses it up front
    dead = SandboxRunner(SandboxCfg(python="/bin/false"), selfcheck=False)
    assert dead.run(req)["status"] == "infrastructure_error"
    with pytest.raises(IsolationUnavailable):
        SandboxRunner(SandboxCfg(python="/bin/false"))


# --------------------------------------------------------------------------- #
def test_t18_public_traces_carry_no_private_inputs(cfg, verifier, tmp_path):
    from tinyperf_chain.adapters.base import RoleRouter
    from tinyperf_chain.adapters.rules import ScriptedAdapter
    from tinyperf_chain.controller import ChainController, EpisodeSpec

    src = "def f(xs):\n    out = []\n    for x in xs:\n        if x not in out:\n            out.append(x)\n    return out\n"
    secret = [(0, [424242, 424242, 7]), (0, [9, 8, 9, 8, 31337])]
    task, priv = make_task(src, ["f"], [[("xs", "list[int]")]], secret, [(0, [5, 5, 6])])
    task.public_examples = {"f": [([1, 2, 3],)]}                       # no duplicates: the public smoke check passes
    wrong = "<EDIT>\ndef f(xs):\n    return list(xs)\n</EDIT>"       # drops the de-duplication: fails only on hidden inputs
    router = RoleRouter({"M1": ScriptedAdapter(["TARGET f00"] * 3), "M2": ScriptedAdapter(["PLAN membership_index; src=out; loop=n00"] * 3),
                         "M3": ScriptedAdapter([wrong] * 3), "M4": ScriptedAdapter([]), "M5": ScriptedAdapter(["ABANDON"] * 3)},
                        "scripted", "S")
    ev = str(tmp_path / "events.jsonl")
    rec = ChainController(cfg, verifier, router).run_episode(task, priv, EpisodeSpec("t18", seed=0, mode="dev", events_path=ev))
    assert rec["accepted_edits"] == 0
    events = [json.loads(l) for l in open(ev) if l.strip()]
    blob = json.dumps(events) + json.dumps({k: v for k, v in rec.items() if k != "events"})
    for leak in ("424242", "31337"):
        assert leak not in blob
    mand = [e for e in events if e["kind"] == "verify" and e["stage"] == "mandatory"]
    assert mand and all(e["n_checked"] == len(secret) for e in mand) and all(e.get("public_feedback") is None for e in mand)
    assert all(e["bundle_sha256"] == "scripted" and "start_mono_ns" in e for e in mand)
    assert not os.path.exists(ev + ".partial")


def test_t11_invalid_probe_selection_consumes_only_the_call(cfg, verifier, unit_count_common):
    from tinyperf_chain.adapters.base import RoleRouter
    from tinyperf_chain.adapters.rules import ScriptedAdapter, rule_router_adapters
    from tinyperf_chain.controller import ChainController, EpisodeSpec

    base = rule_router_adapters()
    u = unit_count_common
    for bad in ("PROBES p01,p01", "PROBES p99", "PROBES p01,p02,p03,p04"):
        ad = dict(base)
        ad["M4"] = ScriptedAdapter([bad] * 8)
        rec = ChainController(cfg, verifier, RoleRouter(ad, "scripted", "S")).run_episode(
            u.public, u.private, EpisodeSpec("t11", seed=1, mode="dev", run_audit=False))
        assert rec["extra_probe_calls"] == 0
        assert rec["control_failure_reasons"].get("M4"), rec["control_failure_reasons"]
        kinds = [e["kind"] for e in rec["events"]]
        assert "verify" in kinds and any(e["kind"] == "verify" and e["stage"] == "mandatory" for e in rec["events"])


# --------------------------------------------------------------------------- #
def test_t22_t23_early_eos_parity_empty_prompt_and_wasted_steps(tiny_model):
    from tinyperf.common.config import GenerationConfig
    from tinyperf.model.generate import generate as legacy_generate
    from tinyperf_chain.serve.engine import EngineOverflow, RoleEngine, SamplingSpec

    m = tiny_model.eval()
    prompts = [[5, 6, 7, 8, 9, 10, 11], [12, 13], [14, 15, 16, 17]]
    spec = SamplingSpec(greedy=True)
    probe = RoleEngine(m, static_cache=True).generate(prompts, 3, spec, [999999], pad_id=0)
    stop = [probe[0].ids[0]]                       # whatever row 0 emits first becomes a stop token
    a = RoleEngine(m, static_cache=True).generate(prompts, 12, spec, stop, pad_id=0)
    b = RoleEngine(m, static_cache=False).generate(prompts, 12, spec, stop, pad_id=0)
    legacy = legacy_generate(m, prompts, GenerationConfig(max_new_tokens=12), pad_id=0, stop_ids=stop, greedy=True)
    assert [r.ids for r in a] == [r.ids for r in b] == legacy.ids
    assert a[0].finished and len(a[0].ids) == 1
    assert any(not r.finished or len(r.ids) > 1 for r in a)       # the other rows kept going (finished-row padding path)
    assert a[0].wasted_steps > 0 and sum(r.wasted_steps for r in a) > 0
    with pytest.raises(EngineOverflow):
        RoleEngine(m).generate([[]], 4, spec, stop, pad_id=0)


def test_per_row_deadlines_never_truncate_neighbours(tiny_model):
    from tinyperf_chain.serve.engine import RoleEngine, SamplingSpec
    from tinyperf_chain.serve.scheduler import RoleScheduler

    m = tiny_model.eval()
    eng = RoleEngine(m)
    spec = SamplingSpec(greedy=True)
    expired = time.monotonic() - 1.0
    outs = eng.generate([[5, 6, 7], [8, 9]], 8, spec, [999999], pad_id=0, deadlines=[expired, None])
    assert outs[0].timed_out and outs[0].ids == [] and not outs[0].finished
    assert not outs[1].timed_out and len(outs[1].ids) == 8
    sched = RoleScheduler({"m": eng}, window_ms=20.0, max_batch=8, max_pending=4)
    try:
        f1 = sched.submit("m", [5, 6, 7], 8, spec, [999999], 1, 0, expired)
        f2 = sched.submit("m", [8, 9], 8, spec, [999999], 2, 0, None)
        r1, r2 = f1.result(timeout=30), f2.result(timeout=30)
        assert r1.timed_out and not r2.timed_out and len(r2.ids) == 8
        assert sched.stats["m"]["batches"] >= 1 and sched.in_flight("m") == 0
    finally:
        sched.close()


# --------------------------------------------------------------------------- #
def test_plan_adherence_labels_only_realised_mechanisms():
    from tinyperf_chain import registry as R

    old = "def ms(xs):\n    best = xs[0]\n    for i in range(len(xs)):\n        for j in range(i, len(xs)):\n            s = sum(xs[i:j + 1])\n            if s > best:\n                best = s\n    return best\n"
    kadane = "def ms(xs):\n    best = cur = xs[0]\n    for x in xs[1:]:\n        cur = max(x, cur + x)\n        best = max(best, cur)\n    return best\n"
    hoisted = "def ms(xs):\n    best = xs[0]\n    n = len(xs)\n    for i in range(n):\n        for j in range(i, n):\n            s = sum(xs[i:j + 1])\n            if s > best:\n                best = s\n    return best\n"
    assert R.label_rewrite(old, kadane) is None
    assert R.label_rewrite(old, hoisted).transform_id == "loop_invariant_hoist"
    keep = "def keep(items, allowed):\n    out = []\n    for x in items:\n        if x in allowed:\n            out.append(x)\n    return out\n"
    fast = "def keep(items, allowed):\n    s = set(allowed)\n    return [x for x in items if x in s]\n"
    assert R.label_rewrite(keep, fast).transform_id == "membership_index"
    for mid in R.MECHANISMS:                      # programmatic proposals always realise their own mechanism
        for c in R.propose(keep, mid):
            assert R.extract_plan(keep, c, mid) is not None


def test_replay_mixture_and_mechanism_balance(cfg, tok):
    from tinyperf_chain.data.project import Projector

    pj = Projector(cfg, tok)
    recs = [{"weight": 1.0, "round": 0, "mechanism": "membership_index"} for _ in range(6)]
    recs += [{"weight": 1.0, "round": 1, "mechanism": "string_join"} for _ in range(2)]
    info = pj._balance_and_replay("M3", recs)
    recent = sum(r["weight"] for r in recs if r["round"] == 1)
    earlier = sum(r["weight"] for r in recs if r["round"] == 0)
    assert abs(recent / (recent + earlier) - 0.7) < 1e-9 and info["replay"]["n_recent"] == 2
    assert all(r["replay"] == "recent" for r in recs if r["round"] == 1)
    assert info["mechanism_counts"] == {"membership_index": 6, "string_join": 2}


def test_spec_17_1_config_and_contract_policies():
    from tinyperf_chain.config import ConfigError, load_chain_config
    from tinyperf_chain.contracts import CONTRACT_VERSION, FunctionContract, ParamSpec, TaskContract

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    c = load_chain_config(os.path.join(repo, "configs", "chain", "report_17_1.yaml")).validate()
    assert c.roles.M3.mode == "sample" and c.roles.M3.temperature == 0.6 and c.roles.M1.mode == "greedy"
    with pytest.raises(ConfigError):
        load_chain_config(os.path.join(repo, "configs", "chain", "report_17_1.yaml"), ["roles.M1.temperature=0.5"]).validate()
    with pytest.raises(ConfigError):
        load_chain_config(os.path.join(repo, "configs", "chain", "report_17_1.yaml"), ["budget.patch_slots=six"])
    with pytest.raises(ConfigError):
        load_chain_config(os.path.join(repo, "configs", "chain", "report_17_1.yaml"), ["serving.timing_workers=2"]).validate()
    tc = TaskContract(CONTRACT_VERSION, [FunctionContract("h", [ParamSpec("a", "list[int]")])], exception_policy="message")
    assert any("exception_policy" in r for r in tc.unsupported_reasons())


def test_promotion_gate_and_rl_revert_reward(cfg):
    from tinyperf_chain.eval.promote import promotion_decision
    from tinyperf_chain.train.rl import rewards_for

    report = {"paired_vs_B4": {"B4r@6": {"success": {"lo": -0.01, "hi": 0.1, "n_clusters": 20}, "log_speedup": {"lo": 0.0, "hi": 0.2}},
                               "B4bad@6": {"success": {"lo": -0.2, "hi": 0.0, "n_clusters": 20}, "log_speedup": {"lo": -0.3, "hi": 0.0}}}}
    assert promotion_decision(report, "B4r", "B4")["promote"]
    assert not promotion_decision(report, "B4bad", "B4")["promote"]
    assert not promotion_decision(report, "B4r", "B4", role_report={"m1": {"bundle": {"mean_regret": 0.5}}},
                                  role_baseline={"m1": {"bundle": {"mean_regret": 0.1}}})["promote"]
    r_rev, inv_rev, _ = rewards_for({"stage": "duplicate", "kind": "duplicate", "accepted": False}, True, False, cfg)
    r_bad, inv_bad, _ = rewards_for({"stage": "mandatory", "kind": "private_mismatch", "accepted": False}, True, False, cfg)
    assert r_rev == -cfg.rl.slot_cost and not inv_rev and inv_bad and r_bad < r_rev


def test_m5_ablation_views_and_release_manifest_fields(cfg, tok, tiny_model, tmp_path):
    from tinyperf_chain.adapters.model import view_for_prompt_role
    from tinyperf_chain.bundle import lock_bundle, load_bundle_json
    from tinyperf_chain.context import M3View, M5View

    v5 = M5View("f00", "def f(x):\n    return x\n", "def f(x):\n    return x + 1\n", [], "x:int", "PLAN loop_invariant_hoist; src=x", {"failure_kind": "return_mismatch"})
    v3 = view_for_prompt_role("M3", "M5", v5)
    assert isinstance(v3, M3View) and v3.target_source == v5.current_source and v3.plan == v5.plan
    assert view_for_prompt_role("M5", "M5", v5) is v5
    ck = str(tmp_path / "m.pt")
    tiny_model.save(ck, extra={"role": "M3", "parent_sha256": "abc", "data_hash": {"train": "x"}})
    b = {"schema_version": "bundle_v1", "system_id": "T", "roles": {r: {"adapter": "neural", "checkpoint": ck} for r in ("M1", "M2", "M3", "M4", "M5")}}
    bp = str(tmp_path / "b.json")
    json.dump(b, open(bp, "w"))
    locked = lock_bundle(bp, str(tmp_path / "b.lock.json"), cfg)
    rel = locked["release"]
    w = rel["weights"]["M3"]
    assert w["parent_sha256"] == "abc" and w["training_data_sha256"] == {"train": "x"} and w["params_recount"] == w["params_formula"]
    assert rel["decoding"]["M3"]["temperature"] == 0.6 and "contracts" in rel["registries"] and "verification" in rel["evaluator"]
    assert load_bundle_json(str(tmp_path / "b.lock.json"))["bundle_sha256"] == locked["bundle_sha256"]


def test_replicate_intervals_across_training_seeds():
    from tinyperf_chain.eval.aggregate import _success, aggregate, replicate_intervals

    recs = []
    for b, seed in (("B4_s1", 1), ("B4_s2", 2), ("B4_s3", 3)):
        for c in range(4):
            for e in range(2):
                ok = (c + e + seed) % 3 != 0
                recs.append({"system_id": b, "system_group": "B4", "training_seed": seed, "cluster_id": f"c{c}", "task_id": f"t{c}",
                             "patch_slots_max": 6, "audit_pass": ok, "raw_audit_speedup": 2.0 if ok else None,
                             "effective_speedup": 2.0 if ok else 1.0, "failed_run": False})
    r = replicate_intervals(recs, _success)
    g = r["B4@6"]
    assert g["n_bundles"] == 3 and g["lo"] <= g["mean"] <= g["hi"] and set(g["per_bundle"]) == {"B4_s1", "B4_s2", "B4_s3"}
    rep = aggregate(recs, reference=None, do_paired=False)
    assert "replicates" in rep and "B4@6" in rep["replicates"]["success"]


def test_rl_resume_round_trip(tmp_path, tiny_model, cfg):
    """T25 for RL: last.pt restores weights, optimizer, RNG and step; a changed state set is refused."""
    import random

    from tinyperf_chain.train.rl import RoleRL

    rl = object.__new__(RoleRL)
    rl.cfg, rl.role, rl.out, rl.model = cfg, "M3", str(tmp_path), tiny_model
    rl.opt = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    rl.rng, rl.step, rl.states_sha256, rl.init_sha = random.Random(3), 7, "states-a", "parent-x"
    # one optimizer step so the optimizer carries state
    loss = tiny_model(torch.tensor([[1, 2, 3]]))[0, -1].sum()
    loss.backward()
    rl.opt.step()
    rl.save("last.pt", with_state=True)
    draw, w0 = rl.rng.random(), tiny_model.wte.weight.detach().clone()
    rl2 = object.__new__(RoleRL)
    rl2.cfg, rl2.role, rl2.out = cfg, "M3", str(tmp_path)
    torch.manual_seed(99)
    rl2.model = type(tiny_model)(tiny_model.cfg)
    rl2.opt = torch.optim.AdamW(rl2.model.parameters(), lr=1e-3)
    rl2.rng, rl2.step, rl2.states_sha256, rl2.init_sha = random.Random(0), 0, "states-a", "parent-x"
    rl2._resume()
    assert rl2.step == 7 and torch.equal(rl2.model.wte.weight, w0) and rl2.rng.random() == draw
    assert rl2.opt.state_dict()["state"] and len(rl2.opt.state_dict()["state"]) == len(rl.opt.state_dict()["state"])
    rl3 = object.__new__(RoleRL)
    rl3.cfg, rl3.role, rl3.out, rl3.model = cfg, "M3", str(tmp_path), type(tiny_model)(tiny_model.cfg)
    rl3.opt = torch.optim.AdamW(rl3.model.parameters(), lr=1e-3)
    rl3.rng, rl3.step, rl3.states_sha256, rl3.init_sha = random.Random(0), 0, "states-b", "parent-x"
    with pytest.raises(RuntimeError):
        rl3._resume()


def test_renamer_respects_shadowing_locals():
    from tinyperf_chain.context import rename_functions

    src = "def f(helper):\n    return helper + 1\n\ndef helper(x):\n    return f(x)\n"
    out = rename_functions(src, {"f": "f00", "helper": "f01"})
    assert "def f00(helper):" in out and "return helper + 1" in out and "def f01(x):" in out and "return f00(x)" in out
