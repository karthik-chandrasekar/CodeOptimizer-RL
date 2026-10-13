"""Measure before optimizing: engine variants, verifier costs and the eq. (10) forecast.

    python -m tinyperf_chain.serve.bench engine --checkpoint ckpt.pt --batch 1 8 32 --prompt-len 512 1536 --max-new 64 \\
        [--compile] [--device auto] [--saturate 64]
    python -m tinyperf_chain.serve.bench checks --config configs/chain/pilot.yaml --suite dev_files --n 100 \\
        [--engine-checkpoint ckpt.pt] [--plan-calls 19 --plan-checks 6 --plan-blocks 90 --plan-files 100 --plan-episodes 4]

``engine`` compares, on identical random prompts and *independently* (the compiled forward is swapped in only while
the compiled variant runs, and always swapped back): the legacy generator (concatenating cache, full prefill
logits), the legacy model with last-position logits, the static-cache eager engine, and optionally ``torch.compile``
of the static-cache forward.  It reports cold compile time separately from warm prefill time and decode tokens/s,
peak host memory (``ru_maxrss``), peak device memory and the KV-cache bytes, greedy-output parity with the legacy
reference (a mismatch near a logit tie is reported, never hidden) and, with ``--saturate N``, the saturated
throughput of N concurrent requests through the role scheduler.

``checks`` runs the deterministic rule writer's candidates through the verifier stages on real tasks, reports
per-stage wall/CPU costs and computes eq. (10)::

    T_serial ~ N_calls * t_model + N_checks * t_check + N_blocks * t_block

for a planned run from measured means *and* p95 values (the p95 forecast is the safety margin), together with the
bottleneck-service bound for a pipelined run.  ``t_model`` is measured with ``--engine-checkpoint``; without it the
forecast reports the model term as missing rather than zero.
"""
from __future__ import annotations

import argparse
import json
import resource
import statistics
import time
from typing import Any, Dict, List, Optional

import torch


def _sync(dev: str) -> None:
    if dev.startswith("cuda"):
        torch.cuda.synchronize()


def _host_peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _p95(x: List[float]) -> Optional[float]:
    return sorted(x)[int(0.95 * (len(x) - 1))] if x else None


def bench_engine(ckpt: str, batches: List[int], prompt_lens: List[int], max_new: int, device: str, compile_: bool,
                 repeats: int = 3, saturate: int = 0, eager_reference: bool = True) -> List[Dict[str, Any]]:
    from tinyperf.common.config import GenerationConfig
    from tinyperf.model.generate import generate as legacy_generate
    from tinyperf.model.transformer import TinyPerfLM
    import tinyperf_chain.serve.engine as engine_mod
    from tinyperf_chain.serve import cache as cache_mod
    from tinyperf_chain.serve.engine import RoleEngine, SamplingSpec

    model = TinyPerfLM.load(ckpt).to(device).eval()
    V = model.cfg.vocab_size
    out = []
    eager_forward = cache_mod.forward_static
    compiled = None
    if compile_:
        try:
            compiled = torch.compile(eager_forward)
        except Exception as e:  # noqa: BLE001
            compiled = None
            print(json.dumps({"compile_error": repr(e)[:200]}), flush=True)
    for B in batches:
        for L in prompt_lens:
            if L + max_new > model.cfg.max_seq_len:
                continue
            g = torch.Generator().manual_seed(B * 7919 + L)
            prompts = [torch.randint(10, V - 1, (L,), generator=g).tolist() for _ in range(B)]
            stop = [V - 1]
            row: Dict[str, Any] = {"batch": B, "prompt_len": L, "max_new": max_new, "host_peak_mb_before": _host_peak_mb()}
            ref_ids = None
            variants = []
            if eager_reference:
                variants.append(("legacy", None, None))
            variants += [("static", RoleEngine(model, device=device, static_cache=True), eager_forward),
                         ("legacy_last", RoleEngine(model, device=device, static_cache=False, last_logits_only=True), eager_forward)]
            if compiled is not None:
                variants.append(("static_compiled", RoleEngine(model, device=device, static_cache=True), compiled))
            for name, eng, fwd in variants:
                times, dec = [], []
                if device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()
                engine_mod._engine_forward_static = fwd if fwd is not None else eager_forward   # only for this variant
                try:
                    if name == "static_compiled":
                        _sync(device)
                        t0 = time.perf_counter()
                        eng.generate(prompts, 2, SamplingSpec(greedy=True), stop, pad_id=0)   # cold start: compile + warm-up
                        _sync(device)
                        row["compile_cold_s"] = time.perf_counter() - t0
                    for _ in range(repeats):
                        _sync(device)
                        t0 = time.perf_counter()
                        if eng is None:
                            res = legacy_generate(model, prompts, GenerationConfig(max_new_tokens=max_new), pad_id=0,
                                                  stop_ids=stop, device=device, greedy=True)
                            ids = res.ids
                            pre = None
                        else:
                            rs = eng.generate(prompts, max_new, SamplingSpec(greedy=True), stop, pad_id=0)
                            ids = [r.ids for r in rs]
                            pre = rs[0].prefill_ns / 1e6
                            dec.append(sum(len(r.ids) for r in rs) / max(1e-9, rs[0].decode_ns / 1e9))
                        _sync(device)
                        times.append(time.perf_counter() - t0)
                finally:
                    engine_mod._engine_forward_static = eager_forward                              # always restored
                row[f"{name}_s"] = statistics.median(times)
                if pre is not None:
                    row[f"{name}_prefill_ms"] = pre
                    row[f"{name}_decode_tok_s"] = statistics.median(dec)
                    row[f"{name}_kv_cache_mb"] = eng.stats["peak_cache_bytes"] / 2 ** 20
                if device.startswith("cuda"):
                    row[f"{name}_device_peak_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
                if ref_ids is None:
                    ref_ids = ids
                else:
                    row[f"{name}_parity"] = ids == ref_ids
            row["host_peak_mb_after"] = _host_peak_mb()
            if saturate > 0:
                row["saturated"] = bench_saturated(model, device, L, max_new, saturate, stop)
            out.append(row)
            print(json.dumps(row), flush=True)
    return out


def bench_saturated(model, device: str, L: int, max_new: int, n_requests: int, stop: List[int]) -> Dict[str, Any]:
    """Saturated throughput: ``n_requests`` concurrent requests through the scheduler; output tokens and completed
    requests divided by the full interval (never summed latencies)."""
    from concurrent.futures import ThreadPoolExecutor

    from tinyperf_chain.serve.engine import RoleEngine, SamplingSpec
    from tinyperf_chain.serve.scheduler import RoleScheduler

    V = model.cfg.vocab_size
    eng = RoleEngine(model, device=device, static_cache=True)
    sched = RoleScheduler({"m": eng}, window_ms=2.0, max_batch=32, max_pending=max(16, n_requests), mode="throughput")
    g = torch.Generator().manual_seed(L * 31 + n_requests)
    prompts = [torch.randint(10, V - 1, (L,), generator=g).tolist() for _ in range(n_requests)]
    lat: List[float] = []

    def one(i: int) -> int:
        t0 = time.perf_counter()
        r = sched.generate("m", prompts[i], max_new, SamplingSpec(greedy=True), stop, i, 0)
        lat.append(time.perf_counter() - t0)
        return len(r.ids)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(n_requests) as ex:
        toks = list(ex.map(one, range(n_requests)))
    interval = time.perf_counter() - t0
    sched.close()
    return {"n_requests": n_requests, "interval_s": interval, "requests_per_s": n_requests / interval,
            "output_tok_per_s": sum(toks) / interval, "latency_s_median": statistics.median(lat), "latency_s_p95": _p95(lat),
            "batches": sched.stats["m"]["batches"], "wasted_steps": sched.stats["m"]["wasted_steps"]}


def _model_cost(ckpt: str, device: str, prompt_len: int, max_new: int, n: int = 5) -> Dict[str, Any]:
    from tinyperf.model.transformer import TinyPerfLM
    from tinyperf_chain.serve.engine import RoleEngine, SamplingSpec

    model = TinyPerfLM.load(ckpt).to(device).eval()
    V = model.cfg.vocab_size
    eng = RoleEngine(model, device=device, static_cache=True)
    g = torch.Generator().manual_seed(1)
    xs = []
    for i in range(n):
        p = torch.randint(10, V - 1, (prompt_len,), generator=g).tolist()
        _sync(device)
        t0 = time.perf_counter()
        eng.generate([p], max_new, SamplingSpec(greedy=True), [V - 1], pad_id=0)
        _sync(device)
        xs.append(time.perf_counter() - t0)
    return {"n": n, "mean_s": statistics.mean(xs), "p95_s": _p95(xs), "prompt_len": prompt_len, "max_new": max_new}


def forecast(rep: Dict[str, Any], n_calls: int, n_checks: int, n_blocks: int, n_files: int, n_episodes: int,
             n_systems: int = 1, workers: int = 1) -> Dict[str, Any]:
    """Eq. (10) for one episode and for the planned run, from means and from p95 values (safety margin), plus the
    bottleneck-service bound for a pipelined run with ``workers`` parallel episodes."""
    def term(stage: str, key: str) -> Optional[float]:
        v = rep.get(stage, {}).get(key)
        return None if v is None else float(v)

    out: Dict[str, Any] = {"inputs": {"n_calls": n_calls, "n_checks": n_checks, "n_blocks": n_blocks, "n_files": n_files,
                                      "n_episodes": n_episodes, "n_systems": n_systems, "workers": workers}}
    for key in ("mean_s", "p95_s"):
        tm = term("model_call", key)
        tc = term("per_candidate_check", key)
        tb = term("timing_block", key)
        if tc is None or tb is None:
            out[key] = {"error": "verifier costs not measured"}
            continue
        model_term = None if tm is None else n_calls * tm
        episode = (model_term or 0.0) + n_checks * tc + n_blocks * tb
        run = episode * n_files * n_episodes * n_systems
        out[key] = {"t_model": tm, "t_check": tc, "t_block": tb, "episode_serial_s": episode,
                    "model_term_s": model_term, "model_term_missing": tm is None,
                    "run_serial_s": run, "run_serial_h": run / 3600.0,
                    "run_pipelined_bound_h": run / 3600.0 / max(1, workers)}
    return out


def bench_checks(config: Optional[str], suite: str, n: int, overrides: List[str], engine_ckpt: Optional[str] = None,
                 plan: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    from tinyperf_chain import registry as R
    from tinyperf_chain.adapters.verifier import Verifier
    from tinyperf_chain.config import load_chain_config
    from tinyperf_chain.program import PatchError, apply_patch
    from tinyperf_chain.serve.engine import resolve_device
    from tinyperf_chain.tasks import TaskStore

    cfg = load_chain_config(config, overrides)
    v = Verifier(cfg, cache_dir=cfg.paths.cache)
    store = TaskStore(cfg.paths.tasks, cfg.paths.private)
    tasks = store.load_public(suite)
    priv = store.load_private(suite)
    stages: Dict[str, List[float]] = {"smoke": [], "mandatory": [], "timing_block": [], "profile": [], "per_candidate_check": []}
    cpu: Dict[str, List[float]] = {"smoke": [], "mandatory": []}
    count = 0
    for t in tasks:
        p0 = t.original()
        t0 = time.perf_counter()
        v.profile(t, p0)
        stages["profile"].append(time.perf_counter() - t0)
        for name in t.functions:
            fn = p0.function_node(name)
            for plan_ in R.static_plans(fn, t.contract.function(name), limit=2):
                for code in R.propose(p0.function_source(name), plan_.transform_id, plan_)[:1]:
                    try:
                        cand = apply_patch(p0, p0.hash, name, code, t.contract.permitted_imports)
                    except PatchError:
                        continue
                    t0 = time.perf_counter()
                    sm = v.smoke_check(t, cand, name)
                    ts = time.perf_counter() - t0
                    stages["smoke"].append(ts)
                    cpu["smoke"].append(sm.cpu_s)
                    t0 = time.perf_counter()
                    mc = v.mandatory_checks(t, priv[t.task_id].online(), cand)
                    tmc = time.perf_counter() - t0
                    stages["mandatory"].append(tmc)
                    cpu["mandatory"].append(mc.cpu_s)
                    stages["per_candidate_check"].append(ts + tmc)
                    if sm.ok and mc.ok:
                        blocks = cfg.verification.train_timing_blocks
                        t0 = time.perf_counter()
                        v.paired_timing(t, priv[t.task_id].online_timing, p0, cand, blocks=blocks, seed=count)
                        stages["timing_block"].append((time.perf_counter() - t0) / blocks)
                    count += 1
                    if count >= n:
                        break
                if count >= n:
                    break
            if count >= n:
                break
        if count >= n:
            break
    rep: Dict[str, Any] = {k: {"n": len(x), "mean_s": statistics.mean(x) if x else None, "p95_s": _p95(x)} for k, x in stages.items()}
    for k, x in cpu.items():
        rep[k]["cpu_mean_s"] = statistics.mean(x) if x else None
    if engine_ckpt:
        d3 = cfg.roles.get("M3")
        rep["model_call"] = _model_cost(engine_ckpt, resolve_device(cfg.serving.device), min(d3.max_input, 512), d3.max_output)
    else:
        rep["model_call"] = {"n": 0, "mean_s": None, "p95_s": None, "note": "pass --engine-checkpoint to measure t_model"}
    rep["sandbox_stats"] = dict(v.runner.stats)
    plan = plan or {}
    b = cfg.budget
    rep["forecast"] = forecast(rep, plan.get("calls", b.role_calls), plan.get("checks", b.patch_slots),
                               plan.get("blocks", b.patch_slots * cfg.verification.development_timing_blocks + cfg.verification.audit_timing_blocks),
                               plan.get("files", 100), plan.get("episodes", 4), plan.get("systems", 1),
                               max(1, cfg.serving.episode_workers))
    rep["forecast"]["note"] = ("T_serial ~ N_calls*t_model + N_checks*t_check + N_blocks*t_block (eq. 10); the p95 row is the "
                               "safety margin; the pipelined bound divides by episode workers and ignores queueing")
    print(json.dumps(rep, indent=1))
    return rep


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("engine")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--batch", type=int, nargs="+", default=[1, 8])
    e.add_argument("--prompt-len", type=int, nargs="+", default=[256, 1024])
    e.add_argument("--max-new", type=int, default=64)
    e.add_argument("--device", default="auto")
    e.add_argument("--compile", action="store_true")
    e.add_argument("--saturate", type=int, default=0, help="also measure saturated throughput with N concurrent requests")
    e.add_argument("--no-eager-reference", action="store_true")
    c = sub.add_parser("checks")
    c.add_argument("--config", default=None)
    c.add_argument("--suite", default="dev_files")
    c.add_argument("--n", type=int, default=100)
    c.add_argument("--engine-checkpoint", default=None, help="measure t_model for the eq. 10 forecast")
    c.add_argument("--plan-calls", type=int, default=None)
    c.add_argument("--plan-checks", type=int, default=None)
    c.add_argument("--plan-blocks", type=int, default=None)
    c.add_argument("--plan-files", type=int, default=100)
    c.add_argument("--plan-episodes", type=int, default=4)
    c.add_argument("--plan-systems", type=int, default=1)
    c.add_argument("overrides", nargs="*")
    a = p.parse_args(argv)
    if a.cmd == "engine":
        from tinyperf_chain.serve.engine import resolve_device

        bench_engine(a.checkpoint, a.batch, a.prompt_len, a.max_new, resolve_device(a.device), a.compile,
                     saturate=a.saturate, eager_reference=not a.no_eager_reference)
    else:
        plan = {k: v for k, v in (("calls", a.plan_calls), ("checks", a.plan_checks), ("blocks", a.plan_blocks),
                                  ("files", a.plan_files), ("episodes", a.plan_episodes), ("systems", a.plan_systems)) if v is not None}
        bench_checks(a.config, a.suite, a.n, a.overrides, a.engine_checkpoint, plan)


if __name__ == "__main__":
    main()
