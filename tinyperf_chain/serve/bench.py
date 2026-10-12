"""Measure before optimizing: engine variants and verifier costs.

    python -m tinyperf_chain.serve.bench engine --checkpoint ckpt.pt --batch 1 8 32 --prompt-len 512 1536 --max-new 64 \\
        [--compile] [--device auto]
    python -m tinyperf_chain.serve.bench checks --config configs/chain/pilot.yaml --suite dev_files --n 100

``engine`` compares, on identical random prompts: the legacy generator (concatenating cache, full prefill logits),
the legacy model with last-position logits, the static-cache engine, and optionally ``torch.compile`` of the
static-cache forward.  It reports prefill time, decode tokens/s, peak device memory and greedy-output parity with
the legacy reference (a mismatch near a logit tie is reported, never hidden).  Compilation time is reported
separately from warm throughput.

``checks`` runs the deterministic rule writer's candidates through the verifier stages on real tasks and reports
per-stage wall/CPU costs and the serial forecast ``T ~ N_calls t_model + N_checks t_check + N_blocks t_block``
(eq. 10) for a planned run.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any, Dict, List, Optional

import torch


def _sync(dev: str) -> None:
    if dev.startswith("cuda"):
        torch.cuda.synchronize()


def bench_engine(ckpt: str, batches: List[int], prompt_lens: List[int], max_new: int, device: str, compile_: bool,
                 repeats: int = 3) -> List[Dict[str, Any]]:
    from tinyperf.common.config import GenerationConfig
    from tinyperf.model.generate import generate as legacy_generate
    from tinyperf.model.transformer import TinyPerfLM
    from tinyperf_chain.serve import cache as cache_mod
    from tinyperf_chain.serve.engine import RoleEngine, SamplingSpec

    model = TinyPerfLM.load(ckpt).to(device).eval()
    V = model.cfg.vocab_size
    out = []
    for B in batches:
        for L in prompt_lens:
            if L + max_new > model.cfg.max_seq_len:
                continue
            g = torch.Generator().manual_seed(B * 7919 + L)
            prompts = [torch.randint(10, V - 1, (L,), generator=g).tolist() for _ in range(B)]
            stop = [V - 1]
            row: Dict[str, Any] = {"batch": B, "prompt_len": L, "max_new": max_new}
            ref_ids = None
            variants = [("legacy", None), ("static", RoleEngine(model, device=device, static_cache=True)),
                        ("legacy_last", RoleEngine(model, device=device, static_cache=False, last_logits_only=True))]
            if compile_:
                try:
                    compiled = torch.compile(cache_mod.forward_static)
                    eng = RoleEngine(model, device=device, static_cache=True)
                    orig = cache_mod.forward_static
                    t0 = time.perf_counter()
                    import tinyperf_chain.serve.engine as engine_mod
                    engine_mod.forward_static = compiled
                    eng.generate(prompts, 2, SamplingSpec(greedy=True), stop, pad_id=0)   # compile / warm-up
                    row["compile_s"] = time.perf_counter() - t0
                    variants.append(("static_compiled", eng))
                except Exception as e:  # noqa: BLE001
                    row["compile_error"] = repr(e)[:200]
            for name, eng in variants:
                times, dec = [], []
                if device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()
                for _ in range(repeats):
                    _sync(device)
                    t0 = time.perf_counter()
                    if eng is None:
                        res = legacy_generate(model, prompts, GenerationConfig(max_new_tokens=max_new), pad_id=0, stop_ids=stop,
                                              device=device, greedy=True)
                        ids = res.ids
                        pre = None
                    else:
                        rs = eng.generate(prompts, max_new, SamplingSpec(greedy=True), stop, pad_id=0)
                        ids = [r.ids for r in rs]
                        pre = rs[0].prefill_ns / 1e6
                        dec.append(sum(len(r.ids) for r in rs) / max(1e-9, rs[0].decode_ns / 1e9))
                    _sync(device)
                    times.append(time.perf_counter() - t0)
                row[f"{name}_s"] = statistics.median(times)
                if pre is not None:
                    row[f"{name}_prefill_ms"] = pre
                    row[f"{name}_decode_tok_s"] = statistics.median(dec)
                if device.startswith("cuda"):
                    row[f"{name}_peak_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
                if ref_ids is None:
                    ref_ids = ids
                else:
                    row[f"{name}_parity"] = ids == ref_ids
            if compile_ and "static_compiled_s" in row:
                import tinyperf_chain.serve.engine as engine_mod
                engine_mod.forward_static = cache_mod.forward_static
            out.append(row)
            print(json.dumps(row), flush=True)
    return out


def bench_checks(config: Optional[str], suite: str, n: int, overrides: List[str]) -> Dict[str, Any]:
    from tinyperf_chain import registry as R
    from tinyperf_chain.adapters.verifier import Verifier
    from tinyperf_chain.config import load_chain_config
    from tinyperf_chain.program import PatchError, apply_patch
    from tinyperf_chain.tasks import TaskStore

    cfg = load_chain_config(config, overrides)
    v = Verifier(cfg, cache_dir=cfg.paths.cache)
    store = TaskStore(cfg.paths.tasks, cfg.paths.private)
    tasks = store.load_public(suite)
    priv = store.load_private(suite)
    stages: Dict[str, List[float]] = {"smoke": [], "mandatory": [], "timing_block": [], "profile": []}
    count = 0
    for t in tasks:
        p0 = t.original()
        t0 = time.perf_counter()
        v.profile(t, p0)
        stages["profile"].append(time.perf_counter() - t0)
        for name in t.functions:
            fn = p0.function_node(name)
            for plan in R.static_plans(fn, t.contract.function(name), limit=2):
                for code in R.propose(p0.function_source(name), plan.transform_id, plan)[:1]:
                    try:
                        cand = apply_patch(p0, p0.hash, name, code, t.contract.permitted_imports)
                    except PatchError:
                        continue
                    t0 = time.perf_counter()
                    sm = v.smoke_check(t, cand, name)
                    stages["smoke"].append(time.perf_counter() - t0)
                    t0 = time.perf_counter()
                    mc = v.mandatory_checks(t, priv[t.task_id], cand)
                    stages["mandatory"].append(time.perf_counter() - t0)
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
    rep = {k: {"n": len(x), "mean_s": statistics.mean(x) if x else None,
               "p95_s": sorted(x)[int(0.95 * (len(x) - 1))] if x else None} for k, x in stages.items()}
    m = rep["mandatory"]["mean_s"] or 0.0
    s = rep["smoke"]["mean_s"] or 0.0
    b = rep["timing_block"]["mean_s"] or 0.0
    rep["forecast_note"] = "T_serial ~ N_calls*t_model + N_checks*(t_smoke+t_mandatory) + N_blocks*t_block"
    rep["per_candidate_check_s"] = s + m
    rep["per_block_s"] = b
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
    c = sub.add_parser("checks")
    c.add_argument("--config", default=None)
    c.add_argument("--suite", default="dev_files")
    c.add_argument("--n", type=int, default=100)
    c.add_argument("overrides", nargs="*")
    a = p.parse_args(argv)
    if a.cmd == "engine":
        from tinyperf_chain.serve.engine import resolve_device

        bench_engine(a.checkpoint, a.batch, a.prompt_len, a.max_new, resolve_device(a.device), a.compile)
    else:
        bench_checks(a.config, a.suite, a.n, a.overrides)


if __name__ == "__main__":
    main()
