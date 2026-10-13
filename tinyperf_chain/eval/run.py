"""Run whole-policy comparisons: paired sessions of systems on the same tasks, seeds and budgets.

    python -m tinyperf_chain.eval.run --config configs/chain/eval_dev.yaml --suite development \\
        --bundle bundles/B3.json bundles/B4.json [--legacy-checkpoint ckpt.pt] --output artifacts/chain/runs/dev
    python -m tinyperf_chain.eval.run --config configs/chain/eval_confirm.yaml --suite confirmation \\
        --bundle bundles/B1.lock.json bundles/B4.lock.json --output artifacts/chain/confirm
    python -m tinyperf_chain.eval.run --config ... --suite development --bundle b.json --calibrate-wall

* ``--suite development`` reads ``dev_files`` (and ``dev_functions`` with ``--kinds files functions``);
  ``--suite confirmation`` reads ``confirm_files`` and requires locked bundles, an unchanged environment and a
  calibrated ``budget.search_wall_seconds``.
* Within a (task, episode seed, patch budget) unit every system runs back to back in a shuffled order, so the
  comparison is paired and shares machine conditions.  Each system is evaluated as one whole policy (controller +
  roles); bundles are frozen for the whole session.
* ``--patch-slots 1 2 4 6`` produces search-matched budget curves.
* ``--calibrate-wall`` runs without a wall cap and writes ``calibration.json`` with per-system p50/p95 search times
  and a suggested cap; it changes no evaluation results.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tinyperf_chain.adapters.verifier import Verifier
from tinyperf_chain.bundle import build_router, environment_manifest, load_bundle_json, verify_lock
from tinyperf_chain.config import ChainConfig, config_hash, load_chain_config
from tinyperf_chain.controller import ChainController, EpisodeSpec
from tinyperf_chain.hashing import sha256_text
from tinyperf_chain.ledger import AppendLedger
from tinyperf_chain.tasks import PrivateSuite, PublicTask, TaskStore

SUITES = {"development": ("dev", "dev"), "dev": ("dev", "dev"), "confirmation": ("confirm", "confirm"),
          "confirm": ("confirm", "confirm")}


class _System:
    def __init__(self, system_id: str, runner, group: Optional[str] = None, training_seed: Optional[int] = None):
        self.system_id = system_id
        self.runner = runner      # callable(task, private, spec) -> record
        self.group = group or system_id          # replicate group (15.5: bundles trained with different seeds)
        self.training_seed = training_seed


def make_systems(cfg: ChainConfig, verifier: Verifier, bundles: Sequence[str], legacy: Sequence[str], legacy_ids: Sequence[str],
                 device: str, use_scheduler: bool, locked: bool, allow_env_mismatch: bool) -> Tuple[List[_System], Dict[str, Any], Dict[str, Any]]:
    """Returns (systems, metadata, resources).  ``resources`` holds the engines and schedulers so the session can
    report their measured stats (prefill/decode/queue-wait/peak cache memory)."""
    systems, meta, resources = [], {}, {"engines": {}, "schedulers": {}}
    for path in bundles:
        b = load_bundle_json(path)
        if locked:
            verify_lock(b, cfg, allow_env_mismatch)
        router, res = build_router(b, cfg, device=device, use_scheduler=use_scheduler)
        ctrl = ChainController(cfg, verifier, router)
        systems.append(_System(router.system_id, ctrl.run_episode, b.get("replicate_of"), b.get("training_seed")))
        meta[router.system_id] = {"bundle": path, "bundle_sha256": router.bundle_hash, "roles": router.describe(),
                                  "replicate_of": b.get("replicate_of"), "training_seed": b.get("training_seed")}
        resources["engines"][router.system_id] = res["engines"]
        if res.get("scheduler") is not None:
            resources["schedulers"][router.system_id] = res["scheduler"]
    if legacy:
        from tinyperf.model.tokenizer import CodeTokenizer
        from tinyperf.model.transformer import TinyPerfLM
        from tinyperf_chain.hashing import sha256_file
        from tinyperf_chain.legacy import LegacyDirectEditSystem
        from tinyperf_chain.serve.engine import RoleEngine

        tok = CodeTokenizer(cfg.model.tokenizer)
        ids = list(legacy_ids) + [f"B0_{i}" for i in range(len(legacy_ids), len(legacy))]
        for ck, sid in zip(legacy, ids):
            eng = RoleEngine(TinyPerfLM.load(ck), device=device, checkpoint_sha256=sha256_file(ck))
            lsys = LegacyDirectEditSystem(cfg, verifier, eng, tok, system_id=sid, bundle_hash=sha256_file(ck))
            systems.append(_System(sid, lsys.run_episode))
            meta[sid] = {"legacy_checkpoint": ck, "sha256": eng.checkpoint_sha256, **lsys.describe()}
            resources["engines"][sid] = {ck: eng}
    ids = [s.system_id for s in systems]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"duplicate system ids {ids}")
    return systems, meta, resources


def _peak_memory() -> Dict[str, Any]:
    import resource

    out: Dict[str, Any] = {}
    try:
        out["host_peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        out["children_peak_rss_mb"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        if torch.cuda.is_available():
            out["device_peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 1e6
            out["device_peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 1e6
    except Exception:  # noqa: BLE001
        pass
    return out


def _resource_stats(resources: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"engines": {}, "schedulers": {}}
    seen = set()
    for sid, engs in resources.get("engines", {}).items():
        for ck, eng in engs.items():
            if id(eng) in seen:
                continue
            seen.add(id(eng))
            out["engines"][f"{sid}:{os.path.basename(ck)}"] = dict(eng.stats)
    for sid, sch in resources.get("schedulers", {}).items():
        out["schedulers"][sid] = {k: dict(v) for k, v in sch.stats.items()}
    return out


def run_session(cfg: ChainConfig, tasks: Sequence[Tuple[PublicTask, PrivateSuite]], systems: Sequence[_System], out_dir: str,
                episodes_per_file: int, patch_slots: Sequence[Optional[int]], mode: str, seed0: int = 0,
                workers: int = 1, events: bool = True) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    ledger = AppendLedger(os.path.join(out_dir, "episodes.jsonl"))
    prior = list(ledger.read())
    # resume (12.5, T25): completed episodes are skipped only when they were produced by the *same* bundle; a record
    # from a different bundle hash means the session is a different experiment and must start separately
    by_sys = {s.system_id: s for s in systems}
    for r in prior:
        sid = r.get("system_id")
        if sid in by_sys and hasattr(by_sys[sid].runner, "__self__"):
            want = getattr(getattr(by_sys[sid].runner.__self__, "router", None), "bundle_hash", None) or \
                getattr(by_sys[sid].runner.__self__, "bundle_hash", None)
            if want is not None and r.get("bundle_sha256") != want:
                raise SystemExit(f"{out_dir}: existing records for {sid} were produced by bundle {r.get('bundle_sha256')!r}, "
                                 f"not {want!r}; start a separately identified session (different --output)")
    finished = {r.get("episode_id") for r in prior}
    units = [(t, p, e, slots) for (t, p) in tasks for e in range(episodes_per_file) for slots in patch_slots]
    lock = threading.Lock()
    done = {"n": 0}
    t0 = time.monotonic()

    def work(u) -> None:
        task, private, e, slots = u
        seed = seed0 * 1000 + e
        order = list(systems)
        random.Random(f"order|{task.task_id}|{seed}|{slots}").shuffle(order)
        for s in order:
            eid = f"{s.system_id}|{task.task_id}|s{seed}|p{slots}"
            if eid in finished:
                continue
            ev = os.path.join(out_dir, "events", sha256_text(eid)[:20] + ".jsonl") if events else None
            rec = s.runner(task, private, EpisodeSpec(eid, seed=seed, mode=mode, patch_slots=slots, events_path=ev))
            rec.pop("events", None)
            rec["episode_index"] = e
            rec["system_group"] = s.group
            rec["training_seed"] = s.training_seed
            ledger.append(rec)
            with lock:
                done["n"] += 1

    with ThreadPoolExecutor(max(1, workers)) as ex:
        list(ex.map(work, units))
    t1 = time.monotonic()
    return {"interval_s": t1 - t0, "episodes": done["n"], "units": len(units)}


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--suite", required=True)
    p.add_argument("--kinds", nargs="+", default=["files"])
    p.add_argument("--bundle", nargs="*", default=[])
    p.add_argument("--legacy-checkpoint", nargs="*", default=[], help="B0 / direct-edit arms (one system per checkpoint)")
    p.add_argument("--legacy-system-id", nargs="*", default=["B0"])
    p.add_argument("--output", default=None, help="default: <paths.runs>/<suite>")
    p.add_argument("--episodes-per-file", type=int, default=4)
    p.add_argument("--patch-slots", nargs="*", default=[])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--calibrate-wall", action="store_true")
    p.add_argument("--allow-env-mismatch", action="store_true")
    p.add_argument("--no-events", action="store_true")
    p.add_argument("overrides", nargs="*")
    a = p.parse_args(argv)
    from tinyperf_chain.config import pop_overrides
    pop_overrides(a, ['bundle', 'kinds', 'patch_slots', 'legacy_checkpoint', 'legacy_system_id'])
    a.patch_slots = [int(x) for x in a.patch_slots]
    cfg = load_chain_config(a.config, a.overrides)
    if not a.bundle and not a.legacy_checkpoint:
        raise SystemExit("nothing to evaluate: pass --bundle bundle.json [...] and/or --legacy-checkpoint ckpt.pt")
    split, mode = SUITES.get(a.suite, (a.suite, "dev"))
    confirm = split == "confirm"
    if a.calibrate_wall:
        if confirm:
            raise SystemExit("calibrate on development data, never on the confirmation suite")
        cfg.budget.search_wall_seconds = None
    cfg.validate("eval:confirmation" if confirm else "eval:development")
    if confirm and any(not b.endswith(".lock.json") for b in a.bundle):
        print("note: confirmation bundles are verified as locked regardless of file name")
    store = TaskStore(cfg.paths.tasks, cfg.paths.private)
    tasks: List[Tuple[PublicTask, PrivateSuite]] = []
    for kind in a.kinds:
        suite = f"{split}_{kind}" if a.suite in SUITES else a.suite
        if not os.path.exists(store.public_path(suite)):
            raise SystemExit(f"task suite {suite!r} not found under {cfg.paths.tasks} (build it with "
                             f"tinyperf_chain.data.split / build_tasks, or point paths.root at the artifact root)")
        pubs = store.load_public(suite, a.limit)
        priv = store.load_private(suite)
        tasks += [(t, priv[t.task_id]) for t in pubs]
    if not tasks:
        raise SystemExit("no tasks")
    from tinyperf_chain.serve.engine import resolve_device

    device = resolve_device(cfg.serving.device)
    verifier = Verifier(cfg, cache_dir=cfg.paths.cache)
    workers = max(1, cfg.serving.episode_workers)
    systems, meta, resources = make_systems(cfg, verifier, a.bundle, a.legacy_checkpoint, a.legacy_system_id, device,
                                            use_scheduler=workers > 1, locked=confirm, allow_env_mismatch=a.allow_env_mismatch)
    slots = a.patch_slots or [None]
    out = a.output or cfg.paths.sub("runs", a.suite)
    os.makedirs(out, exist_ok=True)
    session = {"suite": a.suite, "split": split, "n_tasks": len(tasks), "systems": meta, "config_sha256": config_hash(cfg),
               "environment": environment_manifest(), "episodes_per_file": a.episodes_per_file,
               "patch_slots": slots, "calibrate_wall": a.calibrate_wall,
               "search_wall_seconds": cfg.budget.search_wall_seconds, "started": time.time()}
    with open(os.path.join(out, "session_start.json"), "w") as f:
        json.dump(session, f, indent=1)
    res = run_session(cfg, tasks, systems, out, a.episodes_per_file, slots, "confirm" if confirm else "dev", a.seed,
                      workers, not a.no_events)
    session.update(res)
    session["finished"] = time.time()
    session["verifier_stats"] = verifier.stats
    session["sandbox_stats"] = dict(verifier.runner.stats)
    session["verifier_identity"] = verifier.identity
    session["peak_memory"] = _peak_memory()
    session["serving_stats"] = _resource_stats(resources)
    session["n_systems"] = len(systems)
    for sch in resources.get("schedulers", {}).values():
        sch.close()
    with open(os.path.join(out, "session.json"), "w") as f:
        json.dump(session, f, indent=1)
    if a.calibrate_wall:
        from tinyperf_chain.eval.aggregate import _pct, load_records

        recs = load_records([os.path.join(out, "episodes.jsonl")])
        cal = {}
        for sid in sorted({r["system_id"] for r in recs}):
            w = [r["search_wall_ns"] / 1e9 for r in recs if r["system_id"] == sid and not r.get("failed_run")]
            cal[sid] = {"p50": _pct(w, 0.5), "p95": _pct(w, 0.95), "max": max(w) if w else None}
        allp95 = [v["p95"] for v in cal.values() if v["p95"] == v["p95"]]
        cal["suggested_search_wall_seconds"] = round(1.25 * max(allp95), 2) if allp95 else None
        with open(os.path.join(out, "calibration.json"), "w") as f:
            json.dump(cal, f, indent=1)
        print(json.dumps(cal, indent=1))
    print(json.dumps({k: session[k] for k in ("n_tasks", "episodes", "interval_s")}))


if __name__ == "__main__":
    main()
