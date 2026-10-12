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
    def __init__(self, system_id: str, runner):
        self.system_id = system_id
        self.runner = runner      # callable(task, private, spec) -> record


def make_systems(cfg: ChainConfig, verifier: Verifier, bundles: Sequence[str], legacy: Optional[str], legacy_id: str,
                 device: str, use_scheduler: bool, locked: bool, allow_env_mismatch: bool) -> Tuple[List[_System], Dict[str, Any]]:
    systems, meta = [], {}
    for path in bundles:
        b = load_bundle_json(path)
        if locked:
            verify_lock(b, cfg, allow_env_mismatch)
        router, res = build_router(b, cfg, device=device, use_scheduler=use_scheduler)
        ctrl = ChainController(cfg, verifier, router)
        systems.append(_System(router.system_id, ctrl.run_episode))
        meta[router.system_id] = {"bundle": path, "bundle_sha256": router.bundle_hash, "roles": router.describe()}
    if legacy:
        from tinyperf.model.tokenizer import CodeTokenizer
        from tinyperf.model.transformer import TinyPerfLM
        from tinyperf_chain.hashing import sha256_file
        from tinyperf_chain.legacy import LegacyDirectEditSystem
        from tinyperf_chain.serve.engine import RoleEngine

        tok = CodeTokenizer(cfg.model.tokenizer)
        eng = RoleEngine(TinyPerfLM.load(legacy), device=device, checkpoint_sha256=sha256_file(legacy))
        lsys = LegacyDirectEditSystem(cfg, verifier, eng, tok, system_id=legacy_id, bundle_hash=sha256_file(legacy))
        systems.append(_System(legacy_id, lsys.run_episode))
        meta[legacy_id] = {"legacy_checkpoint": legacy, "sha256": eng.checkpoint_sha256}
    ids = [s.system_id for s in systems]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"duplicate system ids {ids}")
    return systems, meta


def run_session(cfg: ChainConfig, tasks: Sequence[Tuple[PublicTask, PrivateSuite]], systems: Sequence[_System], out_dir: str,
                episodes_per_file: int, patch_slots: Sequence[Optional[int]], mode: str, seed0: int = 0,
                workers: int = 1, events: bool = True) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    ledger = AppendLedger(os.path.join(out_dir, "episodes.jsonl"))
    finished = {r.get("episode_id") for r in ledger.read()}    # resume: an interrupted session skips completed episodes
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
    p.add_argument("--legacy-checkpoint", default=None)
    p.add_argument("--legacy-system-id", default="B0")
    p.add_argument("--output", required=True)
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
    pop_overrides(a, ['bundle', 'kinds', 'patch_slots'])
    a.patch_slots = [int(x) for x in a.patch_slots]
    cfg = load_chain_config(a.config, a.overrides)
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
        pubs = store.load_public(suite, a.limit)
        priv = store.load_private(suite)
        tasks += [(t, priv[t.task_id]) for t in pubs]
    if not tasks:
        raise SystemExit("no tasks")
    from tinyperf_chain.serve.engine import resolve_device

    device = resolve_device(cfg.serving.device)
    verifier = Verifier(cfg, cache_dir=cfg.paths.cache)
    workers = max(1, cfg.serving.episode_workers)
    systems, meta = make_systems(cfg, verifier, a.bundle, a.legacy_checkpoint, a.legacy_system_id, device,
                                 use_scheduler=workers > 1, locked=confirm, allow_env_mismatch=a.allow_env_mismatch)
    slots = a.patch_slots or [None]
    out = a.output
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
