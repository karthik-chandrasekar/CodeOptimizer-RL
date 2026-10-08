"""Compose multi-function *file* tasks from verified single-function tasks.

    python -m tinyperf.data.build_files --config configs/data.yaml \\
        --src artifacts/data_r2/test_seeds.jsonl --out artifacts/data_r2/files_test.jsonl --n 300 --split files_test

Each file holds K functions taken from K different seed functions of ``--src`` (so a test file only
contains held-out functions, an RL file only RL functions).  D of them are in their slow form (the
task's ``C0``), the rest are already fast; the agent has to find and fix the slow ones.  Correctness
and timing cover every function through a hidden dispatcher (:mod:`tinyperf.env.files`), and each
component's perf workload is shrunk until the whole slow file runs in <= ``--c0_cap_ms``, which keeps
RL rollouts affordable.  ``reference_speedup`` is measured by swapping in all fast versions.
"""
from __future__ import annotations

import argparse
import ast
import random
from collections import defaultdict
from typing import Dict, List, Optional

from tinyperf.common.config import load_config
from tinyperf.common.utils import get_logger
from tinyperf.env.executor import Executor
from tinyperf.env.files import ENTRY, make_driver, profile_shares, with_driver
from tinyperf.env.task import Task, load_tasks, save_tasks

log = get_logger("build_files")


def _unparse(parts: List[str]) -> str:
    return ast.unparse(ast.parse("\n\n\n".join(parts)))


def make_specs(tasks: List[Task], n: int, k_range, d_range, max_chars: int, rng: random.Random) -> List[Dict]:
    by_seed: Dict[str, List[Task]] = defaultdict(list)
    for t in tasks:
        if t.fast_source:
            by_seed[t.seed_name].append(t)
    seeds = sorted(by_seed)
    if len(seeds) < k_range[0]:
        raise SystemExit(f"only {len(seeds)} distinct functions in the source split; need >= {k_range[0]}")
    specs, attempts = [], 0
    while len(specs) < n and attempts < 50 * n:
        attempts += 1
        k = rng.randint(k_range[0], min(k_range[1], len(seeds)))
        d = rng.randint(d_range[0], min(d_range[1], k))
        comps = [rng.choice(by_seed[s]) for s in rng.sample(seeds, k)]
        names = [c.func_name for c in comps]
        if len(set(names)) < k:
            continue
        slow_idx = set(rng.sample(range(k), d))
        try:
            slow = _unparse([c.source if i in slow_idx else c.fast_source for i, c in enumerate(comps)])
            fast = _unparse([c.fast_source for c in comps])
        except SyntaxError:
            continue
        if len(slow) > max_chars:
            continue
        specs.append({"comps": comps, "names": names, "slow_idx": sorted(slow_idx), "slow": slow, "fast": fast})
    return specs


def realise(ex: Executor, spec: Dict, idx: int, split: str, c0_cap_ns: float, min_speedup: float) -> Optional[Task]:
    comps, names = spec["comps"], spec["names"]
    driver = make_driver(names)
    correct = [(k, *x) for k, c in enumerate(comps) for x in c.correct_inputs]
    perf_parts = [list(c.perf_inputs) for c in comps]
    b: Dict = {}
    for _ in range(8):
        perf = [(k, *x) for k, part in enumerate(perf_parts) for x in part]
        b = ex.baseline(with_driver(spec["slow"], driver), ENTRY, correct, perf, group_by_first_arg=True)
        if b.get("status") == "ok" and float(b.get("candidate_median_ns") or 0) <= c0_cap_ns:
            break
        if b.get("status") not in ("ok", "timeout", "resource", "worker_error") or all(len(p) <= 1 for p in perf_parts):
            return None
        perf_parts = [p[: max(1, len(p) // 2)] for p in perf_parts]
    else:
        return None
    if b.get("status") != "ok":
        return None
    slow_comps = [comps[i] for i in spec["slow_idx"]]
    task = Task(
        task_id=f"{split}_{idx:05d}", func_name=ENTRY, source=spec["slow"], correct_inputs=correct, perf_inputs=perf,
        expected=b["expected"], baseline_ns=float(b["candidate_median_ns"]),
        seed_name=sorted(c.seed_name for c in slow_comps)[0],          # cluster key for function-level CIs
        families=sorted({f for c in slow_comps for f in c.families}),
        chain=[spec["fast"], spec["slow"]], fast_source=spec["fast"], split=split,
        meta={"kind": "file", "driver": driver, "functions": names, "slow_functions": [names[i] for i in spec["slow_idx"]],
              "component_seeds": [c.seed_name for c in comps], "component_tasks": [c.task_id for c in comps],
              "k": len(comps), "d": len(slow_comps), "profile0": profile_shares(b.get("candidate_group_ns"), names)},
    )
    r = ex.evaluate(spec["fast"], task, baseline_code=spec["slow"])
    if r.get("status") != "ok" or not r.get("ratio"):
        return None
    task.reference_speedup = 1.0 / float(r["ratio"])
    return task if task.reference_speedup >= min_speedup else None


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--k", type=int, nargs=2, default=[2, 4], help="functions per file (min max)")
    p.add_argument("--d", type=int, nargs=2, default=[1, 3], help="slow functions per file (min max)")
    p.add_argument("--max_chars", type=int, default=3000)
    p.add_argument("--c0_cap_ms", type=float, default=100.0)
    p.add_argument("--min_speedup", type=float, default=1.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("overrides", nargs="*")
    a = p.parse_args(argv)
    cfg = load_config(a.config, a.overrides)
    rng = random.Random(a.seed)
    src = load_tasks(a.src)
    ex = Executor(cfg.env)
    out: List[Task] = []
    specs = make_specs(src, int(a.n * 1.4) + 8, a.k, a.d, a.max_chars, rng)
    log.info(f"{len(specs)} candidate files from {len({t.seed_name for t in src})} functions in {a.src}")
    results = ex.map(lambda iv: realise(ex, iv[1], iv[0], a.split, a.c0_cap_ms * 1e6, a.min_speedup), list(enumerate(specs)))
    for t in results:
        if t is not None and len(out) < a.n:
            out.append(t)
    ex.close()
    for i, t in enumerate(out):
        t.task_id = f"{a.split}_{i:05d}"
    save_tasks(a.out, out)
    ks = defaultdict(int)
    for t in out:
        ks[(t.meta["k"], t.meta["d"])] += 1
    log.info(f"wrote {len(out)} file tasks to {a.out} ({len(results) - sum(r is not None for r in results)} rejected); "
             f"(functions, slow) counts: {dict(sorted(ks.items()))}")


if __name__ == "__main__":
    main()
