"""Build the synthetic optimization dataset.

    python -m tinyperf.data.build --config configs/data.yaml [data.n_train=8000 ...]

Pipeline per seed
-----------------
1. Normalize the efficient source ``C_fast``; generate hidden inputs.
2. Record ``C_fast``'s behaviour on X_correct (sandbox) and check the workload
   is large enough to time (adaptive ``n_perf``).
3. BFS over degradation operators (AST + hand-written algorithmic variants).
   Every node is verified in the sandbox: it must be behaviour-equivalent to
   ``C_fast`` and slower than its parent by at least ``min_step_slowdown``.
   Only verified nodes are expanded further.
4. Nodes with cumulative slowdown >= ``min_slowdown`` become tasks.  Each task
   stores its chain [C_fast, ..., C_slow] (for programmatic SFT trajectories),
   the family sequence, and a paired reference speedup.
5. Chains are alpha-renamed (consistently across the chain) for variety and
   deduplicated on the canonical (un-renamed) chain, which is also the unit of
   train/test assignment so renamed copies never straddle splits.

Splits
------
- train / val / test_iid       : familiar families, chains of depth in ``train_depths``
- test_compositional           : deeper chains (``compositional_depths``)
- test_heldout                 : any chain containing a family in ``heldout_families``
- test_seeds                   : every chain of the ``heldout_seed_frac`` seed functions (function-level generalisation)
- test_natural (optional)      : human-written slow functions from ``data.natural_dir``
"""
from __future__ import annotations

import hashlib
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tinyperf.common.config import Config, parse_cli
from tinyperf.common.utils import get_logger, write_jsonl
from tinyperf.data.astutil import function_name, normalize
from tinyperf.data.degradations import ALL_OPERATORS, Algorithmic, applicable
from tinyperf.data.rename import apply_rename, make_rename_map
from tinyperf.common.utils import read_jsonl
from tinyperf.data.seeds import SEEDS, Seed, make_inputs
from tinyperf.env.executor import Executor
from tinyperf.env.task import Task, save_tasks

log = get_logger("build")

MIN_STEP_SLOWDOWN = 1.10     # each degradation step must cost at least this (margin above timing noise)
MIN_BASELINE_NS = 2_000_000  # 2 ms: workloads below this are scaled up
MAX_PERF_INPUTS = 24
MAX_C0_NS = 300_000_000      # 300 ms: C0 perf workloads are shrunk until the slow program fits (keeps env steps affordable)
TIMEOUT_RATIO = 50.0         # nominal slowdown recorded for degradations too slow to benchmark at the seed's workload


@dataclass
class Node:
    src: str
    families: List[str]
    chain: List[str]                 # sources from C_fast to this node (inclusive)
    ratio_to_fast: float             # T(node) / T(fast), paired
    depth: int


@dataclass
class SeedResult:
    seed: Seed
    fast: str
    correct_inputs: list
    perf_inputs: list
    expected_fast: list
    perf_scale: int = 3000
    n_perf: int = 3
    nodes: List[Node] = field(default_factory=list)


# --------------------------------------------------------------------------- #
def _fast_task(sr: SeedResult) -> Task:
    """A Task whose C0 is the fast source (used to verify candidates *against the fast behaviour*)."""
    return Task("fast", function_name(sr.fast), sr.fast, sr.correct_inputs, sr.perf_inputs, sr.expected_fast, 0.0)


def prepare_seed(ex: Executor, s: Seed, rng: random.Random, perf_scale: int) -> Optional[SeedResult]:
    fast = normalize(s.source)
    n_perf = 3
    while True:
        correct, perf = make_inputs(s, rng, perf_scale, n_perf=n_perf)
        b = ex.baseline(fast, function_name(fast), correct, perf)
        if b.get("status") != "ok":
            log.warning(f"seed {s.name}: baseline failed: {b.get('status')} {b.get('message','')[:120]}")
            return None
        if b.get("candidate_median_ns", 0) >= MIN_BASELINE_NS or n_perf >= MAX_PERF_INPUTS:
            break
        n_perf = min(MAX_PERF_INPUTS, n_perf * 2)
    return SeedResult(s, fast, correct, perf, b["expected"], perf_scale=perf_scale, n_perf=n_perf)


def expand_seed(ex: Executor, sr: SeedResult, rng: random.Random, max_depth: int, max_nodes: int, algorithmic_first: bool = True) -> List[Node]:
    """BFS over verified degradations. Returns all verified nodes (depth >= 1)."""
    fast_task = _fast_task(sr)
    alg = Algorithmic({sr.fast: sr.seed.slow_variants})
    alg.family = sr.seed.slow_family
    operators = [alg] if sr.seed.only_slow_variants else ALL_OPERATORS + [alg]
    root = Node(sr.fast, [], [sr.fast], 1.0, 0)
    frontier = [root]
    seen = {sr.fast}
    verified: List[Node] = []
    for depth in range(1, max_depth + 1):
        candidates: List[Tuple[Node, str, str]] = []
        for parent in frontier:
            apps = applicable(parent.src, operators)
            for fam, variants in apps.items():
                for v in variants:
                    if v not in seen and len(v) <= 4000:
                        seen.add(v)
                        candidates.append((parent, fam, v))
        if not candidates:
            break
        rng.shuffle(candidates)
        candidates = candidates[: max(1, max_nodes // max_depth) * 2]

        def _verify(item):
            parent, fam, v = item
            r = ex.evaluate(v, fast_task, baseline_code=parent.src, timing_seed=rng.randrange(1 << 30))
            return item, r

        results = ex.map(_verify, candidates)
        new_frontier: List[Node] = []
        for (parent, fam, v), r in results:
            too_slow = (r.get("status") == "timeout" and "benchmarking" in str(r.get("message", ""))
                        and r.get("n_pass") is not None and r.get("n_pass") == r.get("n_total") and parent.depth == 0)
            if too_slow:
                # correct on every hidden input but too slow to benchmark against the fast source: a large, real
                # regression.  Keep it (C0 workloads are shrunk per task in make_task) but do not expand it further.
                node = Node(v, parent.families + [fam], parent.chain + [v], TIMEOUT_RATIO, depth)
                verified.append(node)
                if len(verified) >= max_nodes:
                    return verified
                continue
            if r.get("status") != "ok" or r.get("ratio") is None:
                continue
            step_ratio = float(r["ratio"])
            if step_ratio < MIN_STEP_SLOWDOWN:
                continue
            node = Node(v, parent.families + [fam], parent.chain + [v], parent.ratio_to_fast * step_ratio, depth)
            verified.append(node)
            new_frontier.append(node)
            if len(verified) >= max_nodes:
                return verified
        frontier = new_frontier
    return verified


def canonical_key(seed_name: str, chain: List[str]) -> str:
    h = hashlib.sha1(("\n---\n".join([seed_name] + chain)).encode()).hexdigest()
    return h[:16]


def make_task(ex: Executor, sr: SeedResult, node: Node, rng: random.Random, task_id: str, rename: bool, split: str, min_slowdown: float) -> Optional[Task]:
    chain = list(node.chain)
    fam = list(node.families)
    fname = function_name(sr.fast)
    if rename:
        mapping = make_rename_map(chain, rng, fname)
        chain = [apply_rename(c, mapping) for c in chain]
        fname = mapping.get(fname, fname)
    slow, fast = chain[-1], chain[0]
    # fresh hidden inputs per task (never share inputs between tasks of the same seed); the perf workload is
    # shrunk (fewer inputs, then smaller inputs) until C0 itself runs in <= MAX_C0_NS
    n_perf, scale = sr.n_perf, sr.perf_scale
    if node.ratio_to_fast >= TIMEOUT_RATIO:  # known to be far too slow at the seed's workload: start small
        n_perf, scale = 1, max(100, sr.perf_scale // 4)
    b: Dict = {}
    for _ in range(10):
        correct, perf = make_inputs(sr.seed, rng, scale, n_perf=n_perf)
        b = ex.baseline(slow, fname, correct, perf)
        st = b.get("status")
        if st == "ok" and float(b.get("candidate_median_ns") or 0) <= MAX_C0_NS:
            break
        if st not in ("ok", "timeout", "resource", "worker_error"):  # C0 was verified correct: these mean "too slow"
            return None
        if n_perf > 1:
            n_perf = max(1, n_perf // 2)
        elif scale > 100:
            scale = max(100, scale // 2)
        else:
            return None
    else:
        return None
    if b.get("status") != "ok":
        return None
    task = Task(task_id, fname, slow, correct, perf, b["expected"], float(b["candidate_median_ns"]),
                seed_name=sr.seed.group or sr.seed.name, families=fam, chain=chain, fast_source=fast, split=split,
                meta={"depth": node.depth, "canonical": canonical_key(sr.seed.name, node.chain), "renamed": rename,
                      "perf_scale": scale, "n_perf": n_perf})
    r = ex.evaluate(fast, task, baseline_code=slow, timing_seed=rng.randrange(1 << 30))
    if r.get("status") != "ok" or r.get("ratio") is None:
        return None  # renamed chain must still be equivalent (paranoia: renaming is verified too)
    task.reference_speedup = 1.0 / max(float(r["ratio"]), 1e-9)
    if task.reference_speedup < min_slowdown:
        return None
    return task


# --------------------------------------------------------------------------- #
def load_natural(ex: Executor, natural_dir: str, rng: random.Random, perf_scale: int) -> List[Task]:
    """Human-written slow functions: each file `name.py` must define the function and a
    module-level `def gen_inputs(rng, n)` returning an argument tuple (see README)."""
    tasks: List[Task] = []
    for p in sorted(Path(natural_dir).glob("*.py")):
        ns: Dict = {}
        try:
            exec(compile(p.read_text(), str(p), "exec"), ns)  # noqa: S102 - trusted local files
        except Exception as e:  # noqa: BLE001
            log.warning(f"natural {p.name}: cannot import: {e}")
            continue
        gen = ns.get("gen_inputs")
        if gen is None:
            log.warning(f"natural {p.name}: no gen_inputs")
            continue
        src_lines = [ln for ln in p.read_text().split("\n")]
        import ast as _ast

        tree = _ast.parse(p.read_text())
        funcs = [n for n in tree.body if isinstance(n, _ast.FunctionDef) and n.name != "gen_inputs"]
        if not funcs:
            continue
        fn = funcs[0]
        src = normalize("\n".join(src_lines[fn.lineno - 1: fn.end_lineno]))
        s = Seed(p.stem, src, gen)
        sr = prepare_seed(ex, s, rng, perf_scale)
        if sr is None:
            continue
        b = ex.baseline(sr.fast, fn.name, sr.correct_inputs, sr.perf_inputs)
        tasks.append(Task(f"natural_{p.stem}", fn.name, sr.fast, sr.correct_inputs, sr.perf_inputs, b["expected"],
                          float(b["candidate_median_ns"]), seed_name=p.stem, families=["natural"], split="test_natural"))
    return tasks


# --------------------------------------------------------------------------- #
def load_mined_seeds(path: str, max_mined: int = 0) -> List[Seed]:
    """Seeds from tinyperf.data.mine: the mined function itself (degraded by the operators) plus, for every
    proposer rewrite, a *natural* seed whose only slow version is the original human-written function."""
    from tinyperf.data.argspec import make_gen

    rows = list(read_jsonl(path))
    if max_mined:
        rows = rows[:max_mined]
    out: List[Seed] = []
    for r in rows:
        gen = make_gen(r["arg_specs"])
        out.append(Seed(r["seed_name"], r["source"], gen, [], [], r.get("min_n", 0), group=r["seed_name"]))
        for j, rew in enumerate(r.get("natural", [])):
            out.append(Seed(f"{r['seed_name']}_nat{j}", rew, gen, [r["source"]], [], r.get("min_n", 0),
                            slow_family="natural", only_slow_variants=True, group=r["seed_name"]))
    return out


def main(argv: Optional[List[str]] = None) -> None:
    cfg: Config = parse_cli(argv)
    d = cfg.data
    rng = random.Random(d.seed)
    ex = Executor(cfg.env)
    out = Path(d.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_depths = list(d.train_depths)
    comp_depths = [x for x in range(1, d.max_degradations + 3) if x not in train_depths]
    max_depth = max(train_depths + comp_depths) if comp_depths else max(train_depths)
    heldout = set(d.heldout_families)

    # 1. expand every seed into verified degradation chains
    seeds = (list(SEEDS) if d.builtin_seeds else []) + (load_mined_seeds(d.mined_seeds, d.max_mined) if d.mined_seeds else [])
    log.info(f"{len(seeds)} seeds ({'with' if d.builtin_seeds else 'without'} the hand-written library)")

    def expand_one(s: Seed, srng: random.Random) -> Optional[SeedResult]:
        try:
            sr = prepare_seed(ex, s, srng, d.perf_scale)
            if sr is None:
                return None
            nodes = expand_seed(ex, sr, srng, max_depth=max_depth, max_nodes=d.max_nodes_per_seed)
        except Exception as e:  # noqa: BLE001 - one bad (mined) seed must not stop the build
            log.warning(f"seed {s.name}: {type(e).__name__}: {e}")
            return None
        sr.nodes = [n for n in nodes if n.ratio_to_fast >= d.min_slowdown]
        log.info(f"seed {s.name:28s} verified chains: {len(sr.nodes):3d} (of {len(nodes)} nodes)  "
                 f"fams={sorted(set(f for n in sr.nodes for f in n.families))}")
        return sr

    seed_results: List[SeedResult] = []
    if d.seed_workers <= 1:
        for s in seeds:
            sr = expand_one(s, rng)
            if sr is not None:
                seed_results.append(sr)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=d.seed_workers) as pool:
            for sr in pool.map(lambda s: expand_one(s, random.Random(f"{d.seed}|{s.name}")), seeds):
                if sr is not None:
                    seed_results.append(sr)

    # 2. assign canonical chains to splits
    #    test_seeds: every chain of a held-out *seed function* (chosen by hashing the seed name), so the
    #    function itself is never seen during training in any renaming; the other splits are chain-level.
    pools: Dict[str, List[Tuple[SeedResult, Node]]] = {k: [] for k in ("train", "val", "test_iid", "test_compositional", "test_heldout", "test_seeds")}
    fam_chains: Dict[str, int] = {}
    for sr in seed_results:
        seed_u = int(hashlib.sha1(f"seed|{sr.seed.group or sr.seed.name}|{d.seed}".encode()).hexdigest()[:16], 16) / 16**16
        for n in sr.nodes:
            for f in set(n.families):
                fam_chains[f] = fam_chains.get(f, 0) + 1
            key = canonical_key(sr.seed.name, n.chain)
            u = int(key, 16) / 16**16
            if seed_u < d.heldout_seed_frac:
                pools["test_seeds"].append((sr, n))
            elif set(n.families) & heldout:
                pools["test_heldout"].append((sr, n))
            elif n.depth in comp_depths:
                pools["test_compositional"].append((sr, n))
            elif u < 0.80:
                pools["train"].append((sr, n))
            elif u < 0.90:
                pools["val"].append((sr, n))
            else:
                pools["test_iid"].append((sr, n))
    for k, v in pools.items():
        log.info(f"pool {k:20s}: {len(v)} canonical chains from {len({sr.seed.name for sr, _ in v})} seeds")
    log.info("chains per family (all pools): " + ", ".join(f"{f}={c}" for f, c in sorted(fam_chains.items(), key=lambda kv: -kv[1])))

    # 3. materialize tasks (renamed copies + fresh inputs) up to the requested sizes
    targets = {"train": d.n_train, "val": d.n_val, "test_iid": d.n_test_iid,
               "test_compositional": d.n_test_compositional, "test_heldout": d.n_test_heldout, "test_seeds": d.n_test_seeds}
    counters: Dict[str, int] = {}
    for split, target in targets.items():
        pool = pools[split]
        tasks: List[Task] = []
        if not pool:
            log.warning(f"split {split}: empty pool")
            save_tasks(out / f"{split}.jsonl", [])
            continue
        attempts = 0
        jobs: List[Tuple[SeedResult, Node, bool, str]] = []
        while len(jobs) < target * 1.15:
            sr, n = pool[attempts % len(pool)] if attempts < len(pool) else rng.choice(pool)
            rename = attempts >= len(pool) or rng.random() < d.rename_prob
            jobs.append((sr, n, rename, f"{split}_{attempts:06d}"))
            attempts += 1

        def _make(job):
            sr, n, rename, tid = job
            local_rng = random.Random(int(hashlib.sha1(f"{tid}|{d.seed}".encode()).hexdigest()[:8], 16))
            return make_task(ex, sr, n, local_rng, tid, rename, split, d.min_slowdown)

        for t in ex.map(_make, jobs):
            if t is not None and len(t.source) <= d.max_source_chars:
                tasks.append(t)
            if len(tasks) >= target:
                break
        save_tasks(out / f"{split}.jsonl", tasks)
        counters[split] = len(tasks)
        log.info(f"split {split:20s}: wrote {len(tasks)} tasks")

    if d.natural_dir:
        nat = load_natural(ex, d.natural_dir, rng, d.perf_scale)
        save_tasks(out / "test_natural.jsonl", nat)
        log.info(f"split test_natural: wrote {len(nat)} tasks")

    ex.close()
    write_jsonl(out / "families.jsonl", [{"family": f, "chains": c} for f, c in sorted(fam_chains.items(), key=lambda kv: -kv[1])])
    log.info(f"done: {counters}")


if __name__ == "__main__":
    main()
