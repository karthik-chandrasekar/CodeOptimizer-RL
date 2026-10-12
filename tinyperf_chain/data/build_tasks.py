"""Construct chain tasks (single functions and multi-function files) from split-assigned source units.

    python -m tinyperf_chain.data.build_tasks --config configs/chain/pilot.yaml \\
        --manifest artifacts/chain/split_manifest.json --splits train dev confirm --kinds functions files

Order of operations (the split comes first, everything else inherits it):

1. read the cluster manifest; take units (seed functions) of one partition;
2. *unit variants*: the unit's hand-written slow variants plus depth-1 degradations of its fast reference
   (legacy operators), each labelled with its degradation family (privileged, never shown to models);
3. draw independent input suites from the unit's own generator with separate seeds: public workload (profiling),
   public examples (smoke), online correctness, online timing, audit correctness, audit timing;
4. derive the contract from the constructor's draws (``origin = generator_derived_v1``), shrink workloads until the
   original runs within ``data.c0_cap_ms`` per pass;
5. eligibility (every exclusion is counted with its reason): contract supported (floats deferred), all inputs
   contract-valid and alias-free, original deterministic under two hash seeds and representable, dependency
   closure supported, target function within ``data.max_target_tokens`` tokens, the fast reference correct on
   both correctness suites and at least ``--min-speedup`` faster;
6. files: units of one partition are grouped into disjoint groups of clusters; a file combines one unit from each of
   K clusters of the same group (D slow, K-D already fast), and its cluster id is the group id, so related files
   never straddle independence units.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from tinyperf.data import degradations
from tinyperf.data.astutil import function_name, normalize
from tinyperf.data.seeds import Seed
from tinyperf_chain.adapters.verifier import InfrastructureError, Verifier
from tinyperf_chain.config import ChainConfig, load_chain_config
from tinyperf_chain.contracts import (CONTRACT_VERSION, FunctionContract, TaskContract, derive_function_contract,
                                      validate_args)
from tinyperf_chain.data.split import all_units, load_manifest
from tinyperf_chain.hashing import sha256_text
from tinyperf_chain.program import Program, dependency_closure
from tinyperf_chain.registry import mechanism_for_family
from tinyperf_chain.tasks import PrivateSuite, PublicTask, TaskStore

SUITES = ("public_workload", "public_examples", "online_correctness", "online_timing", "audit_correctness", "audit_timing")


@dataclass
class Variant:
    unit: Seed
    cluster_id: str
    family: str
    slow: str
    fast: str
    name: str


@dataclass
class Draws:
    suites: Dict[str, List[Tuple]]
    scale: int


@dataclass
class UnitTask:
    variant: Variant
    public: PublicTask
    private: PrivateSuite
    contract: FunctionContract


@dataclass
class Eligibility:
    counts: Counter = field(default_factory=Counter)
    examples: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))

    def exclude(self, reason: str, what: str) -> None:
        key = reason.split(":")[0][:60]
        self.counts[key] += 1
        if len(self.examples[key]) < 5:
            self.examples[key].append(what)


def _seed_int(*parts: Any) -> int:
    return int(sha256_text("|".join(str(p) for p in parts))[:15], 16)


def variants_for(unit: Seed, cluster_id: str, max_variants: int, rng: random.Random) -> List[Variant]:
    try:
        fast = normalize(unit.source)
    except SyntaxError:
        return []
    out: List[Variant] = []
    for v in unit.slow_variants:
        try:
            out.append(Variant(unit, cluster_id, unit.slow_family, normalize(v), fast, function_name(fast)))
        except (SyntaxError, ValueError):
            continue
    if not unit.only_slow_variants:
        fams = degradations.applicable(fast)
        for fam in sorted(fams):
            vs = list(fams[fam])
            v = vs[rng.randrange(len(vs))]
            try:
                out.append(Variant(unit, cluster_id, fam, normalize(v), fast, function_name(fast)))
            except (SyntaxError, ValueError):
                continue
    rng.shuffle(out)
    seen, uniq = set(), []
    for v in out:
        if v.slow in seen or v.slow == v.fast:
            continue
        seen.add(v.slow)
        uniq.append(v)
    return uniq[:max_variants]


def draw_suites(unit: Seed, base_seed: int, scale: int, cfg: ChainConfig) -> Dict[str, List[Tuple]]:
    d = cfg.data
    g = unit.gen

    def r(tag: str) -> random.Random:
        return random.Random(_seed_int(base_seed, tag))

    sizes = [max(unit.min_n, k) for k in (1, 2, 3, 5, 8, 13, 21, 34, 60, 120)][: d.n_correct_random]
    edges = [tuple(e) for e in unit.edge_inputs]
    rc, ra, re, rp, rt, rat = r("online_correctness"), r("audit_correctness"), r("examples"), r("public"), r("online_timing"), r("audit_timing")
    return {
        "online_correctness": edges + [tuple(g(rc, k)) for k in sizes],
        "audit_correctness": edges + [tuple(g(ra, k + ra.randint(0, 3))) for k in sizes],
        "public_examples": [tuple(g(re, max(unit.min_n, k))) for k in (3, 5)],
        "public_workload": [tuple(g(rp, max(unit.min_n, int(scale * rp.uniform(0.7, 1.3))))) for _ in range(d.n_perf)],
        "online_timing": [tuple(g(rt, max(unit.min_n, int(scale * rt.uniform(0.7, 1.3))))) for _ in range(d.n_perf)],
        "audit_timing": [tuple(g(rat, max(unit.min_n, int(scale * rat.uniform(0.7, 1.3))))) for _ in range(d.n_perf)],
    }


def _params(src: str, name: str) -> List[str]:
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
    a = fn.args
    if a.vararg or a.kwarg or a.kwonlyargs or a.defaults:
        raise ValueError("only plain positional parameters are supported by contract v1")
    return [x.arg for x in a.posonlyargs + a.args]


class TaskBuilder:
    def __init__(self, cfg: ChainConfig, verifier: Verifier, tokenizer=None, min_speedup: float = 1.3):
        self.cfg = cfg
        self.v = verifier
        self.tok = tokenizer
        self.min_speedup = min_speedup
        self.elig = Eligibility()

    # ------------------------------------------------------------------ #
    def _ntokens(self, src: str) -> Tuple[int, str]:
        if self.tok is not None:
            return len(self.tok.encode(src)), "tokenizer"
        return int(math.ceil(len(src) / 3.0)), "char_proxy"

    def _time_once(self, prog: Program, workload: Sequence[Tuple]) -> Optional[float]:
        req = self.v.runner.base_request("calibrate")
        req.update(code=prog.executable(), entry="_tp_entry", workload=list(workload), target_ms=1.0)
        resp = self.v.runner.run(req)
        return float(resp["t_once_ns"]) if resp.get("status") == "ok" else None

    def _contract(self, name: str, params: List[str], suites: Dict[str, List[Tuple]]) -> FunctionContract:
        draws = [a for s in SUITES for a in suites[s]]
        fc = derive_function_contract(name, params, draws)
        return fc

    def _eligible_program(self, what: str, p0: Program, fast: Program, contract: TaskContract,
                          calls: Dict[str, List[Tuple]], names: List[str]) -> Optional[float]:
        """Shared eligibility checks; returns the measured reference speedup (privileged) or None."""
        reasons = contract.unsupported_reasons()
        if reasons:
            self.elig.exclude("contract_unsupported:" + reasons[0], what)
            return None
        for s, cs in calls.items():
            for c in cs:
                err = validate_args(contract, names[c[0]], c[1:])
                if err:
                    self.elig.exclude("input_invalid", f"{what}:{s}:{err}")
                    return None
        for n in names:
            cl = dependency_closure(p0, n)
            if cl.unsupported:
                self.elig.exclude("closure_unsupported", what)
                return None
            nt, _ = self._ntokens(p0.function_source(n))
            if nt > self.cfg.data.max_target_tokens:
                self.elig.exclude("target_too_long", what)
                return None
        det = self.v.determinism_check(p0, calls["online_correctness"])
        if det:
            self.elig.exclude("original_" + det.split(":")[0].replace(" ", "_"), what)
            return None
        try:
            for s in ("online_correctness", "audit_correctness", "public_examples"):
                self.v.reference_records(p0, calls[s])
            self.v.reference_records(p0, calls["online_timing"], output="digest")
            self.v.reference_records(p0, calls["audit_timing"], output="digest", hash_seed=self.cfg.verification.audit_hash_seed)
            self.v.reference_records(p0, calls["audit_correctness"], hash_seed=self.cfg.verification.audit_hash_seed)
        except InfrastructureError as e:
            self.elig.exclude("original_unrepresentable", f"{what}: {e}")
            return None
        # privileged reference: the fast version must be correct and clearly faster
        tmp_task = PublicTask("tmp", "tmp", "train", p0.source, list(p0.functions), contract, calls["public_workload"])
        for s in ("online_correctness", "audit_correctness"):
            r = self.v._check("build", p0, fast, calls[s], mismatch_kind="mismatch", public=False)
            if not r.ok:
                self.elig.exclude("reference_incorrect", what)
                return None
        tr = self.v.paired_timing(tmp_task, calls["online_timing"], p0, fast, blocks=self.cfg.verification.train_timing_blocks,
                                  seed=_seed_int(what, "ref"), rule="point_estimate", threshold=self.min_speedup)
        if not tr.ok or tr.verdict is None:
            self.elig.exclude("reference_timing_failed", what)
            return None
        if tr.verdict.speedup < self.min_speedup:
            self.elig.exclude("reference_speedup_below_min", what)
            return None
        return tr.verdict.speedup

    # ------------------------------------------------------------------ #
    def build_unit(self, var: Variant, split: str, idx: int) -> Optional[UnitTask]:
        what = f"{var.unit.name}/{var.family}/{idx}"
        try:
            params = _params(var.slow, var.name)
            if _params(var.fast, var.name) != params:
                self.elig.exclude("signature_changed_by_variant", what)
                return None
        except (ValueError, StopIteration, SyntaxError) as e:
            self.elig.exclude("unsupported_signature", f"{what}: {e}")
            return None
        p0 = Program.from_source(var.slow, [var.name])
        fast = Program.from_source(var.fast, [var.name])
        base = _seed_int(var.unit.name, var.family, sha256_text(var.slow), split, self.cfg.data.split_salt)
        scale = self.cfg.data.perf_scale
        suites = None
        for _ in range(10):
            try:
                suites = draw_suites(var.unit, base, scale, self.cfg)
            except Exception as e:  # noqa: BLE001 - a generator bug excludes the unit, it never aborts the build
                self.elig.exclude("generator_error", f"{what}: {e}")
                return None
            t = self._time_once(p0, [(0, *a) for a in suites["online_timing"]])
            if t is None:
                self.elig.exclude("original_failed_on_workload", what)
                return None
            if t <= self.cfg.data.c0_cap_ms * 1e6 or scale <= 8:
                break
            scale = max(8, scale // 2)
        assert suites is not None
        try:
            fc = self._contract(var.name, params, suites)
        except Exception as e:  # noqa: BLE001
            self.elig.exclude("contract_derivation_failed", f"{what}: {e}")
            return None
        contract = TaskContract(CONTRACT_VERSION, [fc], origin="generator_derived_v1",
                                permitted_imports=list(self.cfg.sandbox.allowed_modules))
        calls = {s: [(0, *a) for a in suites[s]] for s in SUITES if s != "public_examples"}
        calls["public_examples"] = [(0, *a) for a in suites["public_examples"]]
        ref = self._eligible_program(what, p0, fast, contract, calls, [var.name])
        if ref is None:
            return None
        tid = f"{split}_fn_{idx:05d}"
        pub = PublicTask(tid, var.cluster_id, split, p0.source, [var.name], contract, calls["public_workload"],
                         {var.name: suites["public_examples"]}, public_workload_id=f"gen:{var.unit.name}:s{scale}",
                         meta={"kind": "function", "k": 1, "component_units": [var.unit.name]})
        priv = PrivateSuite(tid, calls["online_correctness"], calls["online_timing"], calls["audit_correctness"],
                            calls["audit_timing"], seeds={"base": base, "scale": scale},
                            privileged={"fast_source": fast.source, "slow_functions": [var.name], "families": [var.family],
                                        "mechanism": mechanism_for_family(var.family), "reference_speedup": ref,
                                        "unit": var.unit.name})
        self.elig.counts["eligible_unit"] += 1
        return UnitTask(var, pub, priv, fc)

    # ------------------------------------------------------------------ #
    def build_file(self, comps: List[UnitTask], slow_idx: Sequence[int], split: str, idx: int, group_id: str) -> Optional[Tuple[PublicTask, PrivateSuite]]:
        names = [c.variant.name for c in comps]
        what = f"file:{'+'.join(names)}"
        if len(set(names)) != len(names):
            self.elig.exclude("duplicate_function_names", what)
            return None
        srcs = [c.variant.slow if i in slow_idx else c.variant.fast for i, c in enumerate(comps)]
        try:
            p0 = Program.from_source("\n\n\n".join(srcs), names)
            fast = Program.from_source("\n\n\n".join(c.variant.fast for c in comps), names)
        except SyntaxError:
            self.elig.exclude("compose_syntax", what)
            return None
        if len(p0.source) > 6000:
            self.elig.exclude("file_too_long", what)
            return None
        base = _seed_int("file", split, idx, *names)
        scale = self.cfg.data.perf_scale
        per: List[Dict[str, List[Tuple]]] = []
        for _ in range(10):
            per = [draw_suites(c.variant.unit, _seed_int(base, k), max(8, scale // len(comps)), self.cfg) for k, c in enumerate(comps)]
            wl = [(k, *a) for k, s in enumerate(per) for a in s["online_timing"]]
            t = self._time_once(p0, wl)
            if t is None:
                self.elig.exclude("original_failed_on_workload", what)
                return None
            if t <= self.cfg.data.c0_cap_ms * 1e6 or scale <= 8 * len(comps):
                break
            scale = max(8 * len(comps), scale // 2)
        fcs = []
        for c, s in zip(comps, per):
            try:
                fcs.append(self._contract(c.variant.name, [p.name for p in c.contract.params], s))
            except Exception as e:  # noqa: BLE001
                self.elig.exclude("contract_derivation_failed", f"{what}: {e}")
                return None
        contract = TaskContract(CONTRACT_VERSION, fcs, origin="generator_derived_v1",
                                permitted_imports=list(self.cfg.sandbox.allowed_modules))
        calls = {s: [(k, *a) for k, sd in enumerate(per) for a in sd[s]] for s in SUITES}
        ref = self._eligible_program(what, p0, fast, contract, calls, names)
        if ref is None:
            return None
        tid = f"{split}_file_{idx:05d}"
        pub = PublicTask(tid, group_id, split, p0.source, names, contract, calls["public_workload"],
                         {c.variant.name: per[k]["public_examples"] for k, c in enumerate(comps)},
                         public_workload_id=f"files:{group_id}:s{scale}",
                         meta={"kind": "file", "k": len(comps), "component_units": [c.variant.unit.name for c in comps],
                               "component_clusters": [c.variant.cluster_id for c in comps]})
        priv = PrivateSuite(tid, calls["online_correctness"], calls["online_timing"], calls["audit_correctness"],
                            calls["audit_timing"], seeds={"base": base, "scale": scale},
                            privileged={"fast_source": fast.source, "slow_functions": [names[i] for i in slow_idx],
                                        "families": sorted({comps[i].variant.family for i in slow_idx}),
                                        "mechanisms": {names[i]: mechanism_for_family(comps[i].variant.family) for i in slow_idx},
                                        "reference_speedup": ref, "group_id": group_id})
        self.elig.counts["eligible_file"] += 1
        return pub, priv


# --------------------------------------------------------------------------- #
def group_clusters(cluster_ids: Sequence[str], group_size: int, salt: int) -> Dict[str, str]:
    order = sorted(cluster_ids, key=lambda c: sha256_text(f"group|{salt}|{c}"))
    return {c: f"g_{sha256_text('|'.join(order[i - i % group_size: i - i % group_size + group_size]))[:12]}"
            for i, c in enumerate(order)}


def build_split(cfg: ChainConfig, manifest: Dict[str, Any], split: str, kinds: Sequence[str], verifier: Verifier,
                tokenizer=None, units_limit: int = 0, max_variants: int = 3, group_size: int = 4,
                min_speedup: float = 1.3, mined: Optional[str] = None, workers: int = 4) -> Dict[str, Any]:
    store = TaskStore(cfg.paths.tasks, cfg.paths.private)
    units = [u for u in all_units(mined) if manifest["unit_to_cluster"].get(u.name)
             and manifest["clusters"][manifest["unit_to_cluster"][u.name]]["split"] == split]
    units.sort(key=lambda u: sha256_text(f"order|{u.name}"))
    if units_limit:
        units = units[:units_limit]
    b = TaskBuilder(cfg, verifier, tokenizer, min_speedup)
    rng = random.Random(_seed_int("variants", split, cfg.data.split_salt))
    vars_: List[Variant] = []
    for u in units:
        vars_.extend(variants_for(u, manifest["unit_to_cluster"][u.name], max_variants, rng))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        built = list(ex.map(lambda iv: b.build_unit(iv[1], split, iv[0]), list(enumerate(vars_))))
    unit_tasks = [u for u in built if u is not None]
    for i, u in enumerate(unit_tasks):
        tid = f"{split}_fn_{i:05d}"
        u.public.task_id = tid
        u.private.task_id = tid
    out: Dict[str, Any] = {"split": split, "units": len(units), "variants": len(vars_), "unit_tasks": len(unit_tasks)}
    if "functions" in kinds:
        store.save(f"{split}_functions", [u.public for u in unit_tasks], [u.private for u in unit_tasks])
    if "files" in kinds and unit_tasks:
        by_cluster: Dict[str, List[UnitTask]] = defaultdict(list)
        for u in unit_tasks:
            by_cluster[u.variant.cluster_id].append(u)
        groups = group_clusters(sorted(by_cluster), group_size, cfg.data.split_salt)
        by_group: Dict[str, List[str]] = defaultdict(list)
        for c, g in groups.items():
            by_group[g].append(c)
        n_files = int(cfg.data.n_files_per_split.get(split, 0))
        frng = random.Random(_seed_int("files", split, cfg.data.split_salt))
        kmin, kmax = cfg.data.files_k
        dmin, dmax = cfg.data.files_d
        specs = []
        gids = sorted(g for g, cs in by_group.items() if len(cs) >= kmin)
        attempts = 0
        while gids and len(specs) < int(n_files * 1.5) + 4 and attempts < 50 * max(1, n_files):
            attempts += 1
            g = gids[attempts % len(gids)]
            cs = by_group[g]
            k = frng.randint(kmin, min(kmax, len(cs)))
            chosen = frng.sample(cs, k)
            comps = [frng.choice(by_cluster[c]) for c in chosen]
            d = frng.randint(dmin, min(dmax, k))
            slow_idx = sorted(frng.sample(range(k), d))
            specs.append((comps, slow_idx, g))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            files = list(ex.map(lambda iv: b.build_file(iv[1][0], iv[1][1], split, iv[0], iv[1][2]), list(enumerate(specs))))
        files = [f for f in files if f is not None][:n_files]
        for i, (pub, priv) in enumerate(files):
            tid = f"{split}_file_{i:05d}"
            pub.task_id = priv.task_id = tid
        store.save(f"{split}_files", [p for p, _ in files], [q for _, q in files])
        out["file_tasks"] = len(files)
        out["file_groups"] = len(gids)
    out["eligibility"] = dict(b.elig.counts)
    out["exclusion_examples"] = {k: v for k, v in b.elig.examples.items()}
    return out


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--manifest", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "dev", "confirm"])
    p.add_argument("--kinds", nargs="+", default=["functions", "files"])
    p.add_argument("--mined", default=None)
    p.add_argument("--units-limit", type=int, default=0)
    p.add_argument("--max-variants", type=int, default=3)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--min-speedup", type=float, default=1.3)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("overrides", nargs="*")
    a = p.parse_args(argv)
    from tinyperf_chain.config import pop_overrides
    pop_overrides(a, ['splits', 'kinds'])
    cfg = load_chain_config(a.config, a.overrides).validate()
    manifest = load_manifest(a.manifest)
    tok = None
    tpath = a.tokenizer or cfg.model.tokenizer
    if tpath and os.path.exists(tpath):
        from tinyperf.model.tokenizer import CodeTokenizer

        tok = CodeTokenizer(tpath)
    verifier = Verifier(cfg, cache_dir=cfg.paths.cache)
    report = {"manifest_sha256": manifest["manifest_sha256"], "splits": {}}
    for split in a.splits:
        report["splits"][split] = build_split(cfg, manifest, split, a.kinds, verifier, tok, a.units_limit, a.max_variants,
                                              a.group_size, a.min_speedup, a.mined, cfg.data.workers)
        print(json.dumps(report["splits"][split]["eligibility"]), flush=True)
    os.makedirs(cfg.paths.tasks, exist_ok=True)
    with open(os.path.join(cfg.paths.tasks, "eligibility.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({s: {k: v for k, v in r.items() if k != "exclusion_examples"} for s, r in report["splits"].items()}, indent=1))


if __name__ == "__main__":
    main()
