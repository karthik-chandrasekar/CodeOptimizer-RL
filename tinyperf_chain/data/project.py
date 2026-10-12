"""Project the trace ledger into five role datasets (``role_sft_v1``).

    python -m tinyperf_chain.data.project --config configs/chain/pilot.yaml --roles M1 M2 M3 M4 M5 \\
        --splits train dev [--manifest artifacts/chain/split_manifest.json]

One experience can feed several roles; every projected record inherits its trial's split and cluster (the five
datasets are never split independently).  Prompts are rendered with the same :class:`PromptBuilder` used online, so
training and serving serialization cannot drift; a record whose required context does not fit is counted and
skipped, never truncated.  Labels:

M1  tie set of targets with maximal measured utility (``alternatives``), or NO_TARGET; ``data.id_permutations``
    copies with shuffled display ids and catalog order (breaks position / id shortcuts).
M2  plans with at least one verified accepted patch (tie set) - with and without a failed-attempt history - or
    NO_PLAN (sampling weight ``data.no_plan_weight``).
M3  replay-gated accepted patches conditioned on their plan; a small share of inapplicable-plan ABANDON cases.
M4  cheapest probe that reveals the candidate's discrepancy, then greedy additional probes by *new* discrepancies
    across candidate variants of the same state and plan (max 3); valid candidates get ``PROBES none``
    (weight ``data.m4_valid_frac``).
M5  verified repair (faster than the pre-attempt state) or ABANDON (weight ``data.m5_abandon_weight``); category-only
    failures are used only when ``private_mismatch`` is a repairable failure.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tinyperf_chain import registry as R
from tinyperf_chain.config import ChainConfig, load_chain_config
from tinyperf_chain.context import (CatalogEntry, ContextOverflow, M1View, M2View, M3View, M4View, M5View, PromptBuilder,
                                    calls_bucket, compact_diff, share_bucket)
from tinyperf_chain.hashing import hash_json, sha256_text
from tinyperf_chain.ledger import AppendLedger, ObjectStore
from tinyperf_chain.program import Program, dependency_closure
from tinyperf_chain.schemas import (PatchAction, Plan, ProbeSelection, TargetChoice, format_patch, format_plan,
                                    format_probes, format_target, parse_plan)
from tinyperf_chain.tasks import PublicTask, TaskStore

RECORD_SCHEMA = "role_sft_v1"


class Projector:
    def __init__(self, cfg: ChainConfig, tok, manifest: Optional[Dict[str, Any]] = None, seed: int = 0,
                 require_gate: bool = True):
        self.cfg = cfg
        self.builder = PromptBuilder(tok, cfg.roles, cfg.model.context_tokens)
        self.objects = ObjectStore(cfg.paths.objects)
        self.ledger = AppendLedger(os.path.join(cfg.paths.ledger, "trials.jsonl"))
        self.tasks = TaskStore(cfg.paths.tasks, cfg.paths.private)
        self.manifest = manifest
        self.rng = random.Random(seed)
        self.require_gate = require_gate
        self.stats: Dict[str, Counter] = defaultdict(Counter)
        self._task_cache: Dict[str, PublicTask] = {}

    # ------------------------------------------------------------------ #
    def _tasks(self, split: str) -> Dict[str, PublicTask]:
        out = {}
        for kind in ("functions", "files"):
            suite = f"{split}_{kind}"
            if os.path.exists(self.tasks.public_path(suite)):
                for t in self.tasks.load_public(suite):
                    out[t.task_id] = t
        return out

    def _check_split(self, rec: Dict[str, Any], task: PublicTask) -> None:
        if rec["split"] != task.split or rec["cluster_id"] != task.cluster_id:
            raise ValueError(f"trial {rec.get('trial_id')} does not inherit its task's split/cluster")
        if self.manifest is not None:
            for unit in task.meta.get("component_units", []):
                cid = self.manifest["unit_to_cluster"].get(unit)
                if cid is None or self.manifest["clusters"][cid]["split"] != task.split:
                    raise ValueError(f"{task.task_id}: unit {unit} is not in the {task.split} partition")

    def _record(self, role: str, rec: Dict[str, Any], task: PublicTask, prompt_view: Any, targets: Sequence[str],
                weight: float = 1.0, **extra: Any) -> Optional[Dict[str, Any]]:
        try:
            built = self.builder.build(role, prompt_view)
        except ContextOverflow:
            self.stats[role]["context_overflow"] += 1
            return None
        tgt_refs = [self.objects.put_text(t) for t in targets]
        r = {
            "schema_version": RECORD_SCHEMA, "role": role, "split": task.split, "cluster_id": task.cluster_id,
            "task_id": task.task_id, "parent_program_sha256": rec["state_sha256"],
            "prompt_artifact": self.objects.put_text(built.text), "prompt_tokens": built.n_tokens,
            "target_artifacts": tgt_refs, "loss_mask": "action_tokens_including_stop", "weight": float(weight),
            "verification_record": rec.get("trial_id"), "dropped_sections": built.dropped, **extra,
        }
        r["record_id"] = f"{role}_" + hash_json([r["prompt_artifact"], tgt_refs, extra.get("variant", 0)])[:16]
        self.stats[role]["records"] += 1
        return r

    def _fid(self, task: PublicTask) -> str:
        """Display ids are an episode-level permutation online; sample one the same way here."""
        return f"f{self.rng.randrange(max(1, len(task.functions))):02d}"

    # ------------------------------------------------------------------ #
    def _state(self, rec: Dict[str, Any], task: PublicTask) -> Program:
        return Program(self.objects.get_text(rec["state_ref"]), tuple(task.functions))

    def _deps(self, state: Program, name: str) -> List[str]:
        return [state.function_source(f) for f in dependency_closure(state, name).functions]

    def m1(self, trials: List[Dict[str, Any]], tasks: Dict[str, PublicTask]) -> List[Dict[str, Any]]:
        out = []
        for rec in trials:
            if rec["kind"] != "target_label" or rec["task_id"] not in tasks:
                continue
            task = tasks[rec["task_id"]]
            self._check_split(rec, task)
            state = self._state(rec, task)
            for v in range(max(1, self.cfg.data.id_permutations)):
                rng = random.Random(f"{rec['trial_id']}|{v}")
                ids = list(range(len(task.functions)))
                rng.shuffle(ids)
                disp = {n: f"f{ids[i]:02d}" for i, n in enumerate(task.functions)}
                order = list(task.functions)
                rng.shuffle(order)
                entries = []
                for n in order:
                    fn = state.function_node(n)
                    src = state.function_source(n)
                    p = rec["profile"].get(n, {"ncalls": 0, "self_share": 0.0, "cum_share": 0.0})
                    entries.append(CatalogEntry(disp[n], n, src, share_bucket(p["self_share"]), share_bucket(p["cum_share"]),
                                                calls_bucket(p["ncalls"]), len(src.splitlines()), len(R.anchors(fn)), 0,
                                                summary=f"{src.splitlines()[0]}\n    # loops={R.render_anchors(fn)}"))
                view = M1View(entries, [], self.cfg.budget.patch_slots)
                labels = [format_target(TargetChoice(disp[n])) for n in rec["label"]] or [format_target(TargetChoice(None))]
                self.stats["M1"]["no_target" if not rec["label"] else "target"] += 1
                r = self._record("M1", rec, task, view, labels, variant=v, downstream_bundle=rec.get("downstream_bundle"))
                if r:
                    out.append(r)
        return out

    def m2(self, trials: List[Dict[str, Any]], tasks: Dict[str, PublicTask]) -> List[Dict[str, Any]]:
        out = []
        by_state: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
        for r in trials:
            if r["kind"] == "plan_patch":
                by_state[(r["task_id"], r["state_sha256"], r["function"])].append(r)
        for rec in trials:
            if rec["kind"] != "plan_set" or rec["task_id"] not in tasks:
                continue
            task = tasks[rec["task_id"]]
            self._check_split(rec, task)
            state = self._state(rec, task)
            name = rec["function"]
            fn = state.function_node(name)
            useful = [p["plan"] for p in rec["plans"] if p["useful"]]
            failed = [p["plan"] for p in rec["plans"] if not p["useful"]]
            fid = self._fid(task)
            base = M2View(fid, state.function_source(name), self._deps(state, name), task.contract.render(name),
                          R.render_anchors(fn), R.operand_names(fn), R.applicable_mechanisms(fn), [])
            if useful:
                self.stats["M2"]["plan"] += 1
                r = self._record("M2", rec, task, base, useful, plan_ids=useful)
                if r:
                    out.append(r)
                for fp in failed[:2]:   # the same state after an unsuccessful attempt
                    pr = parse_plan(fp, fid)
                    hist = [f"{pr.payload.transform_id}({';'.join(pr.payload.operands)}):write:rejected"] if pr.ok else []
                    v2 = M2View(**{**base.__dict__, "history": hist})
                    r = self._record("M2", rec, task, v2, useful, plan_ids=useful, variant=1)
                    if r:
                        out.append(r)
            else:
                self.stats["M2"]["no_plan"] += 1
                r = self._record("M2", rec, task, base, [format_plan(Plan(fid, None))], weight=self.cfg.data.no_plan_weight)
                if r:
                    out.append(r)
        return out

    def m3(self, trials: List[Dict[str, Any]], tasks: Dict[str, PublicTask]) -> List[Dict[str, Any]]:
        out = []
        gates = {r["source_trial"]: r for r in trials if r["kind"] == "gate"}
        positives = 0
        abandon_pool = []
        for rec in trials:
            if rec["kind"] not in ("plan_patch", "chain_candidate") or not rec["outcome"].get("accepted") or rec["task_id"] not in tasks:
                continue
            g = gates.get(rec["trial_id"])
            if self.require_gate and rec["kind"] == "plan_patch":
                if g is None:
                    self.stats["M3"]["ungated_skipped"] += 1
                    continue
                if not g["reproduced"]:
                    self.stats["M3"]["gate_failed"] += 1
                    continue
            task = tasks[rec["task_id"]]
            self._check_split(rec, task)
            state = self._state(rec, task)
            name = rec["function"]
            fn = state.function_node(name)
            fid = self._fid(task)
            pr = parse_plan(rec["plan"], fid)
            if not pr.ok or pr.payload.is_no_plan:
                continue
            code = self.objects.get_text(rec["candidate_fn_ref"])
            view = M3View(fid, state.function_source(name), self._deps(state, name), task.contract.render(name),
                          format_plan(pr.payload), R.render_anchors(fn))
            r = self._record("M3", rec, task, view, [format_patch(PatchAction("edit", code))], plan_id=pr.payload.transform_id,
                             proposer=rec.get("proposer"))
            if r:
                out.append(r)
                positives += 1
                abandon_pool.append((rec, task, state, name, fn, pr.payload))
        # inapplicable-plan ABANDON cases: a mechanism with no applicable site on this function
        n_ab = int(round(self.cfg.data.abandon_frac * positives))
        self.rng.shuffle(abandon_pool)
        for rec, task, state, name, fn, plan in abandon_pool[:n_ab]:
            others = [m for m in R.MECHANISMS if not R.sites_for(fn, m)]
            if not others:
                continue
            m = self.rng.choice(others)
            fid = self._fid(task)
            bad = Plan(fid, m, (), R.MECHANISMS[m].preserve)
            view = M3View(fid, state.function_source(name), self._deps(state, name), task.contract.render(name),
                          format_plan(bad), R.render_anchors(fn))
            r = self._record("M3", rec, task, view, [format_patch(PatchAction("abandon"))], plan_id=m, variant=2)
            if r:
                out.append(r)
                self.stats["M3"]["abandon"] += 1
        return out

    def m4(self, trials: List[Dict[str, Any]], tasks: Dict[str, PublicTask]) -> List[Dict[str, Any]]:
        out = []
        groups: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
        for r in trials:
            if r["kind"] == "probe_matrix" and r["task_id"] in tasks:
                groups[(r["task_id"], r["state_sha256"], r["function"], r.get("plan") or "")].append(r)
        for key, recs in groups.items():
            failing = [r for r in recs if any(r["outcomes"].values())]
            for rec in recs:
                task = tasks[rec["task_id"]]
                self._check_split(rec, task)
                state = self._state(rec, task)
                name = rec["function"]
                costs = rec["costs"]
                bad = [p for p, c in rec["outcomes"].items() if c]
                if bad:
                    first = min(bad, key=lambda p: (costs[p], p))
                    chosen = [first]
                    covered = {id(x) for x in failing if x["outcomes"].get(first)}
                    while len(chosen) < self.cfg.budget.max_probes_per_candidate:
                        gains = []
                        for p in rec["outcomes"]:
                            if p in chosen:
                                continue
                            new = sum(1 for x in failing if id(x) not in covered and x["outcomes"].get(p))
                            if new > 0:
                                gains.append((-new, costs.get(p, 0), p))
                        if not gains:
                            break
                        _, _, p = min(gains)
                        chosen.append(p)
                        covered |= {id(x) for x in failing if x["outcomes"].get(p)}
                    label = format_probes(ProbeSelection(tuple(sorted(chosen))))
                    weight = 1.0
                    self.stats["M4"]["discrepancy"] += 1
                else:
                    if self.rng.random() > self.cfg.data.m4_valid_frac:
                        continue
                    label = format_probes(ProbeSelection(()))
                    weight = 1.0
                    self.stats["M4"]["none"] += 1
                cur = state.function_source(name)
                cand = self.objects.get_text(rec["candidate_fn_ref"])
                menu = ",".join(f"{p}:{n}" for p, n in rec["menu"]) or "none"
                view = M4View(self._fid(task), cur, cand, task.contract.render(name), rec.get("plan") or "none", menu, compact_diff(cur, cand))
                r = self._record("M4", rec, task, view, [label], weight=weight, origin=rec.get("origin"))
                if r:
                    out.append(r)
        return out

    def m5(self, trials: List[Dict[str, Any]], tasks: Dict[str, PublicTask]) -> List[Dict[str, Any]]:
        out = []
        allow_private = "private_mismatch" in self.cfg.verification.repairable_failures
        for rec in trials:
            if rec["kind"] != "repair" or rec["task_id"] not in tasks:
                continue
            if rec["feedback_visibility"] == "category_only" and not allow_private:
                self.stats["M5"]["category_only_skipped"] += 1
                continue
            task = tasks[rec["task_id"]]
            self._check_split(rec, task)
            state = self._state(rec, task)
            name = rec["function"]
            failed = self.objects.get_text(rec["failed_fn_ref"])
            tgt = rec["target"]
            if tgt["action"] == "edit":
                label = format_patch(PatchAction("edit", self.objects.get_text(tgt["code_ref"])))
                weight = 1.0
                self.stats["M5"]["repair"] += 1
            else:
                label = format_patch(PatchAction("abandon"))
                weight = self.cfg.data.m5_abandon_weight
                self.stats["M5"]["abandon"] += 1
            view = M5View(self._fid(task), state.function_source(name), failed, self._deps(state, name), task.contract.render(name),
                          rec.get("plan") or "none", rec.get("feedback") or {"failure_kind": rec.get("failure_kind") or "unknown"})
            r = self._record("M5", rec, task, view, [label], weight=weight, feedback_visibility=rec["feedback_visibility"],
                             failure_kind=rec.get("failure_kind"))
            if r:
                out.append(r)
        return out

    # ------------------------------------------------------------------ #
    def run(self, roles: Sequence[str], splits: Sequence[str]) -> Dict[str, Any]:
        all_trials = list(self.ledger.read())
        report: Dict[str, Any] = {}
        for split in splits:
            tasks = self._tasks(split)
            trials = [t for t in all_trials if t.get("split") == split]
            for role in roles:
                recs = getattr(self, role.lower())(trials, tasks)
                path = os.path.join(self.cfg.paths.datasets, role, f"{split}.jsonl")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    for r in recs:
                        f.write(json.dumps(r) + "\n")
                clusters = {r["cluster_id"] for r in recs}
                report[f"{role}/{split}"] = {"records": len(recs), "independent_clusters": len(clusters),
                                             "duplication_ratio": round(len(recs) / max(1, len(clusters)), 2),
                                             "tasks": len({r["task_id"] for r in recs})}
        report["label_mix"] = {k: dict(v) for k, v in self.stats.items()}
        return report

    def review_sample(self, n: int, split: str = "train") -> List[Dict[str, Any]]:
        """~n examples spread over roles / mechanisms / failure classes for manual review."""
        out = []
        for role in ("M1", "M2", "M3", "M4", "M5"):
            path = os.path.join(self.cfg.paths.datasets, role, f"{split}.jsonl")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                recs = [json.loads(l) for l in f if l.strip()]
            strata: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for r in recs:
                strata[str(r.get("plan_id") or r.get("failure_kind") or r.get("origin") or "all")].append(r)
            per = max(1, n // 5)
            keys = sorted(strata)
            for i in range(per):
                if not keys:
                    break
                bucket = strata[keys[i % len(keys)]]
                r = bucket[self.rng.randrange(len(bucket))]
                out.append({"role": role, "record_id": r["record_id"], "prompt": self.objects.get_text(r["prompt_artifact"]),
                            "targets": [self.objects.get_text(t) for t in r["target_artifacts"]],
                            "task_id": r["task_id"], "verification_record": r.get("verification_record")})
        return out


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--roles", nargs="+", default=["M1", "M2", "M3", "M4", "M5"])
    p.add_argument("--splits", nargs="+", default=["train", "dev"])
    p.add_argument("--manifest", default=None)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--no-require-gate", action="store_true")
    p.add_argument("--review", type=int, default=0, help="write N manual-review examples to datasets/review_sample.jsonl")
    p.add_argument("overrides", nargs="*")
    a = p.parse_args(argv)
    from tinyperf_chain.config import pop_overrides
    pop_overrides(a, ['roles', 'splits'])
    cfg = load_chain_config(a.config, a.overrides).validate()
    for s in a.splits:
        cfg.validate(f"collect:{s}")
    from tinyperf.model.tokenizer import CodeTokenizer

    tok = CodeTokenizer(a.tokenizer or cfg.model.tokenizer)
    manifest = None
    if a.manifest:
        from tinyperf_chain.data.split import load_manifest

        manifest = load_manifest(a.manifest)
    pj = Projector(cfg, tok, manifest, require_gate=not a.no_require_gate)
    rep = pj.run(a.roles, a.splits)
    os.makedirs(cfg.paths.datasets, exist_ok=True)
    with open(os.path.join(cfg.paths.datasets, "projection_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    if a.review:
        with open(os.path.join(cfg.paths.datasets, "review_sample.jsonl"), "w") as f:
            for r in pj.review_sample(a.review):
                f.write(json.dumps(r) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
