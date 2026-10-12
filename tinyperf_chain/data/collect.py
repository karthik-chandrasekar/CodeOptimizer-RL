"""Data factory: measured trials for all five roles, appended to one trace ledger.

    python -m tinyperf_chain.data.collect --config configs/chain/pilot.yaml --split train \\
        --stages plans probes repairs targets gate [--downstream reference|bundle.json] [--limit N]
    python -m tinyperf_chain.data.collect --config ... --split train --stages onpolicy --bundle bundle.json

Stages (each appends records to ``<root>/ledger/trials.jsonl``; nothing is ever rewritten):

``plans``    function states x plausible registry plans x up to ``patches_per_plan`` patches from the programmatic
             proposer, the inverse reference (privileged fast version, labelled with the plan it instantiates) and an
             optional teacher; every patch is verified (scope, smoke, mandatory checks against the original, paired
             timing against the state).  Accepted patches spawn the next state (multi-step states).
``probes``   for real candidates (failed or valid) and verified-incorrect near-miss mutations of accepted patches,
             the discrepancy of every menu probe against the original (outcome matrix).
``repairs``  for each repairable failure, a bounded training-only repair search; keep repairs that pass and beat
             the pre-attempt state; correct-but-not-faster is a diagnostic; nothing found -> ABANDON label.
``targets``  file states: for each candidate function an equal downstream budget (plans x patches, repeated
             searches) with a frozen planner/writer (``reference`` rules or a bundle); utilities, tie sets, labels.
``gate``     replay accepted patches on the original's suites and fresh timing seeds (data-quality gate).
``onpolicy`` run the chain with a frozen bundle on training tasks and record the states it actually visits.

All records inherit split and cluster from their task.  A collector refuses the confirmation split.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from tinyperf.data.rename import mutate_bug
from tinyperf_chain import registry as R
from tinyperf_chain.adapters.verifier import InfrastructureError, Verifier
from tinyperf_chain.config import ChainConfig, load_chain_config
from tinyperf_chain.hashing import hash_json, sha256_text
from tinyperf_chain.ledger import AppendLedger, ObjectStore
from tinyperf_chain.program import PatchError, Program, apply_patch, dependency_closure
from tinyperf_chain.schemas import Plan, format_plan
from tinyperf_chain.tasks import PrivateSuite, PublicTask, TaskStore

TRIAL_SCHEMA = "trial_v1"


def _fn_src(program: Program, name: str) -> str:
    return program.function_source(name)


class Collector:
    def __init__(self, cfg: ChainConfig, verifier: Verifier, split: str, seed: int = 0, teacher: Optional[Callable] = None):
        cfg.validate(f"collect:{split}")
        self.cfg = cfg
        self.v = verifier
        self.split = split
        self.seed = seed
        self.teacher = teacher
        self.tasks = TaskStore(cfg.paths.tasks, cfg.paths.private)
        self.objects = ObjectStore(cfg.paths.objects)
        self.ledger = AppendLedger(os.path.join(cfg.paths.ledger, "trials.jsonl"))
        self.workers = max(1, cfg.data.workers)

    # ------------------------------------------------------------------ #
    def load(self, kind: str, limit: int = 0) -> List[Tuple[PublicTask, PrivateSuite]]:
        suite = f"{self.split}_{kind}"
        if not os.path.exists(self.tasks.public_path(suite)):
            return []
        pubs = self.tasks.load_public(suite, limit)
        priv = self.tasks.load_private(suite)
        for t in pubs:
            if t.split != self.split:
                raise ValueError(f"{t.task_id}: task split {t.split} != collector split {self.split}")
        return [(t, priv[t.task_id]) for t in pubs]

    def _base(self, kind: str, task: PublicTask, state: Program, function: Optional[str], **kw: Any) -> Dict[str, Any]:
        rec = {"schema": TRIAL_SCHEMA, "kind": kind, "split": task.split, "cluster_id": task.cluster_id,
               "task_id": task.task_id, "state_sha256": state.hash, "state_ref": self.objects.put_text(state.source),
               "function": function, "t": time.time(), "collector_seed": self.seed}
        rec.update(kw)
        rec["trial_id"] = hash_json({k: v for k, v in rec.items() if k != "t"})[:24]
        return rec

    # ------------------------------------------------------------------ #
    def verify(self, task: PublicTask, private: PrivateSuite, state: Program, name: str, code: str, seed: int,
               blocks: Optional[int] = None) -> Tuple[Dict[str, Any], Optional[Program]]:
        t0 = time.perf_counter_ns()
        try:
            cand = apply_patch(state, state.hash, name, code, task.contract.permitted_imports)
        except PatchError as e:
            return {"stage": "scope", "kind": e.kind, "category": e.kind, "accepted": False,
                    "feedback": {"failure_kind": e.kind, "message": e.message[:160]}, "wall_ns": time.perf_counter_ns() - t0}, None
        if cand.hash == state.hash:
            return {"stage": "duplicate", "kind": "duplicate", "category": "duplicate", "accepted": False,
                    "wall_ns": time.perf_counter_ns() - t0}, cand
        sm = self.v.smoke_check(task, cand, name)
        if not sm.ok:
            return {"stage": "smoke", "kind": sm.kind, "category": sm.category, "accepted": False,
                    "feedback": sm.public_feedback, "wall_ns": time.perf_counter_ns() - t0}, cand
        mc = self.v.mandatory_checks(task, private, cand)
        if not mc.ok:
            return {"stage": "mandatory", "kind": mc.kind, "category": mc.category, "accepted": False,
                    "feedback": {"failure_kind": mc.category or mc.kind}, "wall_ns": time.perf_counter_ns() - t0}, cand
        tr = self.v.paired_timing(task, private.online_timing, state, cand,
                                  blocks=blocks or self.cfg.verification.train_timing_blocks, seed=seed)
        if not tr.ok or tr.verdict is None:
            return {"stage": "timing", "kind": "timing_error", "category": tr.status, "accepted": False,
                    "wall_ns": time.perf_counter_ns() - t0}, cand
        vd = tr.verdict
        return {"stage": "timing", "kind": vd.decision, "category": vd.decision, "accepted": vd.accepted,
                "correct": True, "speedup": vd.speedup, "lo": vd.lo, "hi": vd.hi, "wall_ns": time.perf_counter_ns() - t0}, cand

    # ------------------------------------------------------------------ #
    # plans
    # ------------------------------------------------------------------ #
    def _reference_fn(self, private: PrivateSuite, name: str) -> Optional[str]:
        fast = private.privileged.get("fast_source")
        if not fast:
            return None
        try:
            return Program(fast, (name,)).function_source(name)
        except KeyError:
            return None

    def _plans_for_state(self, task: PublicTask, private: PrivateSuite, state: Program, name: str, depth: int,
                         rng: random.Random) -> List[Dict[str, Any]]:
        d = self.cfg.data
        fn = state.function_node(name)
        fsrc = state.function_source(name)
        fc = task.contract.function(name)
        plans = R.static_plans(fn, fc, limit=d.plans_per_function)
        proposals: Dict[str, List[Tuple[str, str]]] = {p.key(): [] for p in plans}
        by_key = {p.key(): p for p in plans}
        for p in plans:
            for c in R.propose(fsrc, p.transform_id, p)[: d.patches_per_plan]:
                proposals[p.key()].append(("programmatic", c))
        # inverse reference (privileged): labelled with the plan it instantiates, if any
        ref = self._reference_fn(private, name) if depth == 0 else None
        if ref and ref != fsrc:
            for mid in R.MECHANISMS:
                rp = R.extract_plan(fsrc, ref, mid)
                if rp is not None and R.validate_plan(rp, fn, fc).ok:
                    k = rp.key()
                    if k not in by_key:
                        by_key[k] = rp
                        proposals[k] = []
                    proposals[k].append(("inverse_reference", ref))
                    break
        if self.teacher is not None:
            for code in self.teacher(state, name, rng)[: d.patches_per_plan]:
                for mid in R.MECHANISMS:
                    rp = R.extract_plan(fsrc, code, mid)
                    if rp is not None:
                        k = rp.key()
                        if k not in by_key:
                            by_key[k] = rp
                            proposals[k] = []
                        proposals[k].append(("teacher", code))
                        break
        recs = []
        for k, items in proposals.items():
            p = by_key[k]
            for j, (src_kind, code) in enumerate(items):
                seed = int(sha256_text(f"{task.task_id}|{state.hash}|{k}|{j}")[:8], 16)
                out, cand = self.verify(task, private, state, name, code, seed)
                recs.append(self._base("plan_patch", task, state, name, plan=format_plan(Plan("", p.transform_id, p.operands, p.preserve)),
                                       plan_key=k, transform=p.transform_id, proposer=src_kind,
                                       candidate_fn_ref=self.objects.put_text(code),
                                       candidate_ref=self.objects.put_text(cand.source) if cand else None,
                                       outcome=out, depth=depth))
        recs.append(self._base("plan_set", task, state, name, depth=depth,
                               menu=R.applicable_mechanisms(fn),
                               plans=[{"plan": format_plan(Plan("", by_key[k].transform_id, by_key[k].operands, by_key[k].preserve)),
                                       "plan_key": k,
                                       "useful": any(r["outcome"].get("accepted") for r in recs if r.get("plan_key") == k),
                                       "n_patches": len(proposals[k])} for k in proposals]))
        return recs

    def stage_plans(self, limit: int = 0, max_depth: int = 2) -> Dict[str, int]:
        items = self.load("functions", limit)
        rng = random.Random(self.seed)

        def work(tp: Tuple[PublicTask, PrivateSuite]) -> List[Dict[str, Any]]:
            task, private = tp
            out: List[Dict[str, Any]] = []
            state = task.original()
            for depth in range(max_depth):
                try:
                    recs = self._plans_for_state(task, private, state, task.functions[0], depth, random.Random(rng.random()))
                except InfrastructureError as e:
                    out.append(self._base("infrastructure_failure", task, state, None, detail=str(e)[:300]))
                    break
                out.extend(recs)
                acc = [r for r in recs if r["kind"] == "plan_patch" and r["outcome"].get("accepted") and r.get("candidate_ref")]
                if not acc:
                    break
                best = max(acc, key=lambda r: r["outcome"]["speedup"])
                state = Program(self.objects.get_text(best["candidate_ref"]), state.functions)
            return out

        n = 0
        with ThreadPoolExecutor(self.workers) as ex:
            for recs in ex.map(work, items):
                self.ledger.extend(recs)
                n += len(recs)
        return {"records": n, "tasks": len(items)}

    # ------------------------------------------------------------------ #
    # probes
    # ------------------------------------------------------------------ #
    def _task_index(self) -> Dict[str, Tuple[PublicTask, PrivateSuite]]:
        idx = {}
        for kind in ("functions", "files"):
            for t, p in self.load(kind):
                idx[t.task_id] = (t, p)
        return idx

    def _trials(self, kind: str) -> List[Dict[str, Any]]:
        return [r for r in self.ledger.read() if r.get("kind") == kind and r.get("split") == self.split]

    def stage_probes(self, limit: int = 0) -> Dict[str, int]:
        idx = self._task_index()
        trials = [r for r in self._trials("plan_patch") + self._trials("chain_candidate")
                  if r.get("candidate_ref") and r["outcome"]["stage"] in ("smoke", "probe", "mandatory", "timing")]
        if limit:
            trials = trials[:limit]
        rng = random.Random(self.seed + 1)

        def matrix(task: PublicTask, cand: Program, name: str) -> Dict[str, Any]:
            menu = R.probe_menu(task.contract, task.task_id, name, self.cfg.data.menu_size)
            k = task.functions.index(name)
            calls = [(k, *menu.inputs[p]) for p in menu.ids()]
            outs = self.v.outcome_matrix(task, cand, calls)
            return {"menu": menu.entries, "outcomes": dict(zip(menu.ids(), outs)),
                    "costs": {p: len(repr(menu.inputs[p])) for p in menu.ids()}}

        def work(r: Dict[str, Any]) -> List[Dict[str, Any]]:
            if r["task_id"] not in idx:
                return []
            task, private = idx[r["task_id"]]
            state = Program(self.objects.get_text(r["state_ref"]), tuple(task.functions))
            cand = Program(self.objects.get_text(r["candidate_ref"]), tuple(task.functions))
            name = r["function"]
            out = []
            try:
                mx = matrix(task, cand, name)
                valid = r["outcome"]["stage"] == "timing"
                out.append(self._base("probe_matrix", task, state, name, source_trial=r["trial_id"], plan=r.get("plan"),
                                      candidate_fn_ref=r["candidate_fn_ref"], candidate_ref=r["candidate_ref"],
                                      candidate_valid=valid, origin=r["kind"], **mx))
                if valid and r["outcome"].get("accepted"):
                    fsrc = self.objects.get_text(r["candidate_fn_ref"])
                    for j in range(self.cfg.data.near_miss_per_patch):
                        mut = mutate_bug(fsrc, random.Random(rng.random()))
                        if not mut:
                            continue
                        try:
                            mprog = apply_patch(state, state.hash, name, mut, task.contract.permitted_imports)
                        except PatchError:
                            continue
                        mc = self.v.mandatory_checks(task, private, mprog)
                        if mc.ok:
                            continue      # the mutation happens to be behaviour-preserving: not a near miss
                        mmx = matrix(task, mprog, name)
                        out.append(self._base("probe_matrix", task, state, name, source_trial=r["trial_id"], plan=r.get("plan"),
                                              candidate_fn_ref=self.objects.put_text(mut),
                                              candidate_ref=self.objects.put_text(mprog.source), candidate_valid=False,
                                              origin="near_miss", privileged={"mutation_family": "bug_mutation"},
                                              mandatory_category=mc.category, **mmx))
            except InfrastructureError as e:
                out.append(self._base("infrastructure_failure", task, state, name, detail=str(e)[:300]))
            return out

        n = 0
        with ThreadPoolExecutor(self.workers) as ex:
            for recs in ex.map(work, trials):
                self.ledger.extend(recs)
                n += len(recs)
        return {"records": n, "candidates": len(trials)}

    # ------------------------------------------------------------------ #
    # repairs
    # ------------------------------------------------------------------ #
    def stage_repairs(self, limit: int = 0) -> Dict[str, int]:
        idx = self._task_index()
        pp = self._trials("plan_patch") + self._trials("chain_candidate")
        accepted: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
        for r in pp:
            if r["outcome"].get("accepted"):
                accepted.setdefault((r["task_id"], r["state_sha256"], r["function"], r.get("plan", "")), []).append(r)
        failures: List[Dict[str, Any]] = []
        for r in pp:
            o = r["outcome"]
            if o.get("accepted"):
                continue
            if o["stage"] == "scope" and o["kind"] == "syntax_error":
                failures.append({"src": r, "visibility": "syntax", "feedback": o.get("feedback")})
            elif o["stage"] in ("smoke", "probe"):
                failures.append({"src": r, "visibility": f"public_{o['stage']}", "feedback": o.get("feedback")})
            elif o["stage"] == "mandatory":
                failures.append({"src": r, "visibility": "category_only", "feedback": {"failure_kind": o.get("category")}})
        for m in self._trials("probe_matrix"):
            if m.get("origin") == "near_miss":
                bad = [p for p, c in m["outcomes"].items() if c]
                if bad:
                    cheapest = min(bad, key=lambda p: m["costs"][p])
                    failures.append({"src": m, "visibility": "public_probe", "probe": cheapest, "feedback": None})
        if limit:
            failures = failures[:limit]

        def work(f: Dict[str, Any]) -> List[Dict[str, Any]]:
            r = f["src"]
            if r["task_id"] not in idx:
                return []
            task, private = idx[r["task_id"]]
            state = Program(self.objects.get_text(r["state_ref"]), tuple(task.functions))
            name = r["function"]
            plan_txt = r.get("plan") or ""
            fb = f.get("feedback")
            if f.get("probe"):
                cand = Program(self.objects.get_text(r["candidate_ref"]), tuple(task.functions))
                menu = R.probe_menu(task.contract, task.task_id, name, self.cfg.data.menu_size)
                pr = self.v.probe_check(task, cand, name, [(f["probe"], menu.inputs[f["probe"]])])
                fb = pr.public_feedback or {"failure_kind": pr.category}
            failed_code = self.objects.get_text(r["candidate_fn_ref"])
            fsrc = state.function_source(name)
            # repair candidates (training-only search): verified accepted alternatives, programmatic, reference
            cands: List[Tuple[str, str]] = []
            for a in accepted.get((r["task_id"], r["state_sha256"], name, plan_txt), []):
                cands.append(("accepted_alternative", self.objects.get_text(a["candidate_fn_ref"])))
            from tinyperf_chain.schemas import parse_plan

            pres = parse_plan(plan_txt, "")
            plan = pres.payload if pres.ok else None
            if plan is not None and plan.transform_id in R.MECHANISMS:
                for c in R.propose(fsrc, plan.transform_id, plan):
                    cands.append(("programmatic", c))
            refn = self._reference_fn(private, name) if state.hash == task.original().hash else None
            if refn and plan is not None and plan.transform_id in R.MECHANISMS:
                rp = R.extract_plan(fsrc, refn, plan.transform_id)
                if rp is not None and rp.key() == plan.key():
                    cands.append(("inverse_reference", refn))
            seen, tried, target, diag = set(), [], None, None
            for kind, code in cands:
                norm = code.strip()
                if norm in seen or norm == failed_code.strip():
                    continue
                seen.add(norm)
                if len(tried) >= self.cfg.data.repair_candidates:
                    break
                seed = int(sha256_text(f"repair|{r['trial_id']}|{len(tried)}")[:8], 16)
                out, _ = self.verify(task, private, state, name, code, seed)
                tried.append({"proposer": kind, "outcome": {k: out.get(k) for k in ("stage", "kind", "accepted", "speedup")}})
                if out.get("accepted"):
                    target = {"action": "edit", "code_ref": self.objects.put_text(code), "speedup": out.get("speedup"),
                              "proposer": kind}
                    break
                if out.get("correct") and diag is None:
                    diag = "correct_not_faster"
            if target is None:
                target = {"action": "abandon"}
            return [self._base("repair", task, state, name, source_trial=r["trial_id"], plan=plan_txt,
                               failed_fn_ref=r["candidate_fn_ref"], feedback_visibility=f["visibility"], feedback=fb,
                               failure_kind=(fb or {}).get("failure_kind"), repairs_tried=tried, target=target,
                               diagnostic=diag, origin=r["kind"] if "kind" in r else "?")]

        n = 0
        with ThreadPoolExecutor(self.workers) as ex:
            for recs in ex.map(work, failures):
                self.ledger.extend(recs)
                n += len(recs)
        return {"records": n, "failures": len(failures)}

    # ------------------------------------------------------------------ #
    # targets (file states)
    # ------------------------------------------------------------------ #
    def _downstream_reference(self, task: PublicTask, private: PrivateSuite, state: Program, name: str,
                              search: int) -> Dict[str, Any]:
        d = self.cfg.data
        fn = state.function_node(name)
        fsrc = state.function_source(name)
        plans = R.static_plans(fn, task.contract.function(name), limit=d.target_plans)
        best, trials, cost_ns = 0.0, [], 0
        best_src = None
        for p in plans:
            for j, code in enumerate(R.propose(fsrc, p.transform_id, p)[: d.target_patches_per_plan]):
                seed = int(sha256_text(f"target|{task.task_id}|{state.hash}|{name}|{p.key()}|{j}|{search}")[:8], 16)
                out, cand = self.verify(task, private, state, name, code, seed)
                cost_ns += int(out.get("wall_ns", 0))
                g = math.log(out["speedup"]) if out.get("accepted") else 0.0
                trials.append({"plan": format_plan(p), "accepted": bool(out.get("accepted")), "gain": g})
                if g > best:
                    best, best_src = g, cand.source if cand else None
        return {"gain": best, "trials": trials, "cost_ns": cost_ns, "best_source": best_src}

    def stage_targets(self, limit: int = 0, downstream: str = "reference", router=None,
                      from_onpolicy: bool = False) -> Dict[str, int]:
        """Label file states.  Default: the original and the states reached by applying the best labelled target;
        ``from_onpolicy``: the states the chain actually visited (``chain_target_state`` records)."""
        items = self.load("files", limit)
        d = self.cfg.data
        if from_onpolicy:
            seen_states = set()
            visited: Dict[str, List[str]] = {}
            for r in self._trials("chain_target_state"):
                if r["state_sha256"] in seen_states:
                    continue
                seen_states.add(r["state_sha256"])
                visited.setdefault(r["task_id"], []).append(r["state_ref"])
        ds_hash = "reference_rules_v1:" + R.registry_hash()[:12] if downstream == "reference" else (router.bundle_hash if router else downstream)
        n_searches = 1 if downstream == "reference" else d.target_searches   # the reference downstream is deterministic

        def label_state(task: PublicTask, private: PrivateSuite, state: Program, depth: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            prof = self.v.profile(task, state)
            utils: Dict[str, Any] = {}
            for name in task.functions:
                runs = [self._downstream_reference(task, private, state, name, s) if router is None
                        else self._downstream_bundle(task, private, state, name, s, router)
                        for s in range(n_searches)]
                u = sum(r["gain"] for r in runs) / len(runs) - d.utility_lambda_cost * sum(r["cost_ns"] for r in runs) / 1e9 / len(runs)
                utils[name] = {"U": u, "searches": [{"gain": r["gain"], "trials": r["trials"], "cost_ns": r["cost_ns"]} for r in runs],
                               "best_source": max(runs, key=lambda r: r["gain"])["best_source"]}
            umax = max(v["U"] for v in utils.values()) if utils else 0.0
            label = sorted(n for n, v in utils.items() if umax > 1e-9 and v["U"] >= umax - d.utility_tie_tol)
            rec = self._base("target_label", task, state, None, depth=depth, downstream_bundle=ds_hash,
                             profile={k: {"ncalls": v.ncalls, "self_share": v.self_share, "cum_share": v.cum_share}
                                      for k, v in prof.functions.items()},
                             utilities={k: {"U": v["U"], "searches": v["searches"]} for k, v in utils.items()},
                             label=label, n_searches=n_searches, onpolicy=from_onpolicy)
            return rec, utils

        def work_onpolicy(tp: Tuple[PublicTask, PrivateSuite]) -> List[Dict[str, Any]]:
            task, private = tp
            out = []
            for ref in visited.get(task.task_id, [])[: d.max_states_per_file]:
                state = Program(self.objects.get_text(ref), tuple(task.functions))
                try:
                    out.append(label_state(task, private, state, -1)[0])
                except InfrastructureError as e:
                    out.append(self._base("infrastructure_failure", task, state, None, detail=str(e)[:300]))
            return out

        def work(tp: Tuple[PublicTask, PrivateSuite]) -> List[Dict[str, Any]]:
            if from_onpolicy:
                return work_onpolicy(tp)
            task, private = tp
            out = []
            state = task.original()
            for depth in range(d.max_states_per_file):
                try:
                    rec, utils = label_state(task, private, state, depth)
                except InfrastructureError as e:
                    out.append(self._base("infrastructure_failure", task, state, None, detail=str(e)[:300]))
                    break
                out.append(rec)
                label = rec["label"]
                if not label:
                    break
                nxt = utils[label[0]]["best_source"]
                if not nxt:
                    break
                state = Program(nxt, state.functions)
            return out

        n = 0
        with ThreadPoolExecutor(self.workers) as ex:
            for recs in ex.map(work, items):
                self.ledger.extend(recs)
                n += len(recs)
        return {"records": n, "files": len(items)}

    def _downstream_bundle(self, task: PublicTask, private: PrivateSuite, state: Program, name: str, search: int,
                           router) -> Dict[str, Any]:
        """Same downstream budget with the bundle's actual M2/M3 (forced target)."""
        from tinyperf_chain.context import M2View, M3View
        from tinyperf_chain.schemas import RequestEnvelope, SCHEMA_VERSIONS

        d = self.cfg.data
        fn = state.function_node(name)
        fsrc = state.function_source(name)
        fc = task.contract.function(name)
        deps = [state.function_source(f) for f in dependency_closure(state, name).functions]
        fid = "f00"
        best, best_src, trials, cost_ns, attempted = 0.0, None, [], 0, []
        nreq = 0

        def env(role: str, prep) -> RequestEnvelope:
            nonlocal nreq
            nreq += 1
            return RequestEnvelope(role, task.task_id, f"tgt:{task.task_id}:{state.hash[:8]}:{name}:{search}:{nreq}",
                                   state.hash, router.bundle_hash, SCHEMA_VERSIONS[role], prep.max_new_tokens, float("inf"))

        for _ in range(d.target_plans):
            v2 = M2View(fid, fsrc, deps, task.contract.render(name), R.render_anchors(fn), R.operand_names(fn),
                        R.applicable_mechanisms(fn), [])
            ctx2 = {"function_id": fid, "fn_source": fsrc, "fcontract": fc, "attempted": list(attempted)}
            prep = router["M2"].prepare("M2", v2, ctx2)
            r2 = router["M2"].execute(prep, env("M2", prep))
            if not r2.ok or r2.parsed.payload.is_no_plan:
                break
            plan = r2.parsed.payload
            if plan.key() in attempted or not R.validate_plan(plan, fn, fc).ok:
                break
            attempted.append(plan.key())
            for j in range(d.target_patches_per_plan):
                v3 = M3View(fid, fsrc, deps, task.contract.render(name), format_plan(plan), R.render_anchors(fn))
                ctx3 = {"function_id": fid, "function_name": name, "fn_source": fsrc, "plan": plan,
                        "file_source": state.source, "legacy_state": {"best_runtime": 1.0, "last_runtime": "n/a",
                                                                       "last_correct": -1, "last_status": "start",
                                                                       "step": 0, "remaining": 6}}
                prep3 = router["M3"].prepare("M3", v3, ctx3)
                r3 = router["M3"].execute(prep3, env("M3", prep3))
                if not r3.ok or r3.parsed.payload.is_abandon:
                    trials.append({"plan": format_plan(plan), "accepted": False, "gain": 0.0, "status": "abandon_or_malformed"})
                    continue
                seed = int(sha256_text(f"tb|{task.task_id}|{state.hash}|{name}|{plan.key()}|{j}|{search}")[:8], 16)
                out, cand = self.verify(task, private, state, name, r3.parsed.payload.code, seed)
                cost_ns += int(out.get("wall_ns", 0))
                g = math.log(out["speedup"]) if out.get("accepted") else 0.0
                trials.append({"plan": format_plan(plan), "accepted": bool(out.get("accepted")), "gain": g})
                if g > best:
                    best, best_src = g, cand.source if cand else None
        return {"gain": best, "trials": trials, "cost_ns": cost_ns, "best_source": best_src}

    # ------------------------------------------------------------------ #
    # gate
    # ------------------------------------------------------------------ #
    def stage_gate(self, limit: int = 0) -> Dict[str, int]:
        idx = self._task_index()
        done = {r["source_trial"] for r in self._trials("gate")}
        todo = [r for r in self._trials("plan_patch") if r["outcome"].get("accepted") and r["trial_id"] not in done]
        if limit:
            todo = todo[:limit]

        def work(r: Dict[str, Any]) -> List[Dict[str, Any]]:
            if r["task_id"] not in idx:
                return []
            task, private = idx[r["task_id"]]
            state = Program(self.objects.get_text(r["state_ref"]), tuple(task.functions))
            code = self.objects.get_text(r["candidate_fn_ref"])
            seed = int(sha256_text(f"gate|{r['trial_id']}")[:8], 16)
            out, _ = self.verify(task, private, state, r["function"], code, seed)
            return [self._base("gate", task, state, r["function"], source_trial=r["trial_id"],
                               reproduced=bool(out.get("accepted")), speedup_replay=out.get("speedup"),
                               speedup_claimed=r["outcome"].get("speedup"), replay_stage=out.get("stage"))]

        n = 0
        with ThreadPoolExecutor(self.workers) as ex:
            for recs in ex.map(work, todo):
                self.ledger.extend(recs)
                n += len(recs)
        return {"records": n}

    # ------------------------------------------------------------------ #
    # on-policy states from a frozen bundle (training tasks only)
    # ------------------------------------------------------------------ #
    def stage_onpolicy(self, router, kinds: Sequence[str] = ("files", "functions"), limit: int = 0,
                       episodes_per_task: int = 1) -> Dict[str, int]:
        from tinyperf_chain.controller import ChainController, EpisodeSpec

        buf: List[Dict[str, Any]] = []
        items = [x for k in kinds for x in self.load(k, limit)]

        def make_observer(task: PublicTask):
            def obs(rec: Dict[str, Any]) -> None:
                state = Program(rec.pop("state_source"), tuple(task.functions))
                kind = rec.pop("kind")
                if kind == "chain_candidate":
                    cand_src = rec.pop("candidate_source")
                    code = rec.pop("code")
                    o = {"stage": rec["stage"], "kind": rec["check_kind"], "category": rec["category"],
                         "accepted": rec["accepted"], "speedup": rec["speedup"], "feedback": rec["public_feedback"]}
                    if rec["accepted"] or rec["stage"] == "timing":
                        o["correct"] = True
                    buf.append(self._base("chain_candidate", task, state, rec["function"], plan=rec["plan"], role=rec["role"],
                                          candidate_fn_ref=self.objects.put_text(code),
                                          candidate_ref=self.objects.put_text(cand_src) if cand_src else None,
                                          outcome=o, bundle=router.bundle_hash, plan_key=None))
                elif kind == "chain_target_state":
                    buf.append(self._base("chain_target_state", task, state, None, chosen=rec.get("chosen_name"),
                                          catalog=rec["ctx"]["catalog"], bundle=router.bundle_hash))
                elif kind == "chain_plan_state":
                    buf.append(self._base("chain_plan_state", task, state, rec["function"], plan=rec["plan"],
                                          bundle=router.bundle_hash))
            return obs

        n = 0
        for i, (task, private) in enumerate(items):
            for e in range(episodes_per_task):
                ctrl = ChainController(self.cfg, self.v, router, observer=make_observer(task))
                ctrl.run_episode(task, private, EpisodeSpec(f"onpolicy:{task.task_id}:{e}", seed=self.seed * 1000 + e,
                                                            mode="train", run_audit=False))
                self.ledger.extend(buf)
                n += len(buf)
                buf.clear()
        return {"records": n, "tasks": len(items)}


# --------------------------------------------------------------------------- #
def make_teacher(cfg: ChainConfig, k: int = 2):
    """Optional extra patch proposer: the existing optimizer sampled on the legacy observation of a state."""
    if not cfg.data.teacher_checkpoint:
        return None
    import torch

    from tinyperf.env.protocol import EDIT_CLOSE, STOP_TOKEN, format_observation, parse_action
    from tinyperf.model.tokenizer import CodeTokenizer
    from tinyperf.model.transformer import TinyPerfLM
    from tinyperf_chain.serve.engine import RoleEngine, SamplingSpec, resolve_device

    tok = CodeTokenizer(cfg.model.tokenizer)
    dev = resolve_device(cfg.serving.device)
    eng = RoleEngine(TinyPerfLM.load(cfg.data.teacher_checkpoint), device=dev)
    stop = [tok.token_id(EDIT_CLOSE), tok.token_id(STOP_TOKEN), tok.eos_id]

    def propose(state: Program, name: str, rng: random.Random) -> List[str]:
        obs = format_observation(state.source, {"best_runtime": 1.0, "last_runtime": "n/a", "last_correct": -1,
                                                "last_status": "start", "step": 0, "remaining": 6})
        ids = tok.encode(obs, add_bos=True)
        if len(ids) + 768 > eng.max_seq_len:
            return []
        outs = eng.generate([ids] * k, 768, SamplingSpec(temperature=0.8), stop, pad_id=tok.pad_id,
                            seeds=[rng.randrange(1 << 30) for _ in range(k)])
        res = []
        for o in outs:
            a = parse_action(tok.decode(o.ids))
            if a.kind == "edit" and a.code:
                try:
                    for n in ast.parse(a.code).body:
                        if isinstance(n, ast.FunctionDef) and n.name == name:
                            res.append(ast.unparse(n) + "\n")
                except SyntaxError:
                    res.append(a.code)
        return res

    return propose


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--stages", nargs="+", default=["plans", "probes", "repairs", "targets", "gate"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--downstream", default="reference", help="'reference' or a bundle.json for target labels")
    p.add_argument("--bundle", default=None, help="frozen bundle for --stages onpolicy")
    p.add_argument("--episodes-per-task", type=int, default=1)
    p.add_argument("--onpolicy-states", action="store_true", help="targets: label the states recorded by onpolicy")
    p.add_argument("overrides", nargs="*")
    a = p.parse_args(argv)
    from tinyperf_chain.config import pop_overrides
    pop_overrides(a, ['stages'])
    cfg = load_chain_config(a.config, a.overrides)
    cfg.validate(f"collect:{a.split}")
    v = Verifier(cfg, cache_dir=cfg.paths.cache)
    col = Collector(cfg, v, a.split, a.seed, teacher=make_teacher(cfg))
    report: Dict[str, Any] = {}
    router = None
    if a.bundle or (a.downstream not in ("reference",)):
        from tinyperf_chain.bundle import build_router, load_bundle_json
        from tinyperf_chain.serve.engine import resolve_device

        router, _ = build_router(load_bundle_json(a.bundle or a.downstream), cfg, device=resolve_device(cfg.serving.device))
    for st in a.stages:
        t0 = time.time()
        if st == "plans":
            report[st] = col.stage_plans(a.limit)
        elif st == "probes":
            report[st] = col.stage_probes(a.limit)
        elif st == "repairs":
            report[st] = col.stage_repairs(a.limit)
        elif st == "targets":
            report[st] = col.stage_targets(a.limit, a.downstream, router if a.downstream != "reference" else None,
                                           from_onpolicy=a.onpolicy_states)
        elif st == "gate":
            report[st] = col.stage_gate(a.limit)
        elif st == "onpolicy":
            if router is None:
                raise SystemExit("--stages onpolicy needs --bundle")
            report[st] = col.stage_onpolicy(router, limit=a.limit, episodes_per_task=a.episodes_per_task)
        else:
            raise SystemExit(f"unknown stage {st}")
        report[st]["seconds"] = round(time.time() - t0, 1)
        print(json.dumps({st: report[st]}), flush=True)


if __name__ == "__main__":
    main()
