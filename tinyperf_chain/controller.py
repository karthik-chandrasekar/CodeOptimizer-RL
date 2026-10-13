"""Budgeted deterministic controller: M1 -> M2 -> M3 -> (M4 probes, mandatory checks, paired timing) -> M5 repair.

Non-negotiable invariants (enforced here and tested in ``tests/chain``):

* candidates are checked against the immutable original ``P0`` for semantics;
* patches apply only to their recorded base version (the ``program_hash`` of the request envelope); a candidate is
  committed only if that base is still the current best; rejected edits never change the current best ``Pb``;
* repairs consume patch slots (``max_repairs_per_lineage`` per fresh-write lineage, default one); failure after the
  last repair returns to planning;
* hidden audit results never reach a model: the search only ever holds the :class:`OnlineSuite` view of the private
  data, the full suite is handed to the final audit alone, which runs once, after search, on ``Pb``;
* every model call, failed parse and test run is logged with start/end times, budget before/after and the bundle
  hash; model bundles stay fixed during an episode.

The controller owns budgets, program state, input validity, acceptance and stopping.  Roles only choose.
"""
from __future__ import annotations

import ast
import random
import resource
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from tinyperf_chain import registry as R
from tinyperf_chain.adapters.base import Prepared, RoleRouter
from tinyperf_chain.adapters.verifier import CheckResult, InfrastructureError, ProfileSnapshot, TimingResult, Verifier
from tinyperf_chain.budget import BudgetExhausted, BudgetLedger, DeadlineExceeded, SearchStop
from tinyperf_chain.config import ChainConfig
from tinyperf_chain.context import (CatalogEntry, ContextOverflow, M1View, M2View, M3View, M4View, M5View, calls_bucket,
                                    compact_diff, share_bucket)
from tinyperf_chain.hashing import hash_json, sha256_text
from tinyperf_chain.ledger import EventLog
from tinyperf_chain.program import PatchError, Program, apply_patch, dependency_closure, dependency_sources
from tinyperf_chain.schemas import (SCHEMA_VERSIONS, ParseResult, PatchAction, Plan, ProbeSelection, RequestEnvelope,
                                    RoleResponse, TargetChoice, format_plan)
from tinyperf_chain.tasks import OnlineSuite, PrivateSuite, PublicTask

EPISODE_SCHEMA = "chain_episode_v1"
ALLOWED_ACTIONS = {"M1": ("TARGET", "NO_TARGET"), "M2": ("PLAN", "NO_PLAN"), "M3": ("EDIT", "ABANDON"),
                   "M4": ("PROBES",), "M5": ("EDIT", "ABANDON")}


def _peak_rss_mb() -> float:
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:  # noqa: BLE001 - non-posix
        return 0.0


@dataclass
class EpisodeSpec:
    episode_id: str
    seed: int = 0
    mode: str = "dev"                     # train | dev | confirm  (selects the online timing block count)
    patch_slots: Optional[int] = None     # override for patch-budget curves
    events_path: Optional[str] = None
    run_audit: bool = True


@dataclass
class Verdict:
    accept: bool
    repairable: bool
    stage: str
    kind: str
    category: str = ""
    feedback: Optional[Dict[str, Any]] = None
    candidate: Optional[Program] = None
    timing: Optional[TimingResult] = None
    fn_source: str = ""                   # the candidate's version of the target function (for M5)
    base_hash: str = ""                   # the program version the patch was applied to (commit checks it)

    @property
    def outcome(self) -> str:
        return "accepted" if self.accept else (self.category or self.kind)


class _State:
    def __init__(self, task: PublicTask, spec: EpisodeSpec, cfg: ChainConfig):
        self.task = task
        self.original = task.original()
        self.best = self.original
        self.version = 0
        names = [f for f in task.functions]
        rng = random.Random(f"ids|{spec.seed}|{task.task_id}")
        perm = list(range(len(names)))
        rng.shuffle(perm)
        self.display = {n: f"f{perm[i]:02d}" for i, n in enumerate(names)}
        self.by_display = {v: k for k, v in self.display.items()}
        self.attempted: Dict[str, List[str]] = {}
        self.blocked: Set[str] = set()
        self.history_global: List[str] = []
        self.history_fn: Dict[str, List[str]] = {}
        self.rejected_hashes: Set[str] = set()
        self.accepted: List[Dict[str, Any]] = []
        self.control_failures: List[Dict[str, Any]] = []
        self.lineages = 0
        self.profile: Optional[ProfileSnapshot] = None
        self.n_requests = 0
        self.model_latency_ns = 0
        self.prepare_ns = 0
        self.parse_ns = 0
        self.verify_wall_ns = 0
        self.verify_cpu_s = 0.0
        self.patch_attempts = 0
        self.repairs = 0
        self.accepted_repairs = 0
        self.termination = ""
        self.catalog_order_rng = random.Random(f"catalog|{spec.seed}|{task.task_id}")

    def key(self, name: str) -> str:
        return f"{name}@{self.version}"

    def control_failure(self, role: str, reason: str, **extra: Any) -> None:
        self.control_failures.append({"role": role, "reason": reason, **extra})

    def failure_reasons(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for f in self.control_failures:
            out[f["role"]][f["reason"]] += 1
        return {r: dict(v) for r, v in out.items()}


class ChainController:
    def __init__(self, cfg: ChainConfig, verifier: Verifier, router: RoleRouter, observer=None):
        self.cfg = cfg
        self.verifier = verifier
        self.router = router
        self.observer = observer        # training-construction hook (trusted side): receives on-policy states
        self._menus: Dict[Tuple[str, str], R.ProbeMenu] = {}

    def _observe(self, kind: str, **rec: Any) -> None:
        if self.observer is not None:
            try:
                self.observer({"kind": kind, **rec})
            except Exception:  # noqa: BLE001 - data collection must never change the episode
                pass

    # ================================================================== #
    # episode wrapper
    # ================================================================== #
    def run_episode(self, task: PublicTask, private: PrivateSuite, spec: EpisodeSpec) -> Dict[str, Any]:
        if private.task_id != task.task_id:
            raise ValueError("private suite does not belong to the task")
        online = private.online()          # the search never holds audit constructors, seeds or provenance
        log = EventLog(spec.episode_id, spec.events_path)
        log.stamp(bundle_sha256=self.router.bundle_hash, system_id=self.router.system_id, task_id=task.task_id)
        st = _State(task, spec, self.cfg)
        budget = BudgetLedger(self.cfg.budget, patch_slots=spec.patch_slots)
        log.emit("episode_start", cluster_id=task.cluster_id, mode=spec.mode, seed=spec.seed,
                 original_sha256=st.original.hash, display_ids=st.display, budget=budget.snapshot(),
                 verifier_identity_sha256=self.verifier.identity_sha256)
        t0 = time.monotonic_ns()
        infra_error = None
        try:
            try:
                self._search(st, task, online, spec, budget, log)
            except SearchStop as e:
                st.termination = getattr(e, "reason", "stopped")
                log.emit("search_stop", reason=st.termination, detail=str(e))
            except InfrastructureError as e:
                st.termination = "infrastructure_failure"
                infra_error = str(e)
                log.emit("infrastructure_failure", detail=str(e)[:500])
            except Exception as e:  # noqa: BLE001 - a harness bug is a failed run, never a candidate label
                st.termination = "harness_error"
                infra_error = f"harness: {type(e).__name__}: {e}"
                log.emit("infrastructure_failure", detail=infra_error[:500], traceback=traceback.format_exc()[-1500:])
            search_ns = time.monotonic_ns() - t0
            selected = st.best
            log.emit("search_end", selected_sha256=selected.hash, termination=st.termination, budget=budget.snapshot(),
                     **EventLog.span(t0))
            audit = None
            audit_ns = 0
            if spec.run_audit and infra_error is None:
                # no model call, alternative selection or repair after this point; the full private suite is used
                # here and nowhere else
                ta = time.monotonic_ns()
                try:
                    audit = self.verifier.final_audit(task, private, selected, seed=spec.seed)
                except InfrastructureError as e:
                    infra_error = f"audit: {e}"
                audit_ns = time.monotonic_ns() - ta
                log.emit("final_audit", result=audit.to_dict() if audit else None, error=infra_error, **EventLog.span(ta))
            used = budget.snapshot()
            rec = {
                "schema_version": EPISODE_SCHEMA,
                "episode_id": spec.episode_id, "task_id": task.task_id, "cluster_id": task.cluster_id,
                "system_id": self.router.system_id, "bundle_sha256": self.router.bundle_hash, "mode": spec.mode,
                "seed": spec.seed, "original_sha256": st.original.hash, "selected_sha256": selected.hash,
                "termination_reason": st.termination or "unknown",
                "patch_attempts": used["patch_slots"], "patch_slots_max": used["patch_slots_max"],
                "role_calls": used["role_calls"], "calls_by_role": used["calls_by_role"],
                "input_tokens": used["input_tokens"], "output_tokens": used["output_tokens"],
                "extra_probe_calls": used["extra_probe_calls"], "profiles": used["profiles"],
                "accepted_edits": len(st.accepted), "accepted": st.accepted, "repairs": st.repairs,
                "accepted_repairs": st.accepted_repairs,
                "control_failures": len(st.control_failures), "control_failure_reasons": st.failure_reasons(),
                "search_wall_ns": search_ns, "audit_wall_ns": audit_ns,
                "model_latency_ns": st.model_latency_ns, "prepare_ns": st.prepare_ns, "parse_ns": st.parse_ns,
                "verify_wall_ns": st.verify_wall_ns, "verify_cpu_s": st.verify_cpu_s,
                "audit_pass": audit.audit_pass if audit else None,
                "audit_unchanged": audit.unchanged if audit else None,
                "raw_audit_speedup": audit.raw_speedup if audit else None,
                "audit_lo": audit.lo if audit else None, "audit_hi": audit.hi if audit else None,
                "effective_speedup": audit.effective_speedup if audit else None,
                "success_at_1_5": audit.success if audit else None,
                "audit_failure": audit.failure if audit else None,
                "failed_run": infra_error is not None, "infrastructure_error": infra_error,
                "host_peak_rss_mb": _peak_rss_mb(), "verifier_identity_sha256": self.verifier.identity_sha256,
                "events_path": spec.events_path, "selected_source": selected.source,
            }
            log.emit("episode_end", record={k: v for k, v in rec.items() if k not in ("selected_source", "accepted")})
            log.flush()
        finally:
            log.close()
        rec["events"] = None if spec.events_path else log.events
        return rec

    # ================================================================== #
    # role calls
    # ================================================================== #
    def _envelope(self, st: _State, role: str, prep: Prepared, budget: BudgetLedger) -> RequestEnvelope:
        st.n_requests += 1
        d = self.cfg.roles.get(role)
        return RequestEnvelope(
            role=role, task_id=st.task.task_id, request_id=f"{st.task.task_id}:{st.n_requests}", program_hash=st.best.hash,
            bundle_hash=self.router.bundle_hash, schema_version=SCHEMA_VERSIONS[role], max_new_tokens=prep.max_new_tokens,
            deadline_monotonic=budget.deadline if budget.deadline is not None else float("inf"),
            decoding=(("mode", d.mode), ("temperature", d.temperature), ("top_k", d.top_k), ("top_p", d.top_p)),
            allowed_actions=ALLOWED_ACTIONS[role])

    def _call(self, st: _State, role: str, view: Any, ctx: Dict[str, Any], budget: BudgetLedger, log: EventLog) -> Optional[RoleResponse]:
        """Prepare, reserve, execute, commit, log.  Returns None on a predeclared context-overflow outcome."""
        adapter = self.router[role]
        before = budget.snapshot()
        tp = time.monotonic_ns()
        try:
            prep = adapter.prepare(role, view, ctx)
        except ContextOverflow as e:
            # predeclared rule: the call is charged (role call, and a patch slot for M3/M5) with no tokens, and its
            # outcome is "unsupported" (M1: stop, M2: block target, M3/M5: abandon, M4: no extra probes)
            res = budget.reserve_call(role, 0, 0)
            budget.commit_call(res, 0)
            log.emit("role_call", role=role, adapter=adapter.kind, status="context_overflow", detail=str(e),
                     program_sha256=st.best.hash, budget_before=before, budget_after=budget.snapshot(), **EventLog.span(tp))
            return None
        prepare_ns = time.monotonic_ns() - tp
        st.prepare_ns += prepare_ns
        res = budget.reserve_call(role, prep.prompt_tokens, prep.max_new_tokens)
        env = self._envelope(st, role, prep, budget)
        t0 = time.monotonic_ns()
        try:
            resp = adapter.execute(prep, env)
        except Exception as e:  # noqa: BLE001 - an adapter crash is a logged failed call, never an uncharged retry
            budget.charge_failed_call(res)
            log.emit("role_call", role=role, adapter=adapter.kind, request_id=env.request_id, status="adapter_error",
                     detail=repr(e)[:300], program_sha256=st.best.hash, budget_before=before, budget_after=budget.snapshot(),
                     prepare_ns=prepare_ns, **EventLog.span(t0))
            return RoleResponse(role, env.request_id, "", ParseResult(False, error="adapter_error"), adapter.kind, error=repr(e)[:200])
        budget.commit_call(res, resp.output_tokens, prompt_tokens=prep.prompt_tokens)
        lat = time.monotonic_ns() - t0
        st.model_latency_ns += lat
        parse_ns = int(resp.meta.get("parse_ns", 0) or 0)
        st.parse_ns += parse_ns
        status = "ok" if resp.ok else (resp.error or "malformed")
        log.emit("role_call", role=role, adapter=adapter.kind, request_id=env.request_id, status=status,
                 action=self._action_name(role, resp), prompt_tokens=prep.prompt_tokens, output_tokens=resp.output_tokens,
                 finished=resp.finished, dropped_sections=prep.dropped_sections, text=resp.text[:2000],
                 parse_error=resp.parsed.error, latency_ns=lat, prepare_ns=prepare_ns, parse_ns=parse_ns,
                 program_sha256=st.best.hash, prompt_sha256=sha256_text(prep.prompt_text) if prep.prompt_text else None,
                 budget_before=before, budget_after=budget.snapshot(),
                 engine={k: resp.meta[k] for k in ("prefill_ns", "decode_ns", "queue_wait_ns", "batch", "seed", "timed_out")
                         if k in resp.meta},
                 **EventLog.span(t0))
        return resp

    @staticmethod
    def _action_name(role: str, resp: RoleResponse) -> str:
        if not resp.parsed.ok:
            return "malformed"
        p = resp.parsed.payload
        if role == "M1":
            return "NO_TARGET" if p.is_no_target else "TARGET"
        if role == "M2":
            return "NO_PLAN" if p.is_no_plan else "PLAN"
        if role in ("M3", "M5"):
            return "ABANDON" if p.is_abandon else "EDIT"
        return "PROBES_NONE" if not p.probe_ids else "PROBES"

    # ================================================================== #
    # views
    # ================================================================== #
    def _targetable(self, st: _State) -> List[str]:
        out = []
        for name in st.task.functions:
            if st.key(name) in st.blocked:
                continue
            if len(st.attempted.get(st.key(name), [])) >= self.cfg.budget.max_plans_per_function_version:
                continue
            out.append(name)
        return out

    def _m1(self, st: _State, names: List[str], budget: BudgetLedger) -> Tuple[M1View, Dict[str, Any]]:
        entries, ctx_cat = [], []
        order = list(names)
        st.catalog_order_rng.shuffle(order)
        for name in order:
            fn = st.best.function_node(name)
            src = st.best.function_source(name)
            prof = st.profile.functions.get(name) if st.profile else None
            loops = len(R.anchors(fn))
            sig = src.splitlines()[0] if src else name
            fc = st.task.contract.function(name)
            has_plan = bool(R.static_plans(fn, fc, limit=1))
            entries.append(CatalogEntry(
                st.display[name], name, src, share_bucket(prof.self_share) if prof else 0,
                share_bucket(prof.cum_share) if prof else 0, calls_bucket(prof.ncalls) if prof else 0,
                len(src.splitlines()), loops, len(st.attempted.get(st.key(name), [])),
                summary=f"{sig}\n    # lines={len(src.splitlines())} loops={R.render_anchors(fn)}"))
            ctx_cat.append({"display_id": st.display[name], "name": name, "self_share": prof.self_share if prof else 0.0,
                            "cum_share": prof.cum_share if prof else 0.0, "ncalls": prof.ncalls if prof else 0,
                            "has_plan": has_plan, "tried": len(st.attempted.get(st.key(name), []))})
        view = M1View(entries, list(st.history_global), budget.max_patch_slots - budget.used.patch_slots,
                      rename=dict(st.display))
        return view, {"catalog": ctx_cat, "history": list(st.history_global)}

    def _legacy_state(self, st: _State, budget: BudgetLedger) -> Dict[str, Any]:
        """The legacy optimizer's compact <STATE> (runtimes relative to the original, from accepted point estimates)."""
        best = 1.0
        for a in st.accepted:
            best /= max(1e-9, float(a["speedup"]))
        last = st.history_global[-1].rsplit(":", 1)[-1] if st.history_global else "start"
        return {"best_runtime": best, "last_runtime": "n/a", "last_correct": -1 if last == "start" else int(last == "accepted"),
                "last_status": {"accepted": "improved"}.get(last, last), "step": budget.used.patch_slots,
                "remaining": budget.max_patch_slots - budget.used.patch_slots}

    # ================================================================== #
    # search
    # ================================================================== #
    def _search(self, st: _State, task: PublicTask, online: OnlineSuite, spec: EpisodeSpec, budget: BudgetLedger,
                log: EventLog) -> None:
        self._profile(st, task, budget, log)
        while True:
            if not budget.can_request_patch(1, 1):
                st.termination = "patch_budget" if budget.used.patch_slots >= budget.max_patch_slots else "budget"
                return
            names = self._targetable(st)
            if not names:
                st.termination = "no_eligible_target"
                return
            # ---------------- M1 ----------------
            view, ctx = self._m1(st, names, budget)
            r1 = self._call(st, "M1", view, ctx, budget, log)
            if r1 is None:
                st.termination = "context_overflow_m1"
                return
            if not r1.ok:
                st.control_failure("M1", "malformed", error=r1.error or r1.parsed.error)
                st.termination = "invalid_target"
                return
            choice: TargetChoice = r1.parsed.payload
            if choice.is_no_target:
                st.termination = "model_abstained"
                return
            name = st.by_display.get(choice.function_id)
            self._observe("chain_target_state", task_id=task.task_id, state_source=st.best.source, view=view,
                          ctx=ctx, chosen=choice.function_id, chosen_name=name, request_id=r1.request_id,
                          token_ids=r1.token_ids, parse_ok=r1.parsed.ok)
            if name is None or name not in names:
                st.control_failure("M1", "invalid_target", id=choice.function_id)
                st.termination = "invalid_target"
                log.emit("control_failure", role="M1", reason="invalid_target", detail=f"invalid target {choice.function_id}")
                return
            fid = st.display[name]
            # ---------------- M2 ----------------
            fn = st.best.function_node(name)
            fsrc = st.best.function_source(name)
            fcontract = task.contract.function(name)
            closure = dependency_closure(st.best, name)
            if closure.unsupported:
                st.blocked.add(st.key(name))
                log.emit("target_blocked", function=name, reason="unsupported_closure", detail=closure.unsupported)
                continue
            deps = dependency_sources(st.best, closure)
            closure_sha = closure.fingerprint(st.best)
            v2 = M2View(fid, fsrc, deps, task.contract.render(name), R.render_anchors(fn), R.operand_names(fn),
                        R.applicable_mechanisms(fn), list(st.history_fn.get(name, [])))
            attempted = st.attempted.setdefault(st.key(name), [])
            r2 = self._call(st, "M2", v2, {"function_id": fid, "fn_source": fsrc, "fcontract": fcontract,
                                           "attempted": list(attempted)}, budget, log)
            if r2 is None:
                st.blocked.add(st.key(name))
                continue
            if not r2.ok:
                st.control_failure("M2", "malformed", error=r2.error or r2.parsed.error)
                st.blocked.add(st.key(name))
                st.history_global.append(f"{fid}:plan:malformed")
                log.emit("control_failure", role="M2", reason="malformed", detail=r2.parsed.error or r2.error)
                continue
            plan: Plan = r2.parsed.payload
            self._observe("chain_plan_state", task_id=task.task_id, state_source=st.best.source, function=name,
                          view=v2, plan=format_plan(plan), closure_sha256=closure_sha, request_id=r2.request_id,
                          token_ids=r2.token_ids, parse_ok=r2.parsed.ok)
            if plan.is_no_plan:
                st.blocked.add(st.key(name))
                st.history_global.append(f"{fid}:no_plan")
                continue
            chk = R.validate_plan(plan, fn, fcontract)
            if not chk.ok or plan.key() in attempted:
                reason = chk.reason if not chk.ok else "duplicate plan"
                st.control_failure("M2", chk.kind if not chk.ok else "duplicate_plan", detail=reason)
                st.blocked.add(st.key(name))
                st.history_global.append(f"{fid}:{plan.transform_id}:invalid_plan")
                log.emit("control_failure", role="M2", reason=chk.kind if not chk.ok else "duplicate_plan", detail=reason,
                         plan=format_plan(plan))
                continue
            # ---------------- M3 (fresh write) ----------------
            v3 = M3View(fid, fsrc, deps, task.contract.render(name), format_plan(plan), R.render_anchors(fn))
            base_hash = st.best.hash
            r3 = self._call(st, "M3", v3, {"function_id": fid, "function_name": name, "fn_source": fsrc, "plan": plan,
                                           "file_source": st.best.source, "legacy_state": self._legacy_state(st, budget)},
                            budget, log)
            attempted.append(plan.key())
            st.patch_attempts += 1
            st.lineages += 1
            lineage = st.lineages
            verdict = self._patch_outcome(st, task, online, spec, budget, log, r3, name, plan, lineage, "M3", base_hash,
                                          closure_sha)
            if verdict is None:
                continue
            # ---------------- M5 (``max_repairs_per_lineage`` repairs per lineage) ----------------
            repairs_left = self.cfg.budget.max_repairs_per_lineage
            while verdict.repairable and repairs_left > 0 and budget.can_request_patch(1, 1):
                repairs_left -= 1
                fb = verdict.feedback or {"failure_kind": verdict.category or verdict.kind}
                v5 = M5View(fid, fsrc, verdict.fn_source, deps, task.contract.render(name), format_plan(plan), fb)
                r5 = self._call(st, "M5", v5, {"function_id": fid, "fn_source": fsrc, "plan": plan,
                                               "failed_source": verdict.fn_source, "feedback": fb}, budget, log)
                st.repairs += 1
                st.patch_attempts += 1
                rv = self._patch_outcome(st, task, online, spec, budget, log, r5, name, plan, lineage, "M5", base_hash,
                                         closure_sha)
                if rv is None:
                    break
                verdict = rv
                if verdict.accept:
                    st.accepted_repairs += 1
            if verdict.accept:
                self._commit(st, task, budget, log, name, plan, verdict)

    # ------------------------------------------------------------------ #
    def _patch_outcome(self, st: _State, task: PublicTask, online: OnlineSuite, spec: EpisodeSpec, budget: BudgetLedger,
                       log: EventLog, resp: Optional[RoleResponse], name: str, plan: Plan, lineage: int,
                       role: str, base_hash: str, closure_sha: str) -> Optional[Verdict]:
        fid = st.display[name]
        if resp is None:
            outcome = "context_overflow"
        elif not resp.ok:
            outcome = "malformed" if not resp.error else resp.error
        elif resp.parsed.payload.is_abandon:
            outcome = "abandon"
        else:
            outcome = ""
        if outcome:
            self._record(st, fid, name, plan, outcome, role)
            log.emit("patch_outcome", role=role, lineage=lineage, outcome=outcome, function=name)
            self._observe("chain_candidate", task_id=task.task_id, state_source=st.best.source, function=name,
                          function_id=fid, plan=format_plan(plan), role=role, lineage=lineage, code=None, accepted=False,
                          stage="action", check_kind=outcome, category=outcome, repairable=False, public_feedback=None,
                          speedup=None, candidate_source=None, closure_sha256=closure_sha, base_sha256=base_hash,
                          request_id=resp.request_id if resp else None, token_ids=resp.token_ids if resp else None,
                          parse_ok=bool(resp and resp.parsed.ok), parse_error=resp.parsed.error if resp else "context_overflow")
            return None
        base_source = st.best.source
        v = self._assess(st, task, online, spec, budget, log, resp.parsed.payload.code, name, plan, lineage, role, base_hash,
                         closure_sha)
        self._record(st, fid, name, plan, v.outcome, role)
        self._observe("chain_candidate", task_id=task.task_id, state_source=base_source, function=name,
                      function_id=fid, plan=format_plan(plan), role=role, lineage=lineage, code=resp.parsed.payload.code,
                      accepted=v.accept, stage=v.stage, check_kind=v.kind, category=v.category, repairable=v.repairable,
                      public_feedback=v.feedback if v.stage in ("scope", "smoke", "probe") else {"failure_kind": v.category},
                      speedup=v.timing.verdict.speedup if v.timing and v.timing.verdict else None,
                      candidate_source=v.candidate.source if v.candidate else None, closure_sha256=closure_sha,
                      base_sha256=base_hash, request_id=resp.request_id, token_ids=resp.token_ids, parse_ok=True,
                      parse_error="", timing_blocks={"ref_ns": v.timing.ref_ns, "cand_ns": v.timing.cand_ns} if v.timing else None)
        log.emit("patch_outcome", role=role, lineage=lineage, outcome=v.outcome, function=name, stage=v.stage,
                 repairable=v.repairable)
        return v

    def _record(self, st: _State, fid: str, name: str, plan: Plan, outcome: str, role: str) -> None:
        tag = "repair" if role == "M5" else "write"
        st.history_global.append(f"{fid}:{plan.transform_id}:{tag}:{outcome}")
        ops = ";".join(plan.operands)
        st.history_fn.setdefault(name, []).append(f"{plan.transform_id}({ops}):{tag}:{outcome}")

    # ------------------------------------------------------------------ #
    def _menu(self, task: PublicTask, name: str) -> R.ProbeMenu:
        key = (task.task_id, name)
        if key not in self._menus:
            self._menus[key] = R.probe_menu(task.contract, task.task_id, name, self.cfg.data.menu_size)
        return self._menus[key]

    def _stage(self, st: _State, log: EventLog, r: CheckResult, budget: BudgetLedger, t0: int) -> None:
        st.verify_wall_ns += r.wall_ns
        st.verify_cpu_s += r.cpu_s
        log.emit("verify", stage=r.stage, ok=r.ok, check_kind=r.kind, category=r.category, n_checked=r.n_checked,
                 wall_ns=r.wall_ns, cpu_s=r.cpu_s, public_feedback=r.public_feedback, budget=budget.snapshot(), **EventLog.span(t0))

    def _assess(self, st: _State, task: PublicTask, online: OnlineSuite, spec: EpisodeSpec, budget: BudgetLedger,
                log: EventLog, code: str, name: str, plan: Plan, lineage: int, role: str, base_hash: str,
                closure_sha: str) -> Verdict:
        rep = set(self.cfg.verification.repairable_failures)
        dl = budget.deadline
        # 1. scope / static - the patch applies to the *recorded* base version of the request, which is still the
        #    current best (the controller is sequential; the check is explicit so that any future concurrency or a
        #    stale replayed patch fails loudly here, not silently at commit)
        ts = time.monotonic_ns()
        try:
            cand = apply_patch(st.best, base_hash, name, code, task.contract.permitted_imports)
        except PatchError as e:
            log.emit("verify", stage="scope", ok=False, check_kind=e.kind, category=e.kind, detail=e.message[:300],
                     budget=budget.snapshot(), **EventLog.span(ts))
            fb = {"failure_kind": e.kind, "message": e.message[:self.cfg.verification.max_feedback_chars]}
            return Verdict(False, e.kind == "syntax_error" and "syntax_error" in rep, "scope", e.kind, e.kind, fb,
                           fn_source=code, base_hash=base_hash)
        fn_src = cand.function_source(name)
        log.emit("verify", stage="scope", ok=True, candidate_sha256=cand.hash, base_sha256=base_hash,
                 closure_sha256=closure_sha, budget=budget.snapshot(), **EventLog.span(ts))
        if cand.hash == st.best.hash or cand.hash in st.rejected_hashes:
            log.emit("verify", stage="duplicate", ok=False, candidate_sha256=cand.hash)
            return Verdict(False, False, "duplicate", "duplicate", "duplicate", None, cand, fn_source=fn_src, base_hash=base_hash)

        def reject(stage: str, r: CheckResult, public: bool) -> Verdict:
            st.rejected_hashes.add(cand.hash)
            cat = r.kind
            repairable = cat in rep
            fb = r.public_feedback if public else {"failure_kind": r.category or r.kind}
            return Verdict(False, repairable, stage, r.kind, r.category or r.kind, fb, cand, fn_source=fn_src, base_hash=base_hash)

        # 2. quick public checks
        t1 = time.monotonic_ns()
        sm = self.verifier.smoke_check(task, cand, name, deadline=dl)
        self._stage(st, log, sm, budget, t1)
        if not sm.ok:
            return reject("smoke", sm, True)
        # 3. learned / rule probe selection (optional, budgeted)
        menu = self._menu(task, name)
        if menu.entries and budget.probes_left() > 0 and budget.can_call("M4"):
            cur = st.best.function_source(name)
            v4 = M4View(st.display[name], cur, fn_src, task.contract.render(name), format_plan(plan), menu.render(),
                        compact_diff(cur, fn_src))
            r4 = self._call(st, "M4", v4, {"function_id": st.display[name], "menu_ids": menu.ids(), "plan": plan}, budget, log)
            chosen: List[str] = []
            if r4 is not None and r4.ok:
                ids = list(r4.parsed.payload.probe_ids)
                if len(set(ids)) != len(ids) or len(ids) > self.cfg.budget.max_probes_per_candidate or \
                        any(p not in menu.inputs for p in ids):
                    # an invalid selection supplies no extra probes; the role call was charged, nothing else
                    reason = "duplicate_probe" if len(set(ids)) != len(ids) else ("too_many_probes" if len(ids) > self.cfg.budget.max_probes_per_candidate else "unknown_probe")
                    st.control_failure("M4", reason, ids=ids)
                    log.emit("control_failure", role="M4", reason=reason, detail=f"invalid probe selection {ids}")
                else:
                    granted = budget.reserve_probes(len(ids))
                    chosen = ids[:granted]
            elif r4 is not None:
                st.control_failure("M4", "malformed", error=r4.error or r4.parsed.error)
                log.emit("control_failure", role="M4", reason="malformed", detail=r4.parsed.error or r4.error)
            self._observe("chain_probe_state", task_id=task.task_id, function=name, function_id=st.display[name],
                          candidate_source=cand.source, chosen=chosen, menu_ids=menu.ids(), lineage=lineage, role=role,
                          request_id=r4.request_id if r4 else None, token_ids=r4.token_ids if r4 else None,
                          parse_ok=bool(r4 and r4.parsed.ok))
            if chosen:
                t2 = time.monotonic_ns()
                pr = self.verifier.probe_check(task, cand, name, [(p, menu.inputs[p]) for p in chosen], deadline=dl)
                self._stage(st, log, pr, budget, t2)
                if not pr.ok:
                    return reject("probe", pr, True)
        # 4. mandatory private checks (category-only feedback)
        t3 = time.monotonic_ns()
        mc = self.verifier.mandatory_checks(task, online, cand, deadline=dl)
        self._stage(st, log, self.verifier.redacted(mc), budget, t3)
        if not mc.ok:
            return reject("mandatory", self.verifier.redacted(mc), False)
        # 5. paired timing against the current best
        seed = spec.seed * 100003 + st.n_requests
        t4 = time.monotonic_ns()
        tr = self.verifier.paired_timing(task, online.online_timing, st.best, cand, blocks=self.verifier.blocks_for(spec.mode),
                                         seed=seed, deadline=dl)
        st.verify_wall_ns += tr.wall_ns
        log.emit("timing", ok=tr.ok, status=tr.status, reps=tr.reps, warmup=tr.warmup, seed=seed, ref_ns=tr.ref_ns,
                 cand_ns=tr.cand_ns, orders=tr.orders, verdict=tr.verdict.to_dict() if tr.verdict else None, wall_ns=tr.wall_ns,
                 reference_sha256=tr.reference_sha256, candidate_sha256=tr.candidate_sha256, environment=tr.environment,
                 launch=tr.launch, budget=budget.snapshot(), **EventLog.span(t4))
        if not tr.ok or tr.verdict is None:
            st.rejected_hashes.add(cand.hash)
            return Verdict(False, False, "timing", "timing_error", tr.status, None, cand, tr, fn_source=fn_src, base_hash=base_hash)
        extra = self.cfg.verification.confirmation_extra_blocks
        if tr.verdict.decision == "uncertain" and extra > 0:
            t5 = time.monotonic_ns()
            tr2 = self.verifier.paired_timing(task, online.online_timing, st.best, cand, blocks=extra, seed=seed + 7919,
                                              deadline=dl)
            st.verify_wall_ns += tr2.wall_ns
            if tr2.ok:
                from tinyperf_chain.sandbox.timing import timing_verdict
                v = self.cfg.verification
                tr.ref_ns, tr.cand_ns = tr.ref_ns + tr2.ref_ns, tr.cand_ns + tr2.cand_ns
                tr.verdict = timing_verdict(tr.ref_ns, tr.cand_ns, threshold=v.accept_incremental_speedup, rule=v.accept_rule,
                                            draws=v.timing_bootstrap_draws, level=v.timing_interval, seed=seed,
                                            legacy_delta=v.legacy_delta)
            log.emit("timing_confirmation", ok=tr2.ok, extra_blocks=extra, verdict=tr.verdict.to_dict(), budget=budget.snapshot(),
                     **EventLog.span(t5))
        if not tr.verdict.accepted:
            st.rejected_hashes.add(cand.hash)
            return Verdict(False, False, "timing", tr.verdict.decision, tr.verdict.decision, None, cand, tr, fn_source=fn_src,
                           base_hash=base_hash)
        return Verdict(True, False, "timing", "accept", "accepted", None, cand, tr, fn_source=fn_src, base_hash=base_hash)

    # ------------------------------------------------------------------ #
    def _commit(self, st: _State, task: PublicTask, budget: BudgetLedger, log: EventLog, name: str, plan: Plan,
                v: Verdict) -> None:
        cand = v.candidate
        assert cand is not None and v.timing is not None and v.timing.verdict is not None
        # commit_if_base_matches (12.3): the candidate was produced for v.base_hash, which must still be the best
        if v.base_hash != st.best.hash:
            st.control_failure("controller", "stale_base", base=v.base_hash, current=st.best.hash)
            log.emit("control_failure", role="controller", reason="stale_base",
                     detail="candidate base version is no longer the current best; not committed")
            return
        if cand.functions != st.best.functions:
            raise RuntimeError("candidate program does not belong to this task")
        prev = st.best.hash
        st.best = cand
        st.version += 1
        st.rejected_hashes = set()
        tv = v.timing.verdict
        st.accepted.append({"function": name, "plan": format_plan(plan), "speedup": tv.speedup, "lo": tv.lo, "hi": tv.hi,
                            "base_sha256": prev, "new_sha256": cand.hash, "version": st.version})
        log.emit("accept", function=name, plan=format_plan(plan), speedup=tv.speedup, lo=tv.lo, hi=tv.hi,
                 base_sha256=prev, new_sha256=cand.hash, version=st.version, budget=budget.snapshot())
        self._profile(st, task, budget, log)

    def _profile(self, st: _State, task: PublicTask, budget: BudgetLedger, log: EventLog) -> None:
        if not budget.reserve_profile():
            log.emit("profile", ok=False, reason="profile_budget")
            return
        budget.check_deadline("before profiling")
        t0 = time.monotonic_ns()
        snap = self.verifier.profile(task, st.best, deadline=budget.deadline)
        st.profile = snap if snap.ok else st.profile
        log.emit("profile", ok=snap.ok, program_sha256=st.best.hash, message=snap.message,
                 functions={k: {"ncalls": v.ncalls, "self_share": round(v.self_share, 4), "cum_share": round(v.cum_share, 4)}
                            for k, v in snap.functions.items()}, budget=budget.snapshot(), **EventLog.span(t0))
