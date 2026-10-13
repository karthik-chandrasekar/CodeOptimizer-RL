"""B0: the untouched TinyPerf-RL optimizer under the chain's budget, verifier and final audit.

The legacy model sees its own whole-file observation and replies with ``<EDIT>`` (one or more functions, spliced by
name with the legacy splice rule) or ``<STOP>``.  Every action consumes a patch slot and its tokens; candidates go
through the chain's mandatory checks against the immutable original and the same paired timing / acceptance rule,
and the selected program gets the same independent final audit.  The record schema matches
:mod:`tinyperf_chain.controller`, so B0 sits in the same comparison tables.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from tinyperf.env.files import splice_functions
from tinyperf.env.protocol import EDIT_CLOSE, STOP_TOKEN, format_observation, parse_action
from tinyperf_chain.adapters.model import request_seed
from tinyperf_chain.adapters.verifier import InfrastructureError, Verifier
from tinyperf_chain.budget import BudgetLedger, SearchStop
from tinyperf_chain.config import ChainConfig
from tinyperf_chain.controller import EPISODE_SCHEMA, EpisodeSpec, _peak_rss_mb
from tinyperf_chain.ledger import EventLog
from tinyperf_chain.program import Program, normalize_source
from tinyperf_chain.serve.engine import RoleEngine, SamplingSpec
from tinyperf_chain.tasks import PrivateSuite, PublicTask


class LegacyDirectEditSystem:
    def __init__(self, cfg: ChainConfig, verifier: Verifier, engine: RoleEngine, tok, *, system_id: str = "B0",
                 bundle_hash: str = "legacy", temperature: float = 0.6, max_new: int = 768):
        self.cfg, self.verifier, self.engine, self.tok = cfg, verifier, engine, tok
        self.system_id, self.bundle_hash = system_id, bundle_hash
        self.spec = SamplingSpec(greedy=False, temperature=temperature)
        self.max_new = max_new
        self.stop = [tok.token_id(EDIT_CLOSE), tok.token_id(STOP_TOKEN), tok.eos_id]

    def describe(self) -> Dict[str, Any]:
        return {"kind": "legacy_direct_edit", "checkpoint_sha256": self.engine.checkpoint_sha256,
                "decoding": {"mode": "sample", "temperature": self.spec.temperature, "top_k": self.spec.top_k,
                             "top_p": self.spec.top_p, "max_output": self.max_new}, "protocol": "legacy_whole_file_v0"}

    def run_episode(self, task: PublicTask, private: PrivateSuite, spec: EpisodeSpec) -> Dict[str, Any]:
        """Same accounting as the chain controller (H4): every model call, prompt token, check and timing block is
        charged and logged with its latency, so B0's cost is comparable with B1-B6."""
        log = EventLog(spec.episode_id, spec.events_path)
        log.stamp(bundle_sha256=self.bundle_hash, system_id=self.system_id, task_id=task.task_id)
        budget = BudgetLedger(self.cfg.budget, patch_slots=spec.patch_slots)
        online = private.online()
        p0 = task.original()
        best, best_ratio = p0, 1.0
        last = {"last_runtime": "n/a", "last_correct": -1, "last_status": "start"}
        accepted = []
        failures: Dict[str, int] = {}
        termination, infra = "", None
        model_ns = verify_ns = prepare_ns = parse_ns = 0
        verify_cpu = 0.0
        log.emit("episode_start", cluster_id=task.cluster_id, mode=spec.mode, seed=spec.seed, original_sha256=p0.hash,
                 budget=budget.snapshot(), verifier_identity_sha256=self.verifier.identity_sha256)
        t0 = time.monotonic_ns()
        n = 0
        try:
            try:
                while budget.can_request_patch(1, 1):
                    n += 1
                    before = budget.snapshot()
                    tp = time.monotonic_ns()
                    state = {"best_runtime": best_ratio, **last, "step": budget.used.patch_slots,
                             "remaining": budget.max_patch_slots - budget.used.patch_slots}
                    obs = format_observation(best.source, state, feedback="full")
                    ids = self.tok.encode(obs, add_bos=True)
                    prepare_ns += time.monotonic_ns() - tp
                    if len(ids) + self.max_new > self.engine.max_seq_len:
                        res = budget.reserve_call("M3", 0, 0, label="legacy")
                        budget.commit_call(res, 0)
                        log.emit("role_call", role="legacy", status="context_overflow", program_sha256=best.hash,
                                 budget_before=before, budget_after=budget.snapshot(), **EventLog.span(tp))
                        termination = "context_overflow"
                        break
                    res = budget.reserve_call("M3", len(ids), self.max_new, label="legacy")
                    deadline = budget.deadline
                    tm = time.monotonic_ns()
                    out = self.engine.generate([ids], self.max_new, self.spec, self.stop, pad_id=self.tok.pad_id,
                                               seeds=[request_seed(f"{task.task_id}:{n}", spec.seed)], deadline=deadline)[0]
                    lat = time.monotonic_ns() - tm
                    model_ns += lat
                    budget.commit_call(res, len(out.ids))
                    tq = time.monotonic_ns()
                    text = self.tok.decode(out.ids)
                    act = parse_action(text)
                    pn = time.monotonic_ns() - tq
                    parse_ns += pn
                    status = "timeout" if out.timed_out else act.kind
                    log.emit("role_call", role="legacy", status=status, action=act.kind, text=text[:2000], prompt_tokens=len(ids),
                             output_tokens=len(out.ids), finished=out.finished, latency_ns=lat, parse_ns=pn,
                             program_sha256=best.hash, budget_before=before, budget_after=budget.snapshot(),
                             engine={"prefill_ns": out.prefill_ns, "decode_ns": out.decode_ns, "batch": out.batch_size},
                             **EventLog.span(tm))
                    if act.kind == "stop":
                        termination = "model_stopped"
                        break
                    if act.kind != "edit" or not act.code:
                        failures["malformed"] = failures.get("malformed", 0) + 1
                        last = {"last_runtime": "n/a", "last_correct": 0, "last_status": "malformed"}
                        continue
                    try:
                        cand = Program(normalize_source(splice_functions(best.source, act.code)), best.functions)
                    except (SyntaxError, ValueError, RecursionError, MemoryError):
                        failures["syntax_error"] = failures.get("syntax_error", 0) + 1
                        last = {"last_runtime": "n/a", "last_correct": 0, "last_status": "syntax_error"}
                        continue
                    tv = time.monotonic_ns()
                    mc = self.verifier.mandatory_checks(task, online, cand, deadline=budget.deadline)
                    verify_ns += mc.wall_ns
                    verify_cpu += mc.cpu_s
                    red = self.verifier.redacted(mc)
                    log.emit("verify", stage="mandatory", ok=red.ok, check_kind=red.kind, category=red.category,
                             n_checked=red.n_checked, wall_ns=red.wall_ns, cpu_s=red.cpu_s, candidate_sha256=cand.hash,
                             budget=budget.snapshot(), **EventLog.span(tv))
                    if not mc.ok:
                        last = {"last_runtime": "n/a", "last_correct": 0,
                                "last_status": "incorrect" if mc.kind == "private_mismatch" else mc.kind}
                        continue
                    tt = time.monotonic_ns()
                    tr = self.verifier.paired_timing(task, online.online_timing, best, cand,
                                                     blocks=self.verifier.blocks_for(spec.mode), seed=spec.seed * 100003 + n,
                                                     deadline=budget.deadline)
                    verify_ns += tr.wall_ns
                    log.emit("timing", ok=tr.ok, status=tr.status, reps=tr.reps, warmup=tr.warmup, ref_ns=tr.ref_ns,
                             cand_ns=tr.cand_ns, orders=tr.orders, verdict=tr.verdict.to_dict() if tr.verdict else None,
                             wall_ns=tr.wall_ns, reference_sha256=tr.reference_sha256, candidate_sha256=tr.candidate_sha256,
                             environment=tr.environment, launch=tr.launch, budget=budget.snapshot(), **EventLog.span(tt))
                    if not tr.ok or tr.verdict is None:
                        last = {"last_runtime": "n/a", "last_correct": 0, "last_status": tr.status}
                        continue
                    ratio = best_ratio / max(1e-9, tr.verdict.speedup)
                    if tr.verdict.accepted:
                        accepted.append({"speedup": tr.verdict.speedup, "lo": tr.verdict.lo, "hi": tr.verdict.hi,
                                         "base_sha256": best.hash, "new_sha256": cand.hash})
                        prev = best.hash
                        best, best_ratio = cand, ratio
                        last = {"last_runtime": f"{ratio:.3f}", "last_correct": 1, "last_status": "improved"}
                        log.emit("accept", speedup=tr.verdict.speedup, lo=tr.verdict.lo, hi=tr.verdict.hi, base_sha256=prev,
                                 new_sha256=cand.hash, version=len(accepted), budget=budget.snapshot())
                    else:
                        last = {"last_runtime": f"{ratio:.3f}", "last_correct": 1,
                                "last_status": "slower" if tr.verdict.speedup < 1 else "no_gain"}
                else:
                    termination = "patch_budget"
            except SearchStop as e:
                termination = getattr(e, "reason", "stopped")
            except InfrastructureError as e:
                termination, infra = "infrastructure_failure", str(e)
            search_ns = time.monotonic_ns() - t0
            log.emit("search_end", selected_sha256=best.hash, termination=termination, budget=budget.snapshot(), **EventLog.span(t0))
            audit, audit_ns = None, 0
            if spec.run_audit and infra is None:
                ta = time.monotonic_ns()
                try:
                    audit = self.verifier.final_audit(task, private, best, seed=spec.seed)
                except InfrastructureError as e:
                    infra = f"audit: {e}"
                audit_ns = time.monotonic_ns() - ta
                log.emit("final_audit", result=audit.to_dict() if audit else None, error=infra, **EventLog.span(ta))
            u = budget.snapshot()
            rec = {
                "schema_version": EPISODE_SCHEMA, "episode_id": spec.episode_id, "task_id": task.task_id,
                "cluster_id": task.cluster_id, "system_id": self.system_id, "bundle_sha256": self.bundle_hash, "mode": spec.mode,
                "seed": spec.seed, "original_sha256": p0.hash, "selected_sha256": best.hash, "termination_reason": termination,
                "patch_attempts": u["patch_slots"], "patch_slots_max": u["patch_slots_max"], "role_calls": u["role_calls"],
                "calls_by_role": u["calls_by_role"], "input_tokens": u["input_tokens"], "output_tokens": u["output_tokens"],
                "extra_probe_calls": 0, "profiles": 0, "accepted_edits": len(accepted), "accepted": accepted, "repairs": 0,
                "accepted_repairs": 0, "control_failures": sum(failures.values()),
                "control_failure_reasons": {"legacy": failures} if failures else {},
                "search_wall_ns": search_ns, "audit_wall_ns": audit_ns,
                "model_latency_ns": model_ns, "prepare_ns": prepare_ns, "parse_ns": parse_ns,
                "verify_wall_ns": verify_ns, "verify_cpu_s": verify_cpu,
                "audit_pass": audit.audit_pass if audit else None, "audit_unchanged": audit.unchanged if audit else None,
                "raw_audit_speedup": audit.raw_speedup if audit else None, "audit_lo": audit.lo if audit else None,
                "audit_hi": audit.hi if audit else None, "effective_speedup": audit.effective_speedup if audit else None,
                "success_at_1_5": audit.success if audit else None, "audit_failure": audit.failure if audit else None,
                "failed_run": infra is not None, "infrastructure_error": infra, "events_path": spec.events_path,
                "host_peak_rss_mb": _peak_rss_mb(), "verifier_identity_sha256": self.verifier.identity_sha256,
                "selected_source": best.source,
            }
            log.emit("episode_end", record={k: v for k, v in rec.items() if k not in ("selected_source", "accepted")})
            log.flush()
        finally:
            log.close()
        rec["events"] = None if spec.events_path else log.events
        return rec
