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
from tinyperf_chain.controller import EPISODE_SCHEMA, EpisodeSpec
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

    def run_episode(self, task: PublicTask, private: PrivateSuite, spec: EpisodeSpec) -> Dict[str, Any]:
        log = EventLog(spec.episode_id, spec.events_path)
        budget = BudgetLedger(self.cfg.budget, patch_slots=spec.patch_slots)
        p0 = task.original()
        best, best_ratio = p0, 1.0
        last = {"last_runtime": "n/a", "last_correct": -1, "last_status": "start"}
        accepted = []
        termination, infra = "", None
        t0 = time.monotonic_ns()
        n = 0
        try:
            while budget.can_request_patch(1, 1):
                n += 1
                state = {"best_runtime": best_ratio, **last, "step": budget.used.patch_slots,
                         "remaining": budget.max_patch_slots - budget.used.patch_slots}
                obs = format_observation(best.source, state, feedback="full")
                ids = self.tok.encode(obs, add_bos=True)
                if len(ids) + self.max_new > self.engine.max_seq_len:
                    termination = "context_overflow"
                    break
                res = budget.reserve_call("M3", len(ids), self.max_new)
                out = self.engine.generate([ids], self.max_new, self.spec, self.stop, pad_id=self.tok.pad_id,
                                           seeds=[request_seed(f"{task.task_id}:{n}", spec.seed)])[0]
                budget.commit_call(res, len(out.ids))
                text = self.tok.decode(out.ids)
                act = parse_action(text)
                log.emit("role_call", role="legacy", status=act.kind, text=text[:2000], prompt_tokens=len(ids),
                         output_tokens=len(out.ids))
                if act.kind == "stop":
                    termination = "model_stopped"
                    break
                if act.kind != "edit" or not act.code:
                    last = {"last_runtime": "n/a", "last_correct": 0, "last_status": "malformed"}
                    continue
                try:
                    cand = Program(normalize_source(splice_functions(best.source, act.code)), best.functions)
                except (SyntaxError, ValueError, RecursionError, MemoryError):
                    last = {"last_runtime": "n/a", "last_correct": 0, "last_status": "syntax_error"}
                    continue
                mc = self.verifier.mandatory_checks(task, private, cand, deadline=budget.deadline)
                log.emit("verify", stage="mandatory", ok=mc.ok, check_kind=mc.kind, category=mc.category)
                if not mc.ok:
                    last = {"last_runtime": "n/a", "last_correct": 0,
                            "last_status": "incorrect" if mc.kind == "private_mismatch" else mc.kind}
                    continue
                tr = self.verifier.paired_timing(task, private.online_timing, best, cand,
                                                 blocks=self.verifier.blocks_for(spec.mode), seed=spec.seed * 100003 + n,
                                                 deadline=budget.deadline)
                log.emit("timing", ok=tr.ok, verdict=tr.verdict.to_dict() if tr.verdict else None)
                if not tr.ok or tr.verdict is None:
                    last = {"last_runtime": "n/a", "last_correct": 0, "last_status": tr.status}
                    continue
                ratio = best_ratio / max(1e-9, tr.verdict.speedup)
                if tr.verdict.accepted:
                    accepted.append({"speedup": tr.verdict.speedup, "lo": tr.verdict.lo, "hi": tr.verdict.hi,
                                     "base_sha256": best.hash, "new_sha256": cand.hash})
                    best, best_ratio = cand, ratio
                    last = {"last_runtime": f"{ratio:.3f}", "last_correct": 1, "last_status": "improved"}
                    log.emit("accept", speedup=tr.verdict.speedup, new_sha256=cand.hash)
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
        audit, audit_ns = None, 0
        if spec.run_audit and infra is None:
            ta = time.monotonic_ns()
            audit = self.verifier.final_audit(task, private, best, seed=spec.seed)
            audit_ns = time.monotonic_ns() - ta
        u = budget.snapshot()
        rec = {
            "schema_version": EPISODE_SCHEMA, "episode_id": spec.episode_id, "task_id": task.task_id,
            "cluster_id": task.cluster_id, "system_id": self.system_id, "bundle_sha256": self.bundle_hash, "mode": spec.mode,
            "seed": spec.seed, "original_sha256": p0.hash, "selected_sha256": best.hash, "termination_reason": termination,
            "patch_attempts": u["patch_slots"], "patch_slots_max": u["patch_slots_max"], "role_calls": u["role_calls"],
            "calls_by_role": u["calls_by_role"], "input_tokens": u["input_tokens"], "output_tokens": u["output_tokens"],
            "extra_probe_calls": 0, "profiles": 0, "accepted_edits": len(accepted), "accepted": accepted, "repairs": 0,
            "control_failures": 0, "search_wall_ns": search_ns, "audit_wall_ns": audit_ns,
            "model_latency_ns": 0, "verify_wall_ns": 0, "verify_cpu_s": 0.0,
            "audit_pass": audit.audit_pass if audit else None, "audit_unchanged": audit.unchanged if audit else None,
            "raw_audit_speedup": audit.raw_speedup if audit else None, "audit_lo": audit.lo if audit else None,
            "audit_hi": audit.hi if audit else None, "effective_speedup": audit.effective_speedup if audit else None,
            "success_at_1_5": audit.success if audit else None, "audit_failure": audit.failure if audit else None,
            "failed_run": infra is not None, "infrastructure_error": infra, "events_path": spec.events_path,
            "selected_source": best.source,
        }
        log.emit("episode_end", termination=termination)
        log.flush()
        rec["events"] = None if spec.events_path else log.events
        return rec
