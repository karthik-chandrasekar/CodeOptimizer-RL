"""Trusted verifier: correctness against the immutable original, paired timing, profiling and the final audit.

Semantics are always judged against ``P0`` (the immutable original), never only against the current best.
Timing measures the *incremental* improvement over the current best during search and the final improvement over
``P0`` in the audit.  The worker returns plain-value encodings; comparison happens here, in trusted code.

Check stages (each returns a :class:`CheckResult`):

``smoke``      the task's public example inputs for the edited function (public feedback allowed)
``probe``      contract-valid inputs built by the probe registry for M4's selection (public feedback allowed)
``mandatory``  the private online correctness suite + digests of the online timing workload's outputs
               (feedback is the failure *category* only)

A candidate is timed only after it passed every correctness stage.
"""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tinyperf_chain.budget import DeadlineExceeded
from tinyperf_chain.config import ChainConfig
from tinyperf_chain.hashing import hash_json, sha256_text
from tinyperf_chain.program import Program
from tinyperf_chain.sandbox import codec
from tinyperf_chain.sandbox.runner import SandboxRunner
from tinyperf_chain.sandbox.timing import TimingVerdict, median, timing_verdict
from tinyperf_chain.tasks import Call, PrivateSuite, PublicTask


class InfrastructureError(RuntimeError):
    """The harness (not the candidate) failed: worker could not launch, reference program failed, ..."""


@dataclass
class Mismatch:
    index: int
    category: str          # return_mismatch | type_mismatch | exception_mismatch | mutation_mismatch | alias_mismatch
    detail: str            #   | timeout | unsupported_value | missing_record
    call: Optional[Call] = None
    expected: Any = None   # encodings (trusted side only)
    got: Any = None
    expected_exc: Optional[str] = None
    got_exc: Optional[str] = None


@dataclass
class CheckResult:
    ok: bool
    stage: str
    kind: str = "ok"                     # ok | syntax_error | static_error | load_error | timeout | resource | crash
    category: str = ""                   #    | smoke_mismatch | probe_mismatch | private_mismatch | output_too_large
    message: str = ""
    public_feedback: Optional[Dict[str, Any]] = None
    n_checked: int = 0
    wall_ns: int = 0
    cpu_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TimingResult:
    ok: bool
    status: str
    verdict: Optional[TimingVerdict]
    ref_ns: List[int] = field(default_factory=list)
    cand_ns: List[int] = field(default_factory=list)
    orders: List[str] = field(default_factory=list)
    reps: int = 0
    seed: int = 0
    wall_ns: int = 0
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["verdict"] = self.verdict.to_dict() if self.verdict else None
        return d


@dataclass
class FunctionProfile:
    function: str
    ncalls: int
    self_ns: float
    cum_ns: float
    self_share: float
    cum_share: float


@dataclass
class ProfileSnapshot:
    program_hash: str
    functions: Dict[str, FunctionProfile]
    total_ns: float
    ok: bool = True
    message: str = ""
    note: str = "profiler-instrumented public workload; not a benchmark"


@dataclass
class AuditResult:
    audit_pass: Optional[bool]
    unchanged: bool
    raw_speedup: Optional[float]
    lo: Optional[float]
    hi: Optional[float]
    effective_speedup: float
    success: bool
    failure: str = ""
    correctness: Optional[Dict[str, Any]] = None
    timing: Optional[Dict[str, Any]] = None
    wall_ns: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_INFRA = ("infrastructure_error",)
_CANDIDATE_FAIL = {"syntax_error", "static_error", "load_error", "timeout", "resource", "worker_error", "output_too_large",
                   "runtime_error"}


class Verifier:
    def __init__(self, cfg: ChainConfig, runner: Optional[SandboxRunner] = None, cache_dir: Optional[str] = None):
        self.cfg = cfg
        self.v = cfg.verification
        self.runner = runner or SandboxRunner(cfg.sandbox, hash_seed=cfg.verification.hash_seed)
        self.cache_dir = cache_dir
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._corr_sem = threading.BoundedSemaphore(max(1, cfg.serving.correctness_workers))
        n_t = max(1, cfg.serving.timing_workers)
        self._timing_slots: List[List[int]] = []
        cpus = list(cfg.sandbox.timing_cpus)
        for i in range(n_t):
            self._timing_slots.append(cpus[i::n_t] if cpus else [])
        self._timing_free = list(range(n_t))
        self._timing_cv = threading.Condition()
        self.stats = {"runs": 0, "timing_runs": 0, "wall_ns": 0}

    # ------------------------------------------------------------------ #
    # infrastructure
    # ------------------------------------------------------------------ #
    @contextmanager
    def _timing_slot(self):
        with self._timing_cv:
            while not self._timing_free:
                self._timing_cv.wait()
            slot = self._timing_free.pop()
        try:
            yield self._timing_slots[slot]
        finally:
            with self._timing_cv:
                self._timing_free.append(slot)
                self._timing_cv.notify()

    def _timeout(self, deadline: Optional[float], base: Optional[float] = None) -> Tuple[float, bool]:
        base = float(base if base is not None else self.cfg.sandbox.process_timeout_s)
        if deadline is None:
            return base, False
        left = deadline - time.monotonic()
        if left <= 0:
            raise DeadlineExceeded("before sandbox call")
        return (left, True) if left < base else (base, False)

    def _run(self, req: Dict[str, Any], *, deadline: Optional[float], hash_seed: Optional[int] = None,
             timing: bool = False) -> Dict[str, Any]:
        timeout, clipped = self._timeout(deadline)
        if timing:
            with self._timing_slot() as cpus:
                resp = self.runner.run(req, hash_seed=hash_seed, cpus=cpus or None, timeout_s=timeout)
        else:
            with self._corr_sem:
                resp = self.runner.run(req, hash_seed=hash_seed, cpus=self.cfg.sandbox.correctness_cpus or None,
                                       timeout_s=timeout)
        with self._lock:
            self.stats["runs"] += 1
            self.stats["timing_runs"] += int(timing)
            self.stats["wall_ns"] += int(resp.get("_meta", {}).get("wall_ns", 0))
        if resp.get("status") == "timeout" and clipped and resp.get("_meta", {}).get("timed_out"):
            raise DeadlineExceeded("during sandbox call")
        if resp.get("status") in _INFRA:
            raise InfrastructureError(resp.get("message", "sandbox infrastructure failure"))
        return resp

    def _cache_get(self, key: str) -> Any:
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        if self.cache_dir:
            p = os.path.join(self.cache_dir, key[:2], key + ".json")
            if os.path.exists(p):
                with open(p) as f:
                    val = json.load(f)
                with self._lock:
                    self._cache[key] = val
                return val
        return None

    def _cache_put(self, key: str, val: Any) -> None:
        with self._lock:
            self._cache[key] = val
        if self.cache_dir:
            d = os.path.join(self.cache_dir, key[:2])
            os.makedirs(d, exist_ok=True)
            tmp = os.path.join(d, key + f".{os.getpid()}.{threading.get_ident()}.tmp")
            with open(tmp, "w") as f:
                json.dump(val, f)
            os.replace(tmp, os.path.join(d, key + ".json"))

    # ------------------------------------------------------------------ #
    # executing calls
    # ------------------------------------------------------------------ #
    def run_calls(self, program: Program, calls: Sequence[Call], *, output: str = "full", hash_seed: Optional[int] = None,
                  deadline: Optional[float] = None) -> Dict[str, Any]:
        req = self.runner.base_request("run")
        req.update(code=program.executable(), entry="_tp_entry", inputs=list(calls), output=output,
                   dict_order="ignore", stop_on_timeout=True)
        return self._run(req, deadline=deadline, hash_seed=hash_seed)

    def reference_records(self, p0: Program, calls: Sequence[Call], *, output: str = "full",
                          hash_seed: Optional[int] = None, deadline: Optional[float] = None) -> List[Dict[str, Any]]:
        """Behaviour of the immutable original on ``calls`` (cached; the original is deterministic by eligibility)."""
        hs = self.v.hash_seed if hash_seed is None else hash_seed
        key = hash_json(["ref_v1", p0.hash, hash_json(codec.encode(tuple(tuple(c) for c in calls))), output, hs])
        hit = self._cache_get(key)
        if hit is not None:
            return hit
        resp = self.run_calls(p0, calls, output=output, hash_seed=hs, deadline=deadline)
        if resp.get("status") != "ok" or len(resp.get("records", [])) != len(calls):
            raise InfrastructureError(f"reference program failed: {resp.get('status')}: {str(resp.get('message', ''))[:200]}")
        recs = resp["records"]
        if any("unencodable" in r or r.get("exc") == "__timeout__" for r in recs):
            raise InfrastructureError("reference behaviour is not representable under the contract")
        self._cache_put(key, recs)
        return recs

    @staticmethod
    def compare_records(ref: Sequence[Dict[str, Any]], got: Sequence[Dict[str, Any]], calls: Sequence[Call],
                        float_rel: float = 0.0, float_abs: float = 0.0) -> Optional[Mismatch]:
        for i, (e, g) in enumerate(zip(ref, got)):
            call = calls[i] if i < len(calls) else None
            if g.get("exc") == "__timeout__":
                return Mismatch(i, "timeout", "call timed out", call)
            if "unencodable" in g:
                return Mismatch(i, "unsupported_value", str(g["unencodable"])[:120], call)
            if e.get("exc") != g.get("exc"):
                return Mismatch(i, "exception_mismatch", f"expected {e.get('exc')}, got {g.get('exc')}", call,
                                expected_exc=e.get("exc"), got_exc=g.get("exc"),
                                expected=e.get("enc"), got=g.get("enc"))
            if "digest" in e or "digest" in g:
                if e.get("digest") != g.get("digest"):
                    return Mismatch(i, "return_mismatch", "output digest differs", call)
                continue
            ea, eg = e["enc"], g["enc"]
            # enc = ["T", [args_after, ret]]
            args_e, ret_e = ea[1][0], ea[1][1]
            args_g, ret_g = eg[1][0], eg[1][1]
            r = codec.compare(args_e, args_g, float_rel, float_abs)
            if r:
                cat = "alias_mismatch" if "aliasing" in r else "mutation_mismatch"
                return Mismatch(i, cat, r, call, expected=ea, got=eg)
            if e.get("exc") is None:
                r = codec.compare(ret_e, ret_g, float_rel, float_abs)
                if r:
                    detail = r.split(": ", 1)[-1]
                    cat = "alias_mismatch" if "aliasing" in r else ("type_mismatch" if detail.startswith("type ") else "return_mismatch")
                    return Mismatch(i, cat, r, call, expected=ea, got=eg)
        if len(got) < len(ref):
            return Mismatch(len(got), "missing_record", "candidate stopped early", calls[len(got)] if len(got) < len(calls) else None)
        return None

    def _feedback(self, m: Mismatch, probe_id: Optional[str]) -> Dict[str, Any]:
        lim = self.v.max_feedback_chars
        fb: Dict[str, Any] = {"failure_kind": m.category, "probe_id": probe_id}
        redacted = False
        if m.call is not None:
            args = m.call[1:]
            s = repr(args)
            if len(s) > lim:
                s, redacted = s[: lim - 3] + "...", True
            fb["input"] = s
        for side, enc, exc in (("expected", m.expected, m.expected_exc), ("got", m.got, m.got_exc)):
            if exc:
                fb[side] = f"raises {exc}"
            elif enc is not None:
                try:
                    v = codec.decode(enc)[1]          # the return value
                    s = repr(v)
                except codec.CodecError:
                    s = "<unrepresentable>"
                if len(s) > lim:
                    s, redacted = s[: lim - 3] + "...", True
                fb[side] = s
        if m.category == "mutation_mismatch":
            fb["note"] = "argument mutation differs"
        fb["redacted"] = redacted
        return fb

    def _check(self, stage: str, p0: Program, candidate: Program, calls: Sequence[Call], *, mismatch_kind: str,
               public: bool, probe_ids: Optional[Sequence[str]] = None, deadline: Optional[float] = None,
               output: str = "full", hash_seed: Optional[int] = None) -> CheckResult:
        t0 = time.perf_counter_ns()
        if not calls:
            return CheckResult(True, stage, n_checked=0)
        ref = self.reference_records(p0, calls, output=output, hash_seed=hash_seed, deadline=deadline)
        resp = self.run_calls(candidate, calls, output=output, hash_seed=hash_seed, deadline=deadline)
        wall = time.perf_counter_ns() - t0
        cpu = float(resp.get("child_cpu_s", 0.0) or 0.0)
        st = resp.get("status")
        if st != "ok":
            kind = st if st in _CANDIDATE_FAIL else "crash"
            if kind == "worker_error":
                kind = "crash"
            return CheckResult(False, stage, kind, kind, str(resp.get("message", ""))[:300], None, 0, wall, cpu)
        m = self.compare_records(ref, resp.get("records", []), calls, self.v.float_rel_tol, self.v.float_abs_tol)
        if m is None:
            return CheckResult(True, stage, n_checked=len(calls), wall_ns=wall, cpu_s=cpu)
        fb = self._feedback(m, probe_ids[m.index] if probe_ids and m.index < len(probe_ids) else None) if public else None
        msg = m.category if not public else f"{m.category}: {m.detail}"[:300]
        return CheckResult(False, stage, mismatch_kind, m.category, msg, fb, m.index + 1, wall, cpu)

    # ------------------------------------------------------------------ #
    # stages
    # ------------------------------------------------------------------ #
    def smoke_check(self, task: PublicTask, candidate: Program, function: str, *,
                    deadline: Optional[float] = None) -> CheckResult:
        k = task.functions.index(function)
        ex = task.public_examples.get(function, [])[: max(0, self.v.smoke_examples_per_function)]
        calls = [(k, *args) for args in ex]
        return self._check("smoke", task.original(), candidate, calls, mismatch_kind="smoke_mismatch", public=True,
                           deadline=deadline)

    def probe_check(self, task: PublicTask, candidate: Program, function: str, probes: Sequence[Tuple[str, Tuple]], *,
                    deadline: Optional[float] = None) -> CheckResult:
        k = task.functions.index(function)
        calls = [(k, *args) for _, args in probes]
        return self._check("probe", task.original(), candidate, calls, mismatch_kind="probe_mismatch", public=True,
                           probe_ids=[p for p, _ in probes], deadline=deadline)

    def mandatory_checks(self, task: PublicTask, private: PrivateSuite, candidate: Program, *,
                         deadline: Optional[float] = None) -> CheckResult:
        p0 = task.original()
        r = self._check("mandatory", p0, candidate, private.online_correctness, mismatch_kind="private_mismatch",
                        public=False, deadline=deadline)
        if not r.ok or not self.v.check_timing_workload_outputs:
            return r
        r2 = self._check("mandatory_workload", p0, candidate, private.online_timing, mismatch_kind="private_mismatch",
                         public=False, deadline=deadline, output="digest")
        if not r2.ok:
            return r2
        return CheckResult(True, "mandatory", n_checked=r.n_checked + r2.n_checked, wall_ns=r.wall_ns + r2.wall_ns,
                           cpu_s=r.cpu_s + r2.cpu_s)

    def outcome_matrix(self, task: PublicTask, candidate: Program, calls: Sequence[Call], *,
                       deadline: Optional[float] = None) -> List[Optional[str]]:
        """Per-call discrepancy category (None = same behaviour as the original); used to label probe data.
        A candidate-level failure (does not load, crashes) marks every call with that failure kind."""
        if not calls:
            return []
        p0 = task.original()
        ref = self.reference_records(p0, calls, deadline=deadline)
        resp = self.run_calls(candidate, calls, deadline=deadline)
        if resp.get("status") != "ok":
            return [str(resp.get("status"))] * len(calls)
        got = resp.get("records", [])
        out: List[Optional[str]] = []
        for i, c in enumerate(calls):
            if i >= len(got):
                out.append("missing_record")
                continue
            m = self.compare_records([ref[i]], [got[i]], [c], self.v.float_rel_tol, self.v.float_abs_tol)
            out.append(m.category if m else None)
        return out

    def redacted(self, r: CheckResult) -> CheckResult:
        """Category-only copy of a private check result (what may leave the evaluator)."""
        return CheckResult(r.ok, r.stage, r.kind, r.category, r.category, None, r.n_checked, r.wall_ns, r.cpu_s)

    # ------------------------------------------------------------------ #
    # timing
    # ------------------------------------------------------------------ #
    def calibrate(self, p0: Program, workload: Sequence[Call], *, hash_seed: Optional[int] = None,
                  deadline: Optional[float] = None) -> int:
        key = hash_json(["calib_v1", p0.hash, hash_json(codec.encode(tuple(tuple(c) for c in workload))),
                         self.v.block_target_ms])
        hit = self._cache_get(key)
        if hit is not None:
            return int(hit)
        req = self.runner.base_request("calibrate")
        req.update(code=p0.executable(), entry="_tp_entry", workload=list(workload), target_ms=self.v.block_target_ms)
        resp = self._run(req, deadline=deadline, hash_seed=hash_seed, timing=True)
        if resp.get("status") != "ok":
            raise InfrastructureError(f"calibration of the original failed: {resp.get('status')} {resp.get('message', '')}")
        reps = int(resp["reps"])
        self._cache_put(key, reps)
        return reps

    def blocks_for(self, mode: str) -> int:
        return {"train": self.v.train_timing_blocks, "dev": self.v.development_timing_blocks,
                "development": self.v.development_timing_blocks, "confirm": self.v.development_timing_blocks,
                "confirmation": self.v.development_timing_blocks, "audit": self.v.audit_timing_blocks}.get(mode, self.v.train_timing_blocks)

    def paired_timing(self, task: PublicTask, workload: Sequence[Call], reference: Program, candidate: Program, *,
                      blocks: int, seed: int, rule: Optional[str] = None, threshold: Optional[float] = None,
                      hash_seed: Optional[int] = None, deadline: Optional[float] = None) -> TimingResult:
        t0 = time.perf_counter_ns()
        reps = self.calibrate(task.original(), workload, hash_seed=hash_seed, deadline=deadline)
        ref_ns: List[int] = []
        cand_ns: List[int] = []
        orders: List[str] = []
        chunks = [1] * blocks if self.v.fresh_process_per_block else [blocks]
        for ci, nb in enumerate(chunks):
            req = self.runner.base_request("time_pair")
            req.update(code_a=reference.executable(), code_b=candidate.executable(), entry="_tp_entry",
                       workload=list(workload), blocks=nb, reps=reps, warmup=self.v.timing_warmup, seed=seed * 1009 + ci)
            resp = self._run(req, deadline=deadline, hash_seed=hash_seed, timing=True)
            if resp.get("status") != "ok":
                return TimingResult(False, resp.get("status", "error"), None, ref_ns, cand_ns, orders, reps, seed,
                                    time.perf_counter_ns() - t0, str(resp.get("message", ""))[:300])
            ref_ns += [int(x) for x in resp["a_ns"]]
            cand_ns += [int(x) for x in resp["b_ns"]]
            orders += list(resp["orders"])
        verdict = timing_verdict(ref_ns, cand_ns, threshold=threshold or self.v.accept_incremental_speedup,
                                 rule=rule or self.v.accept_rule, draws=self.v.timing_bootstrap_draws,
                                 level=self.v.timing_interval, seed=seed, legacy_delta=self.v.legacy_delta)
        return TimingResult(True, "ok", verdict, ref_ns, cand_ns, orders, reps, seed, time.perf_counter_ns() - t0)

    # ------------------------------------------------------------------ #
    # profiling (public workload)
    # ------------------------------------------------------------------ #
    def profile(self, task: PublicTask, program: Program, *, deadline: Optional[float] = None) -> ProfileSnapshot:
        key = hash_json(["profile_v1", program.hash, task.task_id, hash_json(codec.encode(tuple(tuple(c) for c in task.public_workload)))])
        hit = self._cache_get(key)
        if hit is not None:
            return ProfileSnapshot(hit["program_hash"], {k: FunctionProfile(**v) for k, v in hit["functions"].items()},
                                   hit["total_ns"], hit["ok"], hit["message"])
        req = self.runner.base_request("profile")
        req.update(code=program.executable(), entry="_tp_entry", workload=list(task.public_workload),
                   function_names=program.top_level_functions())
        resp = self._run(req, deadline=deadline)
        if resp.get("status") != "ok":
            return ProfileSnapshot(program.hash, {}, 0.0, False, str(resp.get("message", ""))[:200])
        funcs = resp.get("functions", {})
        total = float(resp.get("profiled_wall_ns", 0.0)) or 1.0
        self_total = sum(float(f["self_ns"]) for f in funcs.values()) or 1.0
        out = {}
        for name in program.top_level_functions():
            f = funcs.get(name, {"ncalls": 0, "self_ns": 0.0, "cum_ns": 0.0})
            out[name] = FunctionProfile(name, int(f["ncalls"]), float(f["self_ns"]), float(f["cum_ns"]),
                                        float(f["self_ns"]) / self_total, min(1.0, float(f["cum_ns"]) / total))
        snap = ProfileSnapshot(program.hash, out, total)
        self._cache_put(key, {"program_hash": snap.program_hash, "functions": {k: asdict(v) for k, v in out.items()},
                              "total_ns": total, "ok": True, "message": ""})
        return snap

    # ------------------------------------------------------------------ #
    # final audit (after search; no feedback to models)
    # ------------------------------------------------------------------ #
    def final_audit(self, task: PublicTask, private: PrivateSuite, selected: Program, *, seed: int = 0) -> AuditResult:
        t0 = time.perf_counter_ns()
        p0 = task.original()
        if selected.hash == p0.hash:
            return AuditResult(True, True, 1.0, None, None, 1.0, False, wall_ns=time.perf_counter_ns() - t0)
        hs = self.v.audit_hash_seed
        c1 = self._check("audit_correctness", p0, selected, private.audit_correctness, mismatch_kind="audit_mismatch",
                         public=False, hash_seed=hs)
        corr = c1.to_dict()
        if c1.ok and self.v.check_timing_workload_outputs:
            c2 = self._check("audit_workload", p0, selected, private.audit_timing, mismatch_kind="audit_mismatch",
                             public=False, hash_seed=hs, output="digest")
            if not c2.ok:
                c1, corr = c2, c2.to_dict()
        if not c1.ok:
            return AuditResult(False, False, None, None, None, 1.0, False, failure=c1.category or c1.kind,
                               correctness=corr, wall_ns=time.perf_counter_ns() - t0)
        tr = self.paired_timing(task, private.audit_timing, p0, selected, blocks=self.v.audit_timing_blocks,
                                seed=100003 + seed, rule="paired_log_median_ci_lower",
                                threshold=self.v.success_final_speedup, hash_seed=hs)
        if not tr.ok or tr.verdict is None:
            return AuditResult(False, False, None, None, None, 1.0, False, failure=f"audit_timing_{tr.status}",
                               correctness=corr, timing=tr.to_dict(), wall_ns=time.perf_counter_ns() - t0)
        s = tr.verdict.speedup
        return AuditResult(True, False, s, tr.verdict.lo, tr.verdict.hi, s, s >= self.v.success_final_speedup,
                           correctness=corr, timing=tr.to_dict(), wall_ns=time.perf_counter_ns() - t0)

    # ------------------------------------------------------------------ #
    def determinism_check(self, p0: Program, calls: Sequence[Call], seeds: Sequence[int] = (0, 1)) -> Optional[str]:
        """Construction-time eligibility: the original must behave identically under different hash seeds."""
        digests = []
        for s in seeds:
            resp = self.run_calls(p0, calls, output="digest", hash_seed=s)
            if resp.get("status") != "ok":
                return f"original failed: {resp.get('status')}"
            recs = resp["records"]
            if len(recs) != len(calls):
                return "original timed out"
            if any("unencodable" in r for r in recs):
                return "original output unsupported: " + next(r["unencodable"] for r in recs if "unencodable" in r)
            if any(r.get("exc") == "__timeout__" for r in recs):
                return "original timed out on an input"
            digests.append([(r.get("exc"), r.get("digest")) for r in recs])
        if any(d != digests[0] for d in digests[1:]):
            return "nondeterministic under hash seed"
        return None
