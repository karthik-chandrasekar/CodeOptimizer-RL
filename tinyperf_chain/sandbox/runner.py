"""Parent side of the sandbox: launch one isolated worker process per request and read its JSON reply.

Layers (defence in depth; none of them alone is a hostile-code security boundary):

1. interpreter flags: ``-P -s -B`` (3.11+) with a scrubbed environment (no PYTHONPATH, fixed PYTHONHASHSEED), or
   ``-I`` on 3.10 (the hash seed is then uncontrolled and the runner records that);
2. a fresh private working directory per call, removed afterwards;
3. a new session / process group, killed as a group on timeout (descendants cannot linger);
4. optional OS isolation: ``unshare`` (no network namespace) or ``bwrap`` (all namespaces unshared, a *minimal*
   read-only filesystem: the interpreter prefix, ``/usr`` and the loader directories, the three worker source
   files, a private ``/tmp`` and the per-call working directory - no home directory, no credentials, no repository);
   availability is probed once and a configured-but-unavailable mode is an error, never a silent downgrade;
5. privilege drop: when the harness runs as root the worker runs as ``sandbox.drop_to_uid`` (default: 65534) unless
   ``sandbox.run_as_root: true`` is set explicitly;
6. inside the worker: legacy AST checks, restricted builtins, import guard, rlimits (address space, CPU, no
   processes, file size), one fresh module namespace per independent trial;
7. output: only size-checked JSON is parsed; nothing candidate-influenced is unpickled.

Infrastructure failures are distinguished from candidate failures: a worker that dies without a JSON reply triggers
one bounded *canary* (``ping``) launch; if the canary also fails the result is ``infrastructure_error`` (a failed-run
record upstream), otherwise the crash is attributed to the candidate.  Every launch records which CPUs were
requested and whether ``taskset`` was actually applied.
"""
from __future__ import annotations

import json
import os
import pickle
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import tinyperf.env.worker as _legacy_worker_mod
from tinyperf_chain.config import SandboxCfg
from tinyperf_chain.sandbox import codec as _codec_mod

WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py")
LEGACY_WORKER_PATH = os.path.abspath(_legacy_worker_mod.__file__)
CODEC_PATH = os.path.abspath(_codec_mod.__file__)
NOBODY_UID = 65534


class IsolationUnavailable(RuntimeError):
    pass


@dataclass
class RunMeta:
    wall_ns: int
    returncode: Optional[int]
    timed_out: bool
    isolation: str
    hash_seed: Optional[int]
    cpus: Optional[List[int]] = None
    taskset_applied: bool = False
    uid: Optional[int] = None
    canary_ok: Optional[bool] = None


_PROBED: Dict[str, bool] = {}
_PROBE_LOCK = threading.Lock()
_SELFCHECKED: Dict[str, Dict[str, Any]] = {}


def _dedupe(paths: List[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        rp = os.path.realpath(p)
        if rp not in out and os.path.exists(rp):
            out.append(rp)
    return out


def bwrap_binds(python: str) -> List[str]:
    """Minimal read-only filesystem for a worker: interpreter prefixes, system libraries, loader config, worker code."""
    prefixes = _dedupe([sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix,
                        os.path.dirname(os.path.dirname(os.path.realpath(python)))])
    args: List[str] = []
    for p in prefixes:
        if p not in ("/", "/usr"):
            args += ["--ro-bind", p, p]
    args += ["--ro-bind", "/usr", "/usr"]
    for d in ("/lib", "/lib64", "/bin", "/sbin"):
        if os.path.islink(d):
            args += ["--symlink", os.readlink(d), d]
        elif os.path.isdir(d):
            args += ["--ro-bind", d, d]
    for f in ("/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/ld.so.conf.d", "/etc/alternatives", "/etc/localtime",
              "/etc/passwd", "/etc/group", "/etc/nsswitch.conf"):
        args += ["--ro-bind-try", f, f]
    for f in (WORKER_PATH, LEGACY_WORKER_PATH, CODEC_PATH):
        args += ["--ro-bind", f, f]
    return args


def _bwrap_prefix(python: str, workdir: Optional[str] = None) -> List[str]:
    cmd = ["bwrap", "--die-with-parent", "--new-session", "--unshare-all", "--cap-drop", "ALL"]
    cmd += bwrap_binds(python)
    cmd += ["--tmpfs", "/tmp", "--dev", "/dev", "--proc", "/proc"]
    if workdir:
        cmd += ["--bind", workdir, workdir, "--chdir", workdir]
    return cmd + ["--"]


def _probe(mode: str, python: str) -> bool:
    key = f"{mode}|{python}"
    with _PROBE_LOCK:
        if key in _PROBED:
            return _PROBED[key]
        ok = False
        try:
            if mode == "unshare" and shutil.which("unshare"):
                ok = subprocess.run(["unshare", "--net", "--map-root-user", "true"], capture_output=True, timeout=10).returncode == 0
            elif mode == "bwrap" and shutil.which("bwrap"):
                ok = subprocess.run(_bwrap_prefix(python) + [python, "-c", "import json"], capture_output=True,
                                    timeout=20).returncode == 0
            elif mode == "rlimits":
                ok = True
        except (OSError, subprocess.SubprocessError):
            ok = False
        _PROBED[key] = ok
        return ok


class SandboxRunner:
    """Thread-safe launcher.  One call = one fresh interpreter."""

    def __init__(self, cfg: SandboxCfg, hash_seed: int = 0, scratch_dir: Optional[str] = None, selfcheck: bool = True):
        self.cfg = cfg
        self.hash_seed = hash_seed
        self.python = cfg.python or sys.executable
        self.scratch = scratch_dir or tempfile.gettempdir()
        if not _probe(cfg.isolation, self.python):
            raise IsolationUnavailable(f"sandbox isolation mode {cfg.isolation!r} is not available on this host")
        self.hash_seed_controlled = sys.version_info >= (3, 11)
        self.uid = self._effective_uid()
        self.taskset_available = shutil.which("taskset") is not None
        self.stats = {"launches": 0, "canaries": 0, "infrastructure_errors": 0}
        self._stats_lock = threading.Lock()
        if selfcheck:
            self.selfcheck()

    # -------------------------------------------------------------- #
    def _effective_uid(self) -> Optional[int]:
        if not hasattr(os, "geteuid") or os.geteuid() != 0:
            return None                     # not root: cannot and need not drop
        if self.cfg.run_as_root:
            return None
        return int(self.cfg.drop_to_uid) if self.cfg.drop_to_uid is not None else NOBODY_UID

    def identity(self) -> Dict[str, Any]:
        """Everything about the execution environment that a cached result depends on (12.5 cache keys)."""
        from tinyperf_chain.sandbox.worker import WORKER_VERSION

        c = self.cfg
        return {"python": platform.python_version(), "implementation": platform.python_implementation(),
                "worker_version": WORKER_VERSION, "codec_version": getattr(_codec_mod, "CODEC_VERSION", "v1"),
                "isolation": c.isolation, "uid": self.uid, "machine": platform.machine(),
                "call_timeout_s": c.call_timeout_s, "workload_timeout_s": c.workload_timeout_s,
                "memory_mb": c.memory_mb, "cpu_seconds": c.cpu_seconds, "hash_seed_controlled": self.hash_seed_controlled}

    def selfcheck(self) -> Dict[str, Any]:
        """One ``ping`` per (interpreter, isolation, uid): a configuration that cannot launch a worker is an error."""
        key = f"{self.python}|{self.cfg.isolation}|{self.uid}"
        with _PROBE_LOCK:
            cached = _SELFCHECKED.get(key)
        if cached is not None:
            return cached
        resp = self.run(self.base_request("ping"), timeout_s=60, canary=False)
        if resp.get("status") != "ok":
            raise IsolationUnavailable(f"sandbox worker cannot run under isolation={self.cfg.isolation!r} uid={self.uid}: "
                                       f"{resp.get('message', '')[:300]}")
        env = resp.get("environment", {})
        with _PROBE_LOCK:
            _SELFCHECKED[key] = env
        return env

    # -------------------------------------------------------------- #
    def _cmd(self, cpus: Optional[List[int]], workdir: str) -> List[str]:
        if self.hash_seed_controlled:
            cmd = [self.python, "-P", "-s", "-B", WORKER_PATH]
        else:  # pragma: no cover - python 3.10
            cmd = [self.python, "-I", "-B", WORKER_PATH]
        if self.cfg.isolation == "unshare":
            cmd = ["unshare", "--net", "--map-root-user", "--"] + cmd
        elif self.cfg.isolation == "bwrap":
            cmd = _bwrap_prefix(self.python, workdir) + cmd
        if cpus and self.taskset_available:
            cmd = ["taskset", "-c", ",".join(str(c) for c in cpus)] + cmd
        return cmd

    def _env(self, hash_seed: int) -> Dict[str, str]:
        return {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
                "PYTHONHASHSEED": str(hash_seed), "PYTHONDONTWRITEBYTECODE": "1", "HOME": "/nonexistent"}

    def base_request(self, mode: str) -> Dict[str, Any]:
        c = self.cfg
        return {
            "mode": mode,
            "legacy_worker_path": LEGACY_WORKER_PATH,
            "codec_path": CODEC_PATH,
            "allowed_modules": list(c.allowed_modules),
            "limits": {"memory_mb": c.memory_mb, "cpu_seconds": c.cpu_seconds, "call_timeout_s": c.call_timeout_s,
                       "workload_timeout_s": c.workload_timeout_s, "recursion_limit": 3000},
        }

    def _bump(self, key: str) -> None:
        with self._stats_lock:
            self.stats[key] += 1

    def run(self, req: Dict[str, Any], *, hash_seed: Optional[int] = None, cpus: Optional[List[int]] = None,
            timeout_s: Optional[float] = None, canary: bool = True) -> Dict[str, Any]:
        hs = self.hash_seed if hash_seed is None else hash_seed
        payload = pickle.dumps(req, protocol=pickle.HIGHEST_PROTOCOL)
        workdir = tempfile.mkdtemp(prefix="tpchain_", dir=self.scratch)
        timeout = float(timeout_s if timeout_s is not None else self.cfg.process_timeout_s)
        kwargs: Dict[str, Any] = {}
        if self.uid is not None:
            kwargs["user"] = self.uid
            kwargs["group"] = self.uid
            os.chmod(workdir, 0o777)
        t0 = time.perf_counter_ns()
        timed_out = False
        rc: Optional[int] = None
        out = b""
        err = b""
        applied = bool(cpus) and self.taskset_available

        def meta(**extra: Any) -> Dict[str, Any]:
            m = RunMeta(time.perf_counter_ns() - t0, rc, timed_out, self.cfg.isolation,
                        hs if self.hash_seed_controlled else None, list(cpus) if cpus else None, applied, self.uid)
            d = asdict(m)
            d.update(extra)
            return d

        self._bump("launches")
        try:
            proc = subprocess.Popen(self._cmd(cpus, workdir), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, cwd=workdir, env=self._env(hs), start_new_session=True, **kwargs)
            try:
                out, err = proc.communicate(payload, timeout=max(0.5, timeout))
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGKILL)     # the whole session: descendants cannot linger
                except (ProcessLookupError, PermissionError):
                    pass
                out, err = proc.communicate()
            rc = proc.returncode
        except OSError as e:
            self._bump("infrastructure_errors")
            return {"status": "infrastructure_error", "message": f"could not launch worker: {e}", "_meta": meta()}
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        if timed_out:
            return {"status": "timeout", "message": "sandbox process exceeded its hard timeout", "_meta": meta()}
        if len(out) > self.cfg.max_output_bytes:
            return {"status": "output_too_large", "message": f"worker output {len(out)} bytes", "_meta": meta()}
        if rc != 0 or not out:
            tail = err.decode(errors="replace")[-400:]
            status = "resource" if ("MemoryError" in tail or rc in (-9, 137)) else "worker_error"
            canary_ok: Optional[bool] = None
            if canary and req.get("mode") not in ("ping", "selftest"):
                # Infrastructure failure differs from candidate failure (12.4): one bounded, visible canary launch.
                self._bump("canaries")
                ping = self.run(self.base_request("ping"), timeout_s=30, canary=False)
                canary_ok = ping.get("status") == "ok"
                if not canary_ok:
                    self._bump("infrastructure_errors")
                    return {"status": "infrastructure_error",
                            "message": f"worker exited with code {rc} and the canary failed too: {tail.strip()}",
                            "_meta": meta(canary_ok=False)}
            return {"status": status, "message": f"worker exited with code {rc}: {tail.strip()}", "_meta": meta(canary_ok=canary_ok)}
        try:
            resp = json.loads(out.decode("utf-8"))
        except (ValueError, UnicodeDecodeError, RecursionError) as e:
            return {"status": "worker_error", "message": f"undecodable worker response: {e}", "_meta": meta()}
        if not isinstance(resp, dict) or "status" not in resp:
            return {"status": "worker_error", "message": "malformed worker response", "_meta": meta()}
        resp["_meta"] = meta()
        return resp

    # -------------------------------------------------------------- #
    def selftest(self) -> Dict[str, Any]:
        """What the OS allows a worker process to do under the configured isolation and limits (trusted probe code)."""
        return self.run(self.base_request("selftest"), timeout_s=30, canary=False)
