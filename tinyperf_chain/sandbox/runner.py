"""Parent side of the sandbox: launch one isolated worker process per request and read its JSON reply.

Layers (defence in depth; none of them alone is a hostile-code security boundary):

1. interpreter flags: ``-P -s -B`` (3.11+) with a scrubbed environment (no PYTHONPATH, fixed PYTHONHASHSEED), or
   ``-I`` on 3.10 (the hash seed is then uncontrolled and the runner records that);
2. a fresh private working directory per call, removed afterwards;
3. a new session / process group, killed as a group on timeout (descendants cannot linger);
4. optional OS isolation: ``unshare`` (no network namespace access) or ``bwrap`` (read-only root, private /tmp, all
   namespaces unshared); availability is probed once and a configured-but-unavailable mode is an error, never a
   silent downgrade;
5. optional privilege drop to an unprivileged uid/gid when running as root;
6. inside the worker: legacy AST checks, restricted builtins, import guard, rlimits (address space, CPU, no
   processes, file size);
7. output: only size-checked JSON is parsed; nothing candidate-influenced is unpickled.
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import tinyperf.env.worker as _legacy_worker_mod
from tinyperf_chain.config import SandboxCfg
from tinyperf_chain.sandbox import codec as _codec_mod

WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py")
LEGACY_WORKER_PATH = os.path.abspath(_legacy_worker_mod.__file__)
CODEC_PATH = os.path.abspath(_codec_mod.__file__)


class IsolationUnavailable(RuntimeError):
    pass


@dataclass
class RunMeta:
    wall_ns: int
    returncode: Optional[int]
    timed_out: bool
    isolation: str
    hash_seed: Optional[int]


_PROBED: Dict[str, bool] = {}
_PROBE_LOCK = threading.Lock()


def _probe(mode: str) -> bool:
    with _PROBE_LOCK:
        if mode in _PROBED:
            return _PROBED[mode]
        ok = False
        try:
            if mode == "unshare" and shutil.which("unshare"):
                ok = subprocess.run(["unshare", "--net", "--map-root-user", "true"], capture_output=True, timeout=10).returncode == 0
            elif mode == "bwrap" and shutil.which("bwrap"):
                ok = subprocess.run(_bwrap_prefix() + ["true"], capture_output=True, timeout=10).returncode == 0
            elif mode == "rlimits":
                ok = True
        except (OSError, subprocess.SubprocessError):
            ok = False
        _PROBED[mode] = ok
        return ok


def _bwrap_prefix() -> List[str]:
    return ["bwrap", "--die-with-parent", "--unshare-all", "--ro-bind", "/", "/", "--tmpfs", "/tmp", "--dev", "/dev",
            "--proc", "/proc", "--"]


class SandboxRunner:
    """Thread-safe launcher.  One call = one fresh interpreter."""

    def __init__(self, cfg: SandboxCfg, hash_seed: int = 0, scratch_dir: Optional[str] = None):
        self.cfg = cfg
        self.hash_seed = hash_seed
        self.python = cfg.python or sys.executable
        self.scratch = scratch_dir or tempfile.gettempdir()
        if not _probe(cfg.isolation):
            raise IsolationUnavailable(f"sandbox isolation mode {cfg.isolation!r} is not available on this host")
        self.hash_seed_controlled = sys.version_info >= (3, 11)

    # -------------------------------------------------------------- #
    def _cmd(self, cpus: Optional[List[int]]) -> List[str]:
        if self.hash_seed_controlled:
            cmd = [self.python, "-P", "-s", "-B", WORKER_PATH]
        else:  # pragma: no cover - python 3.10
            cmd = [self.python, "-I", "-B", WORKER_PATH]
        if self.cfg.isolation == "unshare":
            cmd = ["unshare", "--net", "--map-root-user", "--"] + cmd
        elif self.cfg.isolation == "bwrap":
            cmd = _bwrap_prefix() + cmd
        if cpus and shutil.which("taskset"):
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

    def run(self, req: Dict[str, Any], *, hash_seed: Optional[int] = None, cpus: Optional[List[int]] = None,
            timeout_s: Optional[float] = None) -> Dict[str, Any]:
        hs = self.hash_seed if hash_seed is None else hash_seed
        payload = pickle.dumps(req, protocol=pickle.HIGHEST_PROTOCOL)
        workdir = tempfile.mkdtemp(prefix="tpchain_", dir=self.scratch)
        timeout = float(timeout_s if timeout_s is not None else self.cfg.process_timeout_s)
        kwargs: Dict[str, Any] = {}
        if self.cfg.drop_to_uid is not None and hasattr(os, "geteuid") and os.geteuid() == 0:
            kwargs["user"] = int(self.cfg.drop_to_uid)
            kwargs["group"] = int(self.cfg.drop_to_uid)
            os.chmod(workdir, 0o777)
        t0 = time.perf_counter_ns()
        timed_out = False
        rc: Optional[int] = None
        out = b""
        err = b""
        try:
            proc = subprocess.Popen(self._cmd(cpus), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    cwd=workdir, env=self._env(hs), start_new_session=True, **kwargs)
            try:
                out, err = proc.communicate(payload, timeout=max(0.5, timeout))
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                out, err = proc.communicate()
            rc = proc.returncode
        except OSError as e:
            return {"status": "infrastructure_error", "message": f"could not launch worker: {e}",
                    "_meta": RunMeta(time.perf_counter_ns() - t0, None, False, self.cfg.isolation, hs).__dict__}
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        meta = RunMeta(time.perf_counter_ns() - t0, rc, timed_out, self.cfg.isolation, hs if self.hash_seed_controlled else None)
        if timed_out:
            return {"status": "timeout", "message": "sandbox process exceeded its hard timeout", "_meta": meta.__dict__}
        if len(out) > self.cfg.max_output_bytes:
            return {"status": "output_too_large", "message": f"worker output {len(out)} bytes", "_meta": meta.__dict__}
        if rc != 0 or not out:
            tail = err.decode(errors="replace")[-400:]
            status = "resource" if ("MemoryError" in tail or rc in (-9, 137)) else "worker_error"
            return {"status": status, "message": f"worker exited with code {rc}: {tail.strip()}", "_meta": meta.__dict__}
        try:
            resp = json.loads(out.decode("utf-8"))
        except (ValueError, UnicodeDecodeError, RecursionError) as e:
            return {"status": "worker_error", "message": f"undecodable worker response: {e}", "_meta": meta.__dict__}
        if not isinstance(resp, dict) or "status" not in resp:
            return {"status": "worker_error", "message": "malformed worker response", "_meta": meta.__dict__}
        resp["_meta"] = meta.__dict__
        return resp

    # -------------------------------------------------------------- #
    def selftest(self) -> Dict[str, Any]:
        """What the OS allows a worker process to do under the configured isolation (trusted probe code)."""
        return self.run(self.base_request("selftest"), timeout_s=30)
