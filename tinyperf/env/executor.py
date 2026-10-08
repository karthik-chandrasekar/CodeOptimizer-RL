"""Drives ``tinyperf.env.worker`` subprocesses."""
from __future__ import annotations

import os
import pickle
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from tinyperf.common.config import EnvConfig


class Executor:
    """One call = one fresh interpreter. Thread-safe (subprocess-bound)."""

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self._pool: Optional[ThreadPoolExecutor] = None

    # ------------------------------------------------------------------ #
    def _cmd(self) -> List[str]:
        # The worker is self-contained, so run it by path under an isolated interpreter
        # (-I ignores PYTHONPATH/user site, so the candidate cannot see the project or training deps).
        worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py")
        cmd = [sys.executable, "-I", worker]
        pin = self.cfg.timing.pin_cpu
        if pin is not None and shutil.which("taskset"):
            cmd = ["taskset", "-c", str(pin)] + cmd
        return cmd

    def run(self, req: Dict[str, Any]) -> Dict[str, Any]:
        req = dict(req)
        req.setdefault("allowed_modules", list(self.cfg.allowed_modules))
        req.setdefault("limits", asdict(self.cfg.limits))
        req.setdefault("timing", asdict(self.cfg.timing))
        req.setdefault("float_rel_tol", self.cfg.float_rel_tol)
        req.setdefault("float_abs_tol", self.cfg.float_abs_tol)
        payload = pickle.dumps(req, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            proc = subprocess.run(
                self._cmd(),
                input=payload,
                capture_output=True,
                timeout=self.cfg.limits.process_timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "message": "sandbox process exceeded hard timeout"}
        if proc.returncode != 0 or not proc.stdout:
            err = proc.stderr.decode(errors="replace")[-1500:]
            status = "resource" if ("MemoryError" in err or proc.returncode in (-9, 137)) else "worker_error"
            return {"status": status, "message": f"worker exited with code {proc.returncode}: {err.strip()[-500:]}"}
        try:
            return pickle.loads(proc.stdout)
        except Exception as e:  # noqa: BLE001
            return {"status": "worker_error", "message": f"could not decode worker response: {e}"}

    # ------------------------------------------------------------------ #
    def baseline(self, code: str, func_name: str, correct_inputs: List[Tuple], perf_inputs: List[Tuple],
                 group_by_first_arg: bool = False) -> Dict[str, Any]:
        """Record C0's behaviour on the hidden inputs and its (unpaired) runtime."""
        return self.run({
            "mode": "baseline",
            "code": code,
            "func_name": func_name,
            "group_by_first_arg": group_by_first_arg,
            "correct_inputs": correct_inputs,
            "perf_inputs": perf_inputs,
            "timing": {**asdict(self.cfg.timing), "paired": False},
        })

    def evaluate(self, code: str, task: "Task", baseline_code: Optional[str] = None, correctness_only: bool = False, timing_seed: int = 0) -> Dict[str, Any]:
        from tinyperf.env.task import Task  # local import to avoid cycle

        assert isinstance(task, Task)
        drv = (task.meta or {}).get("driver")
        if drv:  # multi-function file task: measure the whole file through the hidden dispatcher
            from tinyperf.env.files import with_driver
            code = with_driver(code, drv)
            baseline_code = with_driver(baseline_code, drv) if baseline_code else baseline_code
        return self.run({
            "mode": "correctness_only" if correctness_only else "evaluate",
            "code": code,
            "func_name": task.func_name,
            "group_by_first_arg": bool(drv),
            "correct_inputs": task.correct_inputs,
            "expected": task.expected,
            "perf_inputs": task.perf_inputs,
            "baseline_code": baseline_code,
            "timing_seed": timing_seed,
        })

    # ------------------------------------------------------------------ #
    def map(self, fn, items):
        """Run fn over items with up to max_parallel_envs concurrent workers."""
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=max(1, self.cfg.max_parallel_envs))
        return list(self._pool.map(fn, items))

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
