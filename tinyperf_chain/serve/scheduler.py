"""Resident-role request scheduler: one queue and one worker thread per engine (role checkpoint).

Requests from many concurrently running episodes are grouped by (sampling spec, stop ids, output cap, prompt-length
bucket) within a short batching window, up to ``max_batch`` rows, and decoded together.  Queues are bounded
(``max_pending``), so ``submit`` blocks when a role is saturated - backpressure instead of unbounded memory.  The
verifier runs in other threads (subprocess-bound), so generation for one task overlaps with checks of another.

Request-level randomness comes from per-request seeds, so results do not depend on how requests were batched.
"""
from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from tinyperf_chain.serve.engine import GenResult, RoleEngine, SamplingSpec


@dataclass
class _Req:
    prompt: List[int]
    max_new: int
    spec: SamplingSpec
    stop_ids: Tuple[int, ...]
    seed: int
    pad_id: int
    deadline: Optional[float]
    future: Future = field(default_factory=Future)
    t_submit: float = field(default_factory=time.monotonic)

    def group_key(self, buckets: Sequence[int]) -> Tuple:
        n = len(self.prompt)
        b = next((x for x in buckets if n <= x), buckets[-1] if buckets else n)
        return (self.spec.key(), self.stop_ids, self.max_new, b, self.pad_id)


class RoleScheduler:
    def __init__(self, engines: Dict[str, RoleEngine], *, window_ms: float = 2.0, max_batch: int = 32,
                 max_pending: int = 256, length_buckets: Sequence[int] = (256, 512, 1024, 2048)):
        self.engines = engines
        self.window = window_ms / 1000.0
        self.max_batch = max_batch
        self.buckets = sorted(length_buckets)
        self.queues: Dict[str, "queue.Queue[_Req]"] = {k: queue.Queue(maxsize=max_pending) for k in engines}
        self.stats: Dict[str, Dict[str, float]] = {k: {"batches": 0, "rows": 0, "queue_wait_s": 0.0} for k in engines}
        self._stop = threading.Event()
        self._threads = []
        for k in engines:
            th = threading.Thread(target=self._loop, args=(k,), daemon=True, name=f"sched-{k}")
            th.start()
            self._threads.append(th)

    def submit(self, key: str, prompt: Sequence[int], max_new: int, spec: SamplingSpec, stop_ids: Sequence[int],
               seed: int, pad_id: int, deadline: Optional[float] = None) -> Future:
        r = _Req(list(prompt), int(max_new), spec, tuple(stop_ids), int(seed), int(pad_id), deadline)
        self.queues[key].put(r)          # blocks when the role is saturated (backpressure)
        return r.future

    def generate(self, key: str, prompt: Sequence[int], max_new: int, spec: SamplingSpec, stop_ids: Sequence[int],
                 seed: int, pad_id: int, deadline: Optional[float] = None) -> GenResult:
        return self.submit(key, prompt, max_new, spec, stop_ids, seed, pad_id, deadline).result()

    def _loop(self, key: str) -> None:
        q = self.queues[key]
        eng = self.engines[key]
        pending: List[_Req] = []
        while not self._stop.is_set():
            try:
                first = pending.pop(0) if pending else q.get(timeout=0.1)
            except queue.Empty:
                continue
            batch = [first]
            gk = first.group_key(self.buckets)
            t_end = time.monotonic() + self.window
            while len(batch) < self.max_batch:
                rest = t_end - time.monotonic()
                try:
                    r = q.get(timeout=max(0.0, rest)) if rest > 0 else q.get_nowait()
                except queue.Empty:
                    break
                if r.group_key(self.buckets) == gk:
                    batch.append(r)
                else:
                    pending.append(r)
            now = time.monotonic()
            st = self.stats[key]
            st["batches"] += 1
            st["rows"] += len(batch)
            st["queue_wait_s"] += sum(now - r.t_submit for r in batch)
            try:
                dl = [r.deadline for r in batch if r.deadline is not None]
                res = eng.generate([r.prompt for r in batch], first.max_new, first.spec, first.stop_ids,
                                   pad_id=first.pad_id, seeds=[r.seed for r in batch], deadline=min(dl) if dl else None)
                for r, out in zip(batch, res):
                    r.future.set_result(out)
            except Exception as e:  # noqa: BLE001
                for r in batch:
                    if not r.future.done():
                        r.future.set_exception(e)

    def close(self) -> None:
        self._stop.set()
        for th in self._threads:
            th.join(timeout=2)
