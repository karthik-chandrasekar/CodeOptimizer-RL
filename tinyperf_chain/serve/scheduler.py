"""Resident-role request scheduler: one queue and one worker thread per engine (role checkpoint).

Requests from many concurrently running episodes are grouped by (sampling spec, stop ids, output cap, prompt-length
bucket) and decoded together, up to ``max_batch`` rows.  Two modes (14.3):

* ``latency`` (default): a batch is dispatched as soon as the short batching window closes;
* ``throughput``: the worker waits for the window to fill a batch to ``max_batch`` when more work is queued, and
  always serves the largest pending group first.

The number of requests admitted per role (queued or pending) is bounded by ``max_pending`` through a semaphore, so
``submit`` blocks when a role is saturated - backpressure instead of unbounded memory - and pending groups are
batched among themselves rather than one request at a time.  Every request keeps its *own* deadline: the engine
stops a row when that row's deadline passes and never cuts another row short.  Request-level randomness comes from
per-request seeds, so results do not depend on how requests were batched.  The verifier runs in other threads
(subprocess-bound), so generation for one task overlaps with checks of another.
"""
from __future__ import annotations

import queue
import threading
import time
from collections import OrderedDict
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
                 max_pending: int = 256, length_buckets: Sequence[int] = (256, 512, 1024, 2048), mode: str = "latency"):
        if mode not in ("latency", "throughput"):
            raise ValueError("scheduler mode must be latency|throughput")
        self.engines = engines
        self.window = window_ms / 1000.0
        self.max_batch = max_batch
        self.max_pending = max_pending
        self.mode = mode
        self.buckets = sorted(length_buckets)
        self.queues: Dict[str, "queue.Queue[_Req]"] = {k: queue.Queue() for k in engines}
        self._slots: Dict[str, threading.BoundedSemaphore] = {k: threading.BoundedSemaphore(max_pending) for k in engines}
        self.stats: Dict[str, Dict[str, float]] = {k: {"batches": 0, "rows": 0, "queue_wait_s": 0.0, "max_in_flight": 0,
                                                       "wasted_steps": 0, "timed_out_rows": 0} for k in engines}
        self._in_flight: Dict[str, int] = {k: 0 for k in engines}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads = []
        for k in engines:
            th = threading.Thread(target=self._loop, args=(k,), daemon=True, name=f"sched-{k}")
            th.start()
            self._threads.append(th)

    # -------------------------------------------------------------- #
    def submit(self, key: str, prompt: Sequence[int], max_new: int, spec: SamplingSpec, stop_ids: Sequence[int],
               seed: int, pad_id: int, deadline: Optional[float] = None) -> Future:
        r = _Req(list(prompt), int(max_new), spec, tuple(stop_ids), int(seed), int(pad_id), deadline)
        self._slots[key].acquire()          # blocks when the role is saturated (backpressure, bounded admission)
        with self._lock:
            self._in_flight[key] += 1
            self.stats[key]["max_in_flight"] = max(self.stats[key]["max_in_flight"], self._in_flight[key])
        r.future.add_done_callback(lambda _f, k=key: self._release(k))
        self.queues[key].put(r)
        return r.future

    def _release(self, key: str) -> None:
        with self._lock:
            self._in_flight[key] -= 1
        self._slots[key].release()

    def generate(self, key: str, prompt: Sequence[int], max_new: int, spec: SamplingSpec, stop_ids: Sequence[int],
                 seed: int, pad_id: int, deadline: Optional[float] = None) -> GenResult:
        return self.submit(key, prompt, max_new, spec, stop_ids, seed, pad_id, deadline).result()

    def in_flight(self, key: str) -> int:
        with self._lock:
            return self._in_flight[key]

    # -------------------------------------------------------------- #
    def _drain(self, q: "queue.Queue[_Req]", pending: "OrderedDict[Tuple, List[_Req]]", wait_s: float) -> None:
        """Move everything that arrives within ``wait_s`` from the queue into the pending groups."""
        t_end = time.monotonic() + wait_s
        while True:
            rest = t_end - time.monotonic()
            try:
                r = q.get(timeout=max(0.0, rest)) if rest > 0 else q.get_nowait()
            except queue.Empty:
                return
            pending.setdefault(r.group_key(self.buckets), []).append(r)

    def _pick(self, pending: "OrderedDict[Tuple, List[_Req]]") -> Tuple[Tuple, List[_Req]]:
        if self.mode == "throughput":
            gk = max(pending, key=lambda k: len(pending[k]))     # largest group first
        else:
            gk = min(pending, key=lambda k: min(r.t_submit for r in pending[k]))   # oldest request first
        group = pending[gk]
        batch, rest = group[: self.max_batch], group[self.max_batch:]
        if rest:
            pending[gk] = rest
        else:
            del pending[gk]
        return gk, batch

    def _loop(self, key: str) -> None:
        q = self.queues[key]
        eng = self.engines[key]
        pending: "OrderedDict[Tuple, List[_Req]]" = OrderedDict()
        while not self._stop.is_set():
            if not pending:
                try:
                    r = q.get(timeout=0.1)
                except queue.Empty:
                    continue
                pending.setdefault(r.group_key(self.buckets), []).append(r)
            # batching window: in throughput mode keep waiting while no group has filled a full batch
            self._drain(q, pending, self.window)
            if self.mode == "throughput":
                t_end = time.monotonic() + 4 * self.window
                while max(len(v) for v in pending.values()) < self.max_batch and time.monotonic() < t_end:
                    self._drain(q, pending, self.window)
            gk, batch = self._pick(pending)
            first = batch[0]
            now = time.monotonic()
            st = self.stats[key]
            st["batches"] += 1
            st["rows"] += len(batch)
            st["queue_wait_s"] += sum(now - r.t_submit for r in batch)
            try:
                res = eng.generate([r.prompt for r in batch], first.max_new, first.spec, first.stop_ids,
                                   pad_id=first.pad_id, seeds=[r.seed for r in batch], deadlines=[r.deadline for r in batch])
                st["wasted_steps"] += sum(o.wasted_steps for o in res)
                st["timed_out_rows"] += sum(int(o.timed_out) for o in res)
                for r, out in zip(batch, res):
                    out.queue_wait_ns = int((now - r.t_submit) * 1e9)   # type: ignore[attr-defined]
                    r.future.set_result(out)
            except Exception as e:  # noqa: BLE001
                for r in batch:
                    if not r.future.done():
                        r.future.set_exception(e)

    def close(self) -> None:
        self._stop.set()
        for th in self._threads:
            th.join(timeout=2)
