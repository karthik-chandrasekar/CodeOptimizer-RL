"""Append-only execution ledger, content-addressed artifact store and per-episode event logs.

* :class:`ObjectStore` - immutable artifacts addressed by SHA-256 (``objects/ab/abcdef...``), written atomically.
* :class:`AppendLedger` - JSONL records appended under a lock with flush+fsync per batch; never rewritten.
  Rejected candidates stay in the ledger (they are evidence), they are only excluded from imitation targets.
* :class:`EventLog` - one event per role call, verification stage, timing run, acceptance and terminal outcome,
  with monotonic start/end times, process CPU time and budget before/after.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from tinyperf_chain.hashing import sha256_bytes


class ObjectStore:
    def __init__(self, root: str):
        self.root = root

    def _path(self, h: str) -> str:
        return os.path.join(self.root, h[:2], h)

    def put_bytes(self, data: bytes) -> str:
        h = sha256_bytes(data)
        p = self._path(h)
        if not os.path.exists(p):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            tmp = f"{p}.{os.getpid()}.{threading.get_ident()}.tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, p)
        return h

    def put_text(self, s: str) -> str:
        return self.put_bytes(s.encode("utf-8"))

    def put_json(self, obj: Any) -> str:
        return self.put_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8"))

    def get_bytes(self, h: str) -> bytes:
        with open(self._path(h), "rb") as f:
            data = f.read()
        if sha256_bytes(data) != h:
            raise IOError(f"artifact {h} is corrupted")
        return data

    def get_text(self, h: str) -> str:
        return self.get_bytes(h).decode("utf-8")

    def get_json(self, h: str) -> Any:
        return json.loads(self.get_bytes(h))

    def has(self, h: str) -> bool:
        return os.path.exists(self._path(h))


class AppendLedger:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, rec: Dict[str, Any]) -> None:
        self.extend([rec])

    def extend(self, recs: List[Dict[str, Any]]) -> None:
        if not recs:
            return
        lines = "".join(json.dumps(r, separators=(",", ":"), default=str) + "\n" for r in recs)
        with self._lock:
            with open(self.path, "a") as f:
                f.write(lines)
                f.flush()
                os.fsync(f.fileno())

    def read(self) -> Iterator[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue   # a torn final line from an interrupted writer is ignored, never repaired in place


class EventLog:
    """Per-episode event stream: kept in memory and, when a path is given, appended to a ``.partial`` JSONL file as
    each event happens (an interrupted episode leaves its events on disk), then renamed atomically on ``flush``.

    Every event carries ``t_mono_ns`` (emission), ``proc_cpu_s`` and, when the caller measured an interval,
    ``start_mono_ns`` / ``end_mono_ns`` via :meth:`span`.  Episode-wide constants (bundle hash, system id) are
    stamped on every event by :meth:`stamp`.
    """

    def __init__(self, episode_id: str, path: Optional[str] = None):
        self.episode_id = episode_id
        self.path = path
        self.events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._seq = 0
        self._constants: Dict[str, Any] = {}
        self._fh = None
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(path + ".partial", "w")

    def stamp(self, **constants: Any) -> None:
        """Fields copied onto every subsequent event (bundle hash, system id, task id)."""
        self._constants.update(constants)

    def emit(self, kind: str, **fields: Any) -> Dict[str, Any]:
        with self._lock:
            ev = {"episode_id": self.episode_id, "seq": self._seq, "kind": kind,
                  "t_mono_ns": time.monotonic_ns(), "proc_cpu_s": time.process_time(), **self._constants, **fields}
            self._seq += 1
            self.events.append(ev)
            if self._fh is not None:
                self._fh.write(json.dumps(ev, separators=(",", ":"), default=str) + "\n")
                self._fh.flush()
        return ev

    @staticmethod
    def span(start_ns: int, end_ns: Optional[int] = None) -> Dict[str, int]:
        """``start_mono_ns`` / ``end_mono_ns`` / ``elapsed_ns`` fields for an interval measured by the caller."""
        end_ns = time.monotonic_ns() if end_ns is None else end_ns
        return {"start_mono_ns": start_ns, "end_mono_ns": end_ns, "elapsed_ns": end_ns - start_ns}

    def flush(self) -> None:
        if not self.path:
            return
        with self._lock:
            if self._fh is not None:
                self._fh.flush()
                os.fsync(self._fh.fileno())
                self._fh.close()
                self._fh = None
            os.replace(self.path + ".partial", self.path)

    def close(self) -> None:
        """Abandon without the final rename (the ``.partial`` file stays as evidence of an interrupted episode)."""
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
