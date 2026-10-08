"""Task = one optimization problem: (C0, X_correct, X_perf, Y, T0) plus provenance."""
from __future__ import annotations

import base64
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from tinyperf.common.utils import read_jsonl, write_jsonl


def _enc(obj: Any) -> str:
    return base64.b64encode(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")


def _dec(s: str) -> Any:
    return pickle.loads(base64.b64decode(s.encode("ascii")))


@dataclass
class Task:
    task_id: str
    func_name: str
    source: str                                   # C0 - the (slow) starting implementation
    correct_inputs: List[Tuple]                   # hidden X_correct (arg tuples)
    perf_inputs: List[Tuple]                      # hidden X_perf
    expected: List[Dict[str, Any]]                # Y: C0 behaviour records on X_correct
    baseline_ns: float                            # T0 (informational; env uses paired timing)
    # provenance (used for SFT trajectory construction & analysis, never shown to the policy)
    seed_name: str = ""
    families: List[str] = field(default_factory=list)   # degradations applied, in order
    chain: List[str] = field(default_factory=list)      # [C_fast, C_1, ..., C_slow]; chain[-1] == source
    fast_source: Optional[str] = None
    reference_speedup: Optional[float] = None           # T0 / T(fast) measured at build time
    split: str = "train"
    meta: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def to_row(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "func_name": self.func_name,
            "source": self.source,
            "correct_inputs": _enc(self.correct_inputs),
            "perf_inputs": _enc(self.perf_inputs),
            "expected": _enc(self.expected),
            "baseline_ns": self.baseline_ns,
            "seed_name": self.seed_name,
            "families": self.families,
            "chain": self.chain,
            "fast_source": self.fast_source,
            "reference_speedup": self.reference_speedup,
            "split": self.split,
            "meta": self.meta,
        }

    @classmethod
    def from_row(cls, r: Dict[str, Any]) -> "Task":
        return cls(
            task_id=r["task_id"],
            func_name=r["func_name"],
            source=r["source"],
            correct_inputs=_dec(r["correct_inputs"]),
            perf_inputs=_dec(r["perf_inputs"]),
            expected=_dec(r["expected"]),
            baseline_ns=r["baseline_ns"],
            seed_name=r.get("seed_name", ""),
            families=r.get("families", []),
            chain=r.get("chain", []),
            fast_source=r.get("fast_source"),
            reference_speedup=r.get("reference_speedup"),
            split=r.get("split", "train"),
            meta=r.get("meta", {}),
        )


def load_tasks(path: str, limit: int = 0) -> List[Task]:
    return [Task.from_row(r) for r in read_jsonl(path, limit=limit)]


def iter_tasks(path: str) -> Iterator[Task]:
    for r in read_jsonl(path):
        yield Task.from_row(r)


def save_tasks(path: str, tasks: List[Task]) -> int:
    return write_jsonl(path, (t.to_row() for t in tasks))
