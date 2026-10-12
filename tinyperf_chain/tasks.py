"""Chain tasks, split into what models may see and what only the evaluator may see.

:class:`PublicTask`  source of the immutable original, the contract, function order, a public workload used for
                     profiling, a few public example inputs (smoke checks), and non-privileged metadata.
:class:`PrivateSuite` online correctness inputs, online timing workload, independent audit correctness inputs and
                     audit timing workload, their seeds, and privileged provenance (fast reference, which
                     functions were degraded, degradation families).  Privileged fields may create training labels
                     during construction; they never enter prompts.

Inputs are stored with the plain-value codec (JSON), not pickle.  Every input is a dispatcher call ``(k, *args)``
where ``k`` indexes :attr:`PublicTask.functions`.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from tinyperf.env.files import ENTRY
from tinyperf_chain.contracts import TaskContract
from tinyperf_chain.program import Program
from tinyperf_chain.sandbox.codec import decode, encode

Call = Tuple[Any, ...]   # (k, *args)


def _enc_calls(calls: Sequence[Call]) -> List[Any]:
    return [encode(tuple(c)) for c in calls]


def _dec_calls(rows: Sequence[Any]) -> List[Call]:
    return [tuple(decode(r)) for r in rows]


@dataclass
class PublicTask:
    task_id: str
    cluster_id: str                   # independence unit for splits, weights and bootstrap (source-function group)
    split: str                        # train | dev | confirm
    source: str                       # immutable original P0 (normalized, no dispatcher)
    functions: List[str]              # dispatch order
    contract: TaskContract
    public_workload: List[Call]       # profiling only
    public_examples: Dict[str, List[Tuple]] = field(default_factory=dict)   # function -> small arg tuples (no k)
    probe_menu_id: str = "plain_v1"
    public_workload_id: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)          # non-privileged (kind, k, component clusters)

    @property
    def entrypoint(self) -> str:
        return ENTRY

    def original(self) -> Program:
        return Program(self.source, tuple(self.functions))

    def to_row(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id, "cluster_id": self.cluster_id, "split": self.split, "source": self.source,
            "functions": self.functions, "contract": self.contract.to_dict(),
            "public_workload": _enc_calls(self.public_workload),
            "public_examples": {k: _enc_calls(v) for k, v in self.public_examples.items()},
            "probe_menu_id": self.probe_menu_id, "public_workload_id": self.public_workload_id, "meta": self.meta,
        }

    @classmethod
    def from_row(cls, r: Dict[str, Any]) -> "PublicTask":
        return cls(
            task_id=r["task_id"], cluster_id=r["cluster_id"], split=r["split"], source=r["source"],
            functions=list(r["functions"]), contract=TaskContract.from_dict(r["contract"]),
            public_workload=_dec_calls(r["public_workload"]),
            public_examples={k: _dec_calls(v) for k, v in r.get("public_examples", {}).items()},
            probe_menu_id=r.get("probe_menu_id", "plain_v1"), public_workload_id=r.get("public_workload_id", ""),
            meta=r.get("meta", {}),
        )


@dataclass
class PrivateSuite:
    task_id: str
    online_correctness: List[Call]
    online_timing: List[Call]
    audit_correctness: List[Call]
    audit_timing: List[Call]
    seeds: Dict[str, int] = field(default_factory=dict)
    privileged: Dict[str, Any] = field(default_factory=dict)   # fast_source, slow_functions, families, ...

    def to_row(self) -> Dict[str, Any]:
        return {"task_id": self.task_id, "online_correctness": _enc_calls(self.online_correctness),
                "online_timing": _enc_calls(self.online_timing), "audit_correctness": _enc_calls(self.audit_correctness),
                "audit_timing": _enc_calls(self.audit_timing), "seeds": self.seeds, "privileged": self.privileged}

    @classmethod
    def from_row(cls, r: Dict[str, Any]) -> "PrivateSuite":
        return cls(task_id=r["task_id"], online_correctness=_dec_calls(r["online_correctness"]),
                   online_timing=_dec_calls(r["online_timing"]), audit_correctness=_dec_calls(r["audit_correctness"]),
                   audit_timing=_dec_calls(r["audit_timing"]), seeds=r.get("seeds", {}), privileged=r.get("privileged", {}))


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #
def _write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


class TaskStore:
    """``<root>/tasks/<suite>.jsonl`` (public) and ``<root>/private/<suite>.jsonl`` (restricted)."""

    def __init__(self, public_dir: str, private_dir: str):
        self.public_dir = public_dir
        self.private_dir = private_dir

    def public_path(self, suite: str) -> str:
        return os.path.join(self.public_dir, f"{suite}.jsonl")

    def private_path(self, suite: str) -> str:
        return os.path.join(self.private_dir, f"{suite}.jsonl")

    def save(self, suite: str, tasks: Sequence[PublicTask], suites: Sequence[PrivateSuite]) -> None:
        ids = [t.task_id for t in tasks]
        if ids != [s.task_id for s in suites]:
            raise ValueError("public tasks and private suites must align")
        _write_jsonl(self.public_path(suite), [t.to_row() for t in tasks])
        _write_jsonl(self.private_path(suite), [s.to_row() for s in suites])
        try:
            os.chmod(self.private_path(suite), 0o600)
        except OSError:
            pass

    def load_public(self, suite: str, limit: int = 0) -> List[PublicTask]:
        out = []
        for i, r in enumerate(_read_jsonl(self.public_path(suite))):
            if limit and i >= limit:
                break
            out.append(PublicTask.from_row(r))
        return out

    def load_private(self, suite: str) -> Dict[str, PrivateSuite]:
        """Evaluator / verifier / data-construction access only."""
        return {r["task_id"]: PrivateSuite.from_row(r) for r in _read_jsonl(self.private_path(suite))}

    def suites(self) -> List[str]:
        if not os.path.isdir(self.public_dir):
            return []
        return sorted(p[:-6] for p in os.listdir(self.public_dir) if p.endswith(".jsonl"))
