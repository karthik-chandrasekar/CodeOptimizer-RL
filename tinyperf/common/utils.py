from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np


def get_logger(name: str = "tinyperf") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
        log.propagate = False
    return log


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def read_jsonl(path: str | Path, limit: int = 0) -> Iterator[Dict[str, Any]]:
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
            n += 1
    return n


class Timer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0


class MetricLogger:
    """Minimal metrics logger: stdout + jsonl + optional wandb."""

    def __init__(self, out_dir: str, wandb_project: Optional[str] = None, run_name: str = "run", config: Optional[Dict] = None):
        self.log = get_logger()
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self.path = os.path.join(out_dir, "metrics.jsonl")
        self.wandb = None
        if wandb_project:
            try:
                import wandb

                self.wandb = wandb.init(project=wandb_project, name=run_name, config=config)
            except Exception as e:  # pragma: no cover
                self.log.warning(f"wandb disabled: {e}")

    def log_metrics(self, step: int, metrics: Dict[str, Any]) -> None:
        row = {"step": step, **metrics}
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        pretty = " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
        self.log.info(f"step {step} | {pretty}")
        if self.wandb is not None:
            self.wandb.log(row, step=step)
