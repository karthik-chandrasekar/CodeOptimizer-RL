"""Rejection fine-tuning (RFT) data from the policy's own successes on TRAINING tasks.

    python -m tinyperf.train.rft_data --config configs/sft_data.yaml \\
        --tasks artifacts/data_mined/files_msft_rl.jsonl --samples artifacts/rft/samples_trajectories.jsonl \\
        --out artifacts/rft/rft_trajectories.jsonl

``--samples`` is the ``*_trajectories.jsonl`` written by ``tinyperf.eval.evaluate`` when the policy is sampled on
``--tasks`` (training tasks only - never a test split).  A successful episode (best speedup >= ``--threshold``) is
not imitated as-is: its failed edits and edits to already-fast functions are exactly the scattershot behaviour we
do not want to reinforce.  Instead it is *cleaned*:

1. take the functions its final best file changed relative to the original,
2. keep only the task's slow functions (file tasks) and order them hottest-first by the step-0 profile,
3. replay "fix f1, fix f2, ..., STOP" through the real environment.

The result has only useful edits, the first always on the hottest slow function it fixed, with genuine
observations, re-verified correct and fast - in the same row format as SFT trajectories (``method="rft"``).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from tinyperf.common.config import load_config
from tinyperf.common.utils import get_logger
from tinyperf.env.env import PerfEnv
from tinyperf.env.executor import Executor
from tinyperf.env.task import Task, load_tasks
from tinyperf.train.sft_data import replay

log = get_logger("rft_data")


def functions_of(src: str) -> Dict[str, str]:
    return {n.name: ast.unparse(n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}


def cleaned_edits(task: Task, rec: Dict) -> List[str]:
    """The episode's useful edits: changed slow functions, final versions, hottest first."""
    try:
        before, after = functions_of(task.source), functions_of(rec["best_code"])
    except (SyntaxError, KeyError, TypeError, ValueError, RecursionError, MemoryError):
        return []
    slow = set((task.meta or {}).get("slow_functions") or [])
    changed = [n for n in after if n in before and after[n] != before[n] and (not slow or n in slow)]
    prof = (task.meta or {}).get("profile0") or {}
    order = list(before)
    changed.sort(key=lambda n: (-prof.get(n, 0.0), order.index(n)))
    return [after[n] for n in changed]


def select_jobs(tasks: Dict[str, Task], samples_path: str, threshold: float, per_task: int, horizon: int) -> Tuple[List, Dict]:
    by_task: Dict[str, List[Tuple[float, Dict]]] = defaultdict(list)
    stats = {"episodes": 0, "successes": 0, "tasks_sampled": 0, "tasks_with_success": 0}
    seen_tasks = set()
    with open(samples_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["task_id"] not in tasks:
                continue
            stats["episodes"] += 1
            seen_tasks.add(r["task_id"])
            sp = 1.0 / max(float(r["best_ratio"]), 1e-9)
            if sp >= threshold:
                stats["successes"] += 1
                by_task[r["task_id"]].append((sp, r))
    stats["tasks_sampled"] = len(seen_tasks)
    stats["tasks_with_success"] = len(by_task)
    jobs = []
    for tid, lst in by_task.items():
        lst.sort(key=lambda x: -x[0])             # best episodes first
        kept = set()
        for sp, r in lst:
            edits = cleaned_edits(tasks[tid], r)
            if not edits or len(edits) > horizon - 1 or tuple(edits) in kept:
                continue
            kept.add(tuple(edits))
            jobs.append((tid, edits, sp))
            if len(kept) >= per_task:
                break
    return jobs, stats


def build(cfg, tasks_path: str, samples_path: str, out_path: str, threshold: float, per_task: int) -> Dict:
    tasks = {t.task_id: t for t in load_tasks(tasks_path)}
    horizon = cfg.env.horizon
    jobs, stats = select_jobs(tasks, samples_path, threshold, per_task, horizon)
    ex = Executor(cfg.env)

    def work(job) -> Optional[Dict]:
        tid, edits, sp = job
        env = PerfEnv(ex, cfg.env, feedback="full", horizon=horizon)
        row = replay(env, tasks[tid], edits)
        if not row or any(s["status"] not in ("ok", "stop") for s in row["steps"]):
            return None                             # a cleaned edit failed on replay
        if 1.0 / max(row["final_ratio"], 1e-9) < threshold:
            return None                             # timing on replay fell below the bar
        row["method"] = "rft"
        row["sampled_speedup"] = sp
        return row

    rows = [r for r in ex.map(work, jobs) if r]
    ex.close()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    first_hot = 0
    for r in rows:
        t = tasks[r["task_id"]]
        prof = (t.meta or {}).get("profile0") or {}
        slow_prof = {n: prof.get(n, 0.0) for n in (t.meta or {}).get("slow_functions") or []}
        if slow_prof:
            first = functions_of(r["steps"][0]["action"].split("<EDIT>")[-1].split("</EDIT>")[0])
            first_hot += bool(first) and next(iter(first)) == max(slow_prof, key=slow_prof.get)
    stats.update({"jobs": len(jobs), "rows": len(rows), "replay_rejected": len(jobs) - len(rows),
                  "mean_edits": sum(r["n_edits"] for r in rows) / max(1, len(rows)),
                  "first_edit_is_hottest_slow_function": first_hot / max(1, len(rows))})
    with open(os.path.splitext(out_path)[0] + "_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"RFT data: {stats}")
    return stats


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/sft_data.yaml")
    p.add_argument("--tasks", required=True, help="the TRAINING tasks the policy was sampled on")
    p.add_argument("--samples", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, default=1.5)
    p.add_argument("--per_task", type=int, default=2)
    p.add_argument("overrides", nargs="*")
    a = p.parse_args(argv)
    build(load_config(a.config, a.overrides), a.tasks, a.samples, a.out, a.threshold, a.per_task)


if __name__ == "__main__":
    main()
