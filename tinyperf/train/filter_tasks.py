"""Keep only RL tasks of useful difficulty for the current policy.

    python -m tinyperf.train.filter_tasks --tasks artifacts/data/train_rl.jsonl \\
        --probe artifacts/eval/probe_trajectories.jsonl --out artifacts/data/train_rl_filtered.jsonl

``--probe`` is the ``*_trajectories.jsonl`` written by ``tinyperf.eval.evaluate`` when the starting
policy is run on (a prefix of) the task file.  Difficulty is estimated per *chain* (seed function +
degradation families), pooling all renamed copies, and a chain is kept when its success rate lies in
[lo, hi]: always-solved chains give GRPO groups with no signal, never-solved ones give no success to
learn from.  Chains the probe did not cover (or covered with fewer than ``--min_episodes``) are kept.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

Key = Tuple[str, str]


def chain_key(rec: Dict) -> Key:
    return rec["seed_name"], "+".join(rec["families"])


def task_key(rec: Dict) -> Key:
    return rec["task_id"], ""


def chain_success(probe_path: str, threshold: float, key=chain_key) -> Dict[Key, List[int]]:
    acc: Dict[Key, List[int]] = defaultdict(list)
    with open(probe_path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                acc[key(r)].append(int(1.0 / max(float(r["best_ratio"]), 1e-9) >= threshold))
    return acc


def filter_tasks(tasks_path: str, probe_path: str, out_path: str, lo: float, hi: float, threshold: float,
                 min_episodes: int, by: str = "chain", drop_unprobed: bool = False) -> Dict:
    """Keep tasks whose estimated success rate is in [lo, hi].  ``by="chain"`` pools renamed copies of a
    degradation chain; ``by="task"`` judges every task on its own (multi-function files are unique
    compositions).  ``drop_unprobed`` removes tasks the probe did not cover instead of keeping them."""
    key = task_key if by == "task" else chain_key
    succ = chain_success(probe_path, threshold, key)
    decision: Dict[Key, str] = {}
    for k, v in succ.items():
        if len(v) < min_episodes:
            decision[k] = "unprobed"
        else:
            rate = sum(v) / len(v)
            decision[k] = "too_easy" if rate > hi else "too_hard" if rate < lo else "keep"
    counts = defaultdict(int)
    kept_tasks, kept_fns, kept_chains = 0, set(), set()
    with open(tasks_path) as f, open(out_path, "w") as out:
        for line in f:
            if not line.strip():
                continue
            t = json.loads(line)
            k = key(t)
            d = decision.get(k, "unprobed")
            counts[d] += 1
            if d == "keep" or (d == "unprobed" and not drop_unprobed):
                out.write(line if line.endswith("\n") else line + "\n")
                kept_tasks += 1
                kept_fns.add(k[0])
                kept_chains.add(k)
    chains = defaultdict(int)
    for d in decision.values():
        chains[d] += 1
    return {"tasks_by_decision": dict(counts), "chains_by_decision": dict(chains), "kept_tasks": kept_tasks,
            "kept_functions": len(kept_fns), "kept_chains": len(kept_chains), "band": [lo, hi], "threshold": threshold}


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", required=True)
    p.add_argument("--probe", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lo", type=float, default=0.1)
    p.add_argument("--hi", type=float, default=0.9)
    p.add_argument("--threshold", type=float, default=1.5, help="speedup counted as success")
    p.add_argument("--min_episodes", type=int, default=8)
    p.add_argument("--by", choices=["chain", "task"], default="chain")
    p.add_argument("--drop_unprobed", action="store_true")
    a = p.parse_args(argv)
    res = filter_tasks(a.tasks, a.probe, a.out, a.lo, a.hi, a.threshold, a.min_episodes, a.by, a.drop_unprobed)
    print(json.dumps(res, indent=2))
    with open(a.out.replace(".jsonl", "_stats.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
