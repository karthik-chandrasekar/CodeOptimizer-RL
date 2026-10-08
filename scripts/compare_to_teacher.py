"""Does a student break the teacher's ceiling?  Task-by-task comparison on the same held-out tasks.

    python scripts/compare_to_teacher.py --teacher artifacts/opsd_rft/eval/opsd_errloc100_real_trajectories.jsonl \\
        --students s2m_rl=artifacts/distill/eval/s2m_rl_real_trajectories.jsonl s200k_rl=... [--tasks tasks.jsonl]

Per task, each model's best speedup over a full episode is summarised over its samples by the geometric mean (typical
attempt) and the maximum (best attempt).  For each student it reports the share of tasks where it beats the teacher by
more than --margin, the share where its best attempt beats the teacher's best attempt, the geometric-mean speedup
ratio student / teacher with a 95% interval that resamples functions, its overall maximum against the teacher's, and
the tasks with the largest wins.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from typing import Dict, List, Optional


def load(path: str) -> Dict[str, Dict]:
    by_task: Dict[str, Dict] = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        d = by_task.setdefault(r["task_id"], {"logs": [], "cluster": r.get("seed_name") or r["task_id"]})
        d["logs"].append(math.log(1.0 / max(float(r["best_ratio"]), 1e-9)))
    return by_task


def compare(teacher: Dict[str, Dict], student: Dict[str, Dict], margin: float, n_boot: int = 2000, seed: int = 0) -> Dict:
    common = sorted(set(teacher) & set(student))
    lm = math.log(1.0 + margin)
    rows, wins, losses, best_wins = [], 0, 0, 0
    for tid in common:
        t, s = teacher[tid]["logs"], student[tid]["logs"]
        t_gm, s_gm = sum(t) / len(t), sum(s) / len(s)
        wins += s_gm > t_gm + lm
        losses += s_gm < t_gm - lm
        best_wins += max(s) > max(t) + lm
        rows.append((tid, teacher[tid]["cluster"], s_gm - t_gm, math.exp(s_gm), math.exp(t_gm), math.exp(max(s)), math.exp(max(t))))
    by_cluster: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        by_cluster[r[1]].append(r[2])
    cl = [sum(v) / len(v) for v in by_cluster.values()]           # function-weighted, like the budget tables
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(cl) for _ in cl) / len(cl) for _ in range(n_boot)) if cl else [0.0]
    n = max(1, len(rows))
    return {"tasks": len(rows), "functions": len(cl), "win": wins / n, "loss": losses / n, "tie": 1 - (wins + losses) / n,
            "best_attempt_win": best_wins / n,
            "ratio": math.exp(sum(cl) / max(1, len(cl))), "ci": (math.exp(boots[int(0.025 * n_boot)]), math.exp(boots[int(0.975 * n_boot) - 1])),
            "max_student": max((r[5] for r in rows), default=1.0), "max_teacher": max((r[6] for r in rows), default=1.0),
            "top": sorted(rows, key=lambda r: -r[2])[:5]}


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher", required=True)
    p.add_argument("--students", nargs="+", required=True, help="name=trajectories.jsonl")
    p.add_argument("--tasks", default="", help="task file, to show the number of slow functions per task")
    p.add_argument("--margin", type=float, default=0.05, help="a win needs a speedup more than this fraction higher")
    a = p.parse_args(argv)
    slow = {}
    if a.tasks:
        for line in open(a.tasks):
            t = json.loads(line)
            slow[t["task_id"]] = len((t.get("meta") or {}).get("slow_functions") or [])
    teacher = load(a.teacher)
    print(f"student vs teacher, task by task (full episodes; win = more than {a.margin:.0%} faster; "
          f"ratio = geometric-mean speedup student / teacher, 95% CI resampling functions)")
    for spec in a.students:
        name, path = spec.split("=", 1)
        c = compare(teacher, load(path), a.margin)
        print(f"\n== {name}: {c['tasks']} tasks, {c['functions']} functions")
        print(f"   beats the teacher on {c['win']:.1%} of tasks, loses on {c['loss']:.1%}, ties {c['tie']:.1%}; "
              f"best attempt beats the teacher's best on {c['best_attempt_win']:.1%}")
        print(f"   speedup ratio x{c['ratio']:.2f} [x{c['ci'][0]:.2f}, x{c['ci'][1]:.2f}] | max speedup: student "
              f"{c['max_student']:.1f}x, teacher {c['max_teacher']:.1f}x")
        for tid, _, _, s_gm, t_gm, s_max, t_max in c["top"]:
            if s_gm <= t_gm:
                break
            extra = f", {slow[tid]} slow fns" if tid in slow else ""
            print(f"   win: {tid}  student {s_gm:.2f}x (best {s_max:.2f}x) vs teacher {t_gm:.2f}x (best {t_max:.2f}x){extra}")


if __name__ == "__main__":
    main()
