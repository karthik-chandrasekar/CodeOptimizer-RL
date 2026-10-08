"""Hint-following warm-up for OPSD.

    python -m tinyperf.train.hint_sft_data --config configs/sft_data.yaml \\
        --tasks artifacts/data_mined/files_msft_rl.jsonl --mix artifacts/mined_sft/trajectories_all.jsonl \\
        --out artifacts/opsd/hint_train.jsonl

OPSD needs a teacher that *uses* its hint; a 20M model ignores an unfamiliar ``hint=`` line (it ignored the
profile line entirely).  So the policy first learns what hints mean: for each training file we replay an oracle
demonstration (slow functions fixed hottest-first from their known fast versions, then STOP) and put a hint on
every observation, worded exactly as OPSD will word it (:mod:`tinyperf.train.hints`):

* a localization hint ("the slowest unfixed function is g" / "all slow functions are fixed; STOP is right"),
* plus, at random, an outcome hint about the step ("the next edit to g changes its behaviour; it must preserve it
  exactly", "... is not valid ...", "editing h gives no speedup", "editing g speeds the code up", ...);
  some steps get the outcome hint alone, so the teacher also works in error-only mode.

The unhinted half of the mix is the ORIGINAL SFT data (no oracle ordering), so the localization behaviour stays
behind the hint: only OPSD can move it into the unhinted policy - which is what the experiment tests.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
from typing import Dict, List, Optional

from tinyperf.common.config import load_config
from tinyperf.common.utils import get_logger
from tinyperf.env.env import PerfEnv
from tinyperf.env.executor import Executor
from tinyperf.env.files import edited_functions
from tinyperf.env.protocol import parse_action
from tinyperf.env.task import Task, load_tasks
from tinyperf.train.hints import add_hint, code_from_obs, localization_hint, outcome_hint
from tinyperf.train.sft_data import replay

log = get_logger("hint_sft_data")


def _functions(src: str) -> Dict[str, str]:
    return {n.name: ast.unparse(n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}


def hinted_demo(ex: Executor, cfg, task: Task, rng: random.Random) -> Optional[Dict]:
    meta = task.meta or {}
    if not task.fast_source or not meta.get("slow_functions"):
        return None
    fast = _functions(task.fast_source)
    prof = meta.get("profile0") or {}
    slow = sorted([f for f in meta["slow_functions"] if f in fast], key=lambda f: -prof.get(f, 0.0))
    if not slow or len(slow) > cfg.env.horizon - 1:
        return None
    env = PerfEnv(ex, cfg.env, feedback="full", horizon=cfg.env.horizon)
    row = replay(env, task, [fast[f] for f in slow])
    if not row or any(s["status"] not in ("ok", "stop") for s in row["steps"]):
        return None
    fast_fns = [f for f in meta.get("functions", []) if f not in meta["slow_functions"]]
    for s in row["steps"]:
        loc = localization_hint(task, code_from_obs(s["obs"]))
        a = parse_action(s["action"])
        tgt = edited_functions(a.code) if a.kind == "edit" and a.code else []
        if a.kind == "stop":
            pre = outcome_hint("stop", "stop", False, [], []) if rng.random() < 0.5 else None
        else:
            kind = rng.choice(["loc", "careful", "valid", "nogain", "success"])
            pre = {"careful": outcome_hint("edit", "incorrect", False, tgt),
                   "valid": outcome_hint("edit", "syntax_error", False, tgt),
                   "nogain": outcome_hint("edit", "ok", False, [rng.choice(fast_fns)]) if fast_fns else None,
                   "success": outcome_hint("edit", "ok", True, tgt)}.get(kind)
        parts = [pre, loc] if (pre is None or rng.random() < 0.7) else [pre]   # 30%: outcome hint alone
        s["obs"] = add_hint(s["obs"], "; ".join(p for p in parts if p))
    row["method"] = "hint_demo"
    return row


def build(cfg, tasks_path: str, mix_path: str, out_path: str, n_tasks: int, mix_ratio: float, seed: int) -> Dict:
    tasks = load_tasks(tasks_path, limit=n_tasks or None)
    ex = Executor(cfg.env)
    rows = [r for r in ex.map(lambda it: hinted_demo(ex, cfg, it[1], random.Random(seed * 100003 + it[0])),
                               list(enumerate(tasks))) if r]
    ex.close()
    mix: List[str] = []
    if mix_path and os.path.exists(mix_path):
        lines = [l for l in open(mix_path).read().splitlines() if l.strip()]
        random.Random(seed).shuffle(lines)
        mix = lines[:int(len(rows) * mix_ratio)]
    out = [json.dumps(r) for r in rows] + mix
    random.Random(seed + 1).shuffle(out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(out) + "\n")
    stats = {"tasks": len(tasks), "hinted_demos": len(rows), "unhinted_original_sft": len(mix),
             "mean_edits": sum(r["n_edits"] for r in rows) / max(1, len(rows))}
    with open(os.path.splitext(out_path)[0] + "_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"hint warm-up data: {stats}")
    return stats


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/sft_data.yaml")
    p.add_argument("--tasks", required=True, help="TRAINING file tasks")
    p.add_argument("--mix", default="", help="unhinted original SFT trajectories")
    p.add_argument("--out", required=True)
    p.add_argument("--n_tasks", type=int, default=0)
    p.add_argument("--mix_ratio", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("overrides", nargs="*")
    a = p.parse_args(argv)
    build(load_config(a.config, a.overrides), a.tasks, a.mix, a.out, a.n_tasks, a.mix_ratio, a.seed)


if __name__ == "__main__":
    main()
