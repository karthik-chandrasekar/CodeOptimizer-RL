"""Sequence-level knowledge distillation (SeqKD) data from a teacher policy.

    python -m tinyperf.train.seqkd_data --config configs/grpo.yaml --teacher artifacts/.../step0100.pt \\
        --tasks artifacts/data_mined/files_msft_rl.jsonl artifacts/data_mined/train_sft.jsonl \\
        --n_tasks 3000 2000 --k 4 --out artifacts/distill/seqkd.jsonl tokenizer.path=artifacts/tokenizer/tokenizer.json

Kim & Rush (2016): train the student on the teacher's own outputs instead of the original targets.  In this agent
setting the teacher plays complete episodes in the real sandbox on TRAINING tasks; each step's observation and the
teacher's action are stored as text in the SFT row format (``method="seqkd"``), so a student with any tokenizer can
learn from them.  Failed teacher actions keep their sandbox status and are masked from the student's loss by the SFT
trainer; the next observation (which shows the failure) and the teacher's recovery are kept.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from typing import Dict, List, Optional

import torch

from tinyperf.common.config import load_config
from tinyperf.common.utils import get_logger, resolve_device
from tinyperf.env.executor import Executor
from tinyperf.env.protocol import Action, format_action, parse_action
from tinyperf.env.task import load_tasks
from tinyperf.train.common import load_model_and_tokenizer
from tinyperf.train.rollout import Rollout

log = get_logger("seqkd_data")
IMITABLE = ("ok", "stop")


def canonical_action(text: str) -> str:
    """The teacher's action in canonical protocol form; unparseable text is kept raw (its step is masked anyway)."""
    try:
        a = parse_action(text or "")
    except Exception:  # noqa: BLE001
        return text or ""
    if a.kind == "stop":
        return format_action(Action("stop"))
    if a.kind == "edit" and a.code:
        return format_action(a)
    return text or ""


def episode_to_row(ep, horizon: int) -> Dict:
    steps = []
    for ti, (t, rec) in enumerate(zip(ep.turns, ep.summary.steps)):
        steps.append({"step": ti, "remaining": horizon - ti, "obs": t.obs, "action": canonical_action(t.action_text),
                      "status": rec.status, "ratio": rec.ratio, "finished": bool(getattr(t, "finished", True))})
    task = ep.task
    return {"task_id": task.task_id, "source": task.source, "func_name": getattr(task, "entry", ""),
            "families": list(getattr(task, "families", None) or []), "steps": steps,
            "final_ratio": ep.summary.best_ratio, "final_code": ep.summary.best_code,
            "n_edits": ep.summary.n_edits, "n_invalid": ep.summary.n_invalid,
            "method": "seqkd", "teacher_reward": ep.reward}


def build(cfg, teacher: str, task_paths: List[str], n_tasks: List[int], k: int, temperature: float,
          min_speedup: float, out_path: str, batch_tasks: int, seed: int) -> Dict:
    device = resolve_device(cfg.device)            # "auto" -> cuda if available, else cpu (as in the other entry points)
    model, tok = load_model_and_tokenizer(cfg, teacher, device)
    model.eval()
    gen = dataclasses.replace(cfg.grpo.generation, temperature=temperature)
    ex = Executor(cfg.env)
    ro = Rollout(model, tok, ex, cfg.env, cfg.reward, gen, device, horizon=cfg.env.horizon)
    tasks = []
    for i, p in enumerate(task_paths):
        n = n_tasks[i] if i < len(n_tasks) else (n_tasks[-1] if n_tasks else 0)
        tasks += load_tasks(p, limit=n)
    stats = {"teacher": teacher, "tasks": len(tasks), "k": k, "temperature": temperature, "episodes": 0, "kept": 0,
             "steps": 0, "imitable_steps": 0, "truncated_steps": 0, "success_1.05": 0, "success_1.5": 0, "edits": 0}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    t0 = time.time()
    with open(out_path, "w") as f:
        for b in range(0, len(tasks), batch_tasks):
            with torch.no_grad():
                eps = ro.run(tasks[b:b + batch_tasks], group_size=k, seed=seed + b)
            for ep in eps:
                row = episode_to_row(ep, cfg.env.horizon)
                sp = 1.0 / max(ep.summary.best_ratio, 1e-9)
                stats["episodes"] += 1
                stats["success_1.05"] += sp >= 1.05
                stats["success_1.5"] += sp >= 1.5
                stats["edits"] += ep.summary.n_edits
                if sp < min_speedup:
                    continue
                stats["kept"] += 1
                stats["steps"] += len(row["steps"])
                stats["imitable_steps"] += sum(s["status"] in IMITABLE for s in row["steps"])
                stats["truncated_steps"] += sum(not s["finished"] for s in row["steps"])
                f.write(json.dumps(row) + "\n")
            done = min(b + batch_tasks, len(tasks))
            log.info(f"{done}/{len(tasks)} tasks | episodes {stats['episodes']} | teacher success>=1.5 "
                     f"{stats['success_1.5'] / max(1, stats['episodes']):.2f} | {time.time() - t0:.0f}s")
    ex.close()
    n = max(1, stats["episodes"])
    stats.update({"success_1.05": stats["success_1.05"] / n, "success_1.5": stats["success_1.5"] / n,
                  "mean_edits": stats.pop("edits") / n, "seconds": round(time.time() - t0)})
    with open(os.path.splitext(out_path)[0] + "_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"SeqKD data: {stats}")
    return stats


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/grpo.yaml")
    p.add_argument("--teacher", required=True)
    p.add_argument("--tasks", nargs="+", required=True, help="TRAINING task files (never a test split)")
    p.add_argument("--n_tasks", nargs="*", type=int, default=[0], help="per --tasks file; 0 = all")
    p.add_argument("--k", type=int, default=4, help="teacher episodes per task")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--min_speedup", type=float, default=0.0, help="keep only episodes reaching this speedup (0 = all)")
    p.add_argument("--batch_tasks", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("overrides", nargs="*")
    a = p.parse_args(argv)
    build(load_config(a.config, a.overrides), a.teacher, a.tasks, a.n_tasks, a.k, a.temperature, a.min_speedup,
          a.out, a.batch_tasks, a.seed)


if __name__ == "__main__":
    main()
