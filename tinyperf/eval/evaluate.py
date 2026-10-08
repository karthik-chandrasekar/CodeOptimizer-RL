"""Evaluate a checkpoint on a task split.

    python -m tinyperf.eval.evaluate --config configs/eval.yaml \
        eval.checkpoint=artifacts/grpo/final.pt eval.tasks=artifacts/data/test_heldout.jsonl \
        eval.out_path=artifacts/eval/grpo_heldout.json

Overrides such as ``env.horizon=1`` (one-shot) or ``env.feedback=none`` turn the
same script into the horizon / feedback evaluations; the trained-policy
comparison table (Base / SFT-1 / RL-1 / Agent-SFT / Agent-RL) is just this
script run on the corresponding checkpoints (see README).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from tinyperf.common.config import Config, parse_cli, to_dict
from tinyperf.common.utils import get_logger, resolve_device, write_jsonl
from tinyperf.env.executor import Executor
from tinyperf.env.task import load_tasks
from tinyperf.eval.metrics import bootstrap_ci, compute_metrics, per_depth_metrics, per_family_metrics
from tinyperf.train.common import load_model_and_tokenizer
from tinyperf.train.rollout import Rollout, episodes_to_records

log = get_logger("eval")


def run_eval(cfg: Config, checkpoint: Optional[str] = None, tasks_path: Optional[str] = None, out_path: Optional[str] = None,
             n_tasks: Optional[int] = None, quiet: bool = False) -> Dict[str, Any]:
    e = cfg.eval
    checkpoint = checkpoint or e.checkpoint
    tasks_path = tasks_path or e.tasks
    out_path = out_path or e.out_path
    n_tasks = e.n_tasks if n_tasks is None else n_tasks
    device = resolve_device(cfg.device)
    model, tok = load_model_and_tokenizer(cfg, checkpoint, device)
    model.eval()
    tasks = load_tasks(tasks_path, limit=n_tasks)
    if not tasks:
        raise RuntimeError(f"no tasks in {tasks_path}")
    ex = Executor(cfg.env)
    ro = Rollout(model, tok, ex, cfg.env, cfg.reward, e.generation, device)
    episodes = ro.run(tasks, group_size=max(1, e.n_samples), greedy=e.greedy, seed=cfg.grpo.seed)
    ex.close()
    records = episodes_to_records(episodes)
    metrics = compute_metrics(records)
    result: Dict[str, Any] = {
        "checkpoint": checkpoint,
        "tasks": tasks_path,
        "n_tasks": len(tasks),
        "n_samples": e.n_samples,
        "env": {"horizon": cfg.env.horizon, "feedback": cfg.env.feedback, "improvement_delta": cfg.env.improvement_delta},
        "generation": to_dict(e.generation),
        "greedy": e.greedy,
        "metrics": metrics,
        "per_family": per_family_metrics(records),
        "per_depth": per_depth_metrics(records),
        "ci": {k: bootstrap_ci(records, k, n_boot=500) for k in ("success_rate", "gm_speedup")},
    }
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        if e.save_trajectories:
            write_jsonl(os.path.splitext(out_path)[0] + "_trajectories.jsonl", records)
    if not quiet:
        m = metrics
        log.info(f"{os.path.basename(str(checkpoint))} on {os.path.basename(tasks_path)} (H={cfg.env.horizon}, feedback={cfg.env.feedback}): "
                 f"success={m['success_rate']:.3f} gm_speedup={m['gm_speedup']:.3f} p>=1.5x={m['p_speedup_ge_1.5']:.3f} "
                 f"edits={m['mean_edits']:.2f} invalid={m['invalid_edit_rate']:.3f} stop={m['stop_rate']:.3f} "
                 f"recovery={m.get('recovery_rate', float('nan')):.3f} ref_frac={m.get('reference_fraction', float('nan')):.3f}")
        if out_path:
            log.info(f"wrote {out_path}")
    return result


def main(argv: Optional[List[str]] = None) -> None:
    cfg = parse_cli(argv)
    run_eval(cfg)


if __name__ == "__main__":
    main()
