"""Comparison tables and ablations in one command.

    python -m tinyperf.eval.ablations --config configs/eval.yaml \
        --checkpoints base=artifacts/pretrain/final.pt sft1=artifacts/sft1/final.pt agent_sft=artifacts/sft/final.pt agent_rl=artifacts/grpo/final.pt \
        --splits test_iid test_compositional test_heldout \
        --horizons 6 --feedback full none --out artifacts/eval/table.json

* core comparison  : one row per checkpoint (x split)
* feedback ablation: ``--feedback full none`` adds a ``feedback_gain`` block (full minus none) per checkpoint
* horizon ablation : ``--horizons 1 2 4 6 10`` evaluates every checkpoint at every H (H=1 is the one-shot setting)
* model size       : pass one checkpoint per size (``--checkpoints 10m=... 20m=... 50m=... 100m=...``)

Train-time ablations (a policy *trained* without feedback, or with a different H)
are separate GRPO/SFT runs with ``env.feedback=none`` / ``env.horizon=...``; point
this script at their checkpoints to compare them.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Any, Dict, List

from tinyperf.common.config import load_config
from tinyperf.common.utils import get_logger
from tinyperf.eval.evaluate import run_eval
from tinyperf.eval.metrics import compare, format_table

log = get_logger("ablations")
KEYS = ["success_rate", "gm_speedup", "p_speedup_ge_1.5", "mean_edits", "invalid_edit_rate", "recovery_rate", "stop_rate"]


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--checkpoints", nargs="+", required=True, help="name=path ...")
    p.add_argument("--splits", nargs="+", default=["test_iid"])
    p.add_argument("--data_dir", default="artifacts/data")
    p.add_argument("--horizons", nargs="+", type=int, default=None)
    p.add_argument("--feedback", nargs="+", default=None, choices=["full", "none", "profile"])
    p.add_argument("--n_tasks", type=int, default=0)
    p.add_argument("--n_samples", type=int, default=None)
    p.add_argument("--out", default="artifacts/eval/table.json")
    p.add_argument("overrides", nargs="*")
    a = p.parse_args(argv)
    base_cfg = load_config(a.config, [o for o in a.overrides if "=" in o])
    horizons = a.horizons or [base_cfg.env.horizon]
    feedbacks = a.feedback or [base_cfg.env.feedback]
    ckpts = dict(kv.split("=", 1) for kv in a.checkpoints)
    results: Dict[str, Dict[str, Any]] = {}
    rows: Dict[str, Dict[str, Any]] = {}
    for name, path in ckpts.items():
        for split in a.splits:
            for H in horizons:
                for fb in feedbacks:
                    cfg = copy.deepcopy(base_cfg)
                    cfg.env.horizon, cfg.env.feedback = H, fb
                    if a.n_samples:
                        cfg.eval.n_samples = a.n_samples
                    key = f"{name}|{split}|H={H}|{fb}"
                    out_path = os.path.join(os.path.dirname(a.out) or ".", f"{name}_{split}_H{H}_{fb}.json")
                    r = run_eval(cfg, checkpoint=path, tasks_path=os.path.join(a.data_dir, f"{split}.jsonl"), out_path=out_path,
                                 n_tasks=a.n_tasks, quiet=False)
                    results[key] = r["metrics"]
                    rows[key] = r["metrics"]
    table: Dict[str, Any] = {"rows": results}
    # feedback gain (full - none) for every checkpoint/split/H that has both
    if set(feedbacks) >= {"full", "none"}:
        gains = {}
        for name in ckpts:
            for split in a.splits:
                for H in horizons:
                    kf, kn = f"{name}|{split}|H={H}|full", f"{name}|{split}|H={H}|none"
                    if kf in results and kn in results:
                        gains[f"{name}|{split}|H={H}"] = compare(results[kf], results[kn])
        table["feedback_gain"] = gains
    # horizon curves
    if len(horizons) > 1:
        table["horizon_curve"] = {f"{name}|{split}|{fb}": {str(H): results[f"{name}|{split}|H={H}|{fb}"]["gm_speedup"] for H in horizons}
                                  for name in ckpts for split in a.splits for fb in feedbacks}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(table, f, indent=2)
    txt = format_table(rows, KEYS)
    with open(os.path.splitext(a.out)[0] + ".txt", "w") as f:
        f.write(txt + "\n")
    print(txt)
    if "feedback_gain" in table:
        print("\nfeedback gain (full - none):")
        for k, v in table["feedback_gain"].items():
            print(f"  {k:40s} " + " ".join(f"{kk}={vv:+.3f}" for kk, vv in v.items()))
    log.info(f"wrote {a.out}")


if __name__ == "__main__":
    main()
