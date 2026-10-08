"""Split the training tasks by *function* into an SFT part and an RL part.

    python -m tinyperf.data.split_train --data_dir artifacts/data --rl_frac 0.4

RL then practises only on functions SFT never demonstrated, so its groups contain a mix of
successes and failures instead of memorised answers.  Every renamed copy of a function goes to
the same side (hash of the seed-function name), so nothing leaks between the two.
Writes ``train_sft.jsonl`` and ``train_rl.jsonl`` next to ``train.jsonl`` (file order is kept).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Dict, List, Optional


def is_rl_function(seed_name: str, rl_frac: float, salt: int = 0) -> bool:
    u = int(hashlib.sha1(f"rl-split|{seed_name}|{salt}".encode()).hexdigest()[:16], 16) / 16**16
    return u < rl_frac


def split(data_dir: str, rl_frac: float, salt: int = 0) -> Dict[str, Dict[str, int]]:
    src = os.path.join(data_dir, "train.jsonl")
    outs = {k: open(os.path.join(data_dir, f"train_{k}.jsonl"), "w") for k in ("sft", "rl")}
    stats = {k: {"tasks": 0, "functions": set(), "chains": set()} for k in outs}
    with open(src) as f:
        for line in f:
            if not line.strip():
                continue
            t = json.loads(line)
            side = "rl" if is_rl_function(t["seed_name"], rl_frac, salt) else "sft"
            outs[side].write(line if line.endswith("\n") else line + "\n")
            st = stats[side]
            st["tasks"] += 1
            st["functions"].add(t["seed_name"])
            st["chains"].add((t["seed_name"], "+".join(t["families"])))
    for fh in outs.values():
        fh.close()
    return {k: {"tasks": v["tasks"], "functions": len(v["functions"]), "chains": len(v["chains"])} for k, v in stats.items()}


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default="artifacts/data")
    p.add_argument("--rl_frac", type=float, default=0.4)
    p.add_argument("--salt", type=int, default=0)
    a = p.parse_args(argv)
    res = split(a.data_dir, a.rl_frac, a.salt)
    for k, v in res.items():
        print(f"train_{k}.jsonl: {v['tasks']} tasks from {v['functions']} functions ({v['chains']} chains)")
    with open(os.path.join(a.data_dir, "train_split_stats.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
