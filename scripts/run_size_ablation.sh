#!/usr/bin/env bash
# Model-size ablation: runs the full pipeline for each size, then one table.
# Usage: nohup bash scripts/run_size_ablation.sh > logs/sizes.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
for SIZE in tiny10m tiny20m tiny50m tiny100m; do
  SIZE=$SIZE bash scripts/run_pipeline.sh
done
python -u -m tinyperf.eval.ablations --config configs/eval.yaml \
  --checkpoints 10m=artifacts/grpo_tiny10m/final.pt 20m=artifacts/grpo_tiny20m/final.pt 50m=artifacts/grpo_tiny50m/final.pt 100m=artifacts/grpo_tiny100m/final.pt \
  --splits test_iid test_heldout --n_samples 4 --out artifacts/eval/size_ablation.json
