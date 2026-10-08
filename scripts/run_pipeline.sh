#!/usr/bin/env bash
# Full V0 pipeline on one GPU. Each stage is idempotent (resumes from last.pt where applicable).
# Usage: nohup bash scripts/run_pipeline.sh > logs/pipeline.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
mkdir -p logs artifacts
SIZE=${SIZE:-tiny20m}          # tiny10m | tiny20m | tiny50m | tiny100m
MODEL="configs/model/$SIZE.yaml"

echo "== 1. dataset"
[ -f artifacts/data/train.jsonl ] || python -u -m tinyperf.data.build --config configs/data.yaml

echo "== 2. tokenizer"
[ -f artifacts/tokenizer/tokenizer.json ] || python -u -m tinyperf.model.tokenizer --config configs/pretrain.yaml

echo "== 3. code pretraining ($SIZE)"
python -u -m tinyperf.train.pretrain --config configs/pretrain.yaml model="$MODEL" pretrain.out_dir="artifacts/pretrain_$SIZE"

echo "== 4. agent-SFT trajectories"
[ -f artifacts/sft/trajectories.jsonl ] || python -u -m tinyperf.train.sft_data --config configs/sft_data.yaml

echo "== 5a. Agent-SFT"
python -u -m tinyperf.train.sft --config configs/sft.yaml model="$MODEL" sft.init_from="artifacts/pretrain_$SIZE/final.pt" sft.out_dir="artifacts/sft_$SIZE"
echo "== 5b. SFT-1 (one-shot baseline)"
python -u -m tinyperf.train.sft --config configs/sft.yaml model="$MODEL" sft.init_from="artifacts/pretrain_$SIZE/final.pt" sft.out_dir="artifacts/sft1_$SIZE" sft.first_step_only=true
echo "== 5c. Agent-SFT without feedback (control)"
python -u -m tinyperf.train.sft --config configs/sft.yaml model="$MODEL" sft.init_from="artifacts/pretrain_$SIZE/final.pt" sft.out_dir="artifacts/sft_nofb_$SIZE" env.feedback=none

echo "== 6a. Agent-RL (GRPO, H=6)"
python -u -m tinyperf.train.grpo --config configs/grpo.yaml model="$MODEL" grpo.init_from="artifacts/sft_$SIZE/final.pt" grpo.out_dir="artifacts/grpo_$SIZE"
echo "== 6b. RL-1 (one-shot RL, H=1)"
python -u -m tinyperf.train.grpo --config configs/grpo.yaml model="$MODEL" grpo.init_from="artifacts/sft1_$SIZE/final.pt" grpo.out_dir="artifacts/grpo1_$SIZE" env.horizon=1
echo "== 6c. Agent-RL without feedback (control)"
python -u -m tinyperf.train.grpo --config configs/grpo.yaml model="$MODEL" grpo.init_from="artifacts/sft_nofb_$SIZE/final.pt" grpo.out_dir="artifacts/grpo_nofb_$SIZE" env.feedback=none

echo "== 7. core comparison table on all test splits"
python -u -m tinyperf.eval.ablations --config configs/eval.yaml model="$MODEL" \
  --checkpoints base="artifacts/pretrain_$SIZE/final.pt" sft1="artifacts/sft1_$SIZE/final.pt" rl1="artifacts/grpo1_$SIZE/final.pt" \
                agent_sft="artifacts/sft_$SIZE/final.pt" agent_rl="artifacts/grpo_$SIZE/final.pt" \
                agent_sft_nofb="artifacts/sft_nofb_$SIZE/final.pt" agent_rl_nofb="artifacts/grpo_nofb_$SIZE/final.pt" \
  --splits test_iid test_compositional test_heldout test_seeds --feedback full none --n_samples 4 --out "artifacts/eval/core_$SIZE.json"

echo "== 8. horizon ablation (evaluation-time H) for the agent-RL policy"
python -u -m tinyperf.eval.ablations --config configs/eval.yaml model="$MODEL" \
  --checkpoints agent_rl="artifacts/grpo_$SIZE/final.pt" agent_sft="artifacts/sft_$SIZE/final.pt" \
  --splits test_iid --horizons 1 2 4 6 10 --n_samples 4 --out "artifacts/eval/horizon_$SIZE.json"
echo "== done"
