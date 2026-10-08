#!/usr/bin/env bash
# End-to-end CPU smoke test of the whole pipeline on a tiny dataset + tiny model (~2-3 min on a laptop).
# Usage: bash scripts/smoke_test.sh [workdir]
set -euo pipefail
W=${1:-/tmp/tinyperf_smoke}
mkdir -p "$W"
cd "$(dirname "$0")/.."
export PYTHONPATH=.

cat > "$W/model.yaml" <<YAML
name: tinyperf-smoke
vocab_size: 8192
n_layer: 2
n_embd: 64
n_head: 4
mlp_dim: 128
max_seq_len: 1024
YAML

echo "== 1. dataset (tiny)"
python -m tinyperf.data.build --config configs/data.yaml data.out_dir="$W/data" data.n_train=12 data.n_val=4 data.n_test_iid=4 \
  data.n_test_compositional=4 data.n_test_heldout=4 data.max_nodes_per_seed=6 data.perf_scale=400 env.max_parallel_envs=2 env.timing.repeats=3

echo "== 2. tokenizer"
python -m tinyperf.model.tokenizer --config configs/pretrain.yaml tokenizer.path="$W/tok/tokenizer.json" tokenizer.max_files=60 \
  tokenizer.include_tasks="$W/data/train.jsonl"

echo "== 3. pretrain (20 steps)"
python -m tinyperf.train.pretrain --config configs/pretrain.yaml model="$W/model.yaml" device=cpu tokenizer.path="$W/tok/tokenizer.json" \
  tokenizer.max_files=40 pretrain.include_tasks="$W/data/train.jsonl" pretrain.out_dir="$W/pretrain" pretrain.seq_len=128 \
  pretrain.optim.max_steps=20 pretrain.optim.warmup_steps=3 pretrain.optim.batch_size=8 pretrain.optim.grad_accum=1 pretrain.eval_every=10 pretrain.save_every=10 pretrain.log_every=5

echo "== 4. SFT trajectories (programmatic + search)"
python -m tinyperf.train.sft_data --config configs/sft_data.yaml sft_data.tasks="$W/data/train.jsonl" sft_data.out_path="$W/sft/trajectories.jsonl" \
  sft_data.n_tasks=8 env.max_parallel_envs=2 env.timing.repeats=3

echo "== 5. agent SFT (30 steps)"
python -m tinyperf.train.sft --config configs/sft.yaml model="$W/model.yaml" device=cpu tokenizer.path="$W/tok/tokenizer.json" \
  sft.trajectories="$W/sft/trajectories.jsonl" sft.init_from="$W/pretrain/final.pt" sft.out_dir="$W/sft" sft.seq_len=1024 \
  sft.val_frac=0.2 sft.optim.max_steps=30 sft.optim.warmup_steps=2 sft.optim.batch_size=4 sft.eval_every=10 sft.log_every=5

echo "== 6. GRPO (2 steps, B=2 G=4 H=2)"
python -m tinyperf.train.grpo --config configs/grpo.yaml model="$W/model.yaml" device=cpu tokenizer.path="$W/tok/tokenizer.json" \
  grpo.tasks="$W/data/train.jsonl" grpo.init_from="$W/sft/final.pt" grpo.out_dir="$W/grpo" grpo.group_size=4 grpo.tasks_per_step=2 \
  grpo.minibatch_size=4 grpo.max_steps=2 grpo.eval_every=2 grpo.save_every=2 grpo.n_eval_tasks=2 grpo.generation.max_new_tokens=48 \
  env.horizon=2 env.max_parallel_envs=2 env.timing.repeats=3 eval.generation.max_new_tokens=48

echo "== 7. evaluation + ablation table"
python -m tinyperf.eval.ablations --config configs/eval.yaml --checkpoints sft="$W/sft/final.pt" rl="$W/grpo/final.pt" \
  --splits val test_iid --data_dir "$W/data" --horizons 1 2 --feedback full none --n_tasks 2 --out "$W/eval/table.json" \
  model="$W/model.yaml" device=cpu tokenizer.path="$W/tok/tokenizer.json" env.max_parallel_envs=2 env.timing.repeats=3 eval.generation.max_new_tokens=32

echo "== smoke test OK -> $W"
