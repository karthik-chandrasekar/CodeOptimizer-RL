#!/usr/bin/env bash
# Multi-function single-file tasks.
#   1. build file tasks: held-out test files from test_seeds functions, RL files from the RL functions
#   2. zero-shot: evaluate the (single-function) round-2 SFT and RL checkpoints on held-out files
#   3. GRPO on RL files (agent + no-feedback), starting from the round-2 SFT checkpoints
#   4. evaluate on held-out files; budget-matched table incl. a "complete fix" metric (>= 0.8 x reference)
# Usage:  nohup bash scripts/run_files.sh > logs/files.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

D=${D:-artifacts/data_r2}      # round-2 dataset: test_seeds.jsonl, train_rl.jsonl
S=${S:-artifacts/r2}           # round-2 checkpoints (single-function SFT / RL)
A=${A:-artifacts/files}        # this experiment
M=${M:-configs/model/tiny20m.yaml}
H=${H:-6}
GRPO_STEPS=${GRPO_STEPS:-300}
GRPO_ENVS=${GRPO_ENVS:-30}     # sandboxes per GRPO run (two runs in parallel)
N_TEST=${N_TEST:-300}; N_RL=${N_RL:-2000}; N_VAL=${N_VAL:-64}
EVAL_SAMPLES=${EVAL_SAMPLES:-4}
FILE_OVR=${FILE_OVR:-}         # extra overrides for the file builder (smoke tests)
COMMON=${COMMON:-}             # extra overrides for evaluation / GRPO (smoke tests)
T="$A/eval"; mkdir -p "$T" logs
need() { for f in "$@"; do [ -s "$f" ] || { echo "MISSING or empty $f - see logs/files_*.log"; exit 1; }; done; }

echo "== 1. file tasks  $(date +%T)"
bf() { python -u -m tinyperf.data.build_files --config configs/data.yaml --k 2 4 --d 1 3 "$@" $FILE_OVR; }
[ -s "$D/files_test.jsonl" ]   || bf --src "$D/test_seeds.jsonl" --out "$D/files_test.jsonl"   --split files_test   --n $N_TEST --seed 1
[ -s "$D/files_rl.jsonl" ]     || bf --src "$D/train_rl.jsonl"   --out "$D/files_rl.jsonl"     --split files_rl     --n $N_RL   --seed 2
[ -s "$D/files_rl_val.jsonl" ] || bf --src "$D/train_rl.jsonl"   --out "$D/files_rl_val.jsonl" --split files_rl_val --n $N_VAL  --seed 3
need "$D/files_test.jsonl" "$D/files_rl.jsonl" "$D/files_rl_val.jsonl"

E="--config configs/eval.yaml model=$M env.horizon=$H $COMMON"
ev() { python -u -m tinyperf.eval.evaluate $E eval.tasks="$D/files_test.jsonl" eval.n_tasks=0 eval.n_samples=$EVAL_SAMPLES "$@" 2>&1 | grep -E "INFO eval: final|Error|Traceback" || true; }

echo "== 2. zero-shot on held-out files: single-function checkpoints  $(date +%T)"
ev eval.checkpoint="$S/sft_agent/final.pt"                     eval.out_path="$T/sft_agent_files.json"
ev eval.checkpoint="$S/sft_nofb/final.pt"  env.feedback=none   eval.out_path="$T/sft_nofb_files.json"
ev eval.checkpoint="$S/grpo_agent/final.pt"                    eval.out_path="$T/single_rl_agent_files.json"
ev eval.checkpoint="$S/grpo_nofb/final.pt" env.feedback=none   eval.out_path="$T/single_rl_nofb_files.json"

echo "== 3. GRPO on RL-function files  $(date +%T)"
G="--config configs/grpo.yaml model=$M grpo.adv_norm=none grpo.tasks=$D/files_rl.jsonl grpo.val_tasks=$D/files_rl_val.jsonl grpo.max_steps=$GRPO_STEPS env.horizon=$H env.max_parallel_envs=$GRPO_ENVS env.timing.repeats=3 $COMMON"
python scripts/snapshot_ckpts.py --dir "$A" --runs grpo_agent grpo_nofb > logs/files_snapshots.log 2>&1 &
python -u -m tinyperf.train.grpo $G grpo.init_from="$S/sft_agent/final.pt" grpo.out_dir="$A/grpo_agent"                   > logs/files_grpo_agent.log 2>&1 &
python -u -m tinyperf.train.grpo $G grpo.init_from="$S/sft_nofb/final.pt"  grpo.out_dir="$A/grpo_nofb" env.feedback=none > logs/files_grpo_nofb.log 2>&1 &
wait
need "$A/grpo_agent/final.pt" "$A/grpo_nofb/final.pt"

echo "== 4. held-out files after file-RL  $(date +%T)"
ev eval.checkpoint="$A/grpo_agent/final.pt"                    eval.out_path="$T/rl_agent_files.json"
ev eval.checkpoint="$A/grpo_nofb/final.pt" env.feedback=none   eval.out_path="$T/rl_nofb_files.json"
B=$(seq -s ' ' 1 $H)
python -m tinyperf.eval.budget --reference file_rl_agent --budgets $B --out "$T/budget_files.json" \
  --runs file_rl_agent="$T/rl_agent_files_trajectories.jsonl:multi"  file_rl_nofb="$T/rl_nofb_files_trajectories.jsonl:multi" \
         sft_agent="$T/sft_agent_files_trajectories.jsonl:multi"     sft_nofb="$T/sft_nofb_files_trajectories.jsonl:multi" \
         single_rl_agent="$T/single_rl_agent_files_trajectories.jsonl:multi" single_rl_nofb="$T/single_rl_nofb_files_trajectories.jsonl:multi" \
  > "$T/budget_files.txt"
echo "== done  $(date +%T)  ->  $T/budget_files.txt"
