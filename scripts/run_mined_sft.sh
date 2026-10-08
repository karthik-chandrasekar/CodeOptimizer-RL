#!/usr/bin/env bash
# SFT on mined functions, then RL - so RL starts from a model that has seen real-world code.
#   1. split the mined training functions 60% SFT / 40% RL (a function and its natural variant stay together)
#   2. SFT trajectories from the mined SFT functions (complete fixes only)
#   3. SFT from the pretrained model on hand-written + mined trajectories
#   4. RL files: mined RL functions + the 38 hand-written RL functions; real-code test files from the
#      held-out mined functions (never seen by SFT or RL; the 64 validation files come from the same functions)
#   4b. difficulty filter: keep RL files the new SFT model solves 1-3 times out of 4 (never-solved files teach "don't try")
#   5. GRPO (same settings as the 38-function file agent), snapshots every 100 steps
#   6. evaluate new SFT / new RL / old SFT / old 38-function RL on hand-written AND real-code test files
# Usage:  nohup bash scripts/run_mined_sft.sh > logs/mined_sft.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

DM=${DM:-artifacts/data_mined}; D2=${D2:-artifacts/data_r2}; R2=${R2:-artifacts/r2}; F=${F:-artifacts/files}
A=${A:-artifacts/mined_sft}; M=${M:-configs/model/tiny20m.yaml}; PRE=${PRE:-artifacts/pretrain_tiny20m/final.pt}
N_SFT_TASKS=${N_SFT_TASKS:-6000}; N_FILES=${N_FILES:-3000}; N_MTEST=${N_MTEST:-200}; N_PROBE=${N_PROBE:-1200}
FILTER_LO=${FILTER_LO:-0.2}; FILTER_HI=${FILTER_HI:-0.8}; MIN_KEEP=${MIN_KEEP:-150}
GRPO_STEPS=${GRPO_STEPS:-300}; GRPO_ENVS=${GRPO_ENVS:-60}; H=${H:-6}; EVAL_SAMPLES=${EVAL_SAMPLES:-4}
FILE_OVR=${FILE_OVR:-}; COMMON=${COMMON:-}
T="$A/eval"; mkdir -p "$T" logs
need() { for f in "$@"; do [ -s "$f" ] || { echo "MISSING or empty $f - see logs/msft_*.log"; exit 1; }; done; }

echo "== 1. split mined training functions  $(date +%T)"
[ -s "$DM/train_sft.jsonl" ] || python -u -m tinyperf.data.split_train --data_dir "$DM" --rl_frac 0.4
need "$DM/train_sft.jsonl" "$DM/train_rl.jsonl"

echo "== 2. SFT trajectories from mined SFT functions  $(date +%T)"
[ -s "$A/trajectories_mined_stats.json" ] || python -u -m tinyperf.train.sft_data --config configs/sft_data.yaml \
    sft_data.tasks="$DM/train_sft.jsonl" sft_data.out_path="$A/trajectories_mined.jsonl" sft_data.n_tasks=$N_SFT_TASKS sft_data.min_ref_frac=0.8 $COMMON
cat "$A/trajectories_mined_stats.json"; echo
cat "$R2/trajectories.jsonl" "$A/trajectories_mined.jsonl" > "$A/trajectories_all.jsonl"
echo "SFT trajectories: $(wc -l < "$R2/trajectories.jsonl") hand-written + $(wc -l < "$A/trajectories_mined.jsonl") mined"

echo "== 3. SFT (hand-written + mined)  $(date +%T)"
[ -s "$A/sft_agent/final.pt" ] || python -u -m tinyperf.train.sft --config configs/sft.yaml model=$M sft.init_from=$PRE \
    sft.trajectories="$A/trajectories_all.jsonl" sft.out_dir="$A/sft_agent" $COMMON > logs/msft_sft.log 2>&1
need "$A/sft_agent/final.pt"
{ grep -E "SFT data|val_loss" logs/msft_sft.log 2>/dev/null | tail -n 2 | cut -c10-200; } || true

echo "== 4. file tasks  $(date +%T)"
bf() { python -u -m tinyperf.data.build_files --config configs/data.yaml --k 2 4 --d 1 3 "$@" $FILE_OVR; }
cat "$DM/train_rl.jsonl" "$D2/train_rl.jsonl" > "$DM/train_rl_mixed.jsonl"
[ -s "$DM/files_msft_rl.jsonl" ]    || bf --src "$DM/train_rl_mixed.jsonl" --out "$DM/files_msft_rl.jsonl"    --split files_msft_rl    --n $N_FILES --seed 8
[ -s "$DM/files_mined_test.jsonl" ] || bf --src "$DM/test_seeds.jsonl"     --out "$DM/files_mined_test.jsonl" --split files_mined_test --n $N_MTEST --seed 7
need "$DM/files_msft_rl.jsonl" "$DM/files_mined_test.jsonl" "$DM/files_mined_val.jsonl"

echo "== 4b. difficulty filter: keep RL files the new SFT model sometimes solves  $(date +%T)"
# never-solved files only teach "don't try" (every failed edit costs reward); always-solved ones teach nothing
if [ ! -s "$DM/files_msft_rl_filtered.jsonl" ]; then
  python -u -m tinyperf.eval.evaluate --config configs/eval.yaml model=$M env.horizon=$H eval.checkpoint="$A/sft_agent/final.pt" \
      eval.tasks="$DM/files_msft_rl.jsonl" eval.n_tasks=$N_PROBE eval.n_samples=4 eval.generation.temperature=0.8 \
      eval.out_path="$A/probe_files.json" $COMMON > logs/msft_probe.log 2>&1
  grep "INFO eval: final" logs/msft_probe.log | cut -c10-200 || true
  python -u -m tinyperf.train.filter_tasks --tasks "$DM/files_msft_rl.jsonl" --probe "$A/probe_files_trajectories.jsonl" \
      --out "$DM/files_msft_rl_filtered.jsonl" --by task --drop_unprobed --lo $FILTER_LO --hi $FILTER_HI --threshold 1.05 --min_episodes 4
fi
need "$DM/files_msft_rl_filtered.jsonl"
KEPT=$(wc -l < "$DM/files_msft_rl_filtered.jsonl")
echo "RL files after the difficulty filter: $KEPT"
if [ "$KEPT" -lt "$MIN_KEEP" ]; then
  echo "STOP: only $KEPT RL files have mixed outcomes (< $MIN_KEEP). Too few to train on without overfitting - see $A/probe_files.json"; exit 1
fi

echo "== 5. GRPO  $(date +%T)"
python scripts/snapshot_ckpts.py --dir "$A" --runs grpo > logs/msft_snapshots.log 2>&1 &
python -u -m tinyperf.train.grpo --config configs/grpo.yaml model=$M grpo.adv_norm=none grpo.tasks="$DM/files_msft_rl_filtered.jsonl" \
    grpo.val_tasks="$DM/files_mined_val.jsonl" grpo.max_steps=$GRPO_STEPS env.horizon=$H env.max_parallel_envs=$GRPO_ENVS env.timing.repeats=3 \
    grpo.init_from="$A/sft_agent/final.pt" grpo.out_dir="$A/grpo" $COMMON > logs/msft_grpo.log 2>&1
wait
need "$A/grpo/final.pt"

echo "== 6. evaluation: hand-written and real-code test files  $(date +%T)"
E="--config configs/eval.yaml model=$M env.horizon=$H $COMMON"
ev() { local tasks=$1; shift; python -u -m tinyperf.eval.evaluate $E eval.tasks="$tasks" eval.n_tasks=0 eval.n_samples=$EVAL_SAMPLES "$@"; }
HT="$D2/files_test.jsonl"; MT="$DM/files_mined_test.jsonl"
ev "$HT" eval.checkpoint="$A/grpo/final.pt"         eval.out_path="$T/new_rl_hand.json"
ev "$HT" eval.checkpoint="$A/sft_agent/final.pt"    eval.out_path="$T/new_sft_hand.json"
ev "$MT" eval.checkpoint="$A/grpo/final.pt"         eval.out_path="$T/new_rl_real.json"
ev "$MT" eval.checkpoint="$A/sft_agent/final.pt"    eval.out_path="$T/new_sft_real.json"
ev "$MT" eval.checkpoint="$F/grpo_agent/final.pt"   eval.out_path="$T/old_rl_real.json"
ev "$MT" eval.checkpoint="$R2/sft_agent/final.pt"   eval.out_path="$T/old_sft_real.json"
B=$(seq -s ' ' 1 $H)
python -m tinyperf.eval.budget --reference new_rl --budgets $B --out "$T/budget_hand.json" \
  --runs new_rl="$T/new_rl_hand_trajectories.jsonl:multi" new_sft="$T/new_sft_hand_trajectories.jsonl:multi" \
         old_rl="$F/eval/rl_agent_files_trajectories.jsonl:multi" old_sft="$F/eval/sft_agent_files_trajectories.jsonl:multi" > "$T/budget_hand.txt"
python -m tinyperf.eval.budget --reference new_rl --budgets $B --out "$T/budget_real.json" \
  --runs new_rl="$T/new_rl_real_trajectories.jsonl:multi" new_sft="$T/new_sft_real_trajectories.jsonl:multi" \
         old_rl="$T/old_rl_real_trajectories.jsonl:multi" old_sft="$T/old_sft_real_trajectories.jsonl:multi" > "$T/budget_real.txt"
PT="$D2/files_test_prof.jsonl"; [ -s "$PT" ] || PT="$HT"
python scripts/analyze_files.py --tasks "$PT" --runs new_rl="$T/new_rl_hand_trajectories.jsonl" new_sft="$T/new_sft_hand_trajectories.jsonl" \
  old_rl="$F/eval/rl_agent_files_trajectories.jsonl" > "$T/analysis_hand.txt"
python scripts/analyze_files.py --tasks "$MT" --runs new_rl="$T/new_rl_real_trajectories.jsonl" new_sft="$T/new_sft_real_trajectories.jsonl" \
  old_rl="$T/old_rl_real_trajectories.jsonl" old_sft="$T/old_sft_real_trajectories.jsonl" > "$T/analysis_real.txt"
echo "== done  $(date +%T)  ->  $T/budget_hand.txt  $T/budget_real.txt  $T/analysis_*.txt"
