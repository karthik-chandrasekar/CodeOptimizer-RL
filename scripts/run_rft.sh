#!/usr/bin/env bash
# Rejection fine-tuning on top of the Experiment 6 RL policy.
#   1. sample the RL policy on TRAINING files (never a test split), K attempts each, at RL temperature
#   2. keep successful episodes, cleaned to hottest-first slow-function fixes, replayed through the sandbox
#   3. mix with an equal number of original SFT trajectories (limits forgetting)
#   4. fine-tune the RL policy on that mix at a low learning rate
#   5. evaluate on the hand-written and real-code test files against the RL policy and SFT
# Usage:  nohup bash scripts/run_rft.sh > logs/rft.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

DM=${DM:-artifacts/data_mined}; D2=${D2:-artifacts/data_r2}; X=${X:-artifacts/mined_sft}; A=${A:-artifacts/rft}
M=${M:-configs/model/tiny20m.yaml}; POLICY=${POLICY:-$X/grpo/final.pt}
TASKS=${TASKS:-$DM/files_msft_rl.jsonl}; N_RFT_FILES=${N_RFT_FILES:-1500}; K=${K:-4}
THRESH=${THRESH:-1.5}; PER_TASK=${PER_TASK:-2}; MIX=${MIX:-1.0}
RFT_LR=${RFT_LR:-5e-5}; RFT_EPOCHS=${RFT_EPOCHS:-2}; H=${H:-6}; EVAL_SAMPLES=${EVAL_SAMPLES:-4}
HT=${HT:-$D2/files_test.jsonl}; MT=${MT:-$DM/files_mined_test.jsonl}
COMMON=${COMMON:-}
T="$A/eval"; mkdir -p "$T" logs
need() { for f in "$@"; do [ -s "$f" ] || { echo "MISSING or empty $f - see logs/${LOGP:-}rft_*.log"; exit 1; }; done; }

echo "== 1. sample the RL policy on training files  $(date +%T)"
[ -s "$A/samples_trajectories.jsonl" ] || python -u -m tinyperf.eval.evaluate --config configs/eval.yaml model=$M env.horizon=$H \
    eval.checkpoint="$POLICY" eval.tasks="$TASKS" eval.n_tasks=$N_RFT_FILES eval.n_samples=$K eval.generation.temperature=0.8 \
    eval.out_path="$A/samples.json" $COMMON > logs/${LOGP:-}rft_sample.log 2>&1
{ grep "INFO eval: final" logs/${LOGP:-}rft_sample.log 2>/dev/null | cut -c10-200; } || true
need "$A/samples_trajectories.jsonl"

echo "== 2. successful episodes, cleaned and replayed  $(date +%T)"
[ -s "$A/rft_trajectories.jsonl" ] || python -u -m tinyperf.train.rft_data --config configs/sft_data.yaml --tasks "$TASKS" \
    --samples "$A/samples_trajectories.jsonl" --out "$A/rft_trajectories.jsonl" --threshold $THRESH --per_task $PER_TASK $COMMON
cat "$A/rft_trajectories_stats.json" 2>/dev/null || true; echo
need "$A/rft_trajectories.jsonl"

echo "== 3. mix with original SFT trajectories  $(date +%T)"
N_RFT=$(wc -l < "$A/rft_trajectories.jsonl"); N_MIX=$(python -c "print(int($N_RFT * $MIX))")
{ cat "$A/rft_trajectories.jsonl"; python - "$X/trajectories_all.jsonl" $N_MIX <<'PY'
import random, sys
rows = open(sys.argv[1]).read().splitlines()
random.Random(0).shuffle(rows)
print("\n".join(rows[:int(sys.argv[2])]))
PY
} | grep -v '^$' > "$A/rft_train.jsonl"
echo "RFT training rows: $N_RFT cleaned policy successes + $N_MIX original SFT trajectories"

echo "== 4. RFT: fine-tune the RL policy  $(date +%T)"
[ -s "$A/sft/final.pt" ] || python -u -m tinyperf.train.sft --config configs/sft.yaml model=$M sft.init_from="$POLICY" \
    sft.trajectories="$A/rft_train.jsonl" sft.out_dir="$A/sft" sft.epochs=$RFT_EPOCHS sft.optim.lr=$RFT_LR sft.optim.min_lr=5e-6 \
    sft.optim.warmup_steps=50 $COMMON > logs/${LOGP:-}rft_sft.log 2>&1
need "$A/sft/final.pt"
{ grep -E "SFT data|val_loss" logs/${LOGP:-}rft_sft.log 2>/dev/null | tail -n 2 | cut -c10-200; } || true

echo "== 5. evaluation  $(date +%T)"
E="--config configs/eval.yaml model=$M env.horizon=$H $COMMON"
ev() { local tasks=$1; shift; python -u -m tinyperf.eval.evaluate $E eval.tasks="$tasks" eval.n_tasks=0 eval.n_samples=$EVAL_SAMPLES "$@"; }
ev "$HT" eval.checkpoint="$A/sft/final.pt" eval.out_path="$T/rft_hand.json"
ev "$MT" eval.checkpoint="$A/sft/final.pt" eval.out_path="$T/rft_real.json"
need "$T/rft_hand_trajectories.jsonl" "$T/rft_real_trajectories.jsonl"
opt() { [ -s "$2" ] && echo "$1=$2:multi" || true; }
B=$(seq -s ' ' 1 $H)
for S in hand real; do
  python -m tinyperf.eval.budget --reference rft --budgets $B --out "$T/budget_$S.json" --runs rft="$T/rft_${S}_trajectories.jsonl:multi" \
    $(opt rl "$X/eval/new_rl_${S}_trajectories.jsonl") $(opt rl500 "$X/eval/new_rl500_${S}_trajectories.jsonl") \
    $(opt sft "$X/eval/new_sft_${S}_trajectories.jsonl") > "$T/budget_$S.txt"
done
PT="$D2/files_test_prof.jsonl"; [ -s "$PT" ] || PT="$HT"
python scripts/analyze_files.py --tasks "$PT" --runs rft="$T/rft_hand_trajectories.jsonl" \
  $( [ -s "$X/eval/new_rl_hand_trajectories.jsonl" ] && echo rl="$X/eval/new_rl_hand_trajectories.jsonl" ) > "$T/analysis_hand.txt"
python scripts/analyze_files.py --tasks "$MT" --runs rft="$T/rft_real_trajectories.jsonl" \
  $( [ -s "$X/eval/new_rl_real_trajectories.jsonl" ] && echo rl="$X/eval/new_rl_real_trajectories.jsonl" ) > "$T/analysis_real.txt"
echo "== done  $(date +%T)  ->  $T/budget_real.txt  $T/budget_hand.txt  $T/analysis_*.txt"
