#!/usr/bin/env bash
# RFT and OPSD on the distilled students' RL checkpoints (Experiment 9), mirroring Experiments 7-8 on the 22M model.
#   0. evaluate each student's RL checkpoint in this session (the baseline row; earlier-session evaluations can differ)
#   1. RFT  (scripts/run_rft.sh):  sample the student on training files, clean its successes, fine-tune it on them,
#      mixed 1:1 with its own distillation data (the student's counterpart of "original SFT data")
#   2. OPSD (scripts/run_opsd.sh): hint warm-up from the RFT checkpoint, then distillation-only OPSD (frozen teacher,
#      decision tokens only, beta 0.3, 200 steps), arms with outcome hints and with outcome + localization hints
# Student settings (tokenizer, context, action length, learning rates) reach both scripts through their COMMON hook.
# Usage:  nohup bash scripts/run_student_followups.sh > logs/followups.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

D=${D:-artifacts/distill}; OUT=${OUT:-$D/followup}
export DM=${DM:-artifacts/data_mined} D2=${D2:-artifacts/data_r2}
export HT=${HT:-$D2/files_test.jsonl} MT=${MT:-$DM/files_mined_test.jsonl}
STUDENTS=${STUDENTS:-"s2m s200k"}; PARALLEL=${PARALLEL:-1}; BASE_SAMPLES=${BASE_SAMPLES:-4}
export ENVS=${ENVS:-30}                       # OPSD sandboxes per arm (2 arms x 2 students in parallel)
EXTRA=${EXTRA:-}                              # appended to every command's overrides (e.g. for smoke tests)
mkdir -p logs

student_cfg() {
  case $1 in
    s2m)   M=configs/model/student2m.yaml;   TOK=${TEACHER_TOK:-artifacts/tokenizer/tokenizer.json}; SEQ=2048; MAXNEW=768
           LR_RL=6e-5; LR_FT=1.5e-4 ;;
    s200k) M=configs/model/student200k.yaml; TOK=$D/tokenizer1k/tokenizer.json;                      SEQ=4096; MAXNEW=1152
           LR_RL=1e-4; LR_FT=2.5e-4 ;;
    *) echo "unknown student $1"; exit 1 ;;
  esac
}

run_student() {
  local S=$1; student_cfg $S
  local B="$OUT/$S/base"; mkdir -p "$B/eval"
  [ -e "$B/trajectories_all.jsonl" ] || ln -s "$(realpath "$D/seqkd.jsonl")" "$B/trajectories_all.jsonl"
  local C="tokenizer.path=$TOK sft.seq_len=$SEQ grpo.generation.max_new_tokens=$MAXNEW eval.generation.max_new_tokens=$MAXNEW grpo.lr=$LR_RL $EXTRA"
  echo "== [$S] 0. same-session baseline: RL checkpoint  $(date +%T)"
  for set in hand real; do
    tasks="$HT"; [ $set = real ] && tasks="$MT"
    [ -s "$B/eval/new_rl_${set}_trajectories.jsonl" ] || python -u -m tinyperf.eval.evaluate --config configs/eval.yaml model=$M \
        env.horizon=6 eval.tasks="$tasks" eval.n_tasks=0 eval.n_samples=$BASE_SAMPLES eval.checkpoint="$D/$S/rl/final.pt" \
        eval.out_path="$B/eval/new_rl_$set.json" $C > "logs/${S}_followup_base_$set.log" 2>&1
  done
  echo "== [$S] 1. RFT  $(date +%T)"
  X="$B" A="$OUT/$S/rft" M=$M POLICY="$D/$S/rl/final.pt" RFT_LR=$LR_FT LOGP="${S}_" COMMON="$C" bash scripts/run_rft.sh
  echo "== [$S] 2. hint warm-up + OPSD  $(date +%T)"
  X="$B" A="$OUT/$S/opsd" M=$M POLICY="$OUT/$S/rft/sft/final.pt" START_EVAL_PREFIX="$OUT/$S/rft/eval/rft" WARM_LR=$LR_FT \
      LOGP="${S}_" COMMON="$C" bash scripts/run_opsd.sh
  echo "== [$S] done  $(date +%T)"
}

for S in $STUDENTS; do
  if [ "$PARALLEL" = 1 ]; then run_student $S > "logs/followup_$S.log" 2>&1 & else run_student $S 2>&1 | tee "logs/followup_$S.log"; fi
done
wait
for S in $STUDENTS; do
  grep -q "\[$S\] done" "logs/followup_$S.log" || { echo "[$S] did not finish - see logs/followup_$S.log and logs/${S}_*.log"; continue; }
  echo "== $S: RL start -> RFT -> warm-up -> OPSD (rl300 = this student's RL start, start = its RFT checkpoint)"
  for set in hand real; do
    echo "-- $set (largest budget)"
    { awk '/--- budget B=/{blk=""} {blk = blk $0 "\n"} END {printf "%s", blk}' "$OUT/$S/opsd/eval/budget_$set.txt" | grep "^  " | cut -c1-200; } || true
  done
done
echo "== all done  $(date +%T)  ->  $OUT/<student>/{rft,opsd}/eval/"
