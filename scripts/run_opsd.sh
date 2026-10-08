#!/usr/bin/env bash
# On-policy self-distillation (OPSD, reverse KL to the hinted self) on top of a policy checkpoint (POLICY;
# default: the Experiment 6 RL policy, e.g. POLICY=artifacts/rft/sft/final.pt for RL -> RFT -> OPSD).
#   1. hint warm-up data: oracle file demonstrations with OPSD-style hints + unhinted ORIGINAL SFT data
#   2. warm-up SFT from the RL policy, so the teacher (the policy + a hint) actually uses hints
#   3. diagnostic: does the policy follow a localization hint?  (validation files, hint shown vs not)
#   4. from the warmed-up model: OPSD with outcome hints and with outcome + localization hints; MODE=opsd_only (default)
#      trains on the reverse KL alone (rollouts only supply samples + sandbox verdicts), MODE=grpo+opsd adds the reward
#      term and a GRPO-only control arm
#   5. evaluation WITHOUT hints on hand-written and real-code test files, vs the RL-only policy
# Usage:  nohup bash scripts/run_opsd.sh > logs/opsd.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

DM=${DM:-artifacts/data_mined}; D2=${D2:-artifacts/data_r2}; X=${X:-artifacts/mined_sft}; A=${A:-artifacts/opsd}
M=${M:-configs/model/tiny20m.yaml}; POLICY=${POLICY:-$X/grpo/final.pt}
TASKS=${TASKS:-$DM/files_msft_rl.jsonl}; VAL=${VAL:-$DM/files_mined_val.jsonl}
HT=${HT:-$D2/files_test.jsonl}; MT=${MT:-$DM/files_mined_test.jsonl}
N_WARM_TASKS=${N_WARM_TASKS:-0}; WARM_LR=${WARM_LR:-5e-5}; WARM_EPOCHS=${WARM_EPOCHS:-2}
MODE=${MODE:-opsd_only}                  # opsd_only: distillation only (no reward term) | grpo+opsd: GRPO + OPSD arms + GRPO control
STEPS=${STEPS:-200}; ENVS=${ENVS:-40}; OPSD_BETA=${OPSD_BETA:-0.3}
OPSD_TEACHER=${OPSD_TEACHER:-frozen}; OPSD_MAX_TOKENS=${OPSD_MAX_TOKENS:-12}; SAVE_EVERY=${SAVE_EVERY:-25}; EVAL_EVERY=${EVAL_EVERY:-25}
H=${H:-6}; EVAL_SAMPLES=${EVAL_SAMPLES:-4}; DIAG_SAMPLES=${DIAG_SAMPLES:-2}
START_EVAL_PREFIX=${START_EVAL_PREFIX:-}   # e.g. artifacts/rft/eval/rft -> the starting checkpoint's own test evaluations
COMMON=${COMMON:-}
T="$A/eval"; mkdir -p "$T" logs
need() { for f in "$@"; do [ -s "$f" ] || { echo "MISSING or empty $f - see logs/${LOGP:-}opsd_*.log"; exit 1; }; done; }
E="--config configs/eval.yaml model=$M env.horizon=$H $COMMON"
ev() { local tasks=$1; shift; python -u -m tinyperf.eval.evaluate $E eval.tasks="$tasks" eval.n_tasks=0 "$@"; }

echo "== 1. hint warm-up data  $(date +%T)"
[ -s "$A/hint_train.jsonl" ] || python -u -m tinyperf.train.hint_sft_data --config configs/sft_data.yaml --tasks "$TASKS" \
    --mix "$X/trajectories_all.jsonl" --out "$A/hint_train.jsonl" --n_tasks $N_WARM_TASKS $COMMON
cat "$A/hint_train_stats.json"; echo

echo "== 2. hint warm-up SFT from the RL policy  $(date +%T)"
[ -s "$A/warmup/final.pt" ] || python -u -m tinyperf.train.sft --config configs/sft.yaml model=$M sft.init_from="$POLICY" \
    sft.trajectories="$A/hint_train.jsonl" sft.out_dir="$A/warmup" sft.epochs=$WARM_EPOCHS sft.optim.lr=$WARM_LR \
    sft.optim.min_lr=5e-6 sft.optim.warmup_steps=50 $COMMON > logs/${LOGP:-}opsd_warmup.log 2>&1
need "$A/warmup/final.pt"

echo "== 3. hint-following diagnostic (validation files)  $(date +%T)"
for who in policy warmup; do
  ck="$POLICY"; [ $who = warmup ] && ck="$A/warmup/final.pt"
  for hint in none loc; do
    h=""; [ $hint = loc ] && h="env.hint=loc"
    [ -s "$T/diag_${who}_${hint}_trajectories.jsonl" ] || ev "$VAL" eval.checkpoint="$ck" eval.n_samples=$DIAG_SAMPLES $h \
        eval.out_path="$T/diag_${who}_${hint}.json" >> logs/${LOGP:-}opsd_diag.log 2>&1
  done
done
python scripts/analyze_files.py --tasks "$VAL" --runs policy_nohint="$T/diag_policy_none_trajectories.jsonl" \
  policy_hint="$T/diag_policy_loc_trajectories.jsonl" warmup_nohint="$T/diag_warmup_none_trajectories.jsonl" \
  warmup_hint="$T/diag_warmup_loc_trajectories.jsonl" | tee "$T/hint_diag.txt"

if [ "$MODE" = "opsd_only" ]; then ARMS="opsd_err opsd_errloc"; ONLY="grpo.opsd_only=true"; else ARMS="ctrl opsd_err opsd_errloc"; ONLY=""; fi
echo "== 4. training from the warmed-up model (MODE=$MODE: $ARMS)  $(date +%T)"
G="--config configs/grpo.yaml model=$M grpo.adv_norm=none grpo.tasks=$TASKS grpo.val_tasks=$VAL grpo.max_steps=$STEPS env.horizon=$H env.max_parallel_envs=$ENVS env.timing.repeats=3 grpo.init_from=$A/warmup/final.pt grpo.save_every=$SAVE_EVERY grpo.eval_every=$EVAL_EVERY grpo.opsd_teacher=$OPSD_TEACHER grpo.opsd_max_tokens=$OPSD_MAX_TOKENS $COMMON"
python scripts/snapshot_ckpts.py --dir "$A" --runs $ARMS > logs/${LOGP:-}opsd_snapshots.log 2>&1 &
for arm in $ARMS; do
  case $arm in
    ctrl)        extra="" ;;
    opsd_err)    extra="grpo.opsd_beta=$OPSD_BETA grpo.opsd_hints=error $ONLY" ;;
    opsd_errloc) extra="grpo.opsd_beta=$OPSD_BETA grpo.opsd_hints=error+loc $ONLY" ;;
  esac
  python -u -m tinyperf.train.grpo $G grpo.out_dir="$A/$arm" $extra > "logs/${LOGP:-}opsd_$arm.log" 2>&1 &
done
wait
for arm in $ARMS; do need "$A/$arm/final.pt"; done

echo "== 5. evaluation without hints  $(date +%T)"
B=$(seq -s ' ' 1 $H)
opt() { [ -s "$2" ] && echo "$1=$2:multi" || true; }
for S in hand real; do
  tasks="$HT"; [ $S = real ] && tasks="$MT"
  runs=""; aruns=""
  for arm in $ARMS warmup; do
    [ -s "$T/${arm}_${S}_trajectories.jsonl" ] || ev "$tasks" eval.checkpoint="$A/$arm/final.pt" eval.n_samples=$EVAL_SAMPLES eval.out_path="$T/${arm}_$S.json"
    runs="$runs $arm=$T/${arm}_${S}_trajectories.jsonl:multi"; aruns="$aruns $arm=$T/${arm}_${S}_trajectories.jsonl"
  done
  python -m tinyperf.eval.budget --reference opsd_errloc --budgets $B --out "$T/budget_$S.json" --runs $runs \
    $( [ -n "$START_EVAL_PREFIX" ] && opt start "${START_EVAL_PREFIX}_${S}_trajectories.jsonl" ) \
    $(opt rl300 "$X/eval/new_rl_${S}_trajectories.jsonl") > "$T/budget_$S.txt"
  PT="$tasks"; [ $S = hand ] && [ -s "$D2/files_test_prof.jsonl" ] && PT="$D2/files_test_prof.jsonl"
  python scripts/analyze_files.py --tasks "$PT" --runs $aruns \
    $( [ -n "$START_EVAL_PREFIX" ] && [ -s "${START_EVAL_PREFIX}_${S}_trajectories.jsonl" ] && echo start="${START_EVAL_PREFIX}_${S}_trajectories.jsonl" ) > "$T/analysis_$S.txt"
done
echo "== done  $(date +%T)  ->  $T/hint_diag.txt  $T/budget_*.txt  $T/analysis_*.txt"
