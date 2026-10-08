#!/usr/bin/env bash
# Knowledge distillation of the 22M teacher into a 2M and a 200k student, then RL on each student.
#   0. (PRETRAIN=1) brief code pretraining of each student. The 2M student shares the teacher's 8,192-token tokenizer
#      and reuses its tokenized corpus; the 200k student gets its own 1,024-token BPE, because an 8K embedding table
#      alone would exceed 200k parameters
#   1. sequence-level KD: the teacher plays episodes on TRAINING files and functions; each student is fine-tuned on the
#      teacher's actions (failed teacher actions masked, as in SFT)
#   2. GRPO on each student. The reward comes only from the sandbox verifier (correctness on hidden tests, measured
#      speedup over the original code); nothing ties RL to the teacher (no KL to the distilled start, no distillation
#      term), so a student can exceed the teacher.  Same RL file pool and reward function as the teacher's RL
#   3. held-out evaluation (hand-written and real-code test files) of both students after KD and after RL, vs the teacher
# Usage:  nohup bash scripts/run_distill.sh > logs/distill.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

A=${A:-artifacts/distill}; DM=${DM:-artifacts/data_mined}; D2=${D2:-artifacts/data_r2}
TEACHER=${TEACHER:-artifacts/opsd_rft/opsd_errloc/snapshots/step0100.pt}
TEACHER_TOK=${TEACHER_TOK:-artifacts/tokenizer/tokenizer.json}
TEACHER_EVAL_PREFIX=${TEACHER_EVAL_PREFIX:-artifacts/opsd_rft/eval/opsd_errloc100}   # the teacher's existing test evaluations
RL22_EVAL_PREFIX=${RL22_EVAL_PREFIX:-artifacts/mined_sft/eval/new_rl}                # 22M after SFT + RL, for reference
TEACHER_PRETRAIN_DIR=${TEACHER_PRETRAIN_DIR:-artifacts/pretrain_tiny20m}            # its token cache is reused by the 2M student
CORPUS=${CORPUS:-data/corpus/python.jsonl}
STUDENTS=${STUDENTS:-"s2m s200k"}
PRETRAIN=${PRETRAIN:-1}; PRE_STEPS=${PRE_STEPS:-8000}; TOK1K_FILES=${TOK1K_FILES:-20000}; CORPUS1K_FILES=${CORPUS1K_FILES:-150000}
KD_TASKS=${KD_TASKS:-"$DM/files_msft_rl.jsonl $DM/train_sft.jsonl"}; KD_N=${KD_N:-"3000 2000"}
KD_K=${KD_K:-4}; KD_TEMP=${KD_TEMP:-0.6}; KD_MIN_SPEEDUP=${KD_MIN_SPEEDUP:-0}; KD_ENVS=${KD_ENVS:-60}; KD_EPOCHS=${KD_EPOCHS:-3}
RL_STEPS=${RL_STEPS:-300}; RL_ENVS=${RL_ENVS:-40}; RL_CLIP=${RL_CLIP:-4.0};  # reward caps log-speedup at RL_CLIP (4 = ~55x)
 RLT=${RLT:-$DM/files_msft_rl.jsonl}; VAL=${VAL:-$DM/files_mined_val.jsonl}
H=${H:-6}; EVAL_SAMPLES=${EVAL_SAMPLES:-4}; HT=${HT:-$D2/files_test.jsonl}; MT=${MT:-$DM/files_mined_test.jsonl}
COMMON=${COMMON:-}; PRE_OVR=${PRE_OVR:-}; SFT_OVR=${SFT_OVR:-}; RL_OVR=${RL_OVR:-}
T="$A/eval"; mkdir -p "$T" logs
need() { for f in "$@"; do [ -s "$f" ] || { echo "MISSING or empty $f - see logs/distill_*.log"; exit 1; }; done; }

student_cfg() {   # per-student model, tokenizer and learning rates (small models take larger steps)
  case $1 in
    s2m)   M=configs/model/student2m.yaml;   TOK=$TEACHER_TOK;                   PRE_SEQ=1024; SEQ=2048
           LR_PRE=2e-3; LR_SFT=1e-3; LR_RL=${LR_RL_2M:-6e-5};   MAXNEW=768;  CORPUS_FILES=$CORPUS1K_FILES ;;  # used only if the teacher cache is missing
    s200k) M=configs/model/student200k.yaml; TOK=$A/tokenizer1k/tokenizer.json;  PRE_SEQ=2048; SEQ=4096
           LR_PRE=3e-3; LR_SFT=2e-3; LR_RL=${LR_RL_200K:-1e-4}; MAXNEW=1152; CORPUS_FILES=$CORPUS1K_FILES ;;
    *) echo "unknown student $1"; exit 1 ;;
  esac
}

echo "== 0. students: tokenizer and pretraining (PRETRAIN=$PRETRAIN)  $(date +%T)"
if [[ " $STUDENTS " == *" s200k "* ]] && [ ! -s "$A/tokenizer1k/tokenizer.json" ]; then
  python -u -m tinyperf.model.tokenizer --config configs/pretrain.yaml tokenizer.path="$A/tokenizer1k/tokenizer.json" \
      tokenizer.vocab_size=1024 "tokenizer.corpus_jsonl=[$CORPUS]" tokenizer.include_stdlib=false tokenizer.include_tasks=null \
      tokenizer.max_files=$TOK1K_FILES $COMMON > logs/distill_tokenizer1k.log 2>&1
fi
for S in $STUDENTS; do
  student_cfg $S
  python - "$M" "$TOK" <<'PY'
import sys, yaml
from tinyperf.common.config import ModelConfig
from tinyperf.model.tokenizer import CodeTokenizer
from tinyperf.model.transformer import TinyPerfLM
cfg = ModelConfig(**yaml.safe_load(open(sys.argv[1]))); tok = CodeTokenizer(sys.argv[2])
assert tok.vocab_size <= cfg.vocab_size, f"tokenizer has {tok.vocab_size} tokens > model vocab {cfg.vocab_size}"
print(f"{cfg.name}: {TinyPerfLM(cfg).n_params():,} parameters, tokenizer {tok.vocab_size} tokens, context {cfg.max_seq_len}")
PY
  [ "$PRETRAIN" = 1 ] || continue
  D="$A/$S/pretrain"; mkdir -p "$D"
  if [ $S = s2m ] && [ ! -e "$D/tokens_train.npy" ] && [ -s "$TEACHER_PRETRAIN_DIR/tokens_train.npy" ]; then
    ln -s "$(realpath "$TEACHER_PRETRAIN_DIR/tokens_train.npy")" "$D/tokens_train.npy"
    ln -s "$(realpath "$TEACHER_PRETRAIN_DIR/tokens_val.npy")" "$D/tokens_val.npy"
  fi
  [ -s "$D/final.pt" ] || python -u -m tinyperf.train.pretrain --config configs/pretrain.yaml model=$M tokenizer.path="$TOK" \
      pretrain.out_dir="$D" pretrain.seq_len=$PRE_SEQ "pretrain.corpus_jsonl=[$CORPUS]" pretrain.include_stdlib=false \
      pretrain.include_tasks=null tokenizer.max_files=$CORPUS_FILES pretrain.optim.max_steps=$PRE_STEPS pretrain.optim.lr=$LR_PRE \
      pretrain.optim.min_lr=$(python -c "print($LR_PRE / 10)") pretrain.save_every=2000 $COMMON $PRE_OVR > "logs/distill_${S}_pretrain.log" 2>&1
  need "$D/final.pt"
  { grep -E "val_loss" "logs/distill_${S}_pretrain.log" | tail -n 1 | cut -c10-160; } || true
done

echo "== 1. sequence-level KD data from the teacher  $(date +%T)"
[ -s "$A/seqkd.jsonl" ] || python -u -m tinyperf.train.seqkd_data --config configs/grpo.yaml --teacher "$TEACHER" \
    --tasks $KD_TASKS --n_tasks $KD_N --k $KD_K --temperature $KD_TEMP --min_speedup $KD_MIN_SPEEDUP --out "$A/seqkd.jsonl" \
    tokenizer.path="$TEACHER_TOK" env.horizon=$H env.max_parallel_envs=$KD_ENVS env.timing.repeats=3 $COMMON > logs/distill_seqkd.log 2>&1
need "$A/seqkd.jsonl"
cat "$A/seqkd_stats.json" 2>/dev/null || true; echo

echo "== 2. KD: fine-tune each student on the teacher's actions  $(date +%T)"
for S in $STUDENTS; do
  student_cfg $S
  init="sft.init_from=null"; [ "$PRETRAIN" = 1 ] && init="sft.init_from=$A/$S/pretrain/final.pt"
  [ -s "$A/$S/kd/final.pt" ] || python -u -m tinyperf.train.sft --config configs/sft.yaml model=$M tokenizer.path="$TOK" $init \
      sft.trajectories="$A/seqkd.jsonl" sft.out_dir="$A/$S/kd" sft.seq_len=$SEQ sft.epochs=$KD_EPOCHS sft.optim.lr=$LR_SFT \
      sft.optim.min_lr=$(python -c "print($LR_SFT / 10)") sft.mask_failed_actions=true $COMMON $SFT_OVR > "logs/distill_${S}_kd.log" 2>&1
  need "$A/$S/kd/final.pt"
  { grep -E "SFT data|val_loss" "logs/distill_${S}_kd.log" | tail -n 2 | cut -c10-200; } || true
done

echo "== 3. GRPO on each student: reward from the sandbox verifier only  $(date +%T)"
runs=""
for S in $STUDENTS; do
  student_cfg $S
  [ -s "$A/$S/rl/final.pt" ] && continue
  runs="$runs $S/rl"
  python -u -m tinyperf.train.grpo --config configs/grpo.yaml model=$M tokenizer.path="$TOK" grpo.adv_norm=none \
      grpo.tasks="$RLT" grpo.val_tasks="$VAL" grpo.max_steps=$RL_STEPS env.horizon=$H env.max_parallel_envs=$RL_ENVS \
      env.timing.repeats=3 grpo.init_from="$A/$S/kd/final.pt" grpo.out_dir="$A/$S/rl" grpo.lr=$LR_RL \
      grpo.generation.max_new_tokens=$MAXNEW grpo.save_every=50 grpo.eval_every=50 grpo.kl_beta=0 grpo.opsd_beta=0 \
      reward.clip_log_speedup=$RL_CLIP $COMMON $RL_OVR > "logs/distill_${S}_rl.log" 2>&1 &
done
[ -n "$runs" ] && python scripts/snapshot_ckpts.py --dir "$A" --runs $runs > logs/distill_snapshots.log 2>&1 &
wait
for S in $STUDENTS; do need "$A/$S/rl/final.pt"; done

echo "== 4. held-out evaluation  $(date +%T)"
for S in $STUDENTS; do (
  student_cfg $S
  for stage in kd rl; do for set in hand real; do
    tasks="$HT"; [ $set = real ] && tasks="$MT"
    [ -s "$T/${S}_${stage}_${set}_trajectories.jsonl" ] || python -u -m tinyperf.eval.evaluate --config configs/eval.yaml model=$M \
        tokenizer.path="$TOK" env.horizon=$H eval.tasks="$tasks" eval.n_tasks=0 eval.n_samples=$EVAL_SAMPLES \
        eval.checkpoint="$A/$S/$stage/final.pt" eval.generation.max_new_tokens=$MAXNEW eval.out_path="$T/${S}_${stage}_$set.json" $COMMON
  done; done ) > "logs/distill_${S}_eval.log" 2>&1 &
done
wait
opt() { [ -s "$2" ] && echo "$1=$2" || true; }
for set in hand real; do
  runs="$(opt teacher22m "${TEACHER_EVAL_PREFIX}_${set}_trajectories.jsonl") $(opt rl22m "${RL22_EVAL_PREFIX}_${set}_trajectories.jsonl")"
  for S in $STUDENTS; do for stage in rl kd; do runs="$runs $(opt ${S}_$stage "$T/${S}_${stage}_${set}_trajectories.jsonl")"; done; done
  ref=$(echo $runs | awk '{print $1}' | cut -d= -f1)
  python -m tinyperf.eval.budget --reference $ref --budgets $(seq -s ' ' 1 $H) --out "$T/budget_$set.json" \
      --runs $(for r in $runs; do echo "$r:multi"; done) > "$T/budget_$set.txt"
  PT="$MT"; [ $set = hand ] && { PT="$HT"; [ -s "$D2/files_test_prof.jsonl" ] && PT="$D2/files_test_prof.jsonl"; }
  python scripts/analyze_files.py --tasks "$PT" --runs $runs > "$T/analysis_$set.txt"
  TT="${TEACHER_EVAL_PREFIX}_${set}_trajectories.jsonl"
  if [ -s "$TT" ]; then
    sruns=""; for S in $STUDENTS; do for stage in rl kd; do sruns="$sruns $(opt ${S}_$stage "$T/${S}_${stage}_${set}_trajectories.jsonl")"; done; done
    python scripts/compare_to_teacher.py --teacher "$TT" --tasks "$PT" --students $sruns > "$T/vs_teacher_$set.txt"
  fi
done
python - "$A" "$TEACHER" $STUDENTS <<'PY' | tee "$T/params.txt"
import os, sys, torch
from tinyperf.common.config import ModelConfig
from tinyperf.model.transformer import TinyPerfLM
A, teacher, students = sys.argv[1], sys.argv[2], sys.argv[3:]
rows = [("teacher22m", teacher)] + [(f"{s}_{st}", f"{A}/{s}/{st}/final.pt") for s in students for st in ("kd", "rl")]
for name, p in rows:
    if os.path.exists(p):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        n = TinyPerfLM(ModelConfig(**ck["model_cfg"])).n_params()      # counts the tied embedding once
        print(f"{name:12s} {n:>12,} parameters  ({ck['model_cfg'].get('vocab_size')} vocab)  {p}")
PY
echo "== done  $(date +%T)  ->  $T/budget_{hand,real}.txt  $T/vs_teacher_{hand,real}.txt  $T/analysis_{hand,real}.txt  $T/params.txt"
