#!/usr/bin/env bash
# Off-the-shelf Qwen3.5-0.8B baseline against the TinyPerf models, all evaluated in this one session on the same
# held-out test files, with the same sandbox, budgets and sampling temperature.
#   0. a separate virtual environment with a recent transformers (reuses the installed torch; the project's own
#      environment is untouched)
#   1. Qwen with thinking off (THINKING_ARM=1 adds a thinking-on run), prompted with the protocol and two worked
#      examples from a training file; replies parsed leniently, with format failures counted
#   2. in parallel: the 22M teacher, the 22M SFT->RL model, and the 2M and 200k students re-evaluated in this session
#   3. paired budget tables (reference: Qwen), best-of-k, localization analysis, and how Qwen's replies were read
# Usage:  nohup bash scripts/run_qwen_baseline.sh > logs/qwen_baseline.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

QWEN=${QWEN:-Qwen/Qwen3.5-0.8B}; OUT=${OUT:-artifacts/qwen_baseline}; T="$OUT/eval"
DM=${DM:-artifacts/data_mined}; D2=${D2:-artifacts/data_r2}; HT=${HT:-$D2/files_test.jsonl}; MT=${MT:-$DM/files_mined_test.jsonl}
SHOTS=${SHOTS:-$DM/files_msft_rl.jsonl}; SAMPLES=${SAMPLES:-4}; H=${H:-6}; THINKING_ARM=${THINKING_ARM:-0}
QWEN_BATCH=${QWEN_BATCH:-32}; QWEN_MAXNEW=${QWEN_MAXNEW:-1024}; THINK_MAXNEW=${THINK_MAXNEW:-4096}; QWEN_SHOTS=${QWEN_SHOTS:-edit}
EXTRA_NAMES=${EXTRA_NAMES:-}      # more runs to include in the tables if their files exist (e.g. qwen2shot)
TABLES_ONLY=${TABLES_ONLY:-0}     # 1 = skip all evaluation, only rebuild the tables
TEACHER=${TEACHER:-artifacts/opsd_rft/opsd_errloc/snapshots/step0100.pt}; RL22=${RL22:-artifacts/mined_sft/grpo/final.pt}
S2M=${S2M:-artifacts/distill/s2m/rl/final.pt}; S200K=${S200K:-artifacts/distill/s200k/rl/final.pt}
TOK8K=${TOK8K:-artifacts/tokenizer/tokenizer.json}; TOK1K=${TOK1K:-artifacts/distill/tokenizer1k/tokenizer.json}
export HF_HOME=${HF_HOME:-/workspace/hf_cache}; VENV=${VENV:-/workspace/venv_hf}; HF_PY=${HF_PY:-}
COMMON=${COMMON:-}; QWEN_OVR=${QWEN_OVR:-}
mkdir -p "$T" logs

if [ "$TABLES_ONLY" != 1 ]; then
echo "== 0. Hugging Face environment  $(date +%T)"
if [ -z "$HF_PY" ]; then
  if [ ! -x "$VENV/bin/python" ]; then python -m venv --system-site-packages "$VENV"; "$VENV/bin/pip" install -q -U transformers accelerate; fi
  HF_PY="$VENV/bin/python"
fi
"$HF_PY" -c "import transformers, torch; print('transformers', transformers.__version__, '| torch', torch.__version__)"

echo "== 1-2. Qwen (thinking off) and, in parallel, our models in this session  $(date +%T)"
(
  for spec in "teacher22m|$TEACHER|$TOK8K|768" "rl22m|$RL22|$TOK8K|768" "s2m_rl|$S2M|$TOK8K|768" "s200k_rl|$S200K|$TOK1K|1152"; do
    IFS='|' read -r name ck tok maxnew <<< "$spec"
    for set in hand real; do
      tasks="$HT"; [ $set = real ] && tasks="$MT"
      [ -s "$T/${name}_${set}_trajectories.jsonl" ] || python -u -m tinyperf.eval.evaluate --config configs/eval.yaml env.horizon=$H \
          tokenizer.path="$tok" eval.tasks="$tasks" eval.n_tasks=0 eval.n_samples=$SAMPLES eval.checkpoint="$ck" \
          eval.generation.max_new_tokens=$maxnew eval.out_path="$T/${name}_$set.json" $COMMON
    done
  done
) > logs/qwen_baseline_ours.log 2>&1 &
OURS=$!
for set in hand real; do
  tasks="$HT"; [ $set = real ] && tasks="$MT"
  [ -s "$T/qwen_${set}_trajectories.jsonl" ] || "$HF_PY" -u -m tinyperf.eval.evaluate_hf --model "$QWEN" --tasks "$tasks" \
      --n_samples $SAMPLES --shots_from "$SHOTS" --shots $QWEN_SHOTS --out "$T/qwen_$set.json" --max_new_tokens $QWEN_MAXNEW --batch_size $QWEN_BATCH \
      env.horizon=$H $COMMON $QWEN_OVR > "logs/qwen_$set.log" 2>&1
  { grep -h "eval: final" "logs/qwen_$set.log" | cut -c10-300; } || true
  if [ "$THINKING_ARM" = 1 ]; then
    [ -s "$T/qwen_think_${set}_trajectories.jsonl" ] || "$HF_PY" -u -m tinyperf.eval.evaluate_hf --model "$QWEN" --tasks "$tasks" \
        --thinking --n_samples $SAMPLES --shots_from "$SHOTS" --shots $QWEN_SHOTS --out "$T/qwen_think_$set.json" --max_new_tokens $THINK_MAXNEW \
        --batch_size $QWEN_BATCH env.horizon=$H $COMMON $QWEN_OVR > "logs/qwen_think_$set.log" 2>&1
  fi
done
wait $OURS || echo "warning: some of our evaluations failed - see logs/qwen_baseline_ours.log"
fi

echo "== 3. comparison  $(date +%T)"
names="qwen"; [ "$THINKING_ARM" = 1 ] && names="$names qwen_think"; names="$names $EXTRA_NAMES teacher22m rl22m s2m_rl s200k_rl"
for set in hand real; do
  runs=""; for n in $names; do [ -s "$T/${n}_${set}_trajectories.jsonl" ] && runs="$runs $n=$T/${n}_${set}_trajectories.jsonl"; done
  python -m tinyperf.eval.budget --reference qwen --budgets $(seq -s ' ' 1 $H) --out "$T/budget_$set.json" \
      --runs $(for r in $runs; do echo "$r:multi"; done) > "$T/budget_$set.txt"
  python scripts/best_of_k.py --k 1 $SAMPLES --runs $runs > "$T/best_of_k_$set.txt"
  PT="$MT"; [ $set = hand ] && { PT="$HT"; [ -s "$D2/files_test_prof.jsonl" ] && PT="$D2/files_test_prof.jsonl"; }
  python scripts/analyze_files.py --tasks "$PT" --runs $runs > "$T/analysis_$set.txt"
done
python - "$T" $names <<'PY' | tee "$T/qwen_format.txt"
import json, os, sys
T = sys.argv[1]
for n in sys.argv[2:]:
    if not n.startswith("qwen"):
        continue
    for s in ("hand", "real"):
        p = f"{T}/{n}_{s}.json"
        if os.path.exists(p):
            r = json.load(open(p))
            reads = ", ".join(f"{k} {v:.0%}" for k, v in sorted(r["reply_reading"].items(), key=lambda kv: -kv[1]))
            print(f"{n:11s} {s:5s} | {r['n_params'] / 1e6:.0f}M parameters | replies read as: {reads} | cut off at the token limit {r['truncated_replies']:.0%}")
PY
echo "== done  $(date +%T)  ->  $T/budget_{hand,real}.txt  $T/best_of_k_{hand,real}.txt  $T/analysis_{hand,real}.txt  $T/qwen_format.txt"
