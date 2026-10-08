#!/usr/bin/env bash
# Off-the-shelf Hugging Face baselines against the TinyPerf models, all evaluated in this one session on the same
# held-out test files, with the same sandbox, budgets and sampling temperature (no re-prompting for any model).
#   default baselines (chat prompt): Qwen3.5-0.8B, DeepSeek Coder 1.3B Instruct, DeepSeek-Coder-V2-Lite-Instruct (mixture of
#   experts: 15.7B total, 2.4B active; backend vllm is much faster for it). Base models without a chat template can use the
#   completion format; gated models need HF_TOKEN and the licence accepted on the model page
#   in parallel: the 22M teacher, the 22M SFT->RL model and the 2M and 200k students re-evaluated in this session
#   then: paired budget tables (reference REF), best-of-k, localization analysis, and how each baseline's replies were read
# Usage:  HF_TOKEN=hf_... nohup bash scripts/run_hf_baselines.sh > logs/hf_baselines.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

# name|model|prompt format|batch|max new tokens|backend (hf, or vllm for speed; vllm runs in its own environment)
MODELS=${MODELS:-"qwen08|Qwen/Qwen3.5-0.8B|chat|32|1024|hf dscoder13|deepseek-ai/deepseek-coder-1.3b-instruct|chat|16|1024|hf dscoderv2lite|deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct|chat|8|1024|hf"}
OUT=${OUT:-artifacts/hf_baselines}; T="$OUT/eval"; REF=${REF:-s200k_rl}
DM=${DM:-artifacts/data_mined}; D2=${D2:-artifacts/data_r2}; HT=${HT:-$D2/files_test.jsonl}; MT=${MT:-$DM/files_mined_test.jsonl}
SHOTS=${SHOTS:-$DM/files_msft_rl.jsonl}; SAMPLES=${SAMPLES:-4}; H=${H:-6}
TEACHER=${TEACHER:-artifacts/opsd_rft/opsd_errloc/snapshots/step0100.pt}; RL22=${RL22:-artifacts/mined_sft/grpo/final.pt}
S2M=${S2M:-artifacts/distill/s2m/rl/final.pt}; S200K=${S200K:-artifacts/distill/s200k/rl/final.pt}
TOK8K=${TOK8K:-artifacts/tokenizer/tokenizer.json}; TOK1K=${TOK1K:-artifacts/distill/tokenizer1k/tokenizer.json}
export HF_HOME=${HF_HOME:-/workspace/hf_cache}; VENV=${VENV:-/workspace/venv_hf}; HF_PY=${HF_PY:-}; FAST_KERNELS=${FAST_KERNELS:-1}
VLLM_VENV=${VLLM_VENV:-/workspace/venv_vllm}; VLLM_PY=${VLLM_PY:-}
COMMON=${COMMON:-}; HF_OVR=${HF_OVR:-}; TABLES_ONLY=${TABLES_ONLY:-0}
SKIP_OURS=${SKIP_OURS:-0}         # 1 = do not (re)start our models' evaluations, e.g. when relaunching one baseline
mkdir -p "$T" logs
names=""; for spec in $MODELS; do names="$names ${spec%%|*}"; done

if [ "$TABLES_ONLY" != 1 ]; then
echo "== 0. Hugging Face environment  $(date +%T)"
if [ -z "$HF_PY" ]; then
  if [ ! -x "$VENV/bin/python" ]; then python -m venv --system-site-packages "$VENV"; "$VENV/bin/pip" install -q -U transformers accelerate; fi
  if [ "$FAST_KERNELS" = 1 ] && ! "$VENV/bin/python" -c "import fla" 2>/dev/null; then
    "$VENV/bin/pip" install -q -U flash-linear-attention || true; "$VENV/bin/pip" install -q causal-conv1d --no-build-isolation || true
  fi
  HF_PY="$VENV/bin/python"
fi
"$HF_PY" -c "import transformers, torch; print('transformers', transformers.__version__, '| torch', torch.__version__)"
if [[ "$MODELS" == *"|vllm"* ]] && [ -z "$VLLM_PY" ]; then        # vLLM pins its own torch, so it gets a separate environment
  if [ ! -x "$VLLM_VENV/bin/python" ]; then python -m venv "$VLLM_VENV"; "$VLLM_VENV/bin/pip" install -q vllm pyyaml numpy; fi
  VLLM_PY="$VLLM_VENV/bin/python"
  "$VLLM_PY" -c "import vllm; print('vllm', vllm.__version__)"
fi

echo "== 1-2. baselines and, in parallel, our models in this session  $(date +%T)"
OURS=""
[ "$SKIP_OURS" = 1 ] || (
  for spec in "teacher22m|$TEACHER|$TOK8K|768" "rl22m|$RL22|$TOK8K|768" "s2m_rl|$S2M|$TOK8K|768" "s200k_rl|$S200K|$TOK1K|1152"; do
    IFS='|' read -r name ck tok maxnew <<< "$spec"
    for set in hand real; do
      tasks="$HT"; [ $set = real ] && tasks="$MT"
      [ -s "$T/${name}_${set}_trajectories.jsonl" ] || python -u -m tinyperf.eval.evaluate --config configs/eval.yaml env.horizon=$H \
          tokenizer.path="$tok" eval.tasks="$tasks" eval.n_tasks=0 eval.n_samples=$SAMPLES eval.checkpoint="$ck" \
          eval.generation.max_new_tokens=$maxnew eval.out_path="$T/${name}_$set.json" $COMMON
    done
  done
) > "logs/hf_baselines_ours.log" 2>&1 &
[ "$SKIP_OURS" = 1 ] || OURS=$!
for spec in $MODELS; do
  IFS='|' read -r name model fmt batch maxnew backend <<< "$spec"
  backend=${backend:-hf}; PY="$HF_PY"; [ "$backend" = vllm ] && PY="$VLLM_PY"
  if [ ! -e "$model" ] && ! "$HF_PY" -c "from huggingface_hub import model_info; model_info('$model')" > /dev/null 2>&1; then
    echo "skipping $name: cannot access $model (gated? set HF_TOKEN and accept the licence on its Hugging Face page)"; continue
  fi
  for set in hand real; do
    tasks="$HT"; [ $set = real ] && tasks="$MT"
    [ -s "$T/${name}_${set}_trajectories.jsonl" ] || "$PY" -u -m tinyperf.eval.evaluate_hf --model "$model" --format $fmt --backend $backend \
        --tasks "$tasks" --n_samples $SAMPLES --shots_from "$SHOTS" --shots edit --out "$T/${name}_$set.json" \
        --max_new_tokens $maxnew --batch_size $batch env.horizon=$H $COMMON $HF_OVR > "logs/${name}_$set.log" 2>&1 \
        || echo "warning: $name on $set failed - see logs/${name}_$set.log"
    { grep -h "eval: final" "logs/${name}_$set.log" | cut -c10-300; } || true
  done
done
[ -z "$OURS" ] || wait $OURS || echo "warning: some of our evaluations failed - see logs/hf_baselines_ours.log"
fi

echo "== 3. comparison (reference: $REF)  $(date +%T)"
all="$names teacher22m rl22m s2m_rl s200k_rl"
for set in hand real; do
  runs=""; for n in $all; do [ -s "$T/${n}_${set}_trajectories.jsonl" ] && runs="$runs $n=$T/${n}_${set}_trajectories.jsonl"; done
  python -m tinyperf.eval.budget --reference $REF --budgets $(seq -s ' ' 1 $H) --out "$T/budget_$set.json" \
      --runs $(for r in $runs; do echo "$r:multi"; done) > "$T/budget_$set.txt"
  python scripts/best_of_k.py --k 1 $SAMPLES --runs $runs > "$T/best_of_k_$set.txt"
  PT="$MT"; [ $set = hand ] && { PT="$HT"; [ -s "$D2/files_test_prof.jsonl" ] && PT="$D2/files_test_prof.jsonl"; }
  python scripts/analyze_files.py --tasks "$PT" --runs $runs > "$T/analysis_$set.txt"
done
python - "$T" $names <<'PY' | tee "$T/baseline_format.txt"
import json, os, sys
T = sys.argv[1]
for n in sys.argv[2:]:
    for s in ("hand", "real"):
        p = f"{T}/{n}_{s}.json"
        if os.path.exists(p):
            r = json.load(open(p)); m = r["metrics"]
            reads = ", ".join(f"{k} {v:.0%}" for k, v in sorted(r["reply_reading"].items(), key=lambda kv: -kv[1]))
            print(f"{n:14s} {s:5s} | {r['n_params'] / 1e6:.0f}M params (total) | {r.get('format', 'chat')} prompt, {r.get('backend', 'hf')} | edits/episode {m['mean_edits']:.2f} | "
                  f"invalid edits {m['invalid_edit_rate']:.0%} | replies: {reads} | cut off {r['truncated_replies']:.0%}")
PY
echo "== done  $(date +%T)  ->  $T/budget_{hand,real}.txt  $T/best_of_k_{hand,real}.txt  $T/analysis_{hand,real}.txt  $T/baseline_format.txt"
