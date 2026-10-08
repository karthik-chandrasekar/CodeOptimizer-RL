#!/usr/bin/env bash
# Anti-overfitting: RL on file tasks built from thousands of functions mined from the pretraining corpus.
#   1. mine self-contained functions (static filters + execution-confirmed input types)
#   2. build verified single-function tasks from them (degraded + natural); 10% of mined functions held out
#   3. RL files: "mined" (mined functions only) and "mixed" (mined + the 38 hand-written RL functions);
#      validation files from held-out mined functions, so the validation curve measures transfer
#   4. GRPO x2 with the same settings as the 38-function file agent; snapshots every 100 steps
#   5. evaluate at MID_STEP and at the end on the same held-out test files as all earlier file results
# Usage:  nohup bash scripts/run_mined.sh > logs/mined.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

CORPUS=${CORPUS:-data/corpus/python.jsonl}
MINED=${MINED:-artifacts/mined/mined_seeds.jsonl}
D2=${D2:-artifacts/data_r2}; S=${S:-artifacts/r2}; F=${F:-artifacts/files}
DM=${DM:-artifacts/data_mined}; A=${A:-artifacts/files_mined}; M=${M:-configs/model/tiny20m.yaml}
WORKERS=${WORKERS:-$(( $(nproc) - 8 ))}
MAX_CANDIDATES=${MAX_CANDIDATES:-60000}; MAX_MINED=${MAX_MINED:-5000}; SEED_WORKERS=${SEED_WORKERS:-24}
N_FILES=${N_FILES:-3000}; N_VAL=${N_VAL:-64}
GRPO_STEPS=${GRPO_STEPS:-500}; MID_STEP=${MID_STEP:-300}; GRPO_ENVS=${GRPO_ENVS:-30}; H=${H:-6}; EVAL_SAMPLES=${EVAL_SAMPLES:-4}
MINE_OVR=${MINE_OVR:-}; BUILD_OVR=${BUILD_OVR:-}; FILE_OVR=${FILE_OVR:-}; COMMON=${COMMON:-}
T="$A/eval"; mkdir -p "$T" "$(dirname "$MINED")" logs
need() { for f in "$@"; do [ -s "$f" ] || { echo "MISSING or empty $f - see logs/mined_*.log"; exit 1; }; done; }

echo "== 1. mine $CORPUS  $(date +%T)"
[ -s "$MINED" ] || python -u -m tinyperf.data.mine --corpus "$CORPUS" --out "$MINED" --workers $WORKERS --max_candidates $MAX_CANDIDATES $MINE_OVR
need "$MINED"

echo "== 2. verified tasks from mined functions  $(date +%T)"
[ -s "$DM/train.jsonl" ] || python -u -m tinyperf.data.build --config configs/data.yaml data.out_dir="$DM" data.builtin_seeds=false \
    data.mined_seeds="$MINED" data.max_mined=$MAX_MINED data.seed_workers=$SEED_WORKERS data.heldout_seed_frac=0.1 'data.heldout_families=[]' \
    data.n_train=16000 data.n_val=200 data.n_test_iid=100 data.n_test_compositional=100 data.n_test_heldout=0 data.n_test_seeds=600 $BUILD_OVR
need "$DM/train.jsonl" "$DM/test_seeds.jsonl"

echo "== 3. file tasks  $(date +%T)"
bf() { python -u -m tinyperf.data.build_files --config configs/data.yaml --k 2 4 --d 1 3 "$@" $FILE_OVR; }
cat "$DM/train.jsonl" "$D2/train_rl.jsonl" > "$DM/train_mixed.jsonl"
[ -s "$DM/files_mined_rl.jsonl" ]  || bf --src "$DM/train.jsonl"       --out "$DM/files_mined_rl.jsonl"  --split files_mined_rl  --n $N_FILES --seed 4
[ -s "$DM/files_mixed_rl.jsonl" ]  || bf --src "$DM/train_mixed.jsonl" --out "$DM/files_mixed_rl.jsonl"  --split files_mixed_rl  --n $N_FILES --seed 5
[ -s "$DM/files_mined_val.jsonl" ] || bf --src "$DM/test_seeds.jsonl"  --out "$DM/files_mined_val.jsonl" --split files_mined_val --n $N_VAL   --seed 6
need "$DM/files_mined_rl.jsonl" "$DM/files_mixed_rl.jsonl" "$DM/files_mined_val.jsonl"

echo "== 4. GRPO x2  $(date +%T)"
G="--config configs/grpo.yaml model=$M grpo.adv_norm=none grpo.val_tasks=$DM/files_mined_val.jsonl grpo.max_steps=$GRPO_STEPS env.horizon=$H env.max_parallel_envs=$GRPO_ENVS env.timing.repeats=3 grpo.init_from=$S/sft_agent/final.pt $COMMON"
python scripts/snapshot_ckpts.py --dir "$A" --runs grpo_mined grpo_mixed > logs/mined_snapshots.log 2>&1 &
python -u -m tinyperf.train.grpo $G grpo.tasks="$DM/files_mined_rl.jsonl" grpo.out_dir="$A/grpo_mined" > logs/mined_grpo_mined.log 2>&1 &
python -u -m tinyperf.train.grpo $G grpo.tasks="$DM/files_mixed_rl.jsonl" grpo.out_dir="$A/grpo_mixed" > logs/mined_grpo_mixed.log 2>&1 &
wait
MID=$(printf "step%04d.pt" $MID_STEP)
need "$A/grpo_mined/final.pt" "$A/grpo_mixed/final.pt" "$A/grpo_mined/snapshots/$MID" "$A/grpo_mixed/snapshots/$MID"

echo "== 5. held-out test files  $(date +%T)"
TEST="$D2/files_test.jsonl"
E="--config configs/eval.yaml model=$M env.horizon=$H $COMMON"
ev() { python -u -m tinyperf.eval.evaluate $E eval.tasks="$TEST" eval.n_tasks=0 eval.n_samples=$EVAL_SAMPLES "$@"; }
for arm in mined mixed; do
  ev eval.checkpoint="$A/grpo_$arm/snapshots/$MID" eval.out_path="$T/${arm}_mid.json"
  ev eval.checkpoint="$A/grpo_$arm/final.pt"       eval.out_path="$T/${arm}_end.json"
done
need "$T/mined_mid_trajectories.jsonl" "$T/mixed_mid_trajectories.jsonl" "$T/mined_end_trajectories.jsonl" "$T/mixed_end_trajectories.jsonl"
opt() { [ -s "$2" ] && echo "$1=$2:multi" || true; }   # earlier baselines, included when present
B=$(seq -s ' ' 1 $H)
python -m tinyperf.eval.budget --reference mined --budgets $B --out "$T/budget_mid.json" \
  --runs mined="$T/mined_mid_trajectories.jsonl:multi" mixed="$T/mixed_mid_trajectories.jsonl:multi" \
         $(opt hand38 "$F/eval/rl_agent_files_trajectories.jsonl") $(opt sft "$F/eval/sft_agent_files_trajectories.jsonl") > "$T/budget_mid.txt"
python -m tinyperf.eval.budget --reference mined --budgets $B --out "$T/budget_end.json" \
  --runs mined="$T/mined_end_trajectories.jsonl:multi" mixed="$T/mixed_end_trajectories.jsonl:multi" \
         $(opt hand38 "$F/eval/rl_agent_ext_files_trajectories.jsonl") $(opt sft "$F/eval/sft_agent_files_trajectories.jsonl") > "$T/budget_end.txt"
PT="$D2/files_test_prof.jsonl"; [ -s "$PT" ] || PT="$TEST"
python scripts/analyze_files.py --tasks "$PT" --runs mined="$T/mined_mid_trajectories.jsonl" mixed="$T/mixed_mid_trajectories.jsonl" \
  $( [ -s "$F/eval/rl_agent_files_trajectories.jsonl" ] && echo hand38="$F/eval/rl_agent_files_trajectories.jsonl" ) > "$T/analysis.txt"
echo "== done  $(date +%T)  ->  $T/budget_mid.txt  $T/budget_end.txt  $T/analysis.txt"
