#!/usr/bin/env bash
# Profiler feedback on multi-function files.  Three RL arms, identical except for what they observe:
#   none    : the original code + a step counter
#   full    : the current best code + scalar runtime feedback
#   profile : full + each function's share of the current best file's runtime (a profiler)
# All arms use a fixed budget (STOP disabled in training and evaluation), so they always make the same
# number of edits and face identical reward incentives.  Same test files as the earlier file experiment.
# Usage:  nohup bash scripts/run_profile.sh > logs/profile.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

D=${D:-artifacts/data_r2}; S=${S:-artifacts/r2}; A=${A:-artifacts/files_profile}; M=${M:-configs/model/tiny20m.yaml}
H=${H:-6}; GRPO_STEPS=${GRPO_STEPS:-300}; GRPO_ENVS=${GRPO_ENVS:-20}; EVAL_SAMPLES=${EVAL_SAMPLES:-4}
ANN_OVR=${ANN_OVR:-}; COMMON=${COMMON:-}
T="$A/eval"; mkdir -p "$T" logs
need() { for f in "$@"; do [ -s "$f" ] || { echo "MISSING or empty $f - see logs/profile_*.log"; exit 1; }; done; }

echo "== 1. step-0 profiles for the existing file tasks  $(date +%T)"
for f in files_test files_rl files_rl_val; do
  [ -s "$D/${f}_prof.jsonl" ] || python -u -m tinyperf.data.annotate_profiles --config configs/data.yaml --src "$D/$f.jsonl" --out "$D/${f}_prof.jsonl" $ANN_OVR
done
need "$D/files_test_prof.jsonl" "$D/files_rl_prof.jsonl" "$D/files_rl_val_prof.jsonl"

echo "== 2. fixed-budget GRPO x3  $(date +%T)"
G="--config configs/grpo.yaml model=$M grpo.adv_norm=none grpo.tasks=$D/files_rl_prof.jsonl grpo.val_tasks=$D/files_rl_val_prof.jsonl grpo.max_steps=$GRPO_STEPS env.horizon=$H env.max_parallel_envs=$GRPO_ENVS env.timing.repeats=3 grpo.generation.ban_stop=true eval.generation.ban_stop=true $COMMON"
python scripts/snapshot_ckpts.py --dir "$A" --runs grpo_none grpo_full grpo_profile > logs/profile_snapshots.log 2>&1 &
python -u -m tinyperf.train.grpo $G grpo.init_from="$S/sft_nofb/final.pt"  grpo.out_dir="$A/grpo_none"    env.feedback=none    > logs/profile_grpo_none.log 2>&1 &
python -u -m tinyperf.train.grpo $G grpo.init_from="$S/sft_agent/final.pt" grpo.out_dir="$A/grpo_full"    env.feedback=full    > logs/profile_grpo_full.log 2>&1 &
python -u -m tinyperf.train.grpo $G grpo.init_from="$S/sft_agent/final.pt" grpo.out_dir="$A/grpo_profile" env.feedback=profile > logs/profile_grpo_profile.log 2>&1 &
wait
need "$A/grpo_none/final.pt" "$A/grpo_full/final.pt" "$A/grpo_profile/final.pt"

echo "== 3. held-out files, STOP disabled  $(date +%T)"
E="--config configs/eval.yaml model=$M env.horizon=$H eval.generation.ban_stop=true $COMMON"
ev() { python -u -m tinyperf.eval.evaluate $E eval.tasks="$D/files_test_prof.jsonl" eval.n_tasks=0 eval.n_samples=$EVAL_SAMPLES "$@"; }
ev eval.checkpoint="$A/grpo_profile/final.pt" env.feedback=profile eval.out_path="$T/profile.json"
ev eval.checkpoint="$A/grpo_full/final.pt"    env.feedback=full    eval.out_path="$T/full.json"
ev eval.checkpoint="$A/grpo_none/final.pt"    env.feedback=none    eval.out_path="$T/none.json"
ev eval.checkpoint="$S/sft_agent/final.pt"    env.feedback=profile eval.out_path="$T/sft_profile.json"   # before RL, profile shown
need "$T/profile_trajectories.jsonl" "$T/full_trajectories.jsonl" "$T/none_trajectories.jsonl" "$T/sft_profile_trajectories.jsonl"
B=$(seq -s ' ' 1 $H)
python -m tinyperf.eval.budget --reference profile --budgets $B --out "$T/budget.json" \
  --runs profile="$T/profile_trajectories.jsonl:multi" full="$T/full_trajectories.jsonl:multi" none="$T/none_trajectories.jsonl:multi" \
         sft_profile="$T/sft_profile_trajectories.jsonl:multi" > "$T/budget.txt"
python -m tinyperf.eval.budget --reference full --budgets $B --out "$T/budget_full_vs_none.json" \
  --runs full="$T/full_trajectories.jsonl:multi" none="$T/none_trajectories.jsonl:multi" > "$T/budget_full_vs_none.txt"
python scripts/analyze_files.py --tasks "$D/files_test_prof.jsonl" --runs profile="$T/profile_trajectories.jsonl" \
  full="$T/full_trajectories.jsonl" none="$T/none_trajectories.jsonl" sft_profile="$T/sft_profile_trajectories.jsonl" > "$T/analysis.txt"
echo "== done  $(date +%T)  ->  $T/budget.txt  $T/budget_full_vs_none.txt  $T/analysis.txt"
