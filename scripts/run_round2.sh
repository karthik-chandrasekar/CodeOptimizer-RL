#!/usr/bin/env bash
# Round 2.  Changes vs round 1:
#   * 25% of functions held out as test_seeds (was 15%), 500 test tasks
#   * SFT and RL use disjoint functions (60% / 40% of the training functions)
#   * SFT demonstrations must reach >= 80% of the known fast version (no "partial fix, then STOP")
#   * RL tasks filtered to chains the SFT agent solves 10-90% of the time; all three arms use the same tasks
#   * evaluation success = speedup >= 1.5x; best code kept in evaluation records
# Usage:  nohup bash scripts/run_round2.sh > logs/round2.log 2>&1 &
# Re-running skips the dataset build and trajectory generation if their outputs exist.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

R=${R:-r2}
D=${D:-artifacts/data_$R}                         # dataset of this round
A=${A:-artifacts/$R}                              # checkpoints, trajectories, evaluations of this round
M=${M:-configs/model/tiny20m.yaml}
PRE=${PRE:-artifacts/pretrain_tiny20m/final.pt}   # round-1 pretraining is reused
GRPO_STEPS=${GRPO_STEPS:-1000}
PROBE_TASKS=${PROBE_TASKS:-1200}
RL_FRAC=${RL_FRAC:-0.4}
FILTER_LO=${FILTER_LO:-0.1}
FILTER_HI=${FILTER_HI:-0.9}
SPLITS=${SPLITS:-"test_seeds test_heldout"}
BUILD_OVR=${BUILD_OVR:-}                          # extra overrides for the dataset build (smoke tests)
COMMON=${COMMON:-}                                # extra overrides for every other stage (smoke tests)
mkdir -p "$A/eval" logs

need() { for f in "$@"; do [ -s "$f" ] || { echo "MISSING or empty $f - stage failed, see logs/${R}_*.log"; exit 1; }; done; }

echo "== 1. dataset  $(date +%T)"
[ -f "$D/test_seeds.jsonl" ] || python -u -m tinyperf.data.build --config configs/data.yaml \
    data.out_dir="$D" data.heldout_seed_frac=0.25 data.n_test_seeds=500 $BUILD_OVR

echo "== 2. split training functions 60% SFT / 40% RL  $(date +%T)"
python -u -m tinyperf.data.split_train --data_dir "$D" --rl_frac $RL_FRAC

echo "== 3. SFT trajectories (SFT functions only, complete fixes only)  $(date +%T)"
[ -f "$A/trajectories_stats.json" ] || python -u -m tinyperf.train.sft_data --config configs/sft_data.yaml \
    sft_data.tasks="$D/train_sft.jsonl" sft_data.out_path="$A/trajectories.jsonl" sft_data.min_ref_frac=0.8 $COMMON
cat "$A/trajectories_stats.json"; echo

echo "== 4. SFT x3  $(date +%T)"
S="--config configs/sft.yaml model=$M sft.init_from=$PRE sft.trajectories=$A/trajectories.jsonl $COMMON"
python -u -m tinyperf.train.sft $S sft.out_dir="$A/sft_agent"                          > logs/${R}_sft_agent.log 2>&1 &
python -u -m tinyperf.train.sft $S sft.out_dir="$A/sft1"      sft.first_step_only=true > logs/${R}_sft1.log 2>&1 &
python -u -m tinyperf.train.sft $S sft.out_dir="$A/sft_nofb"  env.feedback=none        > logs/${R}_sft_nofb.log 2>&1 &
wait
need "$A/sft_agent/final.pt" "$A/sft1/final.pt" "$A/sft_nofb/final.pt"

echo "== 5. difficulty probe + filter  $(date +%T)"
E="--config configs/eval.yaml model=$M $COMMON"
python -u -m tinyperf.eval.evaluate $E eval.checkpoint="$A/sft_agent/final.pt" eval.tasks="$D/train_rl.jsonl" \
    eval.n_tasks=$PROBE_TASKS eval.n_samples=4 eval.generation.temperature=0.8 eval.out_path="$A/probe.json" > logs/${R}_probe.log 2>&1
python -u -m tinyperf.train.filter_tasks --tasks "$D/train_rl.jsonl" --probe "$A/probe_trajectories.jsonl" \
    --out "$D/train_rl_filtered.jsonl" --lo $FILTER_LO --hi $FILTER_HI --threshold 1.5
need "$D/train_rl_filtered.jsonl"

echo "== 6. GRPO x3 on the same filtered RL tasks  $(date +%T)"
G="--config configs/grpo.yaml model=$M grpo.adv_norm=none grpo.tasks=$D/train_rl_filtered.jsonl grpo.val_tasks=$D/val.jsonl grpo.max_steps=$GRPO_STEPS $COMMON"
python -u -m tinyperf.train.grpo $G grpo.init_from="$A/sft_agent/final.pt" grpo.out_dir="$A/grpo_agent"                   > logs/${R}_grpo_agent.log 2>&1 &
python -u -m tinyperf.train.grpo $G grpo.init_from="$A/sft1/final.pt"      grpo.out_dir="$A/grpo1"     env.horizon=1     > logs/${R}_grpo1.log 2>&1 &
python -u -m tinyperf.train.grpo $G grpo.init_from="$A/sft_nofb/final.pt"  grpo.out_dir="$A/grpo_nofb" env.feedback=none > logs/${R}_grpo_nofb.log 2>&1 &
wait
need "$A/grpo_agent/final.pt" "$A/grpo1/final.pt" "$A/grpo_nofb/final.pt"

echo "== 7. evaluation (final.pt; success = >=1.5x)  $(date +%T)"
T="$A/eval"
for SP in $SPLITS; do
  ev() { python -u -m tinyperf.eval.evaluate $E eval.tasks="$D/$SP.jsonl" eval.n_tasks=0 "$@" 2>&1 | grep -E "INFO eval: final|Error|Traceback" || true; }
  ev eval.checkpoint="$A/grpo_agent/final.pt" eval.n_samples=4                   eval.out_path="$T/rl_agent_$SP.json"
  ev eval.checkpoint="$A/grpo_nofb/final.pt"  eval.n_samples=4 env.feedback=none eval.out_path="$T/rl_nofb_$SP.json"
  ev eval.checkpoint="$A/grpo1/final.pt"      eval.n_samples=8 env.horizon=1     eval.out_path="$T/rl1_$SP.json"
  ev eval.checkpoint="$A/sft_agent/final.pt"  eval.n_samples=4                   eval.out_path="$T/sft_agent_$SP.json"
  ev eval.checkpoint="$A/sft1/final.pt"       eval.n_samples=8 env.horizon=1     eval.out_path="$T/sft1_$SP.json"
  python -m tinyperf.eval.budget --reference agent_rl --budgets 1 2 3 4 6 --out "$T/budget_$SP.json" \
    --runs agent_rl="$T/rl_agent_${SP}_trajectories.jsonl:multi" nofb_rl="$T/rl_nofb_${SP}_trajectories.jsonl:multi" \
           rl1="$T/rl1_${SP}_trajectories.jsonl:oneshot" agent_sft="$T/sft_agent_${SP}_trajectories.jsonl:multi" > "$T/budget_$SP.txt"
  python -m tinyperf.eval.budget --reference rl1 --budgets 1 2 4 8 --out "$T/passk_$SP.json" \
    --runs rl1="$T/rl1_${SP}_trajectories.jsonl:oneshot" sft1="$T/sft1_${SP}_trajectories.jsonl:oneshot" > "$T/passk_$SP.txt"
done
echo "== done  $(date +%T)  ->  $T/budget_*.txt  $T/passk_*.txt"
