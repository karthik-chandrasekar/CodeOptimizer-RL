#!/usr/bin/env bash
# End-to-end CPU smoke run of TinyPerf-Chain on a toy scale (tiny tokenizer, untrained tiny model, few tasks).
# It exercises every stage and every interface; it measures nothing meaningful.
#   bash scripts/chain/smoke.sh            (A=artifacts/chain_smoke by default; ~10-20 minutes on 2 CPUs)
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
A=${A:-artifacts/chain_smoke}
UNITS=${UNITS:-10}
mkdir -p "$A"
C="configs/chain/smoke.yaml paths.root=$A"
CFG="--config configs/chain/smoke.yaml"
OVR="paths.root=$A model.tokenizer=$A/tokenizer.json model.seed_checkpoint=$A/seed.pt"

echo "== 0. toy tokenizer and an untrained toy seed model  $(date +%T)"
[ -s "$A/tokenizer.json" ] || python -m tinyperf.model.tokenizer --config configs/pretrain.yaml tokenizer.path="$A/tokenizer.json" \
    tokenizer.vocab_size=1024 tokenizer.include_tasks=null tokenizer.max_files=400 > "$A/tokenizer.log" 2>&1
[ -s "$A/seed.pt" ] || python - "$A" <<'PY'
import sys, torch
from tinyperf.common.config import ModelConfig
from tinyperf.model.transformer import TinyPerfLM
torch.manual_seed(0)
m = TinyPerfLM(ModelConfig(name="smoke", vocab_size=1024, n_layer=2, n_embd=32, n_head=2, mlp_dim=64, max_seq_len=2048))
m.save(f"{sys.argv[1]}/seed.pt", extra={"stage": "smoke_untrained"})
print("seed params", m.n_params())
PY

echo "== 1. clusters and splits  $(date +%T)"
[ -s "$A/split_manifest.json" ] || python -m tinyperf_chain.data.split $CFG --out "$A/split_manifest.json" --fail-on-overlap $OVR

echo "== 2. tasks (train / dev / confirm)  $(date +%T)"
[ -s "$A/tasks/train_files.jsonl" ] || python -m tinyperf_chain.data.build_tasks $CFG --manifest "$A/split_manifest.json" --units-limit $UNITS --max-variants 2 \
    --group-size 3 --tokenizer "$A/tokenizer.json" $OVR
python -m tinyperf_chain.data.split --check "$A/split_manifest.json" --tasks-root "$A/tasks" --fail-on-overlap

echo "== 3. data factory on train and dev (never confirm)  $(date +%T)"
rm -f "$A/ledger/trials.jsonl"
for S in train dev; do
  python -m tinyperf_chain.data.collect $CFG --split $S --stages plans probes repairs targets gate $OVR
done

echo "== 4. role datasets  $(date +%T)"
python -m tinyperf_chain.data.project $CFG --splits train dev --manifest "$A/split_manifest.json" --review 10 $OVR

echo "== 5. branch the seed into five roles  $(date +%T)"
rm -rf "$A/checkpoints"
python -m tinyperf_chain.train.branch $CFG --out "$A/checkpoints/initial" $OVR

echo "== 6. role SFT (+ shared-role control)  $(date +%T)"
for R in M1 M2 M3 M4 M5 shared; do
  INIT="$A/checkpoints/initial/${R}.pt"; [ $R = shared ] && INIT="$A/checkpoints/initial/M3.pt"
  python -m tinyperf_chain.train.sft $CFG --role $R --init "$INIT" --out "$A/checkpoints/${R}_sft" $OVR > "$A/sft_$R.log" 2>&1 \
    || { echo "SFT $R failed (see $A/sft_$R.log)"; tail -5 "$A/sft_$R.log"; [ $R = shared ] || exit 1; }
done

echo "== 7. one RL update on M3  $(date +%T)"
python -m tinyperf_chain.train.rl $CFG --role M3 --init "$A/checkpoints/M3_sft/final.pt" --out "$A/checkpoints/M3_rl" $OVR \
    > "$A/rl_M3.log" 2>&1 || { echo "RL failed"; tail -20 "$A/rl_M3.log"; exit 1; }

echo "== 8. bundles and a paired development session  $(date +%T)"
python - "$A" <<'PY'
import json, sys
A = sys.argv[1]
roles = {r: {"adapter": "neural", "checkpoint": f"{A}/checkpoints/{r}_sft/final.pt"} for r in ("M1", "M2", "M3", "M4", "M5")}
json.dump({"schema_version": "bundle_v1", "system_id": "B4", "tokenizer": f"{A}/tokenizer.json", "roles": roles},
          open(f"{A}/B4.json", "w"), indent=1)
rules = {"M1": {"adapter": "rule", "rule": "hottest_self"}, "M2": {"adapter": "rule"}, "M3": {"adapter": "rule"},
         "M4": {"adapter": "rule", "rule": "transform_specific"}, "M5": {"adapter": "rule"}}
json.dump({"schema_version": "bundle_v1", "system_id": "R", "roles": rules}, open(f"{A}/R.json", "w"), indent=1)
b3 = dict(rules); b3["M3"] = roles["M3"]; b3["M5"] = {**roles["M3"], "prompt_role": "M5"}
json.dump({"schema_version": "bundle_v1", "system_id": "B3", "tokenizer": f"{A}/tokenizer.json", "roles": b3},
          open(f"{A}/B3.json", "w"), indent=1)
PY
python - "$A" <<'PY'
import json, sys
A = sys.argv[1]
m0 = {"M1": {"adapter": "rule", "rule": "hottest_self"}, "M2": {"adapter": "rule"},
      "M3": {"adapter": "legacy_writer", "checkpoint": f"{A}/seed.pt", "max_new": 96},
      "M4": {"adapter": "rule", "rule": "fixed_edge"}, "M5": {"adapter": "rule"}}
json.dump({"schema_version": "bundle_v1", "system_id": "M0", "tokenizer": f"{A}/tokenizer.json", "roles": m0},
          open(f"{A}/M0.json", "w"), indent=1)
PY
rm -rf "$A/runs/dev"
python -m tinyperf_chain.eval.run $CFG --suite development --bundle "$A/R.json" "$A/B3.json" "$A/B4.json" "$A/M0.json" \
    --legacy-checkpoint "$A/seed.pt" --output "$A/runs/dev" --episodes-per-file 1 --limit 2 --patch-slots 2 6 $OVR
python -m tinyperf_chain.eval.aggregate --runs "$A/runs/dev/episodes.jsonl" --paired --reference R
python -m tinyperf_chain.eval.roles $CFG --bundle "$A/R.json" --split dev --limit 4 $OVR > "$A/roles_R.json"
python -m tinyperf_chain.eval.roles $CFG --bundle "$A/B4.json" --split dev --limit 3 $OVR > "$A/roles_B4.json"
python -c "import json;d=json.load(open('$A/roles_R.json'));print('role diagnostics (rules):', {k: d[k] for k in ('m1', 'm5')})"

echo "== 9. lock a bundle and verify it  $(date +%T)"
python - "$A" <<'PY'
import sys
from tinyperf_chain.bundle import lock_bundle, load_bundle_json, verify_lock
from tinyperf_chain.config import load_chain_config
A = sys.argv[1]
cfg = load_chain_config("configs/chain/smoke.yaml", [f"paths.root={A}", f"model.tokenizer={A}/tokenizer.json"])
b = lock_bundle(f"{A}/B4.json", f"{A}/B4.lock.json", cfg)
verify_lock(load_bundle_json(f"{A}/B4.lock.json"), cfg)
print("locked", b["bundle_sha256"][:16])
PY
echo "== smoke run complete  $(date +%T)"
