# TinyPerf-Chain

Five ~2M role specialists for verified Python optimization, built as an extension package on top of
`tinyperf` (TinyPerf-RL). The legacy package is untouched except for one backward-compatible inference option
(`TinyPerfLM.forward(..., last_logits_only=False)`).

```
M1 target selector   TARGET fNN | NO_TARGET
M2 planner           PLAN <mechanism>; src=<name>; loop=nNN; preserve=... | NO_PLAN
M3 patch writer      <EDIT> one replacement function </EDIT> | ABANDON
M4 probe selector    PROBES pNN,pNN,pNN | PROBES none
M5 repairer          same payload as M3
```

The harness owns everything that decides correctness, speed, budgets, input validity and acceptance. This code is a
specification implementation: it has been run end to end on toy scale (CPU smoke run, tiny untrained models) and
its acceptance tests pass, but **no training study has been run and no result is claimed**. Every number in
`configs/chain/*.yaml` is a proposed default.

## Layout

| Spec section | Module |
|---|---|
| 3 task contracts, information boundaries | `contracts.py`, `tasks.py` (`PublicTask` vs `PrivateSuite`) |
| 3.3 split before mutation | `data/split.py` (union-find clusters, salted assignment, overlap checks) |
| 4 wire formats, context budgets | `schemas.py` (strict parsers, escaping, envelopes), `context.py` (views, preflight) |
| 5-9 roles | `registry.py` (8 mechanisms, anchors, sites, preconditions, proposers; 14 probe recipes), `adapters/` |
| 10 data factory | `data/build_tasks.py`, `data/collect.py`, `data/project.py`, `ledger.py` |
| 11 training | `train/branch.py`, `train/sft.py`, `train/rl.py` |
| 12 controller | `controller.py`, `budget.py` |
| 13 verifier, timing | `adapters/verifier.py`, `sandbox/{worker,runner,codec,timing}.py` |
| 14 inference | `serve/engine.py`, `serve/cache.py`, `serve/scheduler.py`, `serve/bench.py` |
| 15 evaluation | `eval/run.py`, `eval/aggregate.py`, `eval/roles.py`, `legacy.py` (B0) |
| 16-17 bundles, config, runbook | `bundle.py`, `config.py`, `configs/chain/`, `scripts/chain/smoke.sh` |
| Appendix A tests | `tests/chain/` (T01-T25 plus integration tests) |

## Quick start

```bash
pip install -e ".[train,dev]"
pytest tests/chain                 # ~1 minute on CPU (real sandbox subprocesses, tiny models)
bash scripts/chain/smoke.sh        # end-to-end toy run of every stage, ~3 minutes on 2 CPUs
```

## Runbook

```bash
export PYTHONPATH=. TINYPERF_CKPT=artifacts/distill/s2m/rl/final.pt TINYPERF_TOKENIZER=artifacts/tokenizer/tokenizer.json
C=configs/chain/pilot.yaml

# Stage 0 - identity and splits
python -m tinyperf_chain.train.branch --config $C --out artifacts/chain/checkpoints/initial --dense-report
python -m tinyperf_chain.data.split --config $C --out artifacts/chain/split_manifest.json --fail-on-overlap \
    [--mined artifacts/mined/mined_seeds.jsonl] [--relatives relatives.json]
python -m tinyperf_chain.data.build_tasks --config $C --manifest artifacts/chain/split_manifest.json
python -m tinyperf_chain.data.split --check artifacts/chain/split_manifest.json --tasks-root artifacts/chain/tasks --fail-on-overlap

# Stage 1 - first runnable milestone: rule targeting/planning + the existing writer + verifier + trace recorder
python -m tinyperf_chain.eval.run --config $C --suite development --limit 20 --episodes-per-file 1 \
    --bundle configs/chain/bundles/R_rules.json configs/chain/bundles/M0_milestone.json \
    --legacy-checkpoint $TINYPERF_CKPT --output artifacts/chain/runs/milestone
python -m tinyperf_chain.eval.aggregate --runs artifacts/chain/runs/milestone/episodes.jsonl --paired --reference R
python -m tinyperf_chain.serve.bench checks --config $C --suite dev_files --n 100     # cost forecast (eq. 10)

# Stage 2 - data (train and dev only; the collector refuses the confirmation split)
for S in train dev; do
  python -m tinyperf_chain.data.collect --config $C --split $S --stages plans probes repairs targets gate
done
python -m tinyperf_chain.data.project --config $C --splits train dev --manifest artifacts/chain/split_manifest.json --review 40
#   -> read artifacts/chain/datasets/review_sample.jsonl by hand before scaling up

# Stage 3 - SFT: M2/M3 first, then M1, then M4/M5 (+ controls)
for R in M2 M3 M1 M4 M5; do
  python -m tinyperf_chain.train.sft --config configs/chain/sft.yaml --role $R \
      --init artifacts/chain/checkpoints/initial/$R.pt --out artifacts/chain/checkpoints/${R}_sft
done
python -m tinyperf_chain.train.sft --config configs/chain/sft.yaml --role shared --out artifacts/chain/checkpoints/shared_sft   # B1
for k in 1 2 3 4 5; do   # B2: five generalist branches
  python -m tinyperf_chain.train.sft --config configs/chain/sft.yaml --role shared training.seed=$k \
      --out artifacts/chain/checkpoints/shared_sft_seed$k
done

# Stage 4 - learn from the chain's own states (one role at a time, partners frozen)
python -m tinyperf_chain.data.collect --config $C --split train --stages onpolicy --bundle configs/chain/bundles/B4_specialists.json
python -m tinyperf_chain.data.collect --config $C --split train --stages probes repairs gate
python -m tinyperf_chain.data.collect --config $C --split train --stages targets --onpolicy-states \
    --downstream configs/chain/bundles/B4_specialists.json
python -m tinyperf_chain.data.project --config $C --splits train dev
#   retrain the affected role, publish a new bundle, re-run the development comparison

# Stage 5 - selected RL (writer, then repairer)
python -m tinyperf_chain.train.rl --config configs/chain/rl.yaml --role M3 \
    --init artifacts/chain/checkpoints/M3_sft/best.pt --out artifacts/chain/checkpoints/M3_rl

# Stage 6 - development comparison, diagnostics, calibration, lock, confirmation
python -m tinyperf_chain.eval.roles --config $C --bundle configs/chain/bundles/B4_specialists.json --split dev
python -m tinyperf_chain.eval.run --config configs/chain/eval_dev.yaml --suite development --calibrate-wall \
    --bundle configs/chain/bundles/B4_specialists.json --output artifacts/chain/runs/calibration
python -m tinyperf_chain.eval.run --config configs/chain/eval_dev.yaml --suite development --patch-slots 1 2 4 6 \
    --bundle configs/chain/bundles/{B1_shared,B2_generalists,B3_rule_aux,B4_specialists,B5_dense,B6_22m_adapted}.json \
    --legacy-checkpoint $TINYPERF_CKPT --output artifacts/chain/runs/dev
python -m tinyperf_chain.eval.aggregate --runs artifacts/chain/runs/dev/episodes.jsonl --paired --reference B1
# set budget.search_wall_seconds in eval_confirm.yaml from calibration.json, then lock every bundle:
python -c "from tinyperf_chain.bundle import lock_bundle; from tinyperf_chain.config import load_chain_config as L; \
lock_bundle('configs/chain/bundles/B4_specialists.json', 'B4.lock.json', L('configs/chain/eval_confirm.yaml'))"
python -m tinyperf_chain.eval.run --config configs/chain/eval_confirm.yaml --suite confirmation \
    --bundle B1.lock.json B4.lock.json ... --output artifacts/chain/confirm
python -m tinyperf_chain.eval.aggregate --runs artifacts/chain/confirm/episodes.jsonl --paired --reference B1
```

### Baselines

| ID | How it is produced |
|---|---|
| B0 | `--legacy-checkpoint` in `eval.run`: the untouched optimizer, legacy whole-file protocol, chain budget/verifier/audit |
| B1 | `sft --role shared` (uniform role sampling; `training.shared_sampling=token_matched` for the token-matched control) |
| B2 | five `sft --role shared` branches with different `training.seed`, fixed role assignment (`B2_generalists.json`) |
| B3 | learned M3 (and M3 answering the M5 prompt) with rule M1/M2/M4 (`B3_rule_aux.json`) |
| B4 | five role specialists (`B4_specialists.json`) |
| B5 | `configs/model/dense10m.yaml` (10,490,112 parameters) pretrained/distilled with the legacy `tinyperf.train.pretrain` / `seqkd_data` / `sft` scripts, then `sft --role shared` with `model.expect_*` overridden; its training history is reported separately |
| B6 | the existing 22M checkpoint adapted with `sft --role shared` (`model.expect_*` overridden) |

## Decisions worth knowing

* **Contracts.** Tasks built from the seed library get *generator-derived* contracts: the dataset constructor draws
  from its own generator (all suites plus extra sizes) at construction time and freezes the declaration
  (`origin: generator_derived_v1`); declared contracts are supported unchanged. Floats are deferred (excluded and
  counted in `tasks/eligibility.json`).
* **Semantics** are always judged against the immutable original: return value with exact types, exception type,
  mutation of every argument (also after an exception) and aliasing structure (`[[]]*n` vs fresh lists, returning
  the argument itself vs a copy). The child encodes values as tagged JSON; comparison happens in the parent. For the
  large timing-workload outputs the child returns SHA-256 digests of the encodings (defence in depth, documented).
* **Sandbox.** One fresh interpreter per request with the legacy AST policy, restricted builtins, import guard and
  rlimits, a scrubbed environment (fixed `PYTHONHASHSEED`; the audit uses a different seed), a private working
  directory and process-group kill. `sandbox.isolation: unshare|bwrap` adds OS namespaces when the host supports
  them (probed; never silently downgraded); `drop_to_uid` drops root. `SandboxRunner.selftest()` reports what a
  worker can actually do. This is a research harness, not a hostile multi-tenant boundary.
* **Timing.** Paired blocks with randomised order, fresh namespaces, inputs copied outside the timed region, GC off
  inside it, repetitions calibrated on the original and fixed for both programs; acceptance by the bootstrap lower
  bound of the median paired log ratio (>= log 1.03), uncertain -> rejected; `legacy_ratio` reproduces the old rule.
  `verification.confirmation_extra_blocks` is the separately registered single extra-confirmation arm.
* **Budgets** are reserved before every call (prompt + maximum output), unused tokens released, patch slots never
  refunded. A context overflow is a predeclared, charged *unsupported* outcome (M1 stops, M2 blocks the target,
  M3/M5 abandon, M4 adds no probes).
* **Repairs** follow syntax errors and public (smoke/probe) mismatches by default; private failures give
  category-only feedback and are repairable only if `private_mismatch` is listed.
* **RL reward** uses the same acceptance protocol minus learned probes (scope, smoke, mandatory checks, paired timing);
  behaviour log-probs are checked against the training forward before the first update.
* **Engine.** Static KV cache + last-position logits, exact parity with the legacy generator (tested); the serving
  engine runs eager - `serve/bench.py engine --compile` measures compiled paths before anyone enables them.
* **Legacy writer (milestone M0)** maps the old model's output with one explicit, logged policy:
  keep the selected function's `def`, `<STOP>` -> ABANDON.

## Left for local calibration

Checkpoint / tokenizer paths and their audit, installed environment, code commit, supported input contracts,
independent cluster counts (the 208-seed library gives far fewer than 300/60/100 clusters, so the manifest is
labelled a pilot), final workload mix, `budget.search_wall_seconds`, worker counts, reserved timing cores and
device batching settings.
