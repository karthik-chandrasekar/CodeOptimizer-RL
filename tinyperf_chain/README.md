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
| 10 data factory | `data/build_tasks.py`, `data/collect.py` (rounds, on-policy refresh, private store), `data/mutations.py` (bug families), `data/project.py` (curriculum, mechanism balance, 70/30 replay), `ledger.py` |
| 11 training | `train/branch.py`, `train/sft.py`, `train/rl.py` |
| 12 controller | `controller.py`, `budget.py` |
| 13 verifier, timing | `adapters/verifier.py`, `sandbox/{worker,runner,codec,timing}.py` |
| 14 inference | `serve/engine.py`, `serve/cache.py`, `serve/scheduler.py`, `serve/bench.py` |
| 15 evaluation | `eval/run.py`, `eval/aggregate.py` (eq. 9, cost frontier, ablation outcome change, seed replicates), `eval/roles.py` (m1/m2/m2m3/m4/m5), `eval/promote.py` (development promotion gate), `legacy.py` (B0, same cost accounting) |
| 16-17 bundles, config, runbook | `bundle.py` (release manifest), `config.py`, `configs/chain/` (`report_17_1.yaml` is section 17.1 verbatim), `scripts/chain/smoke.sh` |
| Appendix A tests | `tests/chain/` (T01-T25, `test_chain_hardening.py` for the audit fixes, plus integration tests) |

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

# Stage 4 - learn from the chain's own states (one role at a time, partners frozen).  Each refresh is a collection
# *round*; derived records inherit the round of the state they came from, and the projection gives the newest round
# 70 % of the sampling mass (data.replay_recent_frac) with earlier rounds as replay.
python -m tinyperf_chain.data.collect --config $C --split train --stages onpolicy --bundle configs/chain/bundles/B4_specialists.json   # round k
python -m tinyperf_chain.data.collect --config $C --split train --stages plans targets probes repairs gate --onpolicy-states \
    --downstream configs/chain/bundles/B4_specialists.json          # M2 and M1 relabelled on the states the chain visited
python -m tinyperf_chain.data.project --config $C --splits train dev [--stages 1 2 3] [--replay-recent 0.7]
#   retrain the affected role (its dev metrics are reported per round: dev_forgetting_earlier_rounds), publish a candidate
#   bundle, re-run the development comparison, then gate the promotion:
python -m tinyperf_chain.eval.aggregate --runs artifacts/chain/runs/dev/episodes.jsonl --paired --reference B4
python -m tinyperf_chain.eval.promote --report artifacts/chain/runs/dev/report.json --candidate B4_r2 --incumbent B4 \
    --role-report roles_B4_r2.json --role-baseline roles_B4.json

# Stage 5 - selected RL (writer, then repairer)
python -m tinyperf_chain.train.rl --config configs/chain/rl.yaml --role M3 \
    --init artifacts/chain/checkpoints/M3_sft/best.pt --out artifacts/chain/checkpoints/M3_rl

# Stage 6 - development comparison, diagnostics, calibration, lock, confirmation
python -m tinyperf_chain.eval.roles --config $C --bundle configs/chain/bundles/B4_specialists.json --split dev \
    --checks m1 m2 m2m3 m4 m5 [--holdout-families cmp_swap container_swap]
python -m tinyperf_chain.serve.bench checks --config $C --suite dev_files --n 100 \
    --engine-checkpoint artifacts/chain/checkpoints/M3_sft/best.pt --plan-files 100 --plan-episodes 4 --plan-systems 7   # eq. 10
python -m tinyperf_chain.serve.bench engine --checkpoint artifacts/chain/checkpoints/M3_sft/best.pt --compile --saturate 64
python -m tinyperf_chain.eval.run --config configs/chain/eval_dev.yaml --suite development --calibrate-wall \
    --bundle configs/chain/bundles/B4_specialists.json --output artifacts/chain/runs/calibration
python -m tinyperf_chain.eval.run --config configs/chain/eval_dev.yaml --suite development --patch-slots 1 2 4 6 \
    --bundle configs/chain/bundles/{B1_shared,B2_generalists,B3_rule_aux,B4_specialists,B5_dense,B6_22m_adapted}.json \
    --legacy-checkpoint $TINYPERF_CKPT --output artifacts/chain/runs/dev
python -m tinyperf_chain.eval.aggregate --runs artifacts/chain/runs/dev/episodes.jsonl --paired --reference B1 \
    --ablation-pairs B4:B3 B4:B1            # + cost frontier, control-failure reasons, useful repair rate, seed replicates
# set budget.search_wall_seconds in eval_confirm.yaml from calibration.json (it is null until then), then lock every bundle:
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
| B1 | `sft --role shared` (uniform role sampling); `sft --role shared --match-tokens-from <the five role runs>` for the aggregate-token-matched control |
| B2 | five `sft --role shared` branches with different `training.seed`, fixed role assignment (`B2_generalists.json`) |
| B3 | learned M3 (and M3 answering the M5 prompt) with rule M1/M2/M4 (`B3_rule_aux.json`) |
| B4 | five role specialists (`B4_specialists.json`); bundles trained with other seeds carry `"replicate_of": "B4", "training_seed": k` so `eval.aggregate` reports across-seed intervals separately |
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
* **Sandbox.** One fresh interpreter per request and **one fresh module namespace per independent trial** (every
  input in a correctness run, every repetition of a timing block): no module-level cache, counter or memo table
  survives between trials.  The static patch policy closes the same route from the other side: module-level objects
  are read-only for a patch, also through local aliases, interior elements, shallow copies and unbound-method calls
  (`dict.update(CACHE, ...)`).  Legacy AST policy, restricted builtins, import guard and rlimits inside; a scrubbed
  environment (fixed `PYTHONHASHSEED`; the audit uses a different seed), a private working directory and
  process-group kill outside.  A harness running as root runs workers as uid 65534 unless `sandbox.run_as_root` is
  set; `sandbox.isolation: unshare|bwrap` adds OS namespaces (bwrap binds only the interpreter, `/usr`, the loader
  files and the worker sources read-only - no home, no repository), probed and never silently downgraded.  A worker
  that dies without a reply triggers one canary launch: if the canary fails too the episode is a *failed run*
  (infrastructure), never a candidate label.  `SandboxRunner.selftest()` reports what a worker can actually do
  under its limits.  This is a research harness, not a hostile multi-tenant boundary.
* **Timing.** Paired blocks with randomised order, fresh namespaces and inputs per repetition, GC off inside the
  timed region, repetitions calibrated on the *reference of the pair* (current best during search, the original in
  the audit) so blocks stay in the 50-200 ms range after accepted speedups; every result records CPU model,
  affinity, whether `taskset` was applied, warm-up, timer and worker version.  Acceptance by the bootstrap lower
  bound of the median paired log ratio (>= log 1.03), uncertain -> rejected; `legacy_ratio` reproduces the old
  rule.  `verification.confirmation_extra_blocks` is the separately registered single extra-confirmation arm.
  Cache keys for reference records, calibration and profiles include the interpreter, worker and codec versions,
  isolation mode, uid, limits, comparison policy and host CPU, so nothing measured on another machine is reused.
* **Budgets** are reserved before every call (prompt + maximum output), unused tokens released, patch slots never
  refunded. A context overflow is a predeclared, charged *unsupported* outcome (M1 stops, M2 blocks the target,
  M3/M5 abandon, M4 adds no probes).
* **Repairs** follow syntax errors and public (smoke/probe) mismatches by default; private failures give
  category-only feedback and are repairable only if `private_mismatch` is listed.
* **Data labels.** A reference or teacher rewrite is labelled with a mechanism only when it structurally *realises*
  it (`registry.plan_adherence`); otherwise it is an unlabelled rewrite (kept as evidence, never a plan-conditioned
  demonstration).  NO_PLAN distinguishes "nothing applies" and "every searched plan failed" from "could not be
  instantiated" (not evidence).  M1 labels come from one downstream bundle (`data.label_bundle`), with attempt-history
  variants so the online ledger is seen in training.  Mutation families live in `private/trials_private.jsonl` only.
* **RL reward** uses the same acceptance protocol minus learned probes (scope, smoke, fixed-edge probes, mandatory
  checks, paired timing); a reversion to the current program pays the slot cost only; behaviour log-probs are
  checked against the training forward before the first update; `last.pt` resumes a run exactly.
* **Checkpoint selection** for M3/M5 is executable: greedy development outputs go through the acceptance protocol and
  the accepted-speedup rate decides, with dev loss as tie breaker; metrics are reported per collection round.
* **Engine.** Static KV cache + last-position logits, exact parity with the legacy generator (tested, including early
  EOS and finished rows); every request keeps its own deadline in a batch; wasted decode steps on finished rows and
  peak cache bytes are counted; the serving engine runs eager - `serve/bench.py engine --compile` measures the
  compiled path independently (swapped in only for that variant) before anyone enables it.
* **B0** runs under exactly the same accounting as B1-B6 (model latency, prepare/parse time, verifier CPU, per-event
  budgets), so the quality-versus-cost frontier is like for like.
* **Legacy writer (milestone M0)** maps the old model's output with one explicit, logged policy:
  keep the selected function's `def`, `<STOP>` -> ABANDON.

## Left for local calibration

Checkpoint / tokenizer paths and their audit, installed environment, code commit, supported input contracts,
independent cluster counts (the 208-seed library gives far fewer than 300/60/100 clusters, so the manifest is
labelled a pilot), final workload mix, `budget.search_wall_seconds`, worker counts, reserved timing cores and
device batching settings.
