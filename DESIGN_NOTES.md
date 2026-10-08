# Design notes (V0 implementation reference)

A tiny (10M–100M parameter) code-only decoder that learns to make Python functions
faster **from execution feedback**, not from descriptions. The policy sees a function and
a compact runtime state, emits `<EDIT>…</EDIT>` or `<STOP>`, and a sandboxed environment
executes, checks behaviour against the original, benchmarks, rolls back, and reports
normalized runtimes. Training: code pretraining → agent SFT on (state → action) pairs →
GRPO on the terminal reward `log(T0/T_best) − λ·N_edits − μ·N_invalid`.

Everything in the design doc's V0 scope is implemented: sandbox, environment, synthetic
dataset with degradation families and held-out splits, tokenizer, model, generation,
pretraining, three trajectory sources with failure injection, agent SFT (agent / one-shot /
no-feedback variants), GRPO (DAPO-style clipping, token/sequence aggregation, optional KL),
evaluation metrics, and the ablation runner (horizon, feedback, model size).

```
pip install -e ".[train,dev]"          # torch, pytest;  add ",teacher" for the Anthropic proposer
bash scripts/smoke_test.sh              # ~3 min CPU end-to-end run on a toy dataset + toy model
pytest                                  # 37 tests (sandbox, env, degradations, rewrites, model, GRPO, metrics)
nohup bash scripts/run_pipeline.sh > logs/pipeline.log 2>&1 &   # full V0 on one GPU
```

## Layout

```
tinyperf/
  common/config.py      typed configs (YAML + `a.b=c` CLI overrides, `model=configs/model/tiny20m.yaml`)
  common/utils.py       logging, seeding, jsonl, MetricLogger (jsonl + optional wandb)
  env/worker.py         sandbox worker: fresh `python -I` subprocess per candidate, static AST checks,
                        restricted builtins, rlimits, differential correctness, paired median timing
  env/executor.py       subprocess driver + thread pool for parallel envs
  env/protocol.py       <CODE>/<STATE>/<EDIT>/<STOP> text protocol, parse/format
  env/env.py            PerfEnv (best tracking with δ, rollback, statuses, recovery), compute_reward
  env/task.py           Task = (C0, hidden X_correct, X_perf, Y, T0) + provenance; jsonl (de)serialisation
  data/seeds.py         47 efficient seed functions with input generators and hand-written slow variants
  data/degradations.py  14 efficient→slow operators (families) + `applicable()`
  data/rewrites.py      10 slow→efficient inverse rewrites (the Method C proposer)
  data/rename.py        consistent α-renaming; bug / syntax mutations for failure injection
  data/build.py         dataset builder: BFS over degradations, sandbox verification, canonical-chain splits
  model/tokenizer.py    ~8K byte-level BPE with protocol tokens as single tokens
  model/transformer.py  TinyPerfLM (RMSNorm, RoPE, SwiGLU, tied embeddings, KV cache, padded masks)
  model/generate.py     batched left-padded sampling with KV cache and per-row stop tokens
  train/pretrain.py     stage 1: code pretraining
  train/sft_data.py     stage 2a: trajectories (programmatic / search / teacher) replayed through the real env
  train/sft.py          stage 2b: agent SFT (loss on action tokens only)
  train/rollout.py      B×G lock-step multi-turn rollouts with parallel sandbox execution
  train/grpo.py         stage 3: GRPO
  eval/metrics.py       all V0 metrics (+ per-family / per-depth breakdowns, bootstrap CIs)
  eval/evaluate.py      evaluate one checkpoint on one split
  eval/ablations.py     comparison tables: checkpoints × splits × horizons × feedback
configs/                data / pretrain / sft_data / sft / grpo / eval + model sizes 10m,20m,50m,100m
scripts/                smoke_test.sh, run_pipeline.sh, run_size_ablation.sh
tests/                  pytest suite
```

## Pipeline

| stage | command | output |
|---|---|---|
| dataset | `python -m tinyperf.data.build --config configs/data.yaml` | `artifacts/data/{train,val,test_iid,test_compositional,test_heldout,test_seeds}.jsonl` |
| tokenizer | `python -m tinyperf.model.tokenizer --config configs/pretrain.yaml` | `artifacts/tokenizer/tokenizer.json` |
| pretrain | `python -m tinyperf.train.pretrain --config configs/pretrain.yaml` | `artifacts/pretrain/final.pt` |
| trajectories | `python -m tinyperf.train.sft_data --config configs/sft_data.yaml` | `artifacts/sft/trajectories.jsonl` |
| agent SFT | `python -m tinyperf.train.sft --config configs/sft.yaml` | `artifacts/sft/final.pt` |
| GRPO | `python -m tinyperf.train.grpo --config configs/grpo.yaml` | `artifacts/grpo/{final,best,last}.pt`, `metrics.jsonl` |
| eval | `python -m tinyperf.eval.evaluate --config configs/eval.yaml eval.tasks=artifacts/data/test_heldout.jsonl` | `artifacts/eval/results.json` |

Every option can be overridden on the command line, e.g.
`python -m tinyperf.train.grpo --config configs/grpo.yaml env.horizon=4 grpo.group_size=16 model=configs/model/tiny50m.yaml`.
Unknown keys are rejected. Trainers resume from `<out_dir>/last.pt` if present.

### Pretraining corpus
`configs/pretrain.yaml` ships with `include_stdlib: true` (the local CPython stdlib, ~30 MB of
Python) plus all dataset sources, which is enough to smoke the pipeline. For the real V0 run,
point `tokenizer.corpus_dirs` / `pretrain.corpus_dirs` at a directory of `.py` files (or
`corpus_jsonl` at jsonl files with a `content` field), e.g. a Python subset of The Stack.

### The core comparison and the ablations
`scripts/run_pipeline.sh` trains and evaluates every row of the design doc's table for one
model size (`SIZE=tiny20m`, default):

* **Base** – pretrained model, no protocol training (it mostly fails to emit valid actions,
  which is the point of the row);
* **SFT-1** – `sft.first_step_only=true`: one-shot `(s_0 → final fast code)`;
* **RL-1** – GRPO from SFT-1 with `env.horizon=1`;
* **Agent-SFT** – all `(s_t → a_t)` pairs;
* **Agent-RL** – GRPO from Agent-SFT with `H=6`;
* **no-feedback controls** – `env.feedback=none` at SFT time re-renders the *same*
  trajectories as the no-feedback environment shows them (`C0` + step counter); at GRPO/eval
  time the env still tracks the best program but never reveals runtimes or candidates.

`tinyperf.eval.ablations` produces the tables:

```
python -m tinyperf.eval.ablations --config configs/eval.yaml \
  --checkpoints base=... sft1=... rl1=... agent_sft=... agent_rl=... \
  --splits test_iid test_compositional test_heldout --feedback full none --n_samples 4 \
  --horizons 1 2 4 6 10 --out artifacts/eval/table.json
```

* `--feedback full none` adds a `feedback_gain` block (full − none) per checkpoint/split/H;
* `--horizons` gives the evaluation-time horizon curve; training-time horizon ablations are
  separate GRPO runs with `env.horizon=H`;
* `scripts/run_size_ablation.sh` runs the whole pipeline for 10M/20M/50M/100M and tabulates.

Metrics reported (`eval/metrics.py`): behaviour preservation, edit-correct rate, success rate
`P(S ≥ 1.05)`, geometric-mean / median speedup, `P(S ≥ 1.25/1.5/2.0)`, regression rate,
slower-edit rate, mean edits, edits-to-best, wasted edits after best, edits per success,
invalid/malformed rates, STOP rate, failure-episode rate, **recovery rate**, fraction of the
known reference speedup achieved, per-family and per-depth breakdowns, task-level bootstrap CIs.

## Design decisions worth knowing

* **Correctness is relative to `C0`'s behaviour** (return value + type, exception type,
  argument mutation, `math.isclose` for floats) on hidden inputs the policy never sees:
  edge cases plus randomized sizes for correctness, larger randomized workloads for timing.
* **Paired timing.** Candidate and `C0` are interleaved in the same fresh process
  (namespace re-executed per repeat so `lru_cache`/memo tables cannot persist), median of
  `timing.repeats`; the reported ratio is `T_cand / T_C0` directly, so `1.00 = original`.
  Candidates slower than `max_candidate_ratio` are cut short.
* **One subprocess per candidate** (`python -I`, no network/filesystem/threads/time APIs,
  import whitelist, no dunder access, rlimits). Ratios are noisy on a busy box: keep
  `env.max_parallel_envs` ≤ physical cores and consider `env.timing.pin_cpu`.
* **Split assignment is by canonical chain** (before α-renaming), so a renamed copy of a
  training chain can never leak into a test split. `test_compositional` = chains deeper than
  `train_depths`; `test_heldout` = any chain containing a held-out family (`dict_linear_search`,
  `materialize` by default); `test_seeds` = every chain of the 15% of seed *functions* held out entirely
  (hashed by name), the function-level generalisation test.
* **Seed library**: 208 seed functions (`data/seeds.py` + `data/seeds_extra.py`). Degradations too slow to
  benchmark at the seed's workload are kept (correct on all hidden inputs) and each task's `C0` workload is
  shrunk until `C0` runs in ≤300 ms, so env steps stay affordable; `min_slowdown` 1.30 keeps only chains
  well above timing noise.
* **Trajectories are replayed through the real environment**, never hand-written, so the
  observations in SFT data carry genuine feedback strings. Failure injection inserts a
  mutated (bug / syntax) or rejected-by-search edit; by default (`sft.mask_failed_actions`)
  the failed edit itself is *not* an imitation target, only the recovery step that observes it.
* **GRPO details.** Old log-probs are recomputed with a no-grad pass before any update
  (so the first minibatch has ratio exactly 1); the policy is defined as
  `softmax(logits / T)` with the sampling temperature; advantages are broadcast to every
  action token of every turn; asymmetric DAPO clipping (`clip_eps` / `clip_eps_high`);
  `loss_agg=token` (DAPO) or `sequence`; `adv_norm=none` gives the Dr. GRPO variant;
  `kl_beta>0` adds a k3 KL to the frozen SFT policy. Zero-variance groups contribute nothing
  and are logged (`zero_var_groups`).
* **Invalid-edit penalty** defaults to `μ=0.10` per bad edit (the doc's `−1` is large next to
  the `log`-speedup scale of ~0.5–3); it is a config knob (`reward.invalid_penalty`).
* **Reward clipping** at `log(T0/T_best) ≤ 4` (55×) stops timing outliers from dominating groups.

## Natural-code evaluation split

Put human-written slow functions in a directory and set `data.natural_dir`. Each file
defines exactly one function plus an input generator:

```python
# natural/flatten_pairs.py
def flatten_pairs(pairs):
    out = []
    for a, b in pairs:
        out = out + [a, b]
    return out

def gen_inputs(rng, n):            # -> tuple of positional arguments for one call
    return ([(rng.randint(0, 9), rng.randint(0, 9)) for _ in range(n)],)
```

The builder generates hidden inputs, records the function's behaviour and writes
`artifacts/data/test_natural.jsonl`.

## Notes for the H100 run

* `configs/grpo.yaml`: B=8 tasks × G=8 rollouts = 64 episodes/step, up to 6 edits each. With
  `env.max_parallel_envs=16` sandbox time (not the 20M model) dominates; each edit costs one
  fresh interpreter plus `repeats×(cand+C0)` workload evaluations (~2 ms baselines by
  construction, so ~50–150 ms per edit). Raise `max_parallel_envs` on a many-core box.
* `python -u` + `nohup` friendly: metrics go to `<out_dir>/metrics.jsonl` (and wandb if
  `wandb_project` is set); checkpoints `last.pt` (resumable), `best.pt` (by val GM speedup),
  `final.pt`.
* The Anthropic teacher proposer (`sft_data.methods: [programmatic, search, teacher]`) needs
  `pip install anthropic` and `ANTHROPIC_API_KEY`; it sees only code (no descriptions) and
  execution selects, so it cannot inject non-executable knowledge.

## Known gaps (V0)

* The inverse-rewrite proposer covers the 14 synthetic families plus common peepholes; it is
  not a general optimizer, which is intended — search trajectories are meant to be verified
  demonstrations, and the RL stage is where the policy has to generalise beyond them.
* Timing noise under heavy parallelism is real; `improvement_delta` (3%) is the guard.
  For publication-grade numbers evaluate with `env.timing.repeats=7`, `n_samples=4`, pinned
  cores, and report the bootstrap CIs the evaluator writes.
