# TinyPerf-RL — supplementary code

Anonymized for double-blind review.

TinyPerf-RL trains very small, code-only language models (22M, 2M and 0.2M parameters) to make Python
files faster from execution feedback. A model sees the current best version of a file plus a short runtime
state, and replies with an edit (one or more rewritten functions) or STOP. A sandbox runs every edit against
hidden tests and times it; only edits that keep behaviour identical and make the file faster are kept.

This archive contains everything used for the experiments in the paper: the sandbox and environment, data
generation, model, training (pretraining, agent SFT, GRPO, rejection fine-tuning, on-policy self-distillation,
sequence-level distillation), evaluation and analysis, and the off-the-shelf baseline harness. Model
checkpoints, generated datasets and the pretraining corpus are not included because of their size; every one
of them is produced by the scripts below.

## Contents

```
tinyperf/
  env/      sandbox (fresh restricted subprocess per candidate, differential correctness, paired timing),
            text protocol, single-function and multi-function file environments, reward
  data/     seed functions, degradation operators, inverse rewrites, dataset and file-task builders,
            function mining from a code corpus
  model/    byte-level BPE tokenizer, decoder-only transformer (RoPE, SwiGLU, tied embeddings), batched generation
  train/    pretraining, trajectory generation, agent SFT, multi-turn rollouts, GRPO (+ on-policy
            self-distillation), RFT and distillation data builders, hint data
  eval/     evaluation of our checkpoints, evaluation of Hugging Face models (evaluate_hf.py),
            metrics, budget-matched paired comparisons with function-clustered bootstrap intervals
configs/    YAML configs for every stage; model sizes in configs/model/ (tiny20m = 22M, student2m, student200k)
scripts/    one script per experiment (table below) and the analysis tools
tests/      pytest suite (sandbox, environment, data, model, GRPO, metrics, distillation, baseline harness)
DESIGN_NOTES.md   implementation reference: design decisions, sandbox rules, GRPO details, metric definitions
```

## Setup

Python 3.10+ and PyTorch 2.2+ (tested with PyTorch 2.8, CUDA 12.8).

```bash
pip install -e ".[train,dev]"
pytest                          # full test suite, CPU only, under a minute
bash scripts/smoke_test.sh      # end-to-end run on a toy dataset and toy model, CPU, about 5-10 minutes
```

The experiments ran on one NVIDIA H100 80GB with 192 vCPUs. Training and evaluation time is dominated by
the sandbox (every edit is executed and timed in a fresh interpreter), so CPU cores matter more than GPU
speed; `env.max_parallel_envs` sets how many episodes run their sandbox steps in parallel. Timing ratios are
noisy on a heavily loaded machine; the evaluations use paired, interleaved timing with repeated measurements.

Every option can be overridden on the command line (`key=value`, e.g. `env.horizon=6
model=configs/model/student2m.yaml`); unknown keys are rejected. Every script reads its paths and settings
from environment variables with the defaults used in the paper, and skips stages whose outputs already exist.

## Data

* **Hand-written test set** (300 multi-function files over 54 held-out function clusters): built from the
  seed library in `tinyperf/data/seeds.py` and `seeds_extra.py` by `scripts/run_round2.sh` and
  `scripts/run_files.sh`. Held-out functions are chosen by hashing function names, so they never appear in
  training.
* **Real-code test set** (200 files over 74 function clusters) and the mined training pool: functions mined
  from a Python code corpus by `tinyperf/data/mine.py` (static filters, then execution-confirmed input types),
  turned into verified tasks. The scripts expect the corpus at `data/corpus/python.jsonl`, one JSON object per
  source file with a `content` field (`CORPUS=` to change it). The corpus is not redistributed here.
* **Pretraining** uses the same corpus plus the training tasks (`configs/pretrain.yaml`).

## Reproducing the experiments

Run the scripts in this order; each one reads the outputs of the earlier ones from `artifacts/` (the default
paths in each script). Comments inside the scripts refer to experiments by an internal number, listed here.

| # | Experiment | Script | Main outputs |
|---|---|---|---|
| — | Data, tokenizer, pretraining of the 22M model, single-function SFT/GRPO baseline pipeline | `scripts/run_pipeline.sh` | `artifacts/data/`, `artifacts/tokenizer/`, `artifacts/pretrain*/` |
| 1–2 | Single functions; disjoint SFT/RL functions with a difficulty filter | `scripts/run_round2.sh` | `artifacts/data_r2/`, `artifacts/r2/` |
| 3 | Multi-function files: zero-shot transfer, GRPO on file tasks, budget-matched comparison, stopping control | `scripts/run_files.sh` | `artifacts/files/` |
| 4 | Profiler feedback with a fixed edit budget (none / scalar / profile arms) | `scripts/run_profile.sh` | `artifacts/files_profile/` |
| 5 | RL on functions mined from real code | `scripts/run_mined.sh` | `artifacts/data_mined/`, `artifacts/files_mined/` |
| 6 | SFT on mined code, then GRPO (the 22M SFT → RL model) | `scripts/run_mined_sft.sh` | `artifacts/mined_sft/` |
| 7 | Rejection fine-tuning on the RL policy | `scripts/run_rft.sh` | `artifacts/rft/` |
| 8 | Hint warm-up and on-policy self-distillation (the 22M teacher; the paper uses the step-100 checkpoint of the localization arm) | `POLICY=artifacts/rft/sft/final.pt A=artifacts/opsd_rft bash scripts/run_opsd.sh` | `artifacts/opsd_rft/` |
| 9 | Distillation of the teacher into the 2M and 0.2M students (pretraining, sequence-level KD, then verifier-only GRPO); student-vs-teacher comparisons | `scripts/run_distill.sh`, `scripts/compare_to_teacher.py` | `artifacts/distill/` |
| 9 | RFT, hint warm-up and OPSD on the students | `scripts/run_student_followups.sh` | `artifacts/distill/followup/` |
| 9 | Best of k from n verified attempts | `scripts/best_of_k.py` (on evaluations run with `eval.n_samples=16`) | tables |
| 10 | Off-the-shelf baselines in the same sandbox and episodes | `scripts/run_hf_baselines.sh` (and `scripts/run_qwen_baseline.sh`) | `artifacts/hf_baselines/` |

Model-size ablations of the base pipeline: `scripts/run_size_ablation.sh`.

## Evaluation

* All test numbers use six actions per episode (`env.horizon=6`), 4 sampled attempts per file at temperature
  0.6 (`eval.n_samples=4`), and the held-out test files above.
* **Success**: the best kept version is at least 1.5× faster. **Complete fix**: at least 80% of the speedup of
  the known fast reference version. **Speedup**: geometric mean over attempts.
* `python -m tinyperf.eval.budget` builds the budget-matched tables (results after 1–6 actions) with paired
  differences and 95% bootstrap intervals that resample function clusters, so related files are not counted
  as independent evidence. `scripts/analyze_files.py` reports localization (whether an edit targets a slow
  function, or the hottest one) and correct-edit rates.
* `scripts/best_of_k.py` computes best-of-k exactly from n saved attempts (the unbiased pass@k estimator for
  rates, and the order-statistic expectation for speedups).

## Baselines

`tinyperf/eval/evaluate_hf.py` runs any Hugging Face chat model through the same episodes as our models:
the same sandbox, hidden tests, timing, horizon, test files and sampling temperature, with no re-prompting
(a STOP ends the episode). Each turn the model sees exactly the observation our models see, inside its chat
template, with a system prompt describing the protocol and one worked example from a training file. Replies
are read leniently (protocol tags, a fenced or bare code block), and how each reply was read is recorded.

The baselines in the paper, as `scripts/run_hf_baselines.sh` model specs (`name|model|prompt|batch|max new tokens|backend`):

```bash
MODELS="qwen08|Qwen/Qwen3.5-0.8B|chat|32|1024|hf \
        dscoder13|deepseek-ai/deepseek-coder-1.3b-instruct|chat|16|1024|hf \
        qc05|Qwen/Qwen2.5-Coder-0.5B-Instruct|chat|32|1024|hf \
        qc15|Qwen/Qwen2.5-Coder-1.5B-Instruct|chat|16|1024|hf \
        qc3|Qwen/Qwen2.5-Coder-3B-Instruct|chat|32|1024|hf \
        q3c30|Qwen/Qwen3-Coder-30B-A3B-Instruct|chat|64|1024|vllm" \
  bash scripts/run_hf_baselines.sh
```

The script evaluates our four models in the same session (`SKIP_OURS=1` reuses existing results) and builds
the paired tables. Notes:

* Hugging Face models run in a separate virtual environment with a recent `transformers` (created by the
  script; `VENV=` sets its location). Qwen3.5 additionally benefits from `flash-linear-attention`.
* The mixture-of-experts baseline runs on vLLM, in its own environment (`VLLM_VENV=`); set
  `VLLM_WORKER_MULTIPROC_METHOD=spawn`, make the `ninja` build tool available, and give vLLM most of the GPU
  (`VLLM_GPU_UTIL=0.92`) with nothing else running on it.
* Hugging Face weights are cached under `HF_HOME`.

## Notes

* `tinyperf/train/sft_data.py` also contains an optional LLM-API trajectory proposer (`sft_data.methods`
  including `teacher`). It was not used in any experiment; all demonstrations come from the programmatic and
  search proposers and are verified by replaying them through the sandbox.
* Seeds are set in the configs; the sandbox's timing introduces run-to-run variation of a few points, which is
  why every comparison in the paper is made between models evaluated in the same session.
# TinyPerf-CodeOptimizer-
