"""Typed configuration for TinyPerf-RL.

Every stage (dataset build, tokenizer, pretrain, SFT, GRPO, eval) reads a
YAML file into the dataclasses below.  Any field can be overridden from the
command line with ``section.field=value`` tokens, e.g.::

    python -m tinyperf.train.grpo --config configs/grpo.yaml grpo.group_size=16 env.horizon=4
"""
from __future__ import annotations

import dataclasses
import json
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# --------------------------------------------------------------------------- #
# Leaf configs
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    name: str = "tinyperf-20m"
    vocab_size: int = 8192
    n_layer: int = 8
    n_embd: int = 384
    n_head: int = 6
    mlp_dim: int = 1536
    max_seq_len: int = 2048
    dropout: float = 0.0
    tie_embeddings: bool = True
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5


@dataclass
class TokenizerConfig:
    path: str = "artifacts/tokenizer/tokenizer.json"
    vocab_size: int = 8192
    corpus_dirs: List[str] = field(default_factory=list)   # dirs of .py files
    corpus_jsonl: List[str] = field(default_factory=list)  # jsonl with a "content" field
    include_stdlib: bool = True                            # use the local CPython stdlib as free corpus
    include_tasks: Optional[str] = "artifacts/data/train.jsonl"
    max_files: int = 20000


@dataclass
class LimitsConfig:
    """Resource limits applied inside the sandbox worker (best effort)."""
    memory_mb: int = 512
    cpu_seconds: int = 20
    call_timeout_s: float = 2.0       # per correctness call
    workload_timeout_s: float = 8.0   # per timing repeat over the whole perf workload
    process_timeout_s: float = 60.0   # hard subprocess timeout
    max_output_chars: int = 20000


@dataclass
class TimingConfig:
    warmup: int = 1
    repeats: int = 7                # timing repeats; median is used
    paired: bool = True             # interleave baseline/candidate in the same process
    pin_cpu: Optional[int] = None   # taskset core id (Linux only)
    max_candidate_ratio: float = 20.0  # candidates slower than this are cut short


@dataclass
class EnvConfig:
    horizon: int = 6                 # H: max EDIT actions per episode
    improvement_delta: float = 0.03  # δ: candidate must beat best by this fraction
    feedback: str = "full"           # "full" | "none" (feedback ablation) | "profile" (full + per-function runtime shares, file tasks)
    hint: str = ""                   # "" | "loc": show the privileged localization hint to the policy (diagnostics only)
    float_rel_tol: float = 1e-9
    float_abs_tol: float = 1e-12
    allowed_modules: List[str] = field(default_factory=lambda: [
        "math", "itertools", "functools", "collections", "heapq", "bisect",
        "operator", "string", "re", "array", "statistics", "fractions", "decimal",
    ])
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    max_parallel_envs: int = 8       # concurrent sandbox subprocesses


@dataclass
class RewardConfig:
    step_penalty: float = 0.015      # λ
    invalid_penalty: float = 0.10    # μ: per incorrect / malformed / crashing edit
    clip_log_speedup: float = 4.0    # clip log(T0/T_best)
    regression_penalty: float = 0.0  # extra penalty if the returned code is slower than C0 (cannot happen with rollback)


@dataclass
class DataConfig:
    out_dir: str = "artifacts/data"
    mined_seeds: str = ""            # jsonl from tinyperf.data.mine ("" = none)
    max_mined: int = 0               # cap on mined functions used (0 = all)
    builtin_seeds: bool = True       # include the hand-written seed library
    seed_workers: int = 1            # seeds expanded in parallel (each seed gets its own RNG when > 1)
    n_train: int = 8000
    n_val: int = 400
    n_test_iid: int = 400
    n_test_compositional: int = 300
    heldout_families: List[str] = field(default_factory=lambda: ["dict_linear_search", "materialize"])
    n_test_heldout: int = 300
    heldout_seed_frac: float = 0.15  # fraction of *seed functions* reserved for test_seeds (never seen in any form)
    n_test_seeds: int = 300
    min_slowdown: float = 1.30       # C_slow must be at least this much slower than C_fast (margin above timing noise)
    max_degradations: int = 3        # deepest chain explored is max_degradations + 2 (compositional split)
    train_depths: List[int] = field(default_factory=lambda: [1, 2])  # chain depths allowed in train/val/test_iid
    max_nodes_per_seed: int = 60
    max_source_chars: int = 2200     # ≈ 700 tokens
    rename_prob: float = 0.8         # alpha-rename identifiers for variety
    perf_scale: int = 3000           # size of benchmark workloads
    seed: int = 0
    natural_dir: Optional[str] = None  # optional dir of human-written slow functions (see README)


@dataclass
class OptimConfig:
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    warmup_steps: int = 200
    max_steps: int = 20000
    grad_clip: float = 1.0
    batch_size: int = 64
    grad_accum: int = 1


@dataclass
class PretrainConfig:
    seq_len: int = 1024
    corpus_dirs: List[str] = field(default_factory=list)
    corpus_jsonl: List[str] = field(default_factory=list)
    include_stdlib: bool = True
    include_tasks: Optional[str] = "artifacts/data/train.jsonl"
    out_dir: str = "artifacts/pretrain"
    eval_every: int = 500
    save_every: int = 1000
    log_every: int = 20
    optim: OptimConfig = field(default_factory=OptimConfig)


@dataclass
class SFTDataConfig:
    tasks: str = "artifacts/data/train.jsonl"
    out_path: str = "artifacts/sft/trajectories.jsonl"
    methods: List[str] = field(default_factory=lambda: ["programmatic", "search"])
    n_tasks: int = 4000
    programmatic_per_task: int = 2
    search_beam: int = 3
    search_candidates: int = 6
    failure_injection_prob: float = 0.35  # insert an incorrect edit + recovery
    teacher_model: str = "claude-sonnet-4-6"
    teacher_candidates: int = 6
    teacher_max_depth: int = 3
    min_ref_frac: float = 0.0        # keep a trajectory only if its final speedup >= this x the task's reference speedup
                                     # (0 = off; 0.8 drops demonstrations that STOP after a partial fix)


@dataclass
class SFTConfig:
    trajectories: str = "artifacts/sft/trajectories.jsonl"
    init_from: str = "artifacts/pretrain/final.pt"   # "" = train from scratch (no-pretraining ablation)
    out_dir: str = "artifacts/sft"
    seq_len: int = 2048
    epochs: int = 3
    methods: List[str] = field(default_factory=list)  # keep only trajectories from these methods ([] = all)
    first_step_only: bool = False    # SFT-1: one-shot (s_0 -> final fast code) example per trajectory
    mask_failed_actions: bool = True # don't imitate actions the env rejected (incorrect/syntax/...); the *recovery* step after them is kept
    max_trajectories: int = 0        # 0 = all
    val_frac: float = 0.03           # held-out fraction of *tasks* for validation
    eval_every: int = 200
    save_every: int = 500
    log_every: int = 20
    optim: OptimConfig = field(default_factory=lambda: OptimConfig(lr=1e-4, max_steps=6000, batch_size=32))


@dataclass
class GenerationConfig:
    temperature: float = 0.8
    top_p: float = 1.0
    top_k: int = 0
    max_new_tokens: int = 768
    ban_stop: bool = False           # evaluation control: the policy cannot emit <STOP>, so it must use its whole budget


@dataclass
class GRPOConfig:
    tasks: str = "artifacts/data/train.jsonl"
    val_tasks: str = ""               # explicit validation file for periodic eval ("" = derive from tasks path)
    opsd_beta: float = 0.0            # weight of on-policy self-distillation (reverse KL to the hinted self); 0 = off
    opsd_hints: str = "error+loc"     # "error" (sandbox verdicts / negative feedback) | "loc" (privileged localization) | "error+loc"
    opsd_on: str = "all"              # "all" turns or only "problematic" ones (failed, no-gain, premature STOP)
    opsd_max_items: int = 256         # cap on distilled turns per step
    opsd_only: bool = False           # drop the policy-gradient (reward) term: the update is only the OPSD reverse KL
    opsd_teacher: str = "frozen"      # "frozen": a fixed copy of the initial model is the teacher | "self": shared weights
    opsd_max_tokens: int = 0          # distill only the first N tokens of each action (0 = all); ~12 covers STOP vs EDIT + function name
    opsd_skip_truncated: bool = True  # never distill responses that hit the token limit
    init_from: str = "artifacts/sft/final.pt"
    out_dir: str = "artifacts/grpo"
    group_size: int = 8              # G
    tasks_per_step: int = 8          # B: functions per optimizer step  (B*G rollouts)
    ppo_epochs: int = 1
    minibatch_size: int = 16         # (prompt,response) pairs per forward/backward
    clip_eps: float = 0.2
    clip_eps_high: float = 0.28      # DAPO-style asymmetric clipping (set = clip_eps to disable)
    kl_beta: float = 0.0             # KL to reference (SFT) policy; 0 disables
    adv_norm: str = "std"            # "std" (GRPO) | "none" (Dr. GRPO style, mean-centered only)
    loss_agg: str = "token"          # "token" (DAPO) | "sequence"
    max_steps: int = 2000
    log_every: int = 1
    eval_every: int = 50
    save_every: int = 100
    seed: int = 0
    lr: float = 2e-5
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    n_eval_tasks: int = 64


@dataclass
class EvalConfig:
    tasks: str = "artifacts/data/test_iid.jsonl"
    checkpoint: str = "artifacts/grpo/final.pt"
    out_path: str = "artifacts/eval/results.json"
    n_tasks: int = 0                  # 0 = all
    n_samples: int = 1                # rollouts per task (report mean over samples)
    greedy: bool = False
    generation: GenerationConfig = field(default_factory=lambda: GenerationConfig(temperature=0.6))
    save_trajectories: bool = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    data: DataConfig = field(default_factory=DataConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    sft_data: SFTDataConfig = field(default_factory=SFTDataConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    device: str = "auto"
    wandb_project: Optional[str] = None
    run_name: str = "tinyperf"


# --------------------------------------------------------------------------- #
# Loading / overriding
# --------------------------------------------------------------------------- #
def _from_dict(cls, d: Dict[str, Any]):
    kwargs = {}
    hints = typing.get_type_hints(cls)  # resolves the string annotations from `from __future__ import annotations`
    for f in fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        ftype = hints.get(f.name, f.type)
        if is_dataclass(ftype) and isinstance(v, dict):
            kwargs[f.name] = _from_dict(ftype, v)
        else:
            kwargs[f.name] = v
    unknown = set(d) - {f.name for f in fields(cls)}
    if unknown:
        raise KeyError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**kwargs)


def _parse_scalar(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return yaml.safe_load(s)


def apply_overrides(cfg: Any, overrides: List[str]) -> Any:
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override must look like a.b=c, got {ov!r}")
        key, val = ov.split("=", 1)
        obj = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        leaf = parts[-1]
        if not hasattr(obj, leaf):
            raise KeyError(f"Unknown config key {key}")
        setattr(obj, leaf, _parse_scalar(val))
    return cfg


def _set_nested(raw: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    d = raw
    for p in parts[:-1]:
        if not isinstance(d.get(p), dict):
            d[p] = {}
        d = d[p]
    d[parts[-1]] = value


def load_config(path: Optional[str] = None, overrides: Optional[List[str]] = None) -> Config:
    """Load a YAML config, apply ``a.b=c`` overrides on the raw dict, then build the typed Config.

    ``model: configs/model/tiny20m.yaml`` (in the file or as the override ``model=...yaml``) is expanded."""
    raw: Dict[str, Any] = {}
    if path:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override must look like a.b=c, got {ov!r}")
        key, val = ov.split("=", 1)
        _set_nested(raw, key, _parse_scalar(val))
    if isinstance(raw.get("model"), str):
        with open(raw["model"]) as f:
            raw["model"] = yaml.safe_load(f)
    return _from_dict(Config, raw)


def to_dict(cfg: Any) -> Dict[str, Any]:
    return dataclasses.asdict(cfg)


def dump_config(cfg: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(to_dict(cfg), f, sort_keys=False)


def parse_cli(argv: Optional[List[str]] = None):
    """Common CLI: ``--config path [key=value ...]``."""
    import argparse
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    args, rest = p.parse_known_args(argv)
    overrides = [r for r in rest if "=" in r]
    return load_config(args.config, overrides)
