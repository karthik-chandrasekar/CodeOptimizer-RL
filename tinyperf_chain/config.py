"""Strict configuration for TinyPerf-Chain (``configs/chain/*.yaml``).

This is a NEW schema; it is not valid input for the legacy ``tinyperf.common.config`` loader.

* unknown keys anywhere are errors;
* ``base: other.yaml`` merges another file first (path relative to the including file);
* ``${VAR}`` expands environment variables (an unset variable expands to the empty string, which the validators
  reject wherever a value is actually required);
* ``a.b=c`` command-line overrides are applied to the raw mapping before typing;
* :meth:`ChainConfig.validate` rejects incompatible combinations (context budgets above model capacity, final-test
  data reachable by a training collector, mutable bundles during evaluation, stochastic filtering without matching
  log-probabilities, a missing wall-time calibration for confirmatory comparisons, ...).
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from tinyperf.common.config import EnvConfig

ROLE_NAMES = ("M1", "M2", "M3", "M4", "M5")


class ConfigError(ValueError):
    pass


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentCfg:
    name: str = "tinyperf_chain_v1"
    seed: int = 17
    bundle_frozen_per_episode: bool = True
    final_test_access: str = "evaluator_only"


@dataclass
class ModelCfg:
    seed_checkpoint: str = ""
    tokenizer: str = ""
    context_tokens: int = 2048
    add_special_tokens: bool = False
    full_model_finetuning: bool = True
    expect_n_embd: int = 128          # identity checks on the seed student (T01)
    expect_n_layer: int = 4
    expect_n_head: int = 4
    expect_mlp_dim: int = 512
    expect_vocab_size: int = 8192
    expect_tied_embeddings: bool = True


@dataclass
class RoleDecoding:
    max_input: int = 1408
    max_output: int = 640
    mode: str = "greedy"              # "greedy" | "sample"
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0


def _rd(i: int, o: int, mode: str = "greedy", temperature: float = 1.0) -> Any:
    return field(default_factory=lambda: RoleDecoding(i, o, mode, temperature))


@dataclass
class RolesCfg:
    M1: RoleDecoding = _rd(2016, 32)
    M2: RoleDecoding = _rd(1888, 160)
    M3: RoleDecoding = _rd(1408, 640, "sample", 0.6)
    M4: RoleDecoding = _rd(1952, 96)
    M5: RoleDecoding = _rd(1408, 640, "sample", 0.6)

    def get(self, role: str) -> RoleDecoding:
        return getattr(self, role)


@dataclass
class BudgetCfg:
    patch_slots: int = 6
    role_calls: int = 32
    total_input_output_tokens: int = 20000
    extra_probe_calls: int = 18
    max_probes_per_candidate: int = 3
    max_repairs_per_lineage: int = 1
    max_plans_per_function_version: int = 2
    max_profiles: int = 7
    search_wall_seconds: Optional[float] = None   # REQUIRED (calibrated) before confirmatory comparisons
    min_call_seconds: float = 0.0                  # do not start a call with less wall time than this left


@dataclass
class VerificationCfg:
    semantic_reference: str = "immutable_original"
    online_private_feedback: str = "category_only"
    final_audit_feedback: str = "disabled"
    accept_incremental_speedup: float = 1.03
    accept_rule: str = "paired_log_median_ci_lower"   # | "point_estimate" | "legacy_ratio" (reproduction arm)
    legacy_delta: float = 0.03
    timing_bootstrap_draws: int = 1000
    timing_interval: float = 0.95
    success_final_speedup: float = 1.50
    train_timing_blocks: int = 5
    development_timing_blocks: int = 15
    audit_timing_blocks: int = 31
    confirmation_extra_blocks: int = 0               # separately registered arm: extra blocks for marginal cases
    uncertain_timing_policy: str = "reject"
    block_target_ms: float = 50.0
    timing_warmup: int = 1
    fresh_process_per_block: bool = False
    check_timing_workload_outputs: bool = True
    repairable_failures: List[str] = field(default_factory=lambda: ["syntax_error", "smoke_mismatch", "probe_mismatch"])
    smoke_examples_per_function: int = 2
    max_feedback_chars: int = 160
    float_rel_tol: float = 0.0
    float_abs_tol: float = 0.0
    hash_seed: int = 0
    audit_hash_seed: int = 1


@dataclass
class SandboxCfg:
    memory_mb: int = 768
    cpu_seconds: int = 60
    call_timeout_s: float = 2.0
    workload_timeout_s: float = 10.0
    process_timeout_s: float = 180.0
    max_output_bytes: int = 16_000_000
    isolation: str = "rlimits"        # "rlimits" | "unshare" | "bwrap"
    drop_to_uid: Optional[int] = None  # e.g. 65534 when running as root
    python: str = ""                  # interpreter for workers ("" = sys.executable)
    allowed_modules: List[str] = field(default_factory=lambda: list(EnvConfig().allowed_modules))
    timing_cpus: List[int] = field(default_factory=list)       # reserved cores for timing workers (taskset)
    correctness_cpus: List[int] = field(default_factory=list)


@dataclass
class TrainingCfg:
    stage: str = "sft"
    lr: float = 2e-5
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    eps: float = 1e-8
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    min_lr_ratio: float = 0.1
    epochs: int = 3
    microbatch: int = 4
    accumulate: int = 8
    loss: str = "response_mean_per_example"
    gradient_clip: float = 1.0
    checkpoint_every: int = 100
    checkpoint_selection: str = "development_only"
    precision: str = "bf16_autocast"   # | "fp32"
    shared_sampling: str = "uniform_role"   # shared-role control: "uniform_role" | "token_matched"
    token_budget: int = 0              # stop after this many response+prompt tokens (0 = epochs decide)
    tie_mode: str = "sample"           # tied labels: "sample" one per epoch | "sum" of sequence probabilities
    dev_generation_eval_max: int = 64  # greedy-decoded dev prompts per evaluation (role metric)
    eval_every: int = 100
    seed: int = 0
    max_updates: int = 0               # 0 = no cap beyond epochs (smoke tests set a cap)


@dataclass
class RLCfg:
    roles: List[str] = field(default_factory=lambda: ["M3", "M5"])
    states_per_update: int = 2
    group_size: int = 8
    temperature: float = 0.8
    top_k: int = 0
    top_p: float = 1.0
    max_updates: int = 100
    lr: float = 2e-5
    clip_low: float = 0.8
    clip_high: float = 1.28
    gain_cap: float = 4.0
    slot_cost: float = 0.015
    invalid_penalty: float = 0.10
    ratio_tolerance: float = 0.05     # max |log(new/old)| allowed before the first update (numerical tolerance)
    minibatch: int = 8
    save_every: int = 25
    seed: int = 0


@dataclass
class ServingCfg:
    resident_roles: List[str] = field(default_factory=lambda: list(ROLE_NAMES))
    eager_reference_enabled: bool = True
    compile_enabled: bool = False
    static_kv_cache: bool = True
    last_logits_only: bool = True
    per_request_rng: bool = True
    correctness_workers: int = 2
    timing_workers: int = 1
    max_pending_candidates: int = 16
    batch_window_ms: float = 2.0
    max_batch: int = 32
    episode_workers: int = 4
    device: str = "auto"
    length_buckets: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048])


@dataclass
class DataCfg:
    n_train_clusters: int = 300
    n_dev_clusters: int = 60
    n_confirm_clusters: int = 100
    split_salt: int = 0
    max_target_tokens: int = 384
    plans_per_function: int = 3
    patches_per_plan: int = 2
    teacher_checkpoint: str = ""        # optional existing optimizer used as an extra patch proposer
    files_k: List[int] = field(default_factory=lambda: [2, 4])
    files_d: List[int] = field(default_factory=lambda: [1, 3])
    n_files_per_split: Dict[str, int] = field(default_factory=lambda: {"train": 400, "dev": 60, "confirm": 100})
    c0_cap_ms: float = 100.0
    perf_scale: int = 2000
    n_perf: int = 3
    n_correct_random: int = 10
    target_searches: int = 2
    target_plans: int = 2
    target_patches_per_plan: int = 2
    max_states_per_file: int = 3
    near_miss_per_patch: int = 1
    repair_candidates: int = 4
    no_plan_weight: float = 0.25
    abandon_frac: float = 0.05
    m4_valid_frac: float = 0.25
    m5_abandon_weight: float = 0.25
    utility_lambda_cost: float = 0.0
    utility_tie_tol: float = 0.05
    id_permutations: int = 2
    manual_review_n: int = 40
    workers: int = 4
    menu_size: int = 12


@dataclass
class PathsCfg:
    root: str = "artifacts/chain"

    def sub(self, *parts: str) -> str:
        return os.path.join(self.root, *parts)

    @property
    def tasks(self) -> str: return self.sub("tasks")
    @property
    def private(self) -> str: return self.sub("private")          # restricted: hidden suites, audit seeds, privileged provenance
    @property
    def ledger(self) -> str: return self.sub("ledger")
    @property
    def objects(self) -> str: return self.sub("objects")
    @property
    def datasets(self) -> str: return self.sub("datasets")
    @property
    def checkpoints(self) -> str: return self.sub("checkpoints")
    @property
    def runs(self) -> str: return self.sub("runs")
    @property
    def cache(self) -> str: return self.sub("cache")


@dataclass
class ChainConfig:
    experiment: ExperimentCfg = field(default_factory=ExperimentCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    roles: RolesCfg = field(default_factory=RolesCfg)
    budget: BudgetCfg = field(default_factory=BudgetCfg)
    verification: VerificationCfg = field(default_factory=VerificationCfg)
    sandbox: SandboxCfg = field(default_factory=SandboxCfg)
    training: TrainingCfg = field(default_factory=TrainingCfg)
    rl: RLCfg = field(default_factory=RLCfg)
    serving: ServingCfg = field(default_factory=ServingCfg)
    data: DataCfg = field(default_factory=DataCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)

    # -------------------------------------------------------------- #
    def validate(self, purpose: str = "generic") -> "ChainConfig":
        """Raise :class:`ConfigError` on incompatible combinations. ``purpose`` adds stage-specific rules:
        ``collect:<split>``, ``train``, ``eval:development``, ``eval:confirmation``."""
        errs: List[str] = []
        ctx = self.model.context_tokens
        for r in ROLE_NAMES:
            d = self.roles.get(r)
            if d.max_input <= 0 or d.max_output <= 0:
                errs.append(f"roles.{r}: caps must be positive")
            if d.max_input + d.max_output > ctx:
                errs.append(f"roles.{r}: max_input + max_output = {d.max_input + d.max_output} exceeds context {ctx}")
            if d.mode not in ("greedy", "sample"):
                errs.append(f"roles.{r}.mode must be greedy|sample")
            if d.mode == "sample" and d.temperature <= 0:
                errs.append(f"roles.{r}.temperature must be > 0 when sampling")
            if d.top_k < 0 or not (0 < d.top_p <= 1.0):
                errs.append(f"roles.{r}: bad top_k/top_p")
        b = self.budget
        for k in ("patch_slots", "role_calls", "total_input_output_tokens", "extra_probe_calls", "max_profiles",
                  "max_plans_per_function_version"):
            if getattr(b, k) < 0:
                errs.append(f"budget.{k} must be >= 0")
        if b.max_probes_per_candidate < 0 or b.max_probes_per_candidate > 12:
            errs.append("budget.max_probes_per_candidate must be in [0, 12]")
        if b.max_repairs_per_lineage < 0:
            errs.append("budget.max_repairs_per_lineage must be >= 0")
        v = self.verification
        if v.semantic_reference != "immutable_original":
            errs.append("verification.semantic_reference must be immutable_original")
        if v.final_audit_feedback != "disabled":
            errs.append("verification.final_audit_feedback must be disabled (audits never feed back into search)")
        if v.online_private_feedback not in ("category_only",):
            errs.append("verification.online_private_feedback must be category_only")
        if v.accept_incremental_speedup <= 1.0 or v.success_final_speedup <= 1.0:
            errs.append("verification: speedup thresholds must be > 1")
        if v.accept_rule not in ("paired_log_median_ci_lower", "point_estimate", "legacy_ratio"):
            errs.append(f"verification.accept_rule {v.accept_rule!r} unknown")
        if v.uncertain_timing_policy != "reject":
            errs.append("verification.uncertain_timing_policy must be reject")
        if min(v.train_timing_blocks, v.development_timing_blocks, v.audit_timing_blocks) < 1:
            errs.append("verification: timing block counts must be >= 1")
        if v.timing_bootstrap_draws < 100 or not (0.5 < v.timing_interval < 1.0):
            errs.append("verification: bootstrap draws >= 100 and 0.5 < interval < 1")
        if v.hash_seed == v.audit_hash_seed:
            errs.append("verification: the audit must use a hash seed independent of the search")
        unknown_rep = set(v.repairable_failures) - {"syntax_error", "smoke_mismatch", "probe_mismatch", "private_mismatch"}
        if unknown_rep:
            errs.append(f"verification.repairable_failures: unknown {sorted(unknown_rep)}")
        if self.sandbox.isolation not in ("rlimits", "unshare", "bwrap"):
            errs.append("sandbox.isolation must be rlimits|unshare|bwrap")
        if self.experiment.final_test_access != "evaluator_only":
            errs.append("experiment.final_test_access must be evaluator_only")
        if not self.experiment.bundle_frozen_per_episode:
            errs.append("experiment.bundle_frozen_per_episode must be true (no weight changes inside an episode)")
        if self.model.add_special_tokens:
            errs.append("model.add_special_tokens=true is a separate ablation (it resizes the embedding); not supported here")
        if self.serving.compile_enabled:
            errs.append("serving.compile_enabled: the serving engine runs eager; benchmark compiled paths with "
                        "python -m tinyperf_chain.serve.bench --compile before enabling anything")
        if self.serving.timing_workers < 1 or self.serving.correctness_workers < 1:
            errs.append("serving: need at least one timing and one correctness worker")
        if self.rl.top_k < 0 or not (0 < self.rl.top_p <= 1.0) or self.rl.temperature <= 0:
            errs.append("rl: bad sampling settings")
        if not (0 < self.rl.clip_low < 1.0 < self.rl.clip_high):
            errs.append("rl: need clip_low < 1 < clip_high")
        for r in self.rl.roles:
            if r not in ("M1", "M2", "M3", "M4", "M5"):
                errs.append(f"rl.roles: unknown role {r}")
        if self.training.loss != "response_mean_per_example":
            errs.append("training.loss must be response_mean_per_example")
        if self.training.checkpoint_selection != "development_only":
            errs.append("training.checkpoint_selection must be development_only (no final-test selection)")
        if self.training.tie_mode not in ("sample", "sum"):
            errs.append("training.tie_mode must be sample|sum")
        if self.training.shared_sampling not in ("uniform_role", "token_matched"):
            errs.append("training.shared_sampling must be uniform_role|token_matched")
        # ---- purpose-specific ----
        if purpose.startswith("collect"):
            split = purpose.split(":", 1)[1] if ":" in purpose else ""
            if split in ("confirm", "confirmation", "test", "audit"):
                errs.append(f"a training collector may not read the {split!r} split")
        if purpose == "eval:confirmation":
            if b.search_wall_seconds is None:
                errs.append("budget.search_wall_seconds is REQUIRED (calibrate it on development first)")
        if errs:
            raise ConfigError("invalid chain config:\n  - " + "\n  - ".join(errs))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(x: Any) -> Any:
    if isinstance(x, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), ""), x)
    if isinstance(x, list):
        return [_expand_env(v) for v in x]
    if isinstance(x, dict):
        return {k: _expand_env(v) for k, v in x.items()}
    return x


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_raw(path: str, depth: int = 0) -> Dict[str, Any]:
    if depth > 8:
        raise ConfigError("config base: chain too deep")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    base = raw.pop("base", None)
    if base:
        bpath = base if os.path.isabs(base) else os.path.join(os.path.dirname(path), base)
        raw = _deep_merge(_load_raw(bpath, depth + 1), raw)
    return raw


def _typed(cls, d: Any, where: str):
    if not isinstance(d, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(d).__name__}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}")
    kwargs = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        t = hints.get(f.name)
        if is_dataclass(t):
            kwargs[f.name] = _typed(t, v, f"{where}.{f.name}")
        else:
            kwargs[f.name] = v
    try:
        return cls(**kwargs)
    except TypeError as e:  # pragma: no cover - defensive
        raise ConfigError(f"{where}: {e}") from e


def _parse_scalar(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return yaml.safe_load(s)


def _set_nested(raw: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    d = raw
    for p in parts[:-1]:
        if not isinstance(d.get(p), dict):
            d[p] = {}
        d = d[p]
    d[parts[-1]] = value


_OVERRIDE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z0-9_]+)+=")


def pop_overrides(ns: Any, fields_: Sequence[str] = ()) -> List[str]:
    """Move ``a.b=c`` tokens that argparse attached to list-valued options back into ``ns.overrides``."""
    ovs = list(getattr(ns, "overrides", []) or [])
    for name in fields_:
        vals = getattr(ns, name, None)
        if isinstance(vals, list):
            keep = [v for v in vals if not (isinstance(v, str) and _OVERRIDE_RE.match(v))]
            ovs += [v for v in vals if isinstance(v, str) and _OVERRIDE_RE.match(v)]
            setattr(ns, name, keep)
    ns.overrides = ovs
    return ovs


def load_chain_config(path: Optional[str] = None, overrides: Optional[List[str]] = None) -> ChainConfig:
    raw: Dict[str, Any] = _load_raw(path) if path else {}
    for ov in overrides or []:
        if "=" not in ov:
            raise ConfigError(f"override must look like a.b=c, got {ov!r}")
        k, v = ov.split("=", 1)
        _set_nested(raw, k, _parse_scalar(v))
    raw = _expand_env(raw)
    cfg = _typed(ChainConfig, raw, "config")
    return cfg


def dump_chain_config(cfg: ChainConfig, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)


def config_hash(cfg: ChainConfig) -> str:
    from tinyperf_chain.hashing import hash_json

    return hash_json(cfg.to_dict())
