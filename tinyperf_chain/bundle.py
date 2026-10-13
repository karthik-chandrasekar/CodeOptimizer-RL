"""Role bundles: which implementation serves each role, with hashes, decoding settings and environment.

``bundle.json`` (``bundle_v1``)::

    {"schema_version": "bundle_v1", "system_id": "B4", "kind": "specialists",
     "tokenizer": "artifacts/tokenizer/tokenizer.json",
     "roles": {
        "M1": {"adapter": "neural", "checkpoint": "artifacts/chain/checkpoints/M1/final.pt"},
        "M2": {"adapter": "rule", "rule": "first_applicable"},
        "M3": {"adapter": "neural", "checkpoint": "...", "decoding": {"temperature": 0.6}},
        "M4": {"adapter": "rule", "rule": "transform_specific"},
        "M5": {"adapter": "neural", "checkpoint": ".../M3/final.pt", "prompt_role": "M5"},   # M3 repairs
     }}

Adapters: ``neural`` (role checkpoint; several roles may name the same checkpoint, e.g. the shared-role control),
``rule`` (see :mod:`tinyperf_chain.adapters.rules`), ``legacy_writer`` (M3 only: the unchanged TinyPerf-RL model).

``lock`` adds SHA-256 of every referenced file, the registries, the schemas, the config hash, the code commit and an
environment manifest; a locked bundle refuses to load if any referenced file changed.  Loading the same checkpoint
for several roles shares one engine; different checkpoints never share weights or caches.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

from tinyperf_chain import __version__
from tinyperf_chain.adapters.base import RoleAdapter, RoleRouter
from tinyperf_chain.adapters.rules import (ProgrammaticWriter, RulePlanner, RuleProbeSelector, RuleRepairer,
                                           RuleTargetSelector)
from tinyperf_chain.config import ChainConfig, RoleDecoding
from tinyperf_chain.hashing import hash_json, sha256_file
from tinyperf_chain.registry import probe_registry_hash, registry_hash
from tinyperf_chain.schemas import PARSER_VERSION, SCHEMA_VERSIONS

BUNDLE_SCHEMA = "bundle_v1"
ROLE_ORDER = ("M1", "M2", "M3", "M4", "M5")


class BundleError(ValueError):
    pass


def environment_manifest(repo_dir: Optional[str] = None) -> Dict[str, Any]:
    env: Dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
                           "tinyperf_chain": __version__}
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        env["torch"] = None
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    env["cpu"] = line.split(":", 1)[1].strip()
                    break
        env["n_cpus"] = os.cpu_count()
    except OSError:
        pass
    repo = repo_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        env["code_commit"] = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True,
                                            timeout=5).stdout.strip() or None
        env["code_dirty"] = bool(subprocess.run(["git", "-C", repo, "status", "--porcelain"], capture_output=True,
                                                text=True, timeout=5).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        env["code_commit"] = None
    return env


def load_bundle_json(path: str) -> Dict[str, Any]:
    from tinyperf_chain.config import _expand_env

    with open(path) as f:
        b = _expand_env(json.load(f))
    if b.get("schema_version") != BUNDLE_SCHEMA:
        raise BundleError(f"{path}: not a {BUNDLE_SCHEMA} bundle")
    roles = b.get("roles", {})
    missing = [r for r in ROLE_ORDER if r not in roles]
    if missing:
        raise BundleError(f"{path}: roles missing {missing}")
    base = os.path.dirname(os.path.abspath(path))
    for r, spec in roles.items():
        if spec.get("checkpoint") and not os.path.isabs(spec["checkpoint"]) and not os.path.exists(spec["checkpoint"]):
            cand = os.path.join(base, spec["checkpoint"])
            if os.path.exists(cand):
                spec["checkpoint"] = cand
    return b


def referenced_files(b: Dict[str, Any]) -> Dict[str, str]:
    files = {}
    if b.get("tokenizer"):
        files["tokenizer"] = b["tokenizer"]
    for r, spec in b["roles"].items():
        if spec.get("checkpoint"):
            files[f"{r}.checkpoint"] = spec["checkpoint"]
    return files


def checkpoint_manifest(path: str) -> Dict[str, Any]:
    """Architecture counts (eq. 1 and a recount), parent hash and training-data hashes of one checkpoint, read from
    the serialized metadata of a trusted file (16.4).  Weights are referenced by hash, never embedded."""
    import torch

    from tinyperf_chain.train.branch import param_formula

    from tinyperf.model.transformer import TinyPerfLM

    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("model_cfg", ck.get("config", {})) if isinstance(ck, dict) else {}
    extra = ck.get("extra", {}) if isinstance(ck, dict) else {}
    try:
        n_recount = int(TinyPerfLM.load(path, map_location="cpu").n_params())   # the same recount as branch time
    except Exception:  # noqa: BLE001 - a foreign checkpoint: recount unavailable
        n_recount = None
    arch = {k: cfg.get(k) for k in ("vocab_size", "n_embd", "n_layer", "n_head", "mlp_dim", "max_seq_len", "tie_embeddings") if k in cfg}
    formula = None
    if all(k in arch for k in ("vocab_size", "n_embd", "n_layer", "mlp_dim")):
        formula = param_formula(arch["vocab_size"], arch["n_embd"], arch["n_layer"], arch["mlp_dim"])
    return {"sha256": sha256_file(path), "architecture": arch, "params_formula": formula, "params_recount": n_recount,
            "parent_sha256": extra.get("parent_sha256"), "role": extra.get("role"), "stage": extra.get("stage"),
            "training_data_sha256": extra.get("data_hash"), "tokenizer_sha256": extra.get("tokenizer_sha256"),
            "train_step": extra.get("step")}


def release_manifest(b: Dict[str, Any], cfg: ChainConfig) -> Dict[str, Any]:
    """Everything 17.4 asks a released bundle to carry, derived from the bundle and the evaluation config."""
    from tinyperf_chain.contracts import CONTRACT_VERSION, TaskContract

    weights: Dict[str, Any] = {}
    for r in ROLE_ORDER:
        spec = b["roles"][r]
        if spec.get("checkpoint"):
            weights[r] = {"path": spec["checkpoint"], **checkpoint_manifest(spec["checkpoint"])}
    decoding = {}
    for r in ROLE_ORDER:
        spec = b["roles"][r]
        d = replace(cfg.roles.get(r), **spec.get("decoding", {}))
        decoding[r] = {"adapter": spec.get("adapter"), "rule": spec.get("rule"), "prompt_role": spec.get("prompt_role"),
                       **{k: getattr(d, k) for k in ("max_input", "max_output", "mode", "temperature", "top_k", "top_p")}}
    return {
        "weights": weights,
        "tokenizer_sha256": sha256_file(b["tokenizer"]) if b.get("tokenizer") and os.path.exists(b["tokenizer"]) else None,
        "role_schemas": dict(SCHEMA_VERSIONS, parser=PARSER_VERSION),
        "decoding": decoding,
        "registries": {"transforms": registry_hash(), "probes": probe_registry_hash(),
                       "contracts": hash_json([CONTRACT_VERSION, TaskContract.SUPPORTED_POLICIES])},
        "evaluator": {"verification": cfg.verification.__dict__, "budget": cfg.budget.__dict__, "sandbox": {
            k: v for k, v in cfg.sandbox.__dict__.items() if k != "allowed_modules"}},
        "context_tokens": cfg.model.context_tokens,
    }


def lock_bundle(path: str, out: str, cfg: ChainConfig) -> Dict[str, Any]:
    from tinyperf_chain.config import config_hash

    b = load_bundle_json(path)
    files = referenced_files(b)
    b["locked"] = True
    b["file_sha256"] = {k: sha256_file(p) for k, p in files.items()}
    b["registries"] = {"transforms": registry_hash(), "probes": probe_registry_hash()}
    b["schemas"] = dict(SCHEMA_VERSIONS, parser=PARSER_VERSION)
    b["config_sha256"] = config_hash(cfg)
    b["release"] = release_manifest(b, cfg)
    b["environment"] = environment_manifest()
    b["bundle_sha256"] = hash_json({k: v for k, v in b.items() if k not in ("bundle_sha256", "environment")})
    with open(out, "w") as f:
        json.dump(b, f, indent=1)
    return b


def verify_lock(b: Dict[str, Any], cfg: Optional[ChainConfig] = None, allow_env_mismatch: bool = False) -> None:
    if not b.get("locked"):
        raise BundleError("bundle is not locked")
    for k, p in referenced_files(b).items():
        h = sha256_file(p)
        if h != b["file_sha256"].get(k):
            raise BundleError(f"locked bundle: {k} ({p}) changed since locking")
    if b["registries"] != {"transforms": registry_hash(), "probes": probe_registry_hash()}:
        raise BundleError("locked bundle: transformation/probe registries changed since locking")
    want = hash_json({k: v for k, v in b.items() if k not in ("bundle_sha256", "environment")})
    if want != b.get("bundle_sha256"):
        raise BundleError("locked bundle: manifest edited after locking")
    if cfg is not None:
        from tinyperf_chain.config import config_hash

        if b.get("config_sha256") != config_hash(cfg):
            raise BundleError("locked bundle: evaluation config differs from the locked one")
    if not allow_env_mismatch:
        env, cur = b.get("environment", {}), environment_manifest()
        for key in ("python", "torch", "code_commit"):
            if env.get(key) != cur.get(key):
                raise BundleError(f"locked bundle: environment {key} {env.get(key)!r} != current {cur.get(key)!r} "
                                  "(label an environment-upgrade experiment explicitly with --allow-env-mismatch)")


def bundle_hash(b: Dict[str, Any]) -> str:
    if b.get("bundle_sha256"):
        return b["bundle_sha256"]
    files = {k: (sha256_file(p) if os.path.exists(p) else "missing") for k, p in referenced_files(b).items()}
    return hash_json({"bundle": {k: v for k, v in b.items()}, "files": files})


def _rule_adapter(role: str, rule: str, seed: int) -> RoleAdapter:
    if role == "M1":
        return RuleTargetSelector(rule or "hottest_self", seed)
    if role == "M2":
        if rule not in ("", "first_applicable"):
            raise BundleError(f"unknown M2 rule {rule}")
        return RulePlanner()
    if role == "M3":
        return ProgrammaticWriter()
    if role == "M4":
        return RuleProbeSelector(rule or "transform_specific", seed=seed)
    if role == "M5":
        return RuleRepairer()
    raise BundleError(role)


def build_router(b: Dict[str, Any], cfg: ChainConfig, *, device: str = "cpu", use_scheduler: bool = False,
                 seed: int = 0) -> Tuple[RoleRouter, Dict[str, Any]]:
    """Instantiate adapters for every role. Returns (router, resources) where resources hold engines / scheduler."""
    import torch

    from tinyperf.model.tokenizer import CodeTokenizer
    from tinyperf.model.transformer import TinyPerfLM
    from tinyperf_chain.adapters.model import LegacyWriterAdapter, NeuralRoleAdapter
    from tinyperf_chain.context import PromptBuilder
    from tinyperf_chain.serve.engine import RoleEngine
    from tinyperf_chain.serve.scheduler import RoleScheduler

    needs_model = any(s.get("adapter") in ("neural", "legacy_writer") for s in b["roles"].values())
    tok = builder = None
    tok_hash = ""
    if needs_model:
        tpath = b.get("tokenizer") or cfg.model.tokenizer
        if not tpath or not os.path.exists(tpath):
            raise BundleError(f"tokenizer not found: {tpath!r}")
        tok = CodeTokenizer(tpath)
        tok_hash = sha256_file(tpath)
        builder = PromptBuilder(tok, cfg.roles, cfg.model.context_tokens)
    ac = torch.bfloat16 if (str(device).startswith("cuda") and cfg.training.precision == "bf16_autocast") else None
    engines: Dict[str, RoleEngine] = {}

    def _make_engine(ck: str) -> RoleEngine:
        h = sha256_file(ck)
        m = TinyPerfLM.load(ck, map_location="cpu")
        if tok is not None and tok.vocab_size > m.cfg.vocab_size:
            raise BundleError(f"{ck}: tokenizer larger than model vocabulary")
        return RoleEngine(m, device=device, static_cache=cfg.serving.static_kv_cache,
                          last_logits_only=cfg.serving.last_logits_only, autocast_dtype=ac,
                          checkpoint_sha256=h, tokenizer_sha256=tok_hash, name=os.path.basename(ck))

    resident = set(cfg.serving.resident_roles)
    for r in ROLE_ORDER:
        spec = b["roles"][r]
        if spec.get("adapter") in ("neural", "legacy_writer"):
            ck = spec["checkpoint"]
            # resident roles are loaded once, up front (P1); a non-resident role loads on its first call and is
            # then kept (never a per-request reload); the scheduler only batches resident engines
            if ck not in engines and (r in resident or spec.get("adapter") == "legacy_writer" or not use_scheduler):
                engines[ck] = _make_engine(ck)
    sched = None
    if use_scheduler and engines:
        sched = RoleScheduler(engines, window_ms=cfg.serving.batch_window_ms, max_batch=cfg.serving.max_batch,
                              max_pending=max(16, cfg.serving.max_pending_candidates * 4),
                              length_buckets=cfg.serving.length_buckets, mode=cfg.serving.scheduler_mode)
    adapters: Dict[str, RoleAdapter] = {}
    for r in ROLE_ORDER:
        spec = b["roles"][r]
        kind = spec.get("adapter")
        if kind == "rule":
            adapters[r] = _rule_adapter(r, spec.get("rule", ""), seed)
        elif kind == "neural":
            dec = replace(cfg.roles.get(r), **spec.get("decoding", {}))
            ck = spec["checkpoint"]
            if ck in engines:
                adapters[r] = NeuralRoleAdapter(engines[ck], tok, builder, dec, spec.get("prompt_role"), sched, ck, seed,
                                                per_request_rng=cfg.serving.per_request_rng)
            else:   # non-resident: lazy, unbatched
                adapters[r] = NeuralRoleAdapter((lambda c=ck: engines.setdefault(c, _make_engine(c))), tok, builder, dec,
                                                spec.get("prompt_role"), None, ck, seed, per_request_rng=cfg.serving.per_request_rng)
        elif kind == "legacy_writer":
            if r != "M3":
                raise BundleError("legacy_writer can only serve M3")
            adapters[r] = LegacyWriterAdapter(engines[spec["checkpoint"]], tok, spec.get("max_new", 768),
                                              spec.get("temperature", 0.6), seed)
        else:
            raise BundleError(f"{r}: unknown adapter {kind!r}")
    router = RoleRouter(adapters, bundle_hash(b), b.get("system_id", "custom"))
    return router, {"engines": engines, "scheduler": sched, "tokenizer": tok}
