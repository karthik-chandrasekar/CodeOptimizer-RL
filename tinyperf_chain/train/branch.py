"""Branch one existing student checkpoint into five independent role checkpoints.

    TINYPERF_CKPT=artifacts/distill/s2m/rl/final.pt python -m tinyperf_chain.train.branch \\
        --config configs/chain/pilot.yaml --out artifacts/chain/checkpoints/initial

Checks before writing anything (each failure is loud): the seed file exists and loads; architecture matches the
configured expectations (width, layers, heads, MLP width, vocabulary, tied embeddings, context >= the chain context);
the tokenizer's vocabulary fits; the configuration-derived parameter count (eq. 1) equals the recount of the loaded
tensors.  Each branch records its parent hash, tokenizer hash and per-tensor hashes; the seed stays immutable.
No optimizer state is copied between roles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from tinyperf_chain.config import ChainConfig, load_chain_config
from tinyperf_chain.hashing import sha256_file

ROLES = ("M1", "M2", "M3", "M4", "M5")


def param_count_formula(V: int, d: int, L: int, m: int) -> int:
    """P(V,d,L,m) = V d + L (4 d^2 + 3 d m + 2 d) + d   (bias-free, tied embeddings, 2 RMSNorm per block + final)."""
    return V * d + L * (4 * d * d + 3 * d * m + 2 * d) + d


def kv_bytes(L: int, d: int, T: int, bytes_per: int = 2) -> int:
    return 2 * L * d * T * bytes_per


def tensor_hashes(model: torch.nn.Module) -> Dict[str, str]:
    out = {}
    for k, v in model.state_dict().items():
        out[k] = hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]
    return out


def verify_seed(model, cfg: ChainConfig, tok=None) -> Dict[str, Any]:
    mc = model.cfg
    m = cfg.model
    errs = []
    for attr, want in (("n_embd", m.expect_n_embd), ("n_layer", m.expect_n_layer), ("n_head", m.expect_n_head),
                       ("mlp_dim", m.expect_mlp_dim), ("vocab_size", m.expect_vocab_size)):
        if getattr(mc, attr) != want:
            errs.append(f"{attr}={getattr(mc, attr)} expected {want}")
    if bool(mc.tie_embeddings) != m.expect_tied_embeddings:
        errs.append("tie_embeddings mismatch")
    if mc.tie_embeddings and model.lm_head.weight.data_ptr() != model.wte.weight.data_ptr():
        errs.append("embeddings are configured as tied but the tensors are not shared")
    if mc.max_seq_len < m.context_tokens:
        errs.append(f"model context {mc.max_seq_len} < chain context {m.context_tokens}")
    if tok is not None and tok.vocab_size > mc.vocab_size:
        errs.append(f"tokenizer vocab {tok.vocab_size} > model vocab {mc.vocab_size}")
    formula = param_count_formula(mc.vocab_size, mc.n_embd, mc.n_layer, mc.mlp_dim)
    recount = model.n_params()
    if formula != recount:
        errs.append(f"parameter recount {recount} != formula {formula}")
    if errs:
        raise ValueError("seed checkpoint identity check failed: " + "; ".join(errs))
    return {"formula": formula, "recount": recount, "bf16_bytes": recount * 2,
            "kv_bytes_bf16_at_context": kv_bytes(mc.n_layer, mc.n_embd, m.context_tokens)}


def branch(seed_path: str, out_dir: str, cfg: ChainConfig, tokenizer_path: Optional[str] = None,
           check_expectations: bool = True) -> Dict[str, Any]:
    from tinyperf.model.transformer import TinyPerfLM

    seed = Path(seed_path).resolve()
    if not seed.is_file():
        raise FileNotFoundError(seed)
    seed_hash = sha256_file(seed)
    tok = tok_hash = None
    if tokenizer_path:
        from tinyperf.model.tokenizer import CodeTokenizer

        tok = CodeTokenizer(tokenizer_path)
        tok_hash = sha256_file(tokenizer_path)
    probe = TinyPerfLM.load(str(seed), map_location="cpu")
    counts = verify_seed(probe, cfg, tok) if check_expectations else {"recount": probe.n_params()}
    seed_tensors = tensor_hashes(probe)
    os.makedirs(out_dir, exist_ok=True)
    manifest: Dict[str, Any] = {"parent_sha256": seed_hash, "parent_path": str(seed), "tokenizer_sha256": tok_hash,
                                "counts": counts, "five_role_bank_params": 5 * counts["recount"], "roles": {}}
    for role in ROLES:
        model = TinyPerfLM.load(str(seed), map_location="cpu")      # independent tensors per role
        path = os.path.join(out_dir, f"{role}.pt")
        model.save(path, extra={"role": role, "parent_sha256": seed_hash, "stage": "untrained_role_branch",
                                "tokenizer_sha256": tok_hash, "tokenizer_path": tokenizer_path})
        manifest["roles"][role] = {"path": path, "sha256": sha256_file(path), "n_params": model.n_params()}
        if tensor_hashes(model) != seed_tensors:
            raise RuntimeError("branch tensors differ from the seed")
    with open(os.path.join(out_dir, "branch_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--seed-checkpoint", default=None)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--dense-report", action="store_true", help="also print the dense-control parameter accounting")
    p.add_argument("overrides", nargs="*")
    a = p.parse_args(argv)
    cfg = load_chain_config(a.config, a.overrides).validate()
    seed = a.seed_checkpoint or cfg.model.seed_checkpoint or os.environ.get("TINYPERF_CKPT", "")
    if not seed:
        raise SystemExit("no seed checkpoint (set TINYPERF_CKPT or model.seed_checkpoint)")
    m = branch(seed, a.out or cfg.paths.sub("checkpoints", "initial"), cfg, a.tokenizer or cfg.model.tokenizer or None)
    if a.dense_report:
        m["dense_control"] = {"config": "configs/model/dense10m.yaml",
                              "formula_params": param_count_formula(8192, 256, 8, 1024),
                              "bank_minus_dense": 5 * param_count_formula(8192, 128, 4, 512) - param_count_formula(8192, 256, 8, 1024)}
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
