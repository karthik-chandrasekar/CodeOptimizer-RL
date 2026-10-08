"""Bits shared by the pretraining / SFT / GRPO trainers."""
from __future__ import annotations

import contextlib
import math
import os
from typing import Dict, Iterable, Optional, Tuple

import torch

from tinyperf.common.config import Config, ModelConfig, OptimConfig
from tinyperf.common.utils import get_logger
from tinyperf.model.tokenizer import CodeTokenizer
from tinyperf.model.transformer import TinyPerfLM

log = get_logger("train")


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float, betas: Iterable[float] = (0.9, 0.95)) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices only (no decay on norms / embeddings-as-1D)."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    fused = torch.cuda.is_available() and next(model.parameters()).is_cuda
    return torch.optim.AdamW(groups, lr=lr, betas=tuple(betas), fused=fused)


def lr_at(step: int, optim: OptimConfig) -> float:
    """Linear warmup then cosine decay to min_lr."""
    if step < optim.warmup_steps:
        return optim.lr * (step + 1) / max(1, optim.warmup_steps)
    if step >= optim.max_steps:
        return optim.min_lr
    frac = (step - optim.warmup_steps) / max(1, optim.max_steps - optim.warmup_steps)
    return optim.min_lr + 0.5 * (1 + math.cos(math.pi * frac)) * (optim.lr - optim.min_lr)


def set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for g in opt.param_groups:
        g["lr"] = lr


def autocast_ctx(device: str):
    """bf16 autocast on CUDA; no autocast elsewhere (MPS bf16 support is patchy, CPU is for smoke tests)."""
    if device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def autocast_dtype(device: str) -> Optional[torch.dtype]:
    return torch.bfloat16 if device.startswith("cuda") else None


def load_model_and_tokenizer(cfg: Config, checkpoint: Optional[str], device: str) -> Tuple[TinyPerfLM, CodeTokenizer]:
    """Load the tokenizer from cfg.tokenizer.path (or the path stored in the checkpoint) and the model."""
    tok_path = cfg.tokenizer.path
    if checkpoint and os.path.exists(checkpoint):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        extra = ck.get("extra", {})
        if not os.path.exists(tok_path) and extra.get("tokenizer_path") and os.path.exists(extra["tokenizer_path"]):
            tok_path = extra["tokenizer_path"]
        model = TinyPerfLM(ModelConfig(**ck["model_cfg"]))
        model.load_state_dict(ck["state_dict"])
        log.info(f"loaded {checkpoint} (step {extra.get('step', '?')}, {model.n_params()/1e6:.1f}M params)")
    else:
        if checkpoint:
            log.warning(f"checkpoint {checkpoint} not found - initialising from scratch")
        model = TinyPerfLM(cfg.model)
        log.info(f"fresh model: {model.n_params()/1e6:.1f}M params")
    tok = CodeTokenizer(tok_path)
    if tok.vocab_size > model.cfg.vocab_size:
        raise ValueError(f"tokenizer vocab ({tok.vocab_size}) larger than model vocab ({model.cfg.vocab_size})")
    return model.to(device), tok


def grad_norm(model: torch.nn.Module) -> float:
    tot = 0.0
    for p in model.parameters():
        if p.grad is not None:
            tot += float(p.grad.detach().float().norm() ** 2)
    return math.sqrt(tot)


def count_tokens(batches: Iterable[torch.Tensor]) -> int:
    return sum(int(b.numel()) for b in batches)
