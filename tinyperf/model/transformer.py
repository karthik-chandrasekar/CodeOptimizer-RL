"""TinyPerf policy network: a small decoder-only transformer.

Configs (see configs/model/*.yaml):  10M / 20M / 50M / 100M.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinyperf.common.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight.float()).type_as(x)


def rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)  # [T, D/2]
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D]; cos/sin: [T, D/2] (shared positions) or [B, T, D/2] (per-row positions)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    if cos.dim() == 2:
        c, s = cos[None, None, :, :], sin[None, None, :, :]
    else:
        c, s = cos[:, None, :, :], sin[:, None, :, :]
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    return torch.stack((o1, o2), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, attn_mask: Optional[torch.Tensor], cache: Optional[Dict] = None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        is_causal = attn_mask is None and T > 1
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.n_embd, cfg.mlp_dim, bias=False)
        self.w3 = nn.Linear(cfg.n_embd, cfg.mlp_dim, bias=False)
        self.w2 = nn.Linear(cfg.mlp_dim, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ln2 = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, attn_mask, cache=None):
        x = x + self.attn(self.ln1(x), cos, sin, attn_mask, cache)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyPerfLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.wte.weight
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))
        self._rope: Dict[Tuple[str, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def n_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wte.weight.numel()
        return n

    def _rope_for(self, positions: torch.Tensor):
        key = (str(positions.device), self.cfg.max_seq_len)
        if key not in self._rope:
            self._rope[key] = rope_cache(self.cfg.max_seq_len, self.cfg.n_embd // self.cfg.n_head, self.cfg.rope_theta, positions.device)
        cos, sin = self._rope[key]
        return cos[positions], sin[positions]

    def forward(self, idx: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, positions: Optional[torch.Tensor] = None,
                cache: Optional[List[Dict]] = None) -> torch.Tensor:
        """idx: [B, T] -> logits [B, T, V].

        attn_mask: optional bool [B, 1, T, S] (True = attend) for padded batches; when None a causal mask is used.
        positions: [T] or [B, T] absolute positions (per-row positions are needed for left-padded
                   batches / KV-cache decoding); defaults to arange(T).
        """
        B, T = idx.shape
        if positions is None:
            positions = torch.arange(T, device=idx.device)
        cos, sin = self._rope_for(positions)
        x = self.drop(self.wte(idx))
        for i, blk in enumerate(self.blocks):
            x = blk(x, cos, sin, attn_mask, None if cache is None else cache[i])
        return self.lm_head(self.ln_f(x))

    def new_cache(self) -> List[Dict]:
        return [dict() for _ in self.blocks]

    # ------------------------------------------------------------------ #
    def save(self, path: str, extra: Optional[Dict] = None) -> None:
        import os

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"model_cfg": asdict(self.cfg), "state_dict": self.state_dict(), "extra": extra or {}}, path)

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "TinyPerfLM":
        ck = torch.load(path, map_location=map_location, weights_only=False)
        cfg = ModelConfig(**ck["model_cfg"])
        m = cls(cfg)
        m.load_state_dict(ck["state_dict"])
        return m


def causal_padded_mask(attn_valid: torch.Tensor) -> torch.Tensor:
    """attn_valid: [B, T] bool (True = real token). Returns [B,1,T,T] bool mask: causal ∧ key valid."""
    B, T = attn_valid.shape
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=attn_valid.device))
    mask = causal[None, None] & attn_valid[:, None, None, :]
    # make sure every query row attends to at least itself (avoids NaN on fully-masked pad rows)
    eye = torch.eye(T, dtype=torch.bool, device=attn_valid.device)[None, None]
    return mask | eye


def lm_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1), ignore_index=ignore_index)
