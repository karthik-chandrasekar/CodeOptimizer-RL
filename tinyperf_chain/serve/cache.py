"""Static (preallocated) KV cache and a cache-aware forward pass over an unmodified :class:`TinyPerfLM`.

The legacy generator appends keys/values with ``torch.cat`` at every decode step.  Here each layer owns a buffer of
fixed capacity ``[B, H, capacity, D]``; prefill writes slots ``[0, L)`` and decode step ``t`` writes slot ``L + t``
by index.  Validity of every slot is tracked separately from token positions (left padding occupies slots but has no
position), and the attention mask is built explicitly for the absolute slots of the current chunk, so chunked
prefill and mixed prompt lengths stay correct.  A fully masked padding query attends to itself (as in the legacy
``causal_padded_mask``) to avoid NaNs; its output is never used.

Cache identity (checkpoint hash, tokenizer hash, dtype, mask semantics) is recorded in :class:`CacheKey`.  Different
role weights never share a cache, even for identical text.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

from tinyperf.model.transformer import TinyPerfLM, apply_rope


@dataclass(frozen=True)
class CacheKey:
    checkpoint_sha256: str
    tokenizer_sha256: str
    dtype: str
    mask_semantics: str = "causal_keyvalid_selfdiag_v1"


class StaticKVCache:
    def __init__(self, model: TinyPerfLM, batch: int, capacity: int, device, dtype: Optional[torch.dtype] = None,
                 key: Optional[CacheKey] = None):
        cfg = model.cfg
        self.n_layer = cfg.n_layer
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.batch = batch
        self.capacity = capacity
        self.device = device
        self.dtype = dtype
        self.key = key
        self.k: List[Optional[torch.Tensor]] = [None] * self.n_layer
        self.v: List[Optional[torch.Tensor]] = [None] * self.n_layer
        self.valid = torch.zeros(batch, capacity, dtype=torch.bool, device=device)
        self.length = 0               # number of slots written so far (same for every row)

    def _ensure(self, i: int, like: torch.Tensor) -> None:
        if self.k[i] is None:
            shape = (self.batch, self.n_head, self.capacity, self.head_dim)
            dt = self.dtype or like.dtype
            self.k[i] = torch.zeros(shape, dtype=dt, device=self.device)
            self.v[i] = torch.zeros(shape, dtype=dt, device=self.device)

    def write(self, i: int, start: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self._ensure(i, k)
        T = k.size(2)
        if start + T > self.capacity:
            raise IndexError(f"KV cache overflow: {start + T} > capacity {self.capacity}")
        self.k[i][:, :, start:start + T] = k.to(self.k[i].dtype)
        self.v[i][:, :, start:start + T] = v.to(self.v[i].dtype)

    def memory_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v if t is not None)


def chunk_mask(valid: torch.Tensor, start: int, T: int) -> torch.Tensor:
    """Bool [B, 1, T, start+T]: query slot ``start+i`` may attend key slot ``j`` iff ``j <= start+i`` and slot ``j`` is
    valid; every query also attends to its own slot (no all-False rows)."""
    S = start + T
    dev = valid.device
    q = torch.arange(start, S, device=dev)
    k = torch.arange(S, device=dev)
    causal = k[None, :] <= q[:, None]                       # [T, S]
    m = causal[None, None] & valid[:, None, None, :S]       # [B, 1, T, S]
    diag = (k[None, :] == q[:, None])[None, None]
    return m | diag


class CacheIdentityError(RuntimeError):
    """The cache was filled by a different model version / tokenizer / numerical mode than the one using it (14.2)."""


@torch.no_grad()
def forward_static(model: TinyPerfLM, idx: torch.Tensor, positions: torch.Tensor, cache: StaticKVCache,
                   new_valid: torch.Tensor, last_only: bool = True, key: Optional[CacheKey] = None) -> torch.Tensor:
    """Run ``idx`` ([B, T], absolute RoPE ``positions`` [B, T]) at cache slots ``[cache.length, cache.length + T)``.

    ``new_valid`` [B, T] marks which of the new slots hold real tokens.  Returns logits [B, 1|T, V].  When ``key``
    is given it must equal the cache's recorded identity: a cache never survives a model-version change."""
    if key is not None and cache.key is not None and key != cache.key:
        raise CacheIdentityError(f"cache identity {cache.key} != {key}")
    B, T = idx.shape
    start = cache.length
    if start + T > cache.capacity:
        raise IndexError("KV cache capacity exceeded")
    cache.valid[:, start:start + T] = new_valid
    mask = chunk_mask(cache.valid, start, T)
    cos, sin = model._rope_for(positions)
    x = model.wte(idx)
    C = x.size(-1)
    for i, blk in enumerate(model.blocks):
        att = blk.attn
        h = blk.ln1(x)
        q, k, v = att.qkv(h).split(C, dim=2)
        q = q.view(B, T, att.n_head, att.head_dim).transpose(1, 2)
        k = k.view(B, T, att.n_head, att.head_dim).transpose(1, 2)
        v = v.view(B, T, att.n_head, att.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        cache.write(i, start, k, v)
        K = cache.k[i][:, :, :start + T]
        V = cache.v[i][:, :, :start + T]
        y = F.scaled_dot_product_attention(q, K.to(q.dtype), V.to(q.dtype), attn_mask=mask, dropout_p=0.0)
        x = x + att.proj(y.transpose(1, 2).contiguous().view(B, T, C))
        x = x + blk.mlp(blk.ln2(x))
    cache.length = start + T
    if last_only:
        x = x[:, -1:, :]
    return model.lm_head(model.ln_f(x))
