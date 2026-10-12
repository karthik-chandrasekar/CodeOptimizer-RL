"""Role inference engine: batched sampling with exact bookkeeping of the sampling distribution.

* Prompts that do not fit ``len(prompt) + max_new <= max_seq_len`` are refused (:class:`EngineOverflow`); the legacy
  generator's silent tail truncation is never used.
* Prefill projects only the last position to the vocabulary (``last_logits_only``); decoding uses the static KV
  cache (:mod:`tinyperf_chain.serve.cache`) or, as the eager reference, the legacy concatenating cache.
* The recorded log-probability of every sampled token is computed under the distribution actually sampled from:
  ``softmax(filter(logits[~banned] / T))``.  :func:`effective_logprobs` recomputes exactly that distribution for RL
  (with ``top_k = 0`` and ``top_p = 1`` it is plain ``softmax(logits / T)``).  Greedy decoding records log-probs
  under ``softmax(logits)``.
* Per-request generators (``seeds``) make sampling independent of how requests were batched.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from tinyperf.model.generate import left_pad, top_k_top_p_filter
from tinyperf.model.transformer import TinyPerfLM, causal_padded_mask
from tinyperf_chain.hashing import hash_json
from tinyperf_chain.serve.cache import CacheKey, StaticKVCache, forward_static


class EngineOverflow(ValueError):
    pass


@dataclass(frozen=True)
class SamplingSpec:
    greedy: bool = False
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    banned: Tuple[int, ...] = ()
    mask_version: str = "none"

    def key(self) -> str:
        return hash_json(asdict(self))


@dataclass
class GenResult:
    ids: List[int]
    logprobs: List[float]
    finished: bool
    prompt_len: int
    prefill_ns: int = 0
    decode_ns: int = 0
    batch_size: int = 1
    padded_tokens: int = 0


def effective_logits(logits: torch.Tensor, spec: SamplingSpec) -> torch.Tensor:
    lg = logits.float()
    if spec.banned:
        lg = lg.clone()
        lg[..., list(spec.banned)] = float("-inf")
    if spec.greedy:
        return lg
    lg = lg / max(float(spec.temperature), 1e-5)
    if spec.top_k > 0 or spec.top_p < 1.0:
        shp = lg.shape
        lg = top_k_top_p_filter(lg.reshape(-1, shp[-1]), spec.top_k, spec.top_p).reshape(shp)
    return lg


def effective_logprobs(logits: torch.Tensor, targets: torch.Tensor, spec: SamplingSpec) -> torch.Tensor:
    """log pi_e(target) under the effective sampling distribution; logits [..., V], targets [...]."""
    lp = F.log_softmax(effective_logits(logits, spec), dim=-1)
    return lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


class RoleEngine:
    def __init__(self, model: TinyPerfLM, *, device: str = "cpu", static_cache: bool = True, last_logits_only: bool = True,
                 autocast_dtype: Optional[torch.dtype] = None, checkpoint_sha256: str = "", tokenizer_sha256: str = "",
                 name: str = ""):
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.static_cache = static_cache
        self.last_logits_only = last_logits_only
        self.autocast_dtype = autocast_dtype
        self.checkpoint_sha256 = checkpoint_sha256
        self.tokenizer_sha256 = tokenizer_sha256
        self.name = name
        self.max_seq_len = model.cfg.max_seq_len
        self.stats = {"calls": 0, "rows": 0, "prefill_ns": 0, "decode_ns": 0, "output_tokens": 0, "padded_tokens": 0}

    def _ac(self):
        if self.autocast_dtype is not None and str(self.device).startswith("cuda"):
            return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)
        return torch.autocast("cpu", enabled=False)

    @torch.no_grad()
    def generate(self, prompts: Sequence[Sequence[int]], max_new: int, spec: SamplingSpec, stop_ids: Sequence[int], *,
                 pad_id: int = 0, seeds: Optional[Sequence[int]] = None, deadline: Optional[float] = None) -> List[GenResult]:
        if not prompts:
            return []
        for p in prompts:
            if len(p) == 0:
                raise EngineOverflow("empty prompt")
            if len(p) + max_new > self.max_seq_len:
                raise EngineOverflow(f"prompt ({len(p)}) + max_new ({max_new}) exceeds context {self.max_seq_len}")
        B = len(prompts)
        dev = self.device
        ids, valid, positions = left_pad(prompts, pad_id, dev)
        L = ids.size(1)
        prompt_len = valid.sum(-1)
        gens = None
        if seeds is not None and not spec.greedy:
            gens = []
            for s in seeds:
                g = torch.Generator(device="cpu")
                g.manual_seed(int(s) % (2 ** 63 - 1))
                gens.append(g)
        stop = torch.tensor(list(stop_ids), device=dev)
        t0 = time.perf_counter_ns()
        with self._ac():
            if self.static_cache:
                cache = StaticKVCache(self.model, B, L + max_new, dev, key=CacheKey(self.checkpoint_sha256, self.tokenizer_sha256, str(self.autocast_dtype)))
                logits = forward_static(self.model, ids, positions, cache, valid, last_only=True)[:, -1, :]
            else:
                kv = self.model.new_cache()
                logits = self.model(ids, attn_mask=causal_padded_mask(valid), positions=positions, cache=kv,
                                    last_logits_only=self.last_logits_only)[:, -1, :]
        t1 = time.perf_counter_ns()
        out_ids: List[List[int]] = [[] for _ in range(B)]
        out_lp: List[List[float]] = [[] for _ in range(B)]
        finished = torch.zeros(B, dtype=torch.bool, device=dev)
        key_valid = valid
        for t in range(max_new):
            lg = effective_logits(logits, spec)
            if spec.greedy:
                nxt = lg.argmax(-1)
            else:
                probs = F.softmax(lg, -1)
                if gens is None:
                    nxt = torch.multinomial(probs, 1).squeeze(-1)
                else:
                    pc = probs.float().cpu()
                    nxt = torch.stack([torch.multinomial(pc[i], 1, generator=gens[i]) for i in range(B)]).squeeze(-1).to(dev)
            lp = F.log_softmax(lg, -1).gather(-1, nxt[:, None]).squeeze(-1)
            is_stop = torch.isin(nxt, stop)
            nl, ll, fl = nxt.tolist(), lp.tolist(), finished.tolist()
            for i in range(B):
                if not fl[i]:
                    out_ids[i].append(nl[i])
                    out_lp[i].append(ll[i])
            finished = finished | is_stop
            if bool(finished.all()) or t == max_new - 1:
                break
            if deadline is not None and time.monotonic() > deadline:
                break
            step_in = torch.where(finished, torch.full_like(nxt, pad_id), nxt)[:, None]
            pos = (prompt_len + t)[:, None]
            with self._ac():
                if self.static_cache:
                    logits = forward_static(self.model, step_in, pos, cache, torch.ones(B, 1, dtype=torch.bool, device=dev),
                                            last_only=True)[:, -1, :]
                else:
                    key_valid = torch.cat([key_valid, torch.ones(B, 1, dtype=torch.bool, device=dev)], dim=1)
                    logits = self.model(step_in, attn_mask=key_valid[:, None, None, :], positions=pos, cache=kv)[:, -1, :]
        t2 = time.perf_counter_ns()
        fin = finished.tolist()
        padded = int(B * L - int(prompt_len.sum()))
        res = [GenResult(out_ids[i], out_lp[i], bool(fin[i]), int(prompt_len[i]), t1 - t0, t2 - t1, B, padded) for i in range(B)]
        self.stats["calls"] += 1
        self.stats["rows"] += B
        self.stats["prefill_ns"] += t1 - t0
        self.stats["decode_ns"] += t2 - t1
        self.stats["output_tokens"] += sum(len(r.ids) for r in res)
        self.stats["padded_tokens"] += padded
        return res


def resolve_device(name: str) -> str:
    from tinyperf.common.utils import resolve_device as _r

    return _r(name)
