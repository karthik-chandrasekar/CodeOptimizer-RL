"""Batched autoregressive sampling for the policy.

* prompts are left-padded so every row's last prompt token sits at the same
  column; RoPE positions are per-row so padding does not shift positions;
* a KV cache is used for decoding;
* generation stops per row on any of ``stop_ids`` (``</EDIT>``, ``<STOP>``, ``<EOS>``);
* the log-probability of every sampled token under ``softmax(logits / T)`` is
  returned (this is the distribution the policy is defined as during RL; the
  trainer recomputes it with the same temperature).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

from tinyperf.common.config import GenerationConfig
from tinyperf.model.transformer import TinyPerfLM, causal_padded_mask


@dataclass
class GenOutput:
    ids: List[List[int]]          # response token ids per row (stop token included when hit)
    logprobs: List[List[float]]   # sampled-token log-probs under softmax(logits/T)
    finished: List[bool]          # True = ended on a stop token (False = hit max_new_tokens)


def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(-1)
        remove = (cum - probs) > top_p  # always keep the top-1 token
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return logits


def left_pad(prompts: Sequence[Sequence[int]], pad_id: int, device) -> tuple:
    """Return (ids [B,L], valid [B,L] bool, positions [B,L])."""
    B, L = len(prompts), max(len(p) for p in prompts)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    valid = torch.zeros((B, L), dtype=torch.bool)
    for i, p in enumerate(prompts):
        if p:
            ids[i, L - len(p):] = torch.tensor(list(p), dtype=torch.long)
            valid[i, L - len(p):] = True
    positions = (valid.long().cumsum(-1) - 1).clamp(min=0)
    return ids.to(device), valid.to(device), positions.to(device)


@torch.no_grad()
def generate(
    model: TinyPerfLM,
    prompts: Sequence[Sequence[int]],
    gen: GenerationConfig,
    *,
    pad_id: int,
    stop_ids: Sequence[int],
    device: Optional[str] = None,
    greedy: bool = False,
    max_seq_len: Optional[int] = None,
    autocast_dtype: Optional[torch.dtype] = None,
    banned_ids: Optional[Sequence[int]] = None,
) -> GenOutput:
    was_training = model.training
    model.eval()
    device = device or next(model.parameters()).device
    max_seq_len = max_seq_len or model.cfg.max_seq_len
    # keep the *tail* of over-long prompts (the <STATE> block lives at the end)
    max_prompt = max_seq_len - 16
    prompts = [list(p)[-max_prompt:] for p in prompts]
    B = len(prompts)
    ids, valid, positions = left_pad(prompts, pad_id, device)
    L = ids.size(1)
    max_new = max(1, min(gen.max_new_tokens, max_seq_len - L))
    stop = torch.tensor(list(stop_ids), device=device)
    temp = max(float(gen.temperature), 1e-5)

    use_ac = autocast_dtype is not None and str(device).startswith("cuda")
    ac = torch.autocast(device_type="cuda", dtype=autocast_dtype) if use_ac else torch.autocast("cpu", enabled=False)

    cache = model.new_cache()
    with ac:
        logits = model(ids, attn_mask=causal_padded_mask(valid), positions=positions, cache=cache)[:, -1, :]

    out_ids: List[List[int]] = [[] for _ in range(B)]
    out_lp: List[List[float]] = [[] for _ in range(B)]
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    key_valid = valid
    prompt_len = valid.sum(-1)  # [B]

    banned = torch.tensor(list(banned_ids), device=device) if banned_ids else None
    for t in range(max_new):
        lg = logits.float()
        if banned is not None:
            lg[:, banned] = float("-inf")
        if greedy:
            nxt = lg.argmax(-1)
            lp = F.log_softmax(lg, -1).gather(-1, nxt[:, None]).squeeze(-1)
        else:
            lg = lg / temp
            lp_full = F.log_softmax(lg, -1)
            filtered = top_k_top_p_filter(lg, gen.top_k, gen.top_p)
            nxt = torch.multinomial(F.softmax(filtered, -1), 1).squeeze(-1)
            lp = lp_full.gather(-1, nxt[:, None]).squeeze(-1)
        is_stop = torch.isin(nxt, stop)
        nxt_l, lp_l, fin_l, stop_l = nxt.tolist(), lp.tolist(), finished.tolist(), is_stop.tolist()
        for i in range(B):
            if fin_l[i]:
                continue
            out_ids[i].append(nxt_l[i])
            out_lp[i].append(lp_l[i])
        finished = finished | is_stop
        if bool(finished.all()) or t == max_new - 1:
            break
        # ---- decode one step ----
        key_valid = torch.cat([key_valid, torch.ones(B, 1, dtype=torch.bool, device=device)], dim=1)
        pos = (prompt_len + t)[:, None]
        step_in = torch.where(finished, torch.full_like(nxt, pad_id), nxt)[:, None]
        with ac:
            logits = model(step_in, attn_mask=key_valid[:, None, None, :], positions=pos, cache=cache)[:, -1, :]

    if was_training:
        model.train()
    fin = finished.tolist()
    return GenOutput(ids=out_ids, logprobs=out_lp, finished=[bool(f) for f in fin])


def decode_response(tok, ids: List[int]) -> str:
    """Decode a response, cutting at the first stop token (inclusive of its text)."""
    return tok.decode(ids)
