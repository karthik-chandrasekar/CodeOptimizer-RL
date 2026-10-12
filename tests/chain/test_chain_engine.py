"""T22 KV-cache parity, T23 last-logit / engine parity, effective-distribution bookkeeping."""
import torch

from tinyperf.common.config import GenerationConfig
from tinyperf.model.generate import generate as legacy_generate, left_pad
from tinyperf.model.transformer import causal_padded_mask
from tinyperf_chain.serve.cache import StaticKVCache, forward_static
from tinyperf_chain.serve.engine import EngineOverflow, RoleEngine, SamplingSpec, effective_logprobs


PROMPTS = [[5, 6, 7, 8, 9, 10, 11], [12, 13], [14, 15, 16, 17]]


def test_t22_static_cache_matches_full_prefix(tiny_model):
    m = tiny_model.eval()
    ids, valid, pos = left_pad(PROMPTS, 0, "cpu")
    B, L = ids.shape
    cache = StaticKVCache(m, B, L + 6, "cpu")
    with torch.no_grad():
        lg = forward_static(m, ids, pos, cache, valid, last_only=False)
        full = m(ids, attn_mask=causal_padded_mask(valid), positions=pos)
    assert torch.allclose(lg[valid], full[valid], atol=1e-5)
    # chunked continuation (offset causal mask) matches a full recomputation of the extended sequence
    nxt = torch.tensor([[20, 21], [22, 23], [24, 25]])
    npos = pos[:, -1:] + torch.tensor([[1, 2]])
    with torch.no_grad():
        lg2 = forward_static(m, nxt, npos, cache, torch.ones(B, 2, dtype=torch.bool), last_only=False)
        ext = torch.cat([ids, nxt], 1)
        evalid = torch.cat([valid, torch.ones(B, 2, dtype=torch.bool)], 1)
        epos = torch.cat([pos, npos], 1)
        full2 = m(ext, attn_mask=causal_padded_mask(evalid), positions=epos)[:, -2:]
    assert torch.allclose(lg2, full2, atol=1e-5)


def test_t22_t23_engine_parity_and_eos(tiny_model):
    m = tiny_model.eval()
    spec = SamplingSpec(greedy=True)
    stop = [m.cfg.vocab_size - 1, 3]
    a = RoleEngine(m, static_cache=True).generate(PROMPTS, 12, spec, stop, pad_id=0)
    b = RoleEngine(m, static_cache=False).generate(PROMPTS, 12, spec, stop, pad_id=0)
    legacy = legacy_generate(m, PROMPTS, GenerationConfig(max_new_tokens=12), pad_id=0, stop_ids=stop, greedy=True)
    assert [r.ids for r in a] == [r.ids for r in b] == legacy.ids
    for r in a:
        assert r.finished == any(t in stop for t in r.ids)
    ids, valid, pos = left_pad(PROMPTS, 0, "cpu")
    with torch.no_grad():
        full = m(ids, attn_mask=causal_padded_mask(valid), positions=pos)[:, -1]
        last = m(ids, attn_mask=causal_padded_mask(valid), positions=pos, last_logits_only=True)[:, -1]
    assert torch.allclose(full, last, atol=1e-6)


def test_overflow_refused_not_truncated(tiny_model):
    e = RoleEngine(tiny_model)
    try:
        e.generate([[1] * 2040], 16, SamplingSpec(greedy=True), [2], pad_id=0)
    except EngineOverflow:
        return
    raise AssertionError("over-long prompt was not refused")


def test_sampled_logprobs_follow_effective_distribution(tiny_model):
    m = tiny_model.eval()
    spec = SamplingSpec(greedy=False, temperature=0.7, top_k=12, top_p=0.9)
    outs = RoleEngine(m).generate(PROMPTS, 10, spec, [999999], pad_id=0, seeds=[1, 2, 3])
    for p, r in zip(PROMPTS, outs):
        seq = torch.tensor([p + r.ids])
        with torch.no_grad():
            lg = m(seq)[0, len(p) - 1: len(p) - 1 + len(r.ids)]
        rec = effective_logprobs(lg, seq[0, len(p):], spec)
        assert torch.allclose(rec, torch.tensor(r.logprobs), atol=1e-4)
    single = RoleEngine(m).generate([PROMPTS[1]], 10, spec, [999999], pad_id=0, seeds=[2])
    assert single[0].ids == outs[1].ids                       # request-level randomness, batch-invariant
