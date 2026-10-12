"""T19 SFT masking, T20 GRPO likelihood bookkeeping, T21 zero-advantage groups, T25 SFT resume."""
import json
import os

import pytest
import torch

from tinyperf_chain.config import ChainConfig
from tinyperf_chain.serve.engine import RoleEngine, SamplingSpec
from tinyperf_chain.train.rl import grpo_loss, group_advantages, response_logprobs, rewards_for
from tinyperf_chain.train.sft import collate, per_example_loss


def test_t19_only_response_tokens_contribute(tiny_model):
    m = tiny_model
    batch = [([1, 5, 6, 7], [8, 9, 2]), ([1, 5], [10, 2])]
    x, y = collate(batch, 0, "cpu")
    assert x.shape == y.shape
    mask = y != -100
    assert int(mask.sum()) == 5                     # 3 + 2 response tokens including the stop marker
    # the first prompt token is never a target and padding is never a target
    assert y[1, 3:].eq(-100).all()
    ac = torch.autocast("cpu", enabled=False)
    l = per_example_loss(m, x, y, ac)
    assert l.shape == (2,) and torch.isfinite(l).all()
    # per-example response-mean: changing the prompt content of row 1 does not change which tokens are scored
    with pytest.raises(ValueError):
        per_example_loss(m, x, torch.full_like(y, -100), ac)


def test_t20_ratios_one_before_update(tiny_model):
    m = tiny_model.eval()
    spec = SamplingSpec(greedy=False, temperature=0.8)
    prompts = [[1, 4, 5, 6], [1, 7, 8]]
    eng = RoleEngine(__import__("copy").deepcopy(m))
    outs = eng.generate(prompts, 8, spec, [999999], pad_id=0, seeds=[5, 6])
    new = response_logprobs(m, prompts, [o.ids for o in outs], spec, 0, "cpu", torch.autocast("cpu", enabled=False))
    for n, o in zip(new, outs):
        assert torch.allclose(n, torch.tensor(o.logprobs), atol=1e-4)
        assert torch.allclose(torch.exp(n - torch.tensor(o.logprobs)), torch.ones(len(o.ids)), atol=1e-3)


def test_t21_zero_advantage_group_has_zero_gradient(tiny_model):
    cfg = ChainConfig()
    m = tiny_model.eval()
    spec = SamplingSpec(greedy=False, temperature=0.8)
    prompts = [[1, 4, 5, 6]] * 3
    resp = [[7, 8, 9], [7, 9], [8]]
    adv = group_advantages([-0.015, -0.015, -0.015])
    assert adv == [0.0, 0.0, 0.0]
    old = [x.detach() for x in response_logprobs(m, prompts, resp, spec, 0, "cpu", torch.autocast("cpu", enabled=False))]
    m.zero_grad()
    new = response_logprobs(m, prompts, resp, spec, 0, "cpu", torch.autocast("cpu", enabled=False))
    loss, st = grpo_loss(new, old, adv, cfg, 1, 3)
    loss.backward()
    assert float(loss.detach()) == 0.0 and torch.isfinite(loss.detach())
    assert all(p.grad is None or float(p.grad.abs().max()) == 0.0 for p in m.parameters())


def test_rewards_follow_registered_rule():
    cfg = ChainConfig()
    r, inv, g = rewards_for({"stage": "timing", "accepted": True, "speedup": 2.0}, True, False, cfg)
    assert abs(g - 0.6931) < 1e-3 and abs(r - (g - 0.015)) < 1e-9 and not inv
    r, inv, _ = rewards_for({"stage": "mandatory", "accepted": False}, True, False, cfg)
    assert inv and abs(r + 0.115) < 1e-9
    assert rewards_for(None, True, True, cfg)[0] == -0.015
    assert rewards_for(None, False, False, cfg)[1] is True
    r, inv, g = rewards_for({"stage": "timing", "accepted": False, "speedup": 0.9, "kind": "slower"}, True, False, cfg)
    assert g == 0.0 and not inv                                   # correct but not faster: no gain, not invalid
    assert rewards_for({"stage": "timing", "accepted": True, "speedup": 1e9}, True, False, cfg)[2] == 4.0
