"""GRPO / SFT machinery tests (skipped when torch is not installed)."""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from tinyperf.common.config import GenerationConfig, GRPOConfig, ModelConfig  # noqa: E402
from tinyperf.model.generate import generate  # noqa: E402
from tinyperf.model.transformer import TinyPerfLM  # noqa: E402
from tinyperf.train.grpo import Sample, compute_old_logprobs, group_advantages, grpo_loss, pad_samples, token_logprobs  # noqa: E402


def tiny_model(seed: int = 0) -> TinyPerfLM:
    torch.manual_seed(seed)
    return TinyPerfLM(ModelConfig(vocab_size=256, n_layer=2, n_embd=32, n_head=4, mlp_dim=64, max_seq_len=128))


def test_group_advantages_std_and_none():
    r = [1.0, 3.0, 1.0, 3.0, 0.5, 0.5, 0.5, 0.5]
    adv, n_zero = group_advantages(r, group_size=4, norm="std")
    assert n_zero == 1
    assert adv[4:] == [0.0] * 4
    assert abs(sum(adv[:4])) < 1e-9 and adv[1] > 0 > adv[0]
    assert abs(adv[1] - 1.0) < 1e-3  # unit variance within the group
    adv2, _ = group_advantages(r, 4, norm="none")
    assert adv2[:4] == [-1.0, 1.0, -1.0, 1.0]


def test_pad_samples_response_mask_alignment():
    s = [Sample([1, 2, 3], [4, 5], 0.5, 0, 0), Sample([9], [8, 7, 6], -0.5, 1, 0)]
    idx, mask = pad_samples(s, pad_id=0, device="cpu")
    assert idx.shape == (2, 5) and mask.shape == (2, 4)
    # targets idx[:,1:] = [2,3,4,5] / [8,7,6,0]; response targets are 4,5 and 8,7,6
    assert mask[0].tolist() == [False, False, True, True]
    assert mask[1].tolist() == [True, True, True, False]


def test_old_logprobs_match_generation_logprobs():
    """The trainer's recomputed log-probs line up token-for-token with what `generate` sampled."""
    m = tiny_model().eval()
    prompts = [torch.randint(1, 256, (n,)).tolist() for n in (7, 3)]
    gen = GenerationConfig(temperature=0.9, max_new_tokens=6)
    torch.manual_seed(0)
    out = generate(m, prompts, gen, pad_id=0, stop_ids=[255], device="cpu")
    samples = [Sample(p, ids, 1.0, i, 0) for i, (p, ids) in enumerate(zip(prompts, out.ids))]
    old = compute_old_logprobs(m, samples, pad_id=0, temperature=gen.temperature, device="cpu", minibatch=2)
    for o, lps in zip(old, out.logprobs):
        assert torch.allclose(o, torch.tensor(lps), atol=1e-4)


def test_grpo_loss_gradient_sign_and_clipping():
    m = tiny_model()
    samples = [Sample([1, 2, 3], [4, 5, 6], +1.0, 0, 0), Sample([1, 2, 3], [7, 8], -1.0, 1, 0)]
    cfg = GRPOConfig(clip_eps=0.2, clip_eps_high=0.28, kl_beta=0.0, loss_agg="token")
    idx, mask = pad_samples(samples, 0, "cpu")
    old, _ = token_logprobs(m, idx, 1.0, "cpu")
    old = old.detach()
    lp, _ = token_logprobs(m, idx, 1.0, "cpu")
    adv = torch.tensor([1.0, -1.0])
    loss, st = grpo_loss(lp, old, adv, mask, cfg, total_tokens=int(mask.sum()))
    assert st["clip_frac"] == 0.0 and abs(st["ratio_mean"] - 1.0) < 1e-6
    # on-policy: loss == -mean(adv over response tokens)
    expect = -float((adv[:, None] * mask.float()).sum() / mask.sum())
    assert abs(float(loss.detach()) - expect) < 1e-5
    # one SGD step should raise the log-prob of the positive-advantage response and lower the negative one
    opt = torch.optim.SGD(m.parameters(), lr=0.5)
    opt.zero_grad()
    loss.backward()
    opt.step()
    new, _ = token_logprobs(m, idx, 1.0, "cpu")
    d = ((new - old) * mask.float()).sum(1)
    assert d[0] > 0 > d[1], d
    # clipping engages when the ratio leaves [1-eps, 1+eps_high]
    lp2, _ = token_logprobs(m, idx, 1.0, "cpu")
    far_old = old - 1.0  # pretend the old policy was e^1 less likely everywhere -> ratio ~ e
    loss2, st2 = grpo_loss(lp2, far_old, adv, mask, cfg, total_tokens=int(mask.sum()))
    assert st2["clip_frac"] > 0.5


def test_kl_term_is_zero_at_reference_and_positive_away():
    cfg = GRPOConfig(kl_beta=0.1)
    lp = torch.zeros(1, 4)
    mask = torch.ones(1, 4, dtype=torch.bool)
    adv = torch.zeros(1)
    loss0, st0 = grpo_loss(lp, lp, adv, mask, cfg, ref_lp=lp, total_tokens=4)
    assert abs(float(loss0)) < 1e-9 and st0["kl_ref"] == 0.0
    loss1, st1 = grpo_loss(lp, lp, adv, mask, cfg, ref_lp=lp - 0.5, total_tokens=4)
    assert float(loss1) > 0 and st1["kl_ref"] > 0


def test_sft_masks_prompt_tokens():
    from tinyperf.train.sft import Example, collate

    ex = Example(ids=[1, 2, 3, 4, 5], n_prompt=3, task_id="t")
    x, y = collate([ex], pad_id=0, device="cpu")
    assert x.tolist() == [[1, 2, 3, 4]]
    assert y.tolist() == [[-100, -100, 4, 5]]  # loss only on the action tokens (4, 5)


def test_reward_shape():
    from tinyperf.common.config import RewardConfig
    from tinyperf.env.env import EpisodeSummary, compute_reward

    cfg = RewardConfig(step_penalty=0.01, invalid_penalty=0.1, clip_log_speedup=4.0)
    s = EpisodeSummary("t", best_ratio=0.5, best_code="", n_edits=2, n_invalid=1, n_correct=1, stopped=True, recovered=True)
    assert abs(compute_reward(s, cfg) - (math.log(2.0) - 0.02 - 0.1)) < 1e-9
    s0 = EpisodeSummary("t", best_ratio=1.0, best_code="", n_edits=0, n_invalid=0, n_correct=0, stopped=True, recovered=False)
    assert compute_reward(s0, cfg) == 0.0
    huge = EpisodeSummary("t", best_ratio=1e-6, best_code="", n_edits=1, n_invalid=0, n_correct=1, stopped=True, recovered=False)
    assert compute_reward(huge, cfg) <= 4.0
