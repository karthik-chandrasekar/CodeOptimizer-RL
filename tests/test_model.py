"""Model / generation tests (skipped when torch is not installed)."""
from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from tinyperf.common.config import GenerationConfig, ModelConfig  # noqa: E402
from tinyperf.env.protocol import EDIT_CLOSE, STOP_TOKEN  # noqa: E402
from tinyperf.model.generate import generate, left_pad, top_k_top_p_filter  # noqa: E402
from tinyperf.model.transformer import TinyPerfLM, causal_padded_mask  # noqa: E402

TOK_PATH = os.environ.get("TINYPERF_TEST_TOKENIZER", "")


def tiny_model(seed: int = 0) -> TinyPerfLM:
    torch.manual_seed(seed)
    return TinyPerfLM(ModelConfig(vocab_size=512, n_layer=2, n_embd=32, n_head=4, mlp_dim=64, max_seq_len=128)).eval()


def test_forward_shapes_and_param_count():
    m = tiny_model()
    x = torch.randint(0, 512, (3, 11))
    assert m(x).shape == (3, 11, 512)
    assert m.n_params() > m.n_params(non_embedding=True) > 0


def test_left_padding_is_invariant():
    m = tiny_model()
    p1 = list(torch.randint(0, 512, (17,)).tolist())
    p2 = list(torch.randint(0, 512, (6,)).tolist())
    ids, valid, pos = left_pad([p1, p2], pad_id=0, device="cpu")
    with torch.no_grad():
        batched = m(ids, attn_mask=causal_padded_mask(valid), positions=pos)
        single1, single2 = m(torch.tensor([p1])), m(torch.tensor([p2]))
    assert torch.allclose(batched[0], single1[0], atol=1e-5)
    assert torch.allclose(batched[1, -len(p2):], single2[0], atol=1e-5)


def test_kv_cache_generation_matches_full_forward():
    """Sampled-token log-probs from the cached decoder equal a from-scratch forward on prompt+response."""
    m = tiny_model()
    prompts = [torch.randint(1, 512, (n,)).tolist() for n in (9, 4, 13)]
    gen = GenerationConfig(temperature=0.7, max_new_tokens=10)
    torch.manual_seed(1)
    out = generate(m, prompts, gen, pad_id=0, stop_ids=[511], device="cpu")
    for p, ids, lps in zip(prompts, out.ids, out.logprobs):
        seq = torch.tensor([p + ids])
        with torch.no_grad():
            lg = m(seq)[0, len(p) - 1:len(p) - 1 + len(ids)].float() / gen.temperature
        ref = torch.log_softmax(lg, -1).gather(-1, torch.tensor(ids)[:, None]).squeeze(-1)
        assert torch.allclose(ref, torch.tensor(lps), atol=1e-4), (ref, lps)


def test_generation_stops_on_stop_token_and_respects_length():
    m = tiny_model()

    class Biased(torch.nn.Module):  # force token 7 -> every row stops after one token
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            out = self.inner(x).clone()
            out[..., 7] += 100.0
            return out

    m.lm_head = Biased(m.lm_head)
    out = generate(m, [[1, 2, 3], [4, 5]], GenerationConfig(temperature=1.0, max_new_tokens=8), pad_id=0, stop_ids=[7], device="cpu")
    assert out.ids == [[7], [7]] and out.finished == [True, True]
    out2 = generate(m, [[1, 2, 3]], GenerationConfig(temperature=1.0, max_new_tokens=5), pad_id=0, stop_ids=[9999], device="cpu")
    assert len(out2.ids[0]) == 5 and out2.finished == [False]


def test_top_k_top_p_filter():
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    k = top_k_top_p_filter(logits, top_k=2)
    assert torch.isinf(k[0, :2]).all() and not torch.isinf(k[0, 2:]).any()
    p = top_k_top_p_filter(logits, top_p=0.5)
    assert not torch.isinf(p[0, 3]) and torch.isinf(p[0, 0])


def test_checkpoint_roundtrip(tmp_path):
    m = tiny_model()
    path = str(tmp_path / "m.pt")
    m.save(path, extra={"step": 3})
    m2 = TinyPerfLM.load(path)
    x = torch.randint(0, 512, (1, 5))
    with torch.no_grad():
        assert torch.allclose(m(x), m2(x))


@pytest.mark.skipif(not TOK_PATH or not os.path.exists(TOK_PATH), reason="set TINYPERF_TEST_TOKENIZER to a trained tokenizer.json")
def test_protocol_tokens_are_single_tokens():
    from tinyperf.model.tokenizer import CodeTokenizer

    tok = CodeTokenizer(TOK_PATH)
    for t in (EDIT_CLOSE, STOP_TOKEN):
        assert tok.encode(t) == [tok.token_id(t)]


def test_banned_ids_are_never_sampled():
    m = tiny_model()

    class Biased(torch.nn.Module):  # the model strongly prefers token 7 ...
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            out = self.inner(x).clone()
            out[..., 7] += 100.0
            return out

    m.lm_head = Biased(m.lm_head)
    out = generate(m, [[1, 2, 3]] * 4, GenerationConfig(temperature=1.0, max_new_tokens=6), pad_id=0, stop_ids=[9999],
                   device="cpu", banned_ids=[7])                  # ... but it is banned
    assert all(7 not in ids for ids in out.ids) and all(len(ids) == 6 for ids in out.ids)
