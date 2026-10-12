"""T01 checkpoint identity, T02 independent branches; parameter accounting (eq. 1)."""
import pytest
import torch

from tinyperf.common.config import ModelConfig
from tinyperf.model.transformer import TinyPerfLM
from tinyperf_chain.train.branch import branch, kv_bytes, param_count_formula, verify_seed


def test_param_formula_matches_spec_and_models():
    assert param_count_formula(8192, 128, 4, 512) == 2_098_304
    assert 5 * param_count_formula(8192, 128, 4, 512) == 10_491_520
    assert param_count_formula(8192, 256, 8, 1024) == 10_490_112
    assert kv_bytes(4, 128, 2048) == 4_194_304
    for cfg in (ModelConfig(vocab_size=500, n_layer=2, n_embd=32, n_head=2, mlp_dim=64),
                ModelConfig(vocab_size=8192, n_layer=4, n_embd=128, n_head=4, mlp_dim=512)):
        assert TinyPerfLM(cfg).n_params() == param_count_formula(cfg.vocab_size, cfg.n_embd, cfg.n_layer, cfg.mlp_dim)


def test_t01_identity_checks(tmp_path, cfg, tok, tok_path, tiny_model_cfg):
    m = TinyPerfLM(tiny_model_cfg)
    p = str(tmp_path / "seed.pt")
    m.save(p)
    c = cfg
    c.model.expect_vocab_size = tiny_model_cfg.vocab_size
    out = branch(p, str(tmp_path / "br"), c, tok_path)
    assert out["counts"]["formula"] == out["counts"]["recount"]
    assert set(out["roles"]) == {"M1", "M2", "M3", "M4", "M5"}
    wrong = TinyPerfLM(ModelConfig(vocab_size=tiny_model_cfg.vocab_size, n_layer=3, n_embd=32, n_head=2, mlp_dim=64))
    with pytest.raises(ValueError):
        verify_seed(wrong, c, tok)
    with pytest.raises(FileNotFoundError):
        branch(str(tmp_path / "missing.pt"), str(tmp_path / "x"), c)


def test_t02_branches_are_independent(tmp_path, cfg, tiny_model_cfg):
    m = TinyPerfLM(tiny_model_cfg)
    p = str(tmp_path / "seed.pt")
    m.save(p)
    cfg.model.expect_vocab_size = tiny_model_cfg.vocab_size
    out = branch(p, str(tmp_path / "br"), cfg, None)
    models = {r: TinyPerfLM.load(v["path"]) for r, v in out["roles"].items()}
    ptrs = {r: {n: t.data_ptr() for n, t in mm.named_parameters()} for r, mm in models.items()}
    for r, mm in models.items():
        assert mm.lm_head.weight.data_ptr() == mm.wte.weight.data_ptr()          # tied within a model
    for a in ptrs:
        for b in ptrs:
            if a < b:
                assert not set(ptrs[a].values()) & set(ptrs[b].values())         # never aliased across roles
    before = {r: {k: v.clone() for k, v in mm.state_dict().items()} for r, mm in models.items()}
    opt = torch.optim.SGD(models["M1"].parameters(), lr=0.1)
    models["M1"](torch.tensor([[1, 2, 3]])).sum().backward()
    opt.step()
    assert any(not torch.equal(before["M1"][k], v) for k, v in models["M1"].state_dict().items())
    for r in ("M2", "M3", "M4", "M5"):
        assert all(torch.equal(before[r][k], v) for k, v in models[r].state_dict().items())
