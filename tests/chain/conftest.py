"""Fixtures for the TinyPerf-Chain acceptance tests (Appendix A, T01-T25).

Everything runs on CPU: a tiny BPE tokenizer trained on the fly, a tiny transformer, real sandbox subprocesses, and
a few tasks built from the hand-written seed library with small workloads and few timing blocks.
"""
from __future__ import annotations

import os
import random

import pytest

from tinyperf_chain.config import ChainConfig


def fast_cfg(root: str) -> ChainConfig:
    cfg = ChainConfig()
    cfg.paths.root = root
    v = cfg.verification
    v.train_timing_blocks = 3
    v.development_timing_blocks = 3
    v.audit_timing_blocks = 5
    v.block_target_ms = 3.0
    v.timing_bootstrap_draws = 200
    cfg.data.perf_scale = 400
    cfg.data.n_perf = 2
    cfg.serving.correctness_workers = 2
    cfg.model.expect_n_embd, cfg.model.expect_n_layer, cfg.model.expect_n_head, cfg.model.expect_mlp_dim = 32, 2, 2, 64
    return cfg


@pytest.fixture(scope="session")
def chain_root(tmp_path_factory) -> str:
    return str(tmp_path_factory.mktemp("chain"))


@pytest.fixture(scope="session")
def cfg(chain_root) -> ChainConfig:
    return fast_cfg(chain_root)


@pytest.fixture(scope="session")
def verifier(cfg):
    from tinyperf_chain.adapters.verifier import Verifier

    return Verifier(cfg)


@pytest.fixture(scope="session")
def tok_path(tmp_path_factory) -> str:
    from tinyperf.data.seeds import SEEDS
    from tinyperf.model.tokenizer import train_tokenizer

    texts = [s.source for s in SEEDS] + [v for s in SEEDS for v in s.slow_variants]
    texts += ["role=M3 v=chain_v1\nfunction=f00\nplan=PLAN membership_index; src=a; loop=n00\ncontract=xs:list[int]\n"] * 5
    p = str(tmp_path_factory.mktemp("tok") / "tokenizer.json")
    train_tokenizer(texts, 700, p)
    return p


@pytest.fixture(scope="session")
def tok(tok_path):
    from tinyperf.model.tokenizer import CodeTokenizer

    return CodeTokenizer(tok_path)


@pytest.fixture(scope="session")
def tiny_model_cfg(tok):
    from tinyperf.common.config import ModelConfig

    return ModelConfig(name="tiny", vocab_size=tok.vocab_size, n_layer=2, n_embd=32, n_head=2, mlp_dim=64, max_seq_len=2048)


@pytest.fixture()
def tiny_model(tiny_model_cfg):
    import torch

    from tinyperf.model.transformer import TinyPerfLM

    torch.manual_seed(0)
    return TinyPerfLM(tiny_model_cfg)


def _unit(cfg, verifier, seed_name: str, split: str = "dev", family_pref=None):
    from tinyperf.data.seeds import SEED_BY_NAME
    from tinyperf_chain.data.build_tasks import TaskBuilder, variants_for

    b = TaskBuilder(cfg, verifier, min_speedup=1.2)
    vs = variants_for(SEED_BY_NAME[seed_name], f"c_{seed_name}", 8, random.Random(0))
    if family_pref:
        vs.sort(key=lambda v: v.family != family_pref)
    for i, v in enumerate(vs):
        u = b.build_unit(v, split, i)
        if u is not None:
            return u, b
    raise RuntimeError(f"no eligible unit for {seed_name}: {dict(b.elig.counts)}")


@pytest.fixture(scope="session")
def unit_count_common(cfg, verifier):
    return _unit(cfg, verifier, "count_common", family_pref="set_to_list_membership")[0]


@pytest.fixture(scope="session")
def unit_join(cfg, verifier):
    return _unit(cfg, verifier, "join_tokens")[0]


@pytest.fixture(scope="session")
def file_task(cfg, verifier, unit_count_common, unit_join):
    u3, b = _unit(cfg, verifier, "unique_preserve_order")
    r = b.build_file([unit_count_common, unit_join, u3], [0, 1], "dev", 0, "g_test")
    assert r is not None, dict(b.elig.counts)
    return r
