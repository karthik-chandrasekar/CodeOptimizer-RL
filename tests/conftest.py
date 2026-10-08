"""Shared fixtures: a fast executor config and a small verified task built from a seed."""
from __future__ import annotations

import random

import pytest

from tinyperf.common.config import EnvConfig, RewardConfig
from tinyperf.data.astutil import function_name, normalize
from tinyperf.data.seeds import SEEDS, make_inputs
from tinyperf.env.executor import Executor
from tinyperf.env.task import Task


def fast_env_cfg() -> EnvConfig:
    cfg = EnvConfig()
    cfg.timing.repeats = 3
    cfg.timing.warmup = 1
    cfg.limits.process_timeout_s = 60.0
    cfg.max_parallel_envs = 2
    return cfg


@pytest.fixture(scope="session")
def env_cfg() -> EnvConfig:
    return fast_env_cfg()


@pytest.fixture(scope="session")
def reward_cfg() -> RewardConfig:
    return RewardConfig()


@pytest.fixture(scope="session")
def executor(env_cfg) -> Executor:
    ex = Executor(env_cfg)
    yield ex
    ex.close()


def seed_by_name(name: str):
    for s in SEEDS:
        if s.name == name:
            return s
    raise KeyError(name)


def make_task(executor: Executor, seed_name: str = "contains_duplicate", slow_index: int = 0, perf_scale: int = 400) -> Task:
    """A Task whose C0 is a hand-written slow variant of `seed_name` (verified against the fast source)."""
    s = seed_by_name(seed_name)
    rng = random.Random(0)
    fast = normalize(s.source)
    slow = normalize(s.slow_variants[slow_index]) if s.slow_variants else fast
    correct, perf = make_inputs(s, rng, perf_scale, n_perf=3)
    b = executor.baseline(slow, function_name(slow), correct, perf)
    assert b["status"] == "ok", b
    return Task("t_" + seed_name, function_name(slow), slow, correct, perf, b["expected"], float(b["candidate_median_ns"]),
                seed_name=seed_name, families=["algorithmic"], chain=[fast, slow], fast_source=fast)


@pytest.fixture(scope="session")
def dup_task(executor) -> Task:
    return make_task(executor, "contains_duplicate")
