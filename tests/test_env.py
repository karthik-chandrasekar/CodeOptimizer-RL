import math

from tinyperf.env.env import PerfEnv, compute_reward
from tinyperf.env.protocol import EDIT_CLOSE, EDIT_OPEN, STOP_TOKEN


def edit(code: str) -> str:
    return f"{EDIT_OPEN}\n{code}\n{EDIT_CLOSE}"


def test_sandbox_accepts_equivalent_faster_code(executor, dup_task):
    r = executor.evaluate(dup_task.fast_source, dup_task, baseline_code=dup_task.source)
    assert r["status"] == "ok", r
    assert r["ratio"] < 1.0


def test_sandbox_rejects_wrong_and_unsafe_code(executor, dup_task):
    wrong = "def contains_duplicate(xs):\n    return False\n"
    assert executor.evaluate(wrong, dup_task, baseline_code=dup_task.source)["status"] == "incorrect"
    unsafe = "import os\ndef contains_duplicate(xs):\n    return len(xs) != len(set(xs))\n"
    assert executor.evaluate(unsafe, dup_task, baseline_code=dup_task.source)["status"] == "static_error"
    broken = "def contains_duplicate(xs)\n    return 1\n"
    assert executor.evaluate(broken, dup_task, baseline_code=dup_task.source)["status"] == "syntax_error"
    dunder = "def contains_duplicate(xs):\n    return xs.__class__.__mro__[-1]\n"
    assert executor.evaluate(dunder, dup_task, baseline_code=dup_task.source)["status"] == "static_error"


def test_sandbox_rejects_timeouts(executor, dup_task):
    loop = "def contains_duplicate(xs):\n    while True:\n        pass\n"
    assert executor.evaluate(loop, dup_task, baseline_code=dup_task.source)["status"] in ("timeout", "resource", "runtime_error", "worker_error")


def test_episode_rollback_recovery_and_reward(executor, env_cfg, reward_cfg, dup_task):
    env = PerfEnv(executor, env_cfg, horizon=4)
    obs = env.reset(dup_task)
    assert "best_runtime=1.000" in obs and "remaining=4" in obs
    # 1) an incorrect edit is rejected: best code unchanged, feedback says so
    res = env.step(edit("def contains_duplicate(xs):\n    return True\n"))
    assert res.record.status == "incorrect" and not res.done
    assert "last_correct=0" in res.observation and "last_status=incorrect" in res.observation
    assert env.best_code == dup_task.source
    # 2) the fast version is accepted and becomes the new best
    res = env.step(edit(dup_task.fast_source))
    assert res.record.status == "ok" and res.record.improved
    assert env.best_code.rstrip() == dup_task.fast_source.rstrip() and env.best_ratio < 1.0
    assert "last_status=improved" in res.observation
    # 3) STOP ends the episode without consuming an edit
    res = env.step(STOP_TOKEN)
    assert res.done and env.stopped
    s = env.summary()
    assert s.n_edits == 2 and s.n_invalid == 1 and s.n_correct == 1 and s.recovered and s.stopped
    r = compute_reward(s, reward_cfg)
    expected = min(reward_cfg.clip_log_speedup, math.log(1 / s.best_ratio)) - 2 * reward_cfg.step_penalty - reward_cfg.invalid_penalty
    assert abs(r - expected) < 1e-9


def test_horizon_exhaustion_and_malformed(executor, env_cfg, dup_task):
    env = PerfEnv(executor, env_cfg, horizon=2)
    env.reset(dup_task)
    res = env.step("no protocol here")
    assert res.record.status == "malformed" and not res.done
    res = env.step("still nothing")
    assert res.done and not env.stopped
    s = env.summary()
    assert s.n_edits == 2 and s.n_invalid == 2 and s.best_ratio == 1.0 and s.best_code == dup_task.source


def test_no_feedback_mode_hides_state_and_code_changes(executor, env_cfg, dup_task):
    env = PerfEnv(executor, env_cfg, feedback="none", horizon=3)
    obs0 = env.reset(dup_task)
    assert "best_runtime" not in obs0
    res = env.step(edit(dup_task.fast_source))
    assert res.record.improved                       # the env still tracks the best candidate ...
    assert dup_task.source in res.observation        # ... but the policy keeps seeing C0
    assert "last_runtime" not in res.observation and "step=1" in res.observation
