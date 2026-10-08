"""The single-function performance-optimization environment (V0).

The environment remembers far more than the policy sees: full trajectory
history, hidden inputs, best/last candidates.  The policy only ever receives a
fresh compact observation built by :func:`format_observation`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tinyperf.common.config import EnvConfig, RewardConfig
from tinyperf.env.executor import Executor
from tinyperf.env.protocol import Action, format_observation, parse_action
from tinyperf.env.task import Task


@dataclass
class StepRecord:
    step: int
    action_kind: str
    candidate: Optional[str]
    status: str            # ok|incorrect|syntax_error|static_error|load_error|timeout|resource|runtime_error|worker_error|malformed|stop
    correct: bool
    ratio: Optional[float]  # T_candidate / T_0 (None if not measured)
    improved: bool
    best_ratio_after: float
    message: str = ""
    raw_action: str = ""


@dataclass
class EpisodeSummary:
    task_id: str
    best_ratio: float
    best_code: str
    n_edits: int
    n_invalid: int
    n_correct: int
    stopped: bool           # policy emitted STOP (vs. horizon exhausted)
    recovered: bool         # an invalid/incorrect edit was followed later by an improvement
    steps: List[StepRecord] = field(default_factory=list)

    @property
    def speedup(self) -> float:
        return 1.0 / self.best_ratio if self.best_ratio > 0 else float("nan")


class PerfEnv:
    def __init__(self, executor: Executor, cfg: EnvConfig, feedback: Optional[str] = None, horizon: Optional[int] = None):
        self.ex = executor
        self.cfg = cfg
        self.feedback = feedback or cfg.feedback
        self.horizon = horizon or cfg.horizon
        self.task: Optional[Task] = None
        self.reset_state()

    # ------------------------------------------------------------------ #
    def reset_state(self) -> None:
        self.best_code = ""
        self.best_ratio = 1.0
        self.last_ratio: Optional[float] = None
        self.last_correct: Optional[bool] = None
        self.last_status = "start"
        self.step_idx = 0
        self.done = False
        self.stopped = False
        self.history: List[StepRecord] = []
        self._had_failure = False
        self._recovered = False

    def reset(self, task: Task) -> str:
        self.task = task
        self.reset_state()
        self.best_code = task.source
        self.best_profile = (task.meta or {}).get("profile0")   # profile of C0, measured at build time
        return self.observation()

    # ------------------------------------------------------------------ #
    def _state(self) -> Dict[str, Any]:
        return {
            "best_runtime": self.best_ratio,
            "last_runtime": (f"{self.last_ratio:.3f}" if self.last_ratio is not None else "n/a"),
            "last_correct": (-1 if self.last_correct is None else int(self.last_correct)),
            "last_status": self.last_status,
            "step": self.step_idx,
            "remaining": self.horizon - self.step_idx,
        }

    def observation(self) -> str:
        assert self.task is not None
        if self.feedback == "none":
            # No-feedback control: always the original code, no runtime info.
            return format_observation(self.task.source, self._state(), feedback="none")
        state = self._state()
        if self.feedback == "profile" and getattr(self, "best_profile", None):
            state["profile"] = self.best_profile
        obs = format_observation(self.best_code, state, feedback="full")
        if getattr(self.cfg, "hint", "") == "loc":  # diagnostics only: show the privileged localization hint
            from tinyperf.train.hints import add_hint, localization_hint
            obs = add_hint(obs, localization_hint(self.task, self.best_code))
        return obs

    # ------------------------------------------------------------------ #
    def step(self, action_text: str) -> "StepResult":
        assert self.task is not None and not self.done
        action = parse_action(action_text)
        return self.step_action(action)

    def step_action(self, action: Action) -> "StepResult":
        assert self.task is not None and not self.done
        task = self.task
        if action.kind == "stop":
            self.done, self.stopped = True, True
            rec = StepRecord(self.step_idx, "stop", None, "stop", True, None, False, self.best_ratio, raw_action=action.raw)
            self.history.append(rec)
            return StepResult(self.observation(), True, rec)

        self.step_idx += 1
        if action.kind == "malformed":
            rec = StepRecord(self.step_idx, "malformed", None, "malformed", False, None, False, self.best_ratio,
                             "could not parse an <EDIT>...</EDIT> block or <STOP>", raw_action=action.raw)
            self._apply_failure(rec)
        else:
            assert action.code is not None
            code = action.code
            if (task.meta or {}).get("kind") == "file":
                from tinyperf.env.files import splice_functions
                try:
                    code = splice_functions(self.best_code, action.code)
                except (SyntaxError, MemoryError, RecursionError, ValueError):
                    code = action.code  # let the sandbox report the syntax error
            resp = self.ex.evaluate(code, task, baseline_code=task.source, timing_seed=self.step_idx)
            status = resp.get("status", "worker_error")
            if status == "ok" and resp.get("ratio") is not None:
                ratio = float(resp["ratio"])
                improved = ratio < (1.0 - self.cfg.improvement_delta) * self.best_ratio
                rec = StepRecord(self.step_idx, "edit", code, "ok", True, ratio, improved, self.best_ratio, raw_action=action.raw)
                if improved:
                    self.best_code, self.best_ratio = code, ratio
                    if resp.get("candidate_group_ns") and (task.meta or {}).get("functions"):
                        from tinyperf.env.files import profile_shares
                        self.best_profile = profile_shares(resp["candidate_group_ns"], task.meta["functions"])
                    rec.best_ratio_after = ratio
                    if self._had_failure:
                        self._recovered = True
                    self.last_status = "improved"
                else:
                    self.last_status = "slower" if ratio >= self.best_ratio else "no_gain"
                self.last_ratio, self.last_correct = ratio, True
            else:
                rec = StepRecord(self.step_idx, "edit", code, status, False, None, False, self.best_ratio,
                                 str(resp.get("message", ""))[:300], raw_action=action.raw)
                self._apply_failure(rec)
        self.history.append(rec)
        if self.step_idx >= self.horizon:
            self.done = True
        return StepResult(self.observation(), self.done, rec)

    def _apply_failure(self, rec: StepRecord) -> None:
        self.last_ratio, self.last_correct = None, False
        self.last_status = rec.status
        self._had_failure = True

    # ------------------------------------------------------------------ #
    def summary(self) -> EpisodeSummary:
        assert self.task is not None
        edits = [h for h in self.history if h.action_kind != "stop"]
        return EpisodeSummary(
            task_id=self.task.task_id,
            best_ratio=self.best_ratio,
            best_code=self.best_code,
            n_edits=len(edits),
            n_invalid=sum(1 for h in edits if not h.correct),
            n_correct=sum(1 for h in edits if h.correct),
            stopped=self.stopped,
            recovered=self._recovered,
            steps=list(self.history),
        )


@dataclass
class StepResult:
    observation: str
    done: bool
    record: StepRecord


# --------------------------------------------------------------------------- #
# Reward
# --------------------------------------------------------------------------- #
def compute_reward(summary: EpisodeSummary, cfg: RewardConfig) -> float:
    """R = clip(log(T0 / T_best)) - λ·N_edits - μ·N_invalid."""
    perf = math.log(1.0 / max(summary.best_ratio, 1e-9))
    perf = max(-cfg.clip_log_speedup, min(cfg.clip_log_speedup, perf))
    r = perf - cfg.step_penalty * summary.n_edits - cfg.invalid_penalty * summary.n_invalid
    if summary.best_ratio > 1.0:
        r -= cfg.regression_penalty
    return float(r)
