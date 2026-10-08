"""Batched multi-turn rollouts of the policy in :class:`PerfEnv`.

``B`` tasks x ``G`` rollouts each are advanced in lock-step: at every turn the
observations of all *active* episodes are batched through one ``generate``
call, then all their edits are executed in parallel sandboxes.  Every turn
keeps its prompt / response token ids so the GRPO trainer can recompute
log-probabilities; every episode ends with an :class:`EpisodeSummary`, a
terminal reward and an evaluation record (see :mod:`tinyperf.eval.metrics`).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

from tinyperf.common.config import EnvConfig, GenerationConfig, RewardConfig
from tinyperf.env.env import EpisodeSummary, PerfEnv, compute_reward
from tinyperf.env.executor import Executor
from tinyperf.env.protocol import EDIT_CLOSE, STOP_TOKEN
from tinyperf.env.task import Task
from tinyperf.eval.metrics import episode_record
from tinyperf.model.generate import generate
from tinyperf.model.tokenizer import CodeTokenizer
from tinyperf.model.transformer import TinyPerfLM
from tinyperf.train.common import autocast_dtype


@dataclass
class Turn:
    prompt_ids: List[int]              # <BOS> + observation tokens (possibly tail-truncated by `generate`)
    response_ids: List[int]            # sampled action tokens, stop token included
    obs: str
    action_text: str
    status: str                        # env status for the action
    finished: bool                     # generation ended on a stop token (else it hit max_new_tokens)
    sample_logprobs: List[float] = field(default_factory=list)


@dataclass
class Episode:
    task: Task
    group: int                         # index within the task's group of G rollouts
    turns: List[Turn]
    summary: EpisodeSummary
    reward: float
    horizon: int

    @property
    def n_response_tokens(self) -> int:
        return sum(len(t.response_ids) for t in self.turns)

    def record(self) -> Dict[str, Any]:
        r = episode_record(self.summary, self.reward, self.task, self.horizon)
        r["group"] = self.group
        r["n_turns"] = len(self.turns)
        r["response_tokens"] = self.n_response_tokens
        r["truncated_turns"] = sum(1 for t in self.turns if not t.finished)
        r["actions"] = [t.action_text for t in self.turns]
        r["best_code"] = self.summary.best_code
        return r


class Rollout:
    """Runs the policy in the environment.  Thread-safe with respect to the sandbox (one pool per Executor)."""

    def __init__(self, model: TinyPerfLM, tok: CodeTokenizer, executor: Executor, env_cfg: EnvConfig, reward_cfg: RewardConfig,
                 gen_cfg: GenerationConfig, device: str, feedback: Optional[str] = None, horizon: Optional[int] = None,
                 max_batch: int = 64):
        self.model, self.tok, self.ex = model, tok, executor
        self.env_cfg, self.reward_cfg, self.gen_cfg = env_cfg, reward_cfg, gen_cfg
        self.device = device
        self.feedback = feedback or env_cfg.feedback
        self.horizon = horizon or env_cfg.horizon
        self.max_batch = max_batch
        self.stop_ids = [tok.token_id(EDIT_CLOSE), tok.token_id(STOP_TOKEN), tok.eos_id]
        self.banned_ids = [tok.token_id(STOP_TOKEN)] if getattr(gen_cfg, "ban_stop", False) else None

    # ------------------------------------------------------------------ #
    def _prompt(self, obs: str) -> List[int]:
        ids = self.tok.encode(obs, add_bos=True)
        max_prompt = self.model.cfg.max_seq_len - 16  # same tail-truncation rule as `generate`
        return ids[-max_prompt:] if len(ids) > max_prompt else ids

    def run(self, tasks: Sequence[Task], group_size: int = 1, greedy: bool = False, seed: Optional[int] = None) -> List[Episode]:
        """Return ``len(tasks) * group_size`` episodes, ordered task-major (episode i*G+g is rollout g of task i)."""
        if seed is not None:
            torch.manual_seed(seed)
        envs: List[PerfEnv] = []
        meta: List[tuple] = []
        for ti, task in enumerate(tasks):
            for g in range(group_size):
                env = PerfEnv(self.ex, self.env_cfg, feedback=self.feedback, horizon=self.horizon)
                env.reset(task)
                envs.append(env)
                meta.append((task, g))
        turns: List[List[Turn]] = [[] for _ in envs]
        active = list(range(len(envs)))
        while active:
            # sub-batch so that generation memory stays bounded
            for start in range(0, len(active), self.max_batch):
                chunk = active[start:start + self.max_batch]
                obs = [envs[i].observation() for i in chunk]
                prompts = [self._prompt(o) for o in obs]
                gen = generate(self.model, prompts, self.gen_cfg, pad_id=self.tok.pad_id, stop_ids=self.stop_ids,
                               device=self.device, greedy=greedy, autocast_dtype=autocast_dtype(self.device), banned_ids=self.banned_ids)
                texts = [self.tok.decode(ids) for ids in gen.ids]

                def _step(item):
                    i, text = item
                    return envs[i].step(text)

                results = self.ex.map(_step, list(zip(chunk, texts)))
                for k, i in enumerate(chunk):
                    turns[i].append(Turn(prompts[k], gen.ids[k], obs[k], texts[k], results[k].record.status, gen.finished[k], gen.logprobs[k]))
            active = [i for i in active if not envs[i].done]
        episodes: List[Episode] = []
        for i, env in enumerate(envs):
            summ = env.summary()
            task, g = meta[i]
            episodes.append(Episode(task, g, turns[i], summ, compute_reward(summ, self.reward_cfg), self.horizon))
        return episodes


def sample_tasks(tasks: Sequence[Task], n: int, rng: random.Random) -> List[Task]:
    """Sample n distinct tasks (with replacement only if n > len(tasks))."""
    if n >= len(tasks):
        return list(tasks) if n == len(tasks) else [rng.choice(tasks) for _ in range(n)]
    return rng.sample(list(tasks), n)


def episodes_to_records(episodes: Sequence[Episode]) -> List[Dict[str, Any]]:
    return [e.record() for e in episodes]
