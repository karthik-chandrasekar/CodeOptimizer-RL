"""Stage 3 - GRPO in the execution-feedback environment.

    python -m tinyperf.train.grpo --config configs/grpo.yaml [env.horizon=4 env.feedback=none ...]

Algorithm (per optimizer step)
------------------------------
1. sample B functions; roll out G independent episodes per function
   (:class:`tinyperf.train.rollout.Rollout`), each up to H edits + STOP;
2. terminal reward per episode (:func:`tinyperf.env.env.compute_reward`),
   group-relative advantage  A = (R - mean_G) / (std_G + eps)  (``adv_norm=std``)
   or mean-centred only (``adv_norm=none``); A is broadcast to every action
   token of every turn of that episode (observation tokens get no loss);
3. old log-probs are recomputed with a no-grad forward pass *before* any update
   (numerically identical to the training forward, so ratio == 1 on the first
   minibatch), then ``ppo_epochs`` passes of PPO-clipped updates with
   asymmetric (DAPO) clipping and token-level or sequence-level aggregation;
4. optional k3 KL penalty against the frozen SFT policy.
"""
from __future__ import annotations

import copy
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from tinyperf.common.config import Config, GRPOConfig, dump_config, parse_cli
from tinyperf.common.utils import MetricLogger, get_logger, resolve_device, seed_everything, write_jsonl
from tinyperf.env.executor import Executor
from tinyperf.env.task import Task, load_tasks
from tinyperf.eval.metrics import compute_metrics
from tinyperf.model.transformer import TinyPerfLM
from tinyperf.train.common import autocast_ctx, build_optimizer, load_model_and_tokenizer
from tinyperf.train.rollout import Episode, Rollout, episodes_to_records, sample_tasks

log = get_logger("grpo")


# --------------------------------------------------------------------------- #
# Advantages
# --------------------------------------------------------------------------- #
def group_advantages(rewards: Sequence[float], group_size: int, norm: str = "std", eps: float = 1e-6) -> Tuple[List[float], int]:
    """Group-relative advantages. Returns (advantages, n_zero_variance_groups). Rewards are group-major."""
    assert len(rewards) % group_size == 0
    adv: List[float] = []
    n_zero = 0
    for g in range(0, len(rewards), group_size):
        r = list(rewards[g:g + group_size])
        mu = sum(r) / len(r)
        centred = [x - mu for x in r]
        if norm == "std":
            var = sum(c * c for c in centred) / len(r)
            sd = math.sqrt(var)
            if sd < 1e-8:
                n_zero += 1
                adv.extend([0.0] * len(r))
            else:
                adv.extend([c / (sd + eps) for c in centred])
        else:
            if max(abs(c) for c in centred) < 1e-8:
                n_zero += 1
            adv.extend(centred)
    return adv, n_zero


@dataclass
class Sample:
    prompt_ids: List[int]
    response_ids: List[int]
    advantage: float
    episode: int
    turn: int


@dataclass
class OPSDItem:
    student_ids: List[int]     # <BOS> + observation (what the policy saw)
    teacher_ids: List[int]     # <BOS> + observation with a hint line (what the teacher sees)
    response_ids: List[int]    # the policy's own sampled action


def build_opsd_items(episodes, tok, max_seq_len: int, mode: str, on: str, max_items: int, rng,
                     max_tokens: int = 0, skip_truncated: bool = True) -> List[OPSDItem]:
    """One item per selected turn, from ALL turns (including zero-advantage groups, where GRPO learns nothing)."""
    from tinyperf.train.hints import add_hint, build_hint, remaining_slow, code_from_obs

    items: List[OPSDItem] = []
    for ep in episodes:
        steps = ep.summary.steps
        for ti, t in enumerate(ep.turns):
            if ti >= len(steps) or not t.response_ids or (skip_truncated and not getattr(t, "finished", True)):
                continue
            rec = steps[ti]
            if on == "problematic":
                bad = rec.status != "ok" or not rec.improved
                if rec.action_kind == "stop":
                    bad = bool(remaining_slow(ep.task, code_from_obs(t.obs)))
                if not bad:
                    continue
            hint = build_hint(ep.task, t.obs, rec.action_kind, rec.status, rec.improved, t.action_text, mode)
            if not hint:
                continue
            budget = max_seq_len - len(t.response_ids) - 1
            if budget < 16:
                continue
            teacher = tok.encode(add_hint(t.obs, hint), add_bos=True)[-budget:]
            student = list(t.prompt_ids)[-budget:]
            resp = list(t.response_ids)[:max_tokens] if max_tokens > 0 else list(t.response_ids)
            items.append(OPSDItem(student, teacher, resp))
    if len(items) > max_items:
        items = rng.sample(items, max_items)
    return items


def _response_logits(model, prompts: Sequence[List[int]], responses: Sequence[List[int]], pad_id: int, device: str) -> torch.Tensor:
    """Logits at the positions that predict each response token, concatenated: [sum(len(r)), V]."""
    seqs = [list(p) + list(r) for p, r in zip(prompts, responses)]
    idx = torch.full((len(seqs), max(len(x) for x in seqs)), pad_id, dtype=torch.long)
    for j, x in enumerate(seqs):
        idx[j, :len(x)] = torch.tensor(x, dtype=torch.long)
    with autocast_ctx(device):
        logits = model(idx.to(device))       # right padding: causal attention never sees the pads
    return torch.cat([logits[j, len(p) - 1:len(p) - 1 + len(r)] for j, (p, r) in enumerate(zip(prompts, responses))], 0).float()


def opsd_reverse_kl(model, items: Sequence[OPSDItem], pad_id: int, device: str, temperature: float = 1.0, teacher=None):
    """Sum over response tokens of KL(student || teacher), exact over the vocabulary, on the student's own samples.
    The teacher is the same model with the hint, detached: gradients flow only through the student."""
    s = _response_logits(model, [it.student_ids for it in items], [it.response_ids for it in items], pad_id, device)
    with torch.no_grad():
        t = _response_logits(teacher if teacher is not None else model, [it.teacher_ids for it in items],
                             [it.response_ids for it in items], pad_id, device)
    ls, lt = torch.log_softmax(s / temperature, -1), torch.log_softmax(t / temperature, -1)
    kl = (ls.exp() * (ls - lt)).sum(-1)
    return kl.sum(), kl.numel()


def episodes_to_samples(episodes: Sequence[Episode], advantages: Sequence[float], drop_zero: bool = True) -> List[Sample]:
    out: List[Sample] = []
    for ei, (ep, a) in enumerate(zip(episodes, advantages)):
        if drop_zero and abs(a) < 1e-12:
            continue
        for ti, t in enumerate(ep.turns):
            if t.response_ids:
                out.append(Sample(t.prompt_ids, t.response_ids, float(a), ei, ti))
    return out


# --------------------------------------------------------------------------- #
# Log-probs
# --------------------------------------------------------------------------- #
def pad_samples(samples: Sequence[Sample], pad_id: int, device: str):
    """Right-pad prompt+response. Returns idx [N,T], resp_mask [N,T-1] (True where the *target* token is a response token)."""
    seqs = [s.prompt_ids + s.response_ids for s in samples]
    T = max(len(x) for x in seqs)
    N = len(seqs)
    idx = torch.full((N, T), pad_id, dtype=torch.long)
    mask = torch.zeros((N, T - 1), dtype=torch.bool)
    for i, (s, seq) in enumerate(zip(samples, seqs)):
        idx[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        p = len(s.prompt_ids)
        mask[i, p - 1:len(seq) - 1] = True
    return idx.to(device), mask.to(device)


def token_logprobs(model: TinyPerfLM, idx: torch.Tensor, temperature: float, device: str, with_entropy: bool = False):
    """Log-probs of idx[:, 1:] given idx[:, :-1] under softmax(logits / temperature). Returns [N, T-1] (and entropy)."""
    with autocast_ctx(device):
        logits = model(idx[:, :-1])
    logits = logits.float() / max(temperature, 1e-5)
    logp = F.log_softmax(logits, dim=-1)
    tgt = idx[:, 1:]
    lp = logp.gather(-1, tgt[..., None]).squeeze(-1)
    if with_entropy:
        ent = -(logp.exp() * logp).sum(-1)
        return lp, ent
    return lp, None


@torch.no_grad()
def compute_old_logprobs(model: TinyPerfLM, samples: List[Sample], pad_id: int, temperature: float, device: str, minibatch: int) -> List[torch.Tensor]:
    model.eval()
    out: List[torch.Tensor] = []
    for i in range(0, len(samples), minibatch):
        mb = samples[i:i + minibatch]
        idx, mask = pad_samples(mb, pad_id, device)
        lp, _ = token_logprobs(model, idx, temperature, device)
        for j, s in enumerate(mb):
            p, n = len(s.prompt_ids), len(s.response_ids)
            out.append(lp[j, p - 1:p - 1 + n].detach().cpu())
    model.train()
    return out


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def grpo_loss(lp: torch.Tensor, old_lp: torch.Tensor, adv: torch.Tensor, mask: torch.Tensor, cfg: GRPOConfig,
              ref_lp: Optional[torch.Tensor] = None, total_tokens: int = 1, total_seqs: int = 1) -> Tuple[torch.Tensor, Dict[str, float]]:
    """lp/old_lp/ref_lp: [N, T-1]; adv: [N]; mask: [N, T-1]. Loss is scaled so that summing over minibatches gives the
    global token-mean (``loss_agg=token``) or the global sequence-mean (``loss_agg=sequence``)."""
    m = mask.float()
    a = adv[:, None]
    log_ratio = (lp - old_lp) * m
    ratio = log_ratio.exp()
    lo, hi = 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps_high
    surr1 = ratio * a
    surr2 = ratio.clamp(lo, hi) * a
    pg = -torch.min(surr1, surr2)
    per_tok = pg
    if cfg.kl_beta > 0 and ref_lp is not None:
        d = (ref_lp - lp) * m
        k3 = d.exp() - d - 1.0
        per_tok = per_tok + cfg.kl_beta * k3
    else:
        k3 = torch.zeros_like(per_tok)
    if cfg.loss_agg == "sequence":
        per_seq = (per_tok * m).sum(1) / m.sum(1).clamp_min(1.0)
        loss = per_seq.sum() / max(total_seqs, 1)
    else:
        loss = (per_tok * m).sum() / max(total_tokens, 1)
    with torch.no_grad():
        n = m.sum().clamp_min(1.0)
        clipped = (((ratio < lo) | (ratio > hi)) & (mask)).float().sum() / n
        approx_kl = ((old_lp - lp) * m).sum() / n
        stats = {"clip_frac": float(clipped), "approx_kl_old": float(approx_kl), "kl_ref": float((k3 * m).sum() / n),
                 "ratio_mean": float((ratio * m).sum() / n)}
    return loss, stats


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class GRPOTrainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        g = cfg.grpo
        self.device = resolve_device(cfg.device)
        seed_everything(g.seed)
        os.makedirs(g.out_dir, exist_ok=True)
        dump_config(cfg, os.path.join(g.out_dir, "config.yaml"))
        self.model, self.tok = load_model_and_tokenizer(cfg, g.init_from, self.device)
        self.opsd_teacher = None
        if g.opsd_beta > 0 and g.opsd_teacher == "frozen":      # fixed copy of the initial model: cannot drift with the student
            self.opsd_teacher = copy.deepcopy(self.model).eval()
            for p_ in self.opsd_teacher.parameters():
                p_.requires_grad_(False)
        self.model.train()
        self.ref: Optional[TinyPerfLM] = None
        if g.kl_beta > 0:
            self.ref = copy.deepcopy(self.model).eval()
            for p in self.ref.parameters():
                p.requires_grad_(False)
        self.opt = build_optimizer(self.model, g.lr, g.weight_decay)
        self.ex = Executor(cfg.env)
        self.rollout = Rollout(self.model, self.tok, self.ex, cfg.env, cfg.reward, g.generation, self.device,
                               max_batch=max(1, min(64, g.group_size * g.tasks_per_step)))
        self.eval_rollout = Rollout(self.model, self.tok, self.ex, cfg.env, cfg.reward, cfg.eval.generation, self.device,
                                    max_batch=64)
        self.train_tasks = load_tasks(g.tasks)
        if not self.train_tasks:
            raise RuntimeError(f"no tasks in {g.tasks}")
        val_path = g.val_tasks or g.tasks.replace("train.jsonl", "val.jsonl")
        self.val_tasks = load_tasks(val_path) if val_path != g.tasks and os.path.exists(val_path) else self.train_tasks[: g.n_eval_tasks]
        self.rng = random.Random(g.seed)
        self.ml = MetricLogger(g.out_dir, cfg.wandb_project, f"{cfg.run_name}-grpo", config=None)
        self.step = 0
        self.best_eval = -float("inf")
        self._order: List[Task] = []
        last = os.path.join(g.out_dir, "last.pt")
        if os.path.exists(last):
            ck = torch.load(last, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ck["state_dict"])
            self.opt.load_state_dict(ck["extra"]["optimizer"])
            self.step = int(ck["extra"]["step"])
            self.best_eval = float(ck["extra"].get("best_eval", -float("inf")))
            log.info(f"resumed from {last} at step {self.step}")
        log.info(f"GRPO: {len(self.train_tasks)} train tasks, {len(self.val_tasks)} val tasks, device={self.device}, "
                 f"B={g.tasks_per_step} G={g.group_size} H={cfg.env.horizon} feedback={cfg.env.feedback}")

    # ------------------------------------------------------------------ #
    def next_tasks(self) -> List[Task]:
        b = self.cfg.grpo.tasks_per_step
        if len(self._order) < b:
            fresh = list(self.train_tasks)
            self.rng.shuffle(fresh)
            self._order.extend(fresh)
        out, self._order = self._order[:b], self._order[b:]
        return out

    def train_step(self) -> Dict[str, float]:
        g = self.cfg.grpo
        t0 = time.time()
        tasks = self.next_tasks()
        episodes = self.rollout.run(tasks, group_size=g.group_size)
        t_roll = time.time() - t0
        rewards = [e.reward for e in episodes]
        adv, n_zero = group_advantages(rewards, g.group_size, g.adv_norm)
        samples = episodes_to_samples(episodes, adv)
        records = episodes_to_records(episodes)
        em = compute_metrics(records)
        metrics: Dict[str, float] = {
            "reward": float(sum(rewards) / len(rewards)),
            "reward_std": float(torch.tensor(rewards).std()) if len(rewards) > 1 else 0.0,
            "success_rate": em["success_rate"], "gm_speedup": em["gm_speedup"], "mean_edits": em["mean_edits"],
            "invalid_edit_rate": em["invalid_edit_rate"], "stop_rate": em["stop_rate"], "malformed_rate": em["malformed_rate"],
            "recovery_rate": em.get("recovery_rate", float("nan")),
            "zero_var_groups": n_zero / max(1, len(episodes) // g.group_size),
            "n_samples": len(samples), "response_tokens": sum(len(s.response_ids) for s in samples),
            "truncated_turns": sum(1 for e in episodes for t in e.turns if not t.finished),
            "t_rollout": t_roll,
        }
        if g.opsd_only:
            metrics["pg_samples_unused"] = len(samples)
            samples = []          # pure OPSD: rollouts only supply the student's samples and the sandbox verdicts for hints
        opsd_items = []
        if g.opsd_beta > 0:
            opsd_items = build_opsd_items(episodes, self.tok, self.model.cfg.max_seq_len, g.opsd_hints, g.opsd_on,
                                          g.opsd_max_items, random.Random(len(rewards) * 7919 + int(t0)),
                                          g.opsd_max_tokens, g.opsd_skip_truncated)
            metrics["opsd_items"] = len(opsd_items)
        if not samples and not opsd_items:
            metrics.update({"loss": 0.0, "skipped": 1.0})
            return metrics
        # old (and reference) log-probs before any update
        temp = g.generation.temperature
        old = compute_old_logprobs(self.model, samples, self.tok.pad_id, temp, self.device, g.minibatch_size) if samples else []
        ref = compute_old_logprobs(self.ref, samples, self.tok.pad_id, temp, self.device, g.minibatch_size) if (self.ref is not None and samples) else None
        total_tokens = max(1, sum(len(s.response_ids) for s in samples))
        total_seqs = max(1, len(samples))
        t1 = time.time()
        agg: Dict[str, List[float]] = {"loss": [], "clip_frac": [], "approx_kl_old": [], "kl_ref": [], "entropy": [], "grad_norm": []}
        order = list(range(len(samples)))
        for _ in range(g.ppo_epochs):
            self.rng.shuffle(order)
            self.opt.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for i in range(0, len(order), g.minibatch_size):
                ids = order[i:i + g.minibatch_size]
                mb = [samples[j] for j in ids]
                idx, mask = pad_samples(mb, self.tok.pad_id, self.device)
                lp, ent = token_logprobs(self.model, idx, temp, self.device, with_entropy=True)
                old_lp = torch.zeros_like(lp)
                ref_lp = torch.zeros_like(lp) if ref is not None else None
                for j, sid in enumerate(ids):
                    p, n = len(samples[sid].prompt_ids), len(samples[sid].response_ids)
                    old_lp[j, p - 1:p - 1 + n] = old[sid].to(self.device)
                    if ref_lp is not None:
                        ref_lp[j, p - 1:p - 1 + n] = ref[sid].to(self.device)
                advs = torch.tensor([s.advantage for s in mb], device=self.device)
                loss, st = grpo_loss(lp, old_lp, advs, mask, g, ref_lp, total_tokens, total_seqs)
                loss.backward()
                loss_sum += float(loss.detach())
                for k, v in st.items():
                    agg.setdefault(k, []).append(v)
                agg["entropy"].append(float((ent * mask.float()).sum() / mask.float().sum().clamp_min(1.0)))
            if opsd_items:
                opsd_tokens = sum(len(it.response_ids) for it in opsd_items)
                for i in range(0, len(opsd_items), g.minibatch_size):
                    kl_sum, n_tok = opsd_reverse_kl(self.model, opsd_items[i:i + g.minibatch_size], self.tok.pad_id, self.device, temp,
                                                    teacher=self.opsd_teacher)
                    loss_o = g.opsd_beta * kl_sum / opsd_tokens
                    loss_o.backward()
                    loss_sum += float(loss_o.detach())
                    agg.setdefault("opsd_kl", []).append(float(kl_sum.detach()) / max(1, n_tok))
            gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), g.grad_clip)
            self.opt.step()
            agg["loss"].append(loss_sum)
            agg["grad_norm"].append(float(gn))
        for k, v in agg.items():
            if v:
                metrics[k] = float(sum(v) / len(v))
        metrics["t_update"] = time.time() - t1
        return metrics

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate(self, n_tasks: Optional[int] = None) -> Dict[str, float]:
        n = n_tasks or self.cfg.grpo.n_eval_tasks
        tasks = self.val_tasks[:n]
        episodes = self.eval_rollout.run(tasks, group_size=1, greedy=self.cfg.eval.greedy, seed=1234)
        m = compute_metrics(episodes_to_records(episodes))
        keys = ("success_rate", "gm_speedup", "mean_edits", "invalid_edit_rate", "stop_rate", "recovery_rate", "mean_reward", "p_speedup_ge_1.5")
        return {f"eval_{k}": float(m[k]) for k in keys if k in m}

    def save(self, name: str) -> str:
        path = os.path.join(self.cfg.grpo.out_dir, name)
        extra = {"step": self.step, "tokenizer_path": self.cfg.tokenizer.path, "stage": "grpo", "best_eval": self.best_eval,
                 "env": {"horizon": self.cfg.env.horizon, "feedback": self.cfg.env.feedback}}
        if name == "last.pt":
            extra["optimizer"] = self.opt.state_dict()
        self.model.save(path, extra=extra)
        return path

    def train(self) -> str:
        g = self.cfg.grpo
        while self.step < g.max_steps:
            metrics = self.train_step()
            self.step += 1
            if self.step % g.log_every == 0:
                self.ml.log_metrics(self.step, metrics)
            if g.eval_every and self.step % g.eval_every == 0:
                ev = self.evaluate()
                self.ml.log_metrics(self.step, ev)
                score = ev.get("eval_gm_speedup", -float("inf"))
                if score > self.best_eval:
                    self.best_eval = score
                    self.save("best.pt")
            if g.save_every and self.step % g.save_every == 0:
                self.save("last.pt")
        final = self.save("final.pt")
        self.save("last.pt")
        self.ex.close()
        log.info(f"saved {final}")
        return final


def main(argv: Optional[List[str]] = None) -> None:
    cfg = parse_cli(argv)
    GRPOTrainer(cfg).train()


if __name__ == "__main__":
    main()
