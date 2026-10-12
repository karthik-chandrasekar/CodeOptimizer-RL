"""Frozen-partner, one-proposal verifier RL for the writer (M3) and the repairer (M5).

    python -m tinyperf_chain.train.rl --config configs/chain/rl.yaml --role M3 \\
        --init artifacts/chain/checkpoints/M3_sft/best.pt --out artifacts/chain/checkpoints/M3_rl

Per update: sample ``states_per_update`` training conditioning states (M3: state + plan; M5: state + plan + failed
candidate + allowed feedback); freeze a behaviour snapshot; sample ``group_size`` responses per state at the RL
temperature, storing token ids, the sampling spec, per-token behaviour log-probs, the conditioning-state hash and
the policy hash.  Each response is evaluated once against the fixed acceptance protocol (scope, smoke, mandatory
checks against the original, paired timing against the state)::

    g_i = min(log(T_s / T'_i), gain_cap)       (T'_i = T_s unless accepted; correct-but-not-faster -> 0)
    R_i = g_i - slot_cost - invalid_penalty * 1{malformed | out of scope | semantic failure}
    A_i = R_i - mean_group(R)                  (no std scaling; equal rewards -> zero loss and gradient)

    L = - 1/G sum_i 1/K_i sum_t min(r_it A_i, clip(r_it, clip_low, clip_high) A_i),
        r_it = pi_e,theta(y_it | h_it) / pi_e,old(y_it | h_it)

with ``pi_e`` the *effective* sampling distribution (temperature, top-k/top-p, bans).  Old log-probs are recomputed
with the training forward pass in eval mode and compared with the behaviour log-probs before the first update
(``rl.ratio_tolerance``).  One optimizer pass per update; at most ``rl.max_updates`` updates before review.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from tinyperf_chain import registry as R
from tinyperf_chain.adapters.verifier import Verifier
from tinyperf_chain.config import ChainConfig, dump_chain_config, load_chain_config
from tinyperf_chain.context import M3View, M5View, PromptBuilder
from tinyperf_chain.data.collect import Collector
from tinyperf_chain.hashing import hash_json, sha256_file, sha256_text
from tinyperf_chain.ledger import AppendLedger, ObjectStore
from tinyperf_chain.program import Program, dependency_closure
from tinyperf_chain.schemas import format_plan, parse_patch, parse_plan
from tinyperf_chain.serve.engine import RoleEngine, SamplingSpec, effective_logprobs
from tinyperf_chain.tasks import PrivateSuite, PublicTask


@dataclass
class CondState:
    role: str
    task_id: str
    state: Program
    function: str
    plan: str
    prompt_ids: List[int]
    failed: str = ""
    key: str = ""


def rewards_for(outcome: Optional[Dict[str, Any]], parsed_ok: bool, abandon: bool, cfg: ChainConfig) -> Tuple[float, bool, float]:
    """(R, invalid, g) for one response under eqs. (4)-(5)."""
    rl = cfg.rl
    if not parsed_ok:
        return -rl.slot_cost - rl.invalid_penalty, True, 0.0
    if abandon:
        return -rl.slot_cost, False, 0.0
    assert outcome is not None
    invalid = outcome["stage"] in ("scope", "smoke", "probe", "mandatory") or outcome.get("kind") == "duplicate"
    g = min(math.log(outcome["speedup"]), rl.gain_cap) if outcome.get("accepted") else 0.0
    return g - rl.slot_cost - (rl.invalid_penalty if invalid else 0.0), invalid, g


def group_advantages(rewards: Sequence[float]) -> List[float]:
    mu = sum(rewards) / len(rewards)
    return [r - mu for r in rewards]


def response_logprobs(model, prompts: Sequence[List[int]], responses: Sequence[List[int]], spec: SamplingSpec,
                      pad_id: int, device: str, ac) -> List[torch.Tensor]:
    """Effective-distribution log-probs of each response token (right padding: causal attention never sees pads)."""
    seqs = [list(p) + list(r) for p, r in zip(prompts, responses)]
    T = max(len(s) for s in seqs)
    x = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    x = x.to(device)
    with ac:
        logits = model(x)
    out = []
    for i, (p, r) in enumerate(zip(prompts, responses)):
        lg = logits[i, len(p) - 1: len(p) - 1 + len(r)].float()
        out.append(effective_logprobs(lg, torch.tensor(r, device=device), spec))
    return out


def grpo_loss(new_lp: Sequence[torch.Tensor], old_lp: Sequence[torch.Tensor], adv: Sequence[float], cfg: ChainConfig,
              n_groups: int, group_size: int) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Eq. (7): per-response token mean, groups averaged; returns (loss, stats)."""
    rl = cfg.rl
    total = None
    clipped, ntok = 0, 0
    for lp, olp, a in zip(new_lp, old_lp, adv):
        if lp.numel() == 0:
            continue
        ratio = torch.exp(lp - olp)
        a_t = torch.tensor(float(a), device=lp.device)
        s1 = ratio * a_t
        s2 = ratio.clamp(rl.clip_low, rl.clip_high) * a_t
        term = -torch.min(s1, s2).mean() / (group_size * n_groups)
        total = term if total is None else total + term
        clipped += int(((ratio < rl.clip_low) | (ratio > rl.clip_high)).sum())
        ntok += lp.numel()
    if total is None:
        total = torch.zeros((), requires_grad=True)
    return total, {"clip_frac": clipped / max(1, ntok), "tokens": ntok}


class RoleRL:
    def __init__(self, cfg: ChainConfig, role: str, init: str, out_dir: str, device: str, tok, verifier: Verifier):
        from tinyperf.model.transformer import TinyPerfLM
        from tinyperf_chain.train.sft import build_optimizer

        if role not in cfg.rl.roles:
            raise ValueError(f"RL is configured for {cfg.rl.roles}, not {role}")
        self.cfg, self.role, self.out, self.device, self.tok, self.v = cfg, role, out_dir, device, tok, verifier
        os.makedirs(out_dir, exist_ok=True)
        dump_chain_config(cfg, os.path.join(out_dir, "config.yaml"))
        self.model = TinyPerfLM.load(init, map_location="cpu").to(device)
        self.init_sha = sha256_file(init)
        c2 = copy.deepcopy(cfg)
        c2.training.lr = cfg.rl.lr
        c2.training.weight_decay = 0.0
        self.opt = build_optimizer(self.model, c2)
        for g in self.opt.param_groups:
            g["lr"] = cfg.rl.lr
        self.builder = PromptBuilder(tok, cfg.roles, cfg.model.context_tokens)
        self.collector = Collector(cfg, verifier, "train", seed=cfg.rl.seed)
        self.objects = ObjectStore(cfg.paths.objects)
        self.rollouts = AppendLedger(os.path.join(out_dir, "rollouts.jsonl"))
        self.metrics_path = os.path.join(out_dir, "metrics.jsonl")
        self.rng = random.Random(cfg.rl.seed)
        torch.manual_seed(cfg.rl.seed)
        self.ac = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (device.startswith("cuda") and cfg.training.precision == "bf16_autocast") \
            else torch.autocast("cpu", enabled=False)
        self.spec = SamplingSpec(greedy=False, temperature=cfg.rl.temperature, top_k=cfg.rl.top_k, top_p=cfg.rl.top_p)
        self.tasks: Dict[str, Tuple[PublicTask, PrivateSuite]] = self.collector._task_index()
        self.states = self._load_states()
        if not self.states:
            raise RuntimeError(f"no training conditioning states for {role}")
        self.step = 0

    # -------------------------------------------------------------- #
    def _load_states(self) -> List[CondState]:
        out: List[CondState] = []
        seen = set()
        trials = list(self.collector.ledger.read())
        for r in trials:
            if r.get("split") != "train" or r["task_id"] not in self.tasks:
                continue
            task, _ = self.tasks[r["task_id"]]
            if self.role == "M3" and r["kind"] in ("plan_patch", "chain_plan_state") and r.get("plan"):
                key = (r["task_id"], r["state_sha256"], r["function"], r["plan"])
            elif self.role == "M5" and r["kind"] == "repair" and (r["feedback_visibility"] != "category_only"
                                                                  or "private_mismatch" in self.cfg.verification.repairable_failures):
                key = (r["task_id"], r["state_sha256"], r["function"], r["plan"], r["failed_fn_ref"])
            else:
                continue
            if key in seen:
                continue
            seen.add(key)
            state = Program(self.objects.get_text(r["state_ref"]), tuple(task.functions))
            name = r["function"]
            pr = parse_plan(r["plan"], "f00")
            if not pr.ok or pr.payload.is_no_plan:
                continue
            fn = state.function_node(name)
            deps = [state.function_source(f) for f in dependency_closure(state, name).functions]
            fid = f"f{self.rng.randrange(len(task.functions)):02d}"
            if self.role == "M3":
                view = M3View(fid, state.function_source(name), deps, task.contract.render(name), format_plan(pr.payload),
                              R.render_anchors(fn))
                failed = ""
            else:
                failed = self.objects.get_text(r["failed_fn_ref"])
                view = M5View(fid, state.function_source(name), failed, deps, task.contract.render(name), r["plan"],
                              r.get("feedback") or {"failure_kind": r.get("failure_kind") or "unknown"})
            try:
                built = self.builder.build(self.role, view)
            except Exception:  # noqa: BLE001 - context overflow: not a usable conditioning state
                continue
            out.append(CondState(self.role, r["task_id"], state, name, r["plan"], built.ids, failed, hash_json(list(key))))
        return out

    def log(self, **m: Any) -> None:
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps({"step": self.step, **m}) + "\n")

    def policy_hash(self) -> str:
        h = sha256_text("".join(f"{k}:{float(v.float().sum()):.6e}" for k, v in self.model.state_dict().items()))
        return h[:16]

    # -------------------------------------------------------------- #
    def update(self) -> Dict[str, float]:
        from tinyperf.env.protocol import EDIT_CLOSE, STOP_TOKEN

        rl = self.cfg.rl
        G = rl.group_size
        states = [self.states[self.rng.randrange(len(self.states))] for _ in range(rl.states_per_update)]
        self.model.eval()                                   # dropout off for rollout, recompute and update
        behaviour = RoleEngine(copy.deepcopy(self.model), device=self.device)
        pol = self.policy_hash()
        stop = [self.tok.eos_id, self.tok.token_id(STOP_TOKEN), self.tok.token_id(EDIT_CLOSE)]
        prompts, responses, beh_lp, advs, rows = [], [], [], [], []
        n_zero, n_invalid, n_accept, rewards_all = 0, 0, 0, []
        t0 = time.time()
        for si, cs in enumerate(states):
            seeds = [self.rng.randrange(1 << 40) for _ in range(G)]
            gens = behaviour.generate([cs.prompt_ids] * G, self.cfg.roles.get(self.role).max_output, self.spec, stop,
                                      pad_id=self.tok.pad_id, seeds=seeds)
            task, private = self.tasks[cs.task_id]
            R_ = []
            for gi, g in enumerate(gens):
                text = self.tok.decode(g.ids)
                pr = parse_patch(text)
                outcome = None
                if pr.ok and not pr.payload.is_abandon:
                    seed = int(sha256_text(f"rl|{self.step}|{si}|{gi}")[:8], 16)
                    outcome, _ = self.collector.verify(task, private, cs.state, cs.function, pr.payload.code, seed)
                r, invalid, gain = rewards_for(outcome, pr.ok, pr.ok and pr.payload.is_abandon, self.cfg)
                R_.append(r)
                n_invalid += int(invalid)
                n_accept += int(bool(outcome and outcome.get("accepted")))
                rows.append({"update": self.step, "state": cs.key, "policy": pol, "sampling": self.spec.__dict__,
                             "seed": seeds[gi], "token_ids": g.ids, "behaviour_logprobs": g.logprobs, "finished": g.finished,
                             "reward": r, "gain": gain, "invalid": invalid, "outcome": outcome and {k: outcome.get(k) for k in ("stage", "kind", "accepted", "speedup")}})
            A = group_advantages(R_)
            if max(abs(a) for a in A) < 1e-12:
                n_zero += 1
            rewards_all += R_
            for g, a in zip(gens, A):
                prompts.append(cs.prompt_ids)
                responses.append(g.ids)
                beh_lp.append(torch.tensor(g.logprobs, device=self.device))
                advs.append(a)
        t_roll = time.time() - t0
        self.rollouts.extend(rows)
        # recompute old log-probs with the training forward (no grad) and check the behaviour bookkeeping
        mb = max(1, rl.minibatch)
        old: List[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, len(prompts), mb):
                old += [x.detach() for x in response_logprobs(self.model, prompts[i:i + mb], responses[i:i + mb], self.spec,
                                                              self.tok.pad_id, self.device, self.ac)]
        diffs = [float((o - b).abs().max()) for o, b in zip(old, beh_lp) if o.numel()]
        max_diff = max(diffs) if diffs else 0.0
        if max_diff > rl.ratio_tolerance:
            raise RuntimeError(f"behaviour/recomputed log-prob mismatch {max_diff:.3g} > tolerance {rl.ratio_tolerance}; "
                               "sampling filters and the likelihood recomputation disagree")
        # one optimizer pass
        self.opt.zero_grad(set_to_none=True)
        stats = {"clip_frac": 0.0, "tokens": 0}
        loss_total = 0.0
        n_groups = len(states)
        for i in range(0, len(prompts), mb):
            new = response_logprobs(self.model, prompts[i:i + mb], responses[i:i + mb], self.spec, self.tok.pad_id,
                                    self.device, self.ac)
            loss, st = grpo_loss(new, old[i:i + mb], advs[i:i + mb], self.cfg, n_groups, rl.group_size)
            if loss.requires_grad:
                loss.backward()
            loss_total += float(loss.detach())
            stats["clip_frac"] += st["clip_frac"] * st["tokens"]
            stats["tokens"] += st["tokens"]
        gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.gradient_clip)
        self.opt.step()
        self.step += 1
        mean_r = sum(rewards_all) / len(rewards_all)
        m = {"loss": loss_total, "reward_mean": mean_r,
             "reward_var": sum((r - mean_r) ** 2 for r in rewards_all) / len(rewards_all),
             "zero_advantage_groups": n_zero, "invalid_rate": n_invalid / len(rewards_all),
             "accept_rate": n_accept / len(rewards_all), "clip_frac": stats["clip_frac"] / max(1, stats["tokens"]),
             "response_len": sum(len(r) for r in responses) / len(responses), "grad_norm": float(gn),
             "max_logprob_diff": max_diff, "t_rollout": t_roll}
        self.log(**m)
        return m

    def train(self) -> str:
        for _ in range(self.cfg.rl.max_updates):
            self.update()
            if self.cfg.rl.save_every and self.step % self.cfg.rl.save_every == 0:
                self.save(f"step{self.step:04d}.pt")
        return self.save("final.pt")

    def save(self, name: str) -> str:
        path = os.path.join(self.out, name)
        self.model.save(path, extra={"role": self.role, "stage": "rl", "parent_sha256": self.init_sha, "step": self.step,
                                     "tokenizer_path": self.cfg.model.tokenizer, "rl": self.cfg.rl.__dict__})
        return path


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--role", required=True, choices=["M3", "M5"])
    p.add_argument("--init", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("overrides", nargs="*")
    a = p.parse_args(argv)
    cfg = load_chain_config(a.config, a.overrides).validate("collect:train")
    from tinyperf.model.tokenizer import CodeTokenizer
    from tinyperf_chain.serve.engine import resolve_device

    tok = CodeTokenizer(cfg.model.tokenizer)
    v = Verifier(cfg, cache_dir=cfg.paths.cache)
    tr = RoleRL(cfg, a.role, a.init, a.out or cfg.paths.sub("checkpoints", f"{a.role}_rl"), resolve_device(cfg.serving.device), tok, v)
    print(tr.train())


if __name__ == "__main__":
    main()
