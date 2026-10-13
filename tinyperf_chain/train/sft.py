"""Role SFT (and the shared-role control) with a per-example response-mean objective.

    python -m tinyperf_chain.train.sft --config configs/chain/sft.yaml --role M3 \\
        [--init artifacts/chain/checkpoints/initial/M3.pt] [--out artifacts/chain/checkpoints/M3_sft]
    python -m tinyperf_chain.train.sft --config configs/chain/sft.yaml --role shared   # B1: one model, all roles

Loss (eq. 3)::

    L = -(1/B) sum_i (1/K_i) sum_t log p(y_it | x_i, y_i<t)

over response tokens *including* the terminal marker; prompts and padding are masked; logits/targets are shifted
exactly once; no multi-example packing.  A batch with no valid response token is an error.  Tied labels (several
correct targets) are either sampled one per epoch (``tie_mode=sample``) or trained on the log of their summed
sequence probabilities (``tie_mode=sum``).  Record weights are *sampling* weights (NO_PLAN, PROBES none, M5 ABANDON
shares).  The shared-role control samples roles uniformly per example (``shared_sampling=uniform_role``) or matches
aggregate response tokens (``token_matched``) - two different budget questions, reported separately.

Checkpoints every ``checkpoint_every`` optimizer updates and at each epoch end contain optimizer, scheduler and RNG
state (CPU and CUDA), dataset and tokenizer hashes; a run resumes from ``last.pt``.  Selection uses development data
only and is *executable* for the patch roles: M3/M5 greedy outputs on development states are run through the fixed
acceptance protocol (scope, smoke, fixed probes, mandatory checks against the original, paired timing) and the
checkpoint with the best accepted-speedup rate wins; M1/M2/M4 use their measured-outcome labels (greedy exact match
/ valid-output rate) with dev loss as tie breaker.  Development metrics are also reported per collection round, so
forgetting on earlier-round data is visible when a role is refreshed on-policy (10.4).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from tinyperf_chain.config import ChainConfig, dump_chain_config, load_chain_config
from tinyperf_chain.hashing import hash_json, sha256_file
from tinyperf_chain.ledger import ObjectStore
from tinyperf_chain.schemas import parse_role_output

ALL_ROLES = ("M1", "M2", "M3", "M4", "M5")


@dataclass
class Example:
    role: str
    prompt: List[int]
    targets: List[List[int]]          # alternatives (response ids incl. EOS)
    weight: float
    record_id: str
    target_texts: List[str]
    round: int = 0
    task_id: str = ""
    parent_sha256: str = ""
    verification_record: str = ""
    plan_id: Optional[str] = None


def load_examples(cfg: ChainConfig, tok, role: str, split: str, objects: ObjectStore) -> Tuple[List[Example], Dict[str, int], str]:
    path = os.path.join(cfg.paths.datasets, role, f"{split}.jsonl")
    stats = {"records": 0, "too_long": 0}
    out: List[Example] = []
    if not os.path.exists(path):
        return out, stats, ""
    ctx = cfg.model.context_tokens
    dec = cfg.roles.get(role)
    with open(path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    for r in rows:
        stats["records"] += 1
        if r["split"] != split:
            raise ValueError(f"{r['record_id']}: split {r['split']} in the {split} file")
        prompt = tok.encode(objects.get_text(r["prompt_artifact"]), add_bos=True)
        texts = [objects.get_text(t) for t in r["target_artifacts"]]
        tg = [tok.encode(t, add_eos=True) for t in texts]
        tg = [t for t in tg if len(t) <= dec.max_output and len(prompt) + len(t) <= ctx]
        if not tg or len(prompt) > dec.max_input:
            stats["too_long"] += 1
            continue
        out.append(Example(role, prompt, tg, float(r.get("weight", 1.0)), r["record_id"], texts, int(r.get("round", 0) or 0),
                           r.get("task_id", ""), r.get("parent_program_sha256", ""), r.get("verification_record") or "",
                           r.get("plan_id")))
    return out, stats, sha256_file(path)


def tokens_of_runs(dirs: Sequence[str]) -> int:
    """Total prompt+response tokens processed by other runs (their final ``tokens`` metric), for the
    aggregate-token-matched shared control (11.1)."""
    total = 0
    for d in dirs:
        path = os.path.join(d, "metrics.jsonl")
        last = None
        with open(path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if "tokens" in rec:
                        last = rec["tokens"]
        if last is None:
            raise ValueError(f"{path}: no token count")
        total += int(last)
    return total


def epoch_items(examples: Sequence[Example], rng: random.Random) -> List[Example]:
    """Weights are sampling weights: w<1 keeps with probability w, w>=1 repeats floor(w) (+1 w.p. frac)."""
    items = []
    for e in examples:
        w = e.weight
        n = int(math.floor(w)) + (1 if rng.random() < (w - math.floor(w)) else 0)
        items.extend([e] * n)
    rng.shuffle(items)
    return items


def collate(batch: Sequence[Tuple[List[int], List[int]]], pad_id: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    T = max(len(p) + len(r) for p, r in batch)
    x = torch.full((len(batch), T), pad_id, dtype=torch.long)
    y = torch.full((len(batch), T), -100, dtype=torch.long)
    for i, (p, r) in enumerate(batch):
        seq = p + r
        x[i, :len(seq)] = torch.tensor(seq)
        y[i, len(p):len(seq)] = torch.tensor(r)
    return x[:, :-1].to(device), y[:, 1:].to(device)       # shift exactly once


def per_example_loss(model, x: torch.Tensor, y: torch.Tensor, ac) -> torch.Tensor:
    """Response-mean negative log-likelihood per row: [B]."""
    mask = y != -100
    if int(mask.sum()) == 0:
        raise ValueError("batch has no valid response tokens")
    with ac:
        logits = model(x)
    ce = F.cross_entropy(logits.float().transpose(1, 2), y.clamp_min(0), reduction="none")
    ce = ce * mask
    k = mask.sum(1)
    if bool((k == 0).any()):
        raise ValueError("an example has no response tokens")
    return ce.sum(1) / k


def seq_logprob_sum(model, x: torch.Tensor, y: torch.Tensor, ac) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = y != -100
    with ac:
        logits = model(x)
    lp = -F.cross_entropy(logits.float().transpose(1, 2), y.clamp_min(0), reduction="none") * mask
    return lp.sum(1), mask.sum(1)


def build_optimizer(model, cfg: ChainConfig) -> torch.optim.AdamW:
    t = cfg.training
    decay, no_decay = [], []
    for _n, p in model.named_parameters():
        if p.requires_grad:
            (decay if p.dim() >= 2 else no_decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": t.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
                             lr=t.lr, betas=tuple(t.betas), eps=t.eps)


def lr_at(step: int, total: int, cfg: ChainConfig) -> float:
    t = cfg.training
    warm = max(1, int(round(t.warmup_frac * total)))
    if step < warm:
        return t.lr * (step + 1) / warm
    frac = min(1.0, (step - warm) / max(1, total - warm))
    lo = t.lr * t.min_lr_ratio
    return lo + 0.5 * (1 + math.cos(math.pi * frac)) * (t.lr - lo)


class RoleSFT:
    def __init__(self, cfg: ChainConfig, role: str, init: str, out_dir: str, device: str, tok):
        from tinyperf.model.transformer import TinyPerfLM

        self.cfg, self.role, self.out = cfg, role, out_dir
        self.device = device
        self.tok = tok
        self.objects = ObjectStore(cfg.paths.objects)
        os.makedirs(out_dir, exist_ok=True)
        dump_chain_config(cfg, os.path.join(out_dir, "config.yaml"))
        self.init_path = init
        self.model = TinyPerfLM.load(init, map_location="cpu").to(device)
        self.init_sha = sha256_file(init)
        self.tok_sha = sha256_file(cfg.model.tokenizer) if cfg.model.tokenizer and os.path.exists(cfg.model.tokenizer) else None
        roles = ALL_ROLES if role == "shared" else (role,)
        self.train: Dict[str, List[Example]] = {}
        self.dev: Dict[str, List[Example]] = {}
        self.data_hash = {}
        self.stats = {}
        for r in roles:
            self.train[r], st, h1 = load_examples(cfg, tok, r, "train", self.objects)
            self.dev[r], st2, h2 = load_examples(cfg, tok, r, "dev", self.objects)
            self.data_hash[r] = {"train": h1, "dev": h2}
            self.stats[r] = {"train": st, "dev": st2, "n_train": len(self.train[r]), "n_dev": len(self.dev[r])}
        if not any(self.train.values()):
            raise RuntimeError(f"no training examples for {role}")
        self.opt = build_optimizer(self.model, cfg)
        self.ac = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (device.startswith("cuda") and cfg.training.precision == "bf16_autocast") \
            else torch.autocast("cpu", enabled=False)
        self.step = 0
        self.epoch = 0
        self.pos = 0                 # position inside the current epoch's stream (exact resume)
        self.tokens = 0
        self.best = (-float("inf"), float("inf"))
        self.rng = random.Random(cfg.training.seed)
        torch.manual_seed(cfg.training.seed)
        self.metrics_path = os.path.join(out_dir, "metrics.jsonl")
        self.exec_eval = cfg.training.exec_eval and any(r in ("M3", "M5") for r in roles) and any(self.dev.get(r) for r in ("M3", "M5"))
        self._exec_index: Dict[str, Any] = {}
        self._collector = None
        self.earlier_best_exact: Optional[float] = None
        self._resume()

    # -------------------------------------------------------------- #
    def _exec_setup(self) -> None:
        """Index the development states behind M3/M5 records so greedy outputs can be executed (7.1, 11.1)."""
        if self._collector is not None or not self.exec_eval:
            return
        from tinyperf_chain.adapters.verifier import Verifier
        from tinyperf_chain.data.collect import Collector

        self._collector = Collector(self.cfg, Verifier(self.cfg, cache_dir=self.cfg.paths.cache), "dev", self.cfg.training.seed)
        tasks = self._collector._task_index()
        trials = {r["trial_id"]: r for r in self._collector.ledger.read() if r.get("trial_id")}
        for role in ("M3", "M5"):
            for e in self.dev.get(role, []):
                tr = trials.get(e.verification_record)
                if tr is None or tr["task_id"] not in tasks:
                    continue
                task, private = tasks[tr["task_id"]]
                from tinyperf_chain.program import Program

                self._exec_index[e.record_id] = (task, private, Program(self._collector.objects.get_text(tr["state_ref"]), tuple(task.functions)),
                                                 tr["function"])

    def _state(self) -> Dict[str, Any]:
        st = {"step": self.step, "epoch": self.epoch, "pos": self.pos, "tokens": self.tokens, "best": list(self.best),
              "optimizer": self.opt.state_dict(), "rng_python": self.rng.getstate(), "rng_torch": torch.get_rng_state(),
              "data_hash": self.data_hash, "tokenizer_sha256": self.tok_sha, "parent_sha256": self.init_sha, "role": self.role,
              "earlier_best_exact": self.earlier_best_exact}
        if torch.cuda.is_available():
            st["rng_cuda"] = torch.cuda.get_rng_state_all()
        return st

    def save(self, name: str, with_state: bool = True) -> str:
        path = os.path.join(self.out, name)
        extra = {"role": self.role, "stage": "sft", "parent_sha256": self.init_sha, "tokenizer_sha256": self.tok_sha,
                 "tokenizer_path": self.cfg.model.tokenizer, "step": self.step, "epoch": self.epoch, "data_hash": self.data_hash}
        if with_state:
            extra["train_state"] = self._state()
        tmp = path + ".tmp"
        self.model.save(tmp, extra=extra)
        os.replace(tmp, path)
        return path

    def _resume(self) -> None:
        last = os.path.join(self.out, "last.pt")
        if not os.path.exists(last):
            return
        ck = torch.load(last, map_location="cpu", weights_only=False)
        st = ck["extra"].get("train_state")
        if not st:
            return
        if st["data_hash"] != self.data_hash or st.get("parent_sha256") != self.init_sha:
            raise RuntimeError("cannot resume: datasets or parent checkpoint changed; start a separately identified run")
        self.model.load_state_dict(ck["state_dict"])
        self.opt.load_state_dict(st["optimizer"])
        self.step, self.epoch, self.tokens = st["step"], st["epoch"], st["tokens"]
        self.pos = int(st.get("pos", 0))
        self.best = tuple(st["best"])
        self.rng.setstate(tuple(st["rng_python"]) if isinstance(st["rng_python"], list) else st["rng_python"])
        torch.set_rng_state(st["rng_torch"])
        if st.get("rng_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["rng_cuda"])
        self.earlier_best_exact = st.get("earlier_best_exact")

    def log(self, **m: Any) -> None:
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps({"step": self.step, **m}) + "\n")

    # -------------------------------------------------------------- #
    def _epoch_stream(self, epoch: int) -> List[Tuple[Example, int]]:
        rng = random.Random(f"{self.cfg.training.seed}|epoch{epoch}")
        if self.role != "shared":
            items = epoch_items(self.train[self.role], rng)
        else:
            pools = {r: epoch_items(v, rng) for r, v in self.train.items() if v}
            n = sum(len(v) for v in pools.values())
            roles = sorted(pools)
            if self.cfg.training.shared_sampling == "uniform_role":
                items = [rng.choice(pools[rng.choice(roles)]) for _ in range(n)]
            else:   # token_matched: sample examples proportionally to their response tokens
                flat = [e for v in pools.values() for e in v]
                w = [len(e.targets[0]) for e in flat]
                items = rng.choices(flat, weights=w, k=n)
        out = []
        for e in items:
            alt = rng.randrange(len(e.targets)) if self.cfg.training.tie_mode == "sample" else -1
            out.append((e, alt))
        return out

    def _loss(self, batch: List[Tuple[Example, int]]) -> Tuple[torch.Tensor, int]:
        t = self.cfg.training
        single = [(e.prompt, e.targets[a]) for e, a in batch if a >= 0 or len(e.targets) == 1]
        multi = [e for e, a in batch if a < 0 and len(e.targets) > 1]
        losses = []
        ntok = 0
        if single:
            x, y = collate(single, self.tok.pad_id, self.device)
            losses.append(per_example_loss(self.model, x, y, self.ac))
            ntok += int(x.numel())
        for e in multi:   # tie_mode=sum: -log sum_a p(a), scaled by the mean alternative length
            x, y = collate([(e.prompt, r) for r in e.targets], self.tok.pad_id, self.device)
            s, k = seq_logprob_sum(self.model, x, y, self.ac)
            losses.append((-torch.logsumexp(s, 0) / k.float().mean()).reshape(1))
            ntok += int(x.numel())
        return torch.cat(losses), ntok

    @torch.no_grad()
    def evaluate(self) -> Dict[str, Any]:
        from tinyperf_chain.serve.engine import RoleEngine, SamplingSpec
        from tinyperf.env.protocol import EDIT_CLOSE, STOP_TOKEN
        import copy
        import math as _m

        self.model.eval()
        self._exec_setup()
        out: Dict[str, Any] = {}
        losses, n = 0.0, 0
        exact, valid, total = 0, 0, 0
        by_round: Dict[int, Dict[str, float]] = {}
        ex_n = ex_valid = ex_correct = ex_accept = 0
        ex_logs: List[float] = []
        for role, exs in self.dev.items():
            for i in range(0, len(exs), self.cfg.training.microbatch):
                chunk = exs[i:i + self.cfg.training.microbatch]
                b = [(e.prompt, e.targets[0]) for e in chunk]
                x, y = collate(b, self.tok.pad_id, self.device)
                pl = per_example_loss(self.model, x, y, self.ac)
                losses += float(pl.sum())
                n += len(b)
                for e, l in zip(chunk, pl.tolist()):
                    d = by_round.setdefault(e.round, {"loss": 0.0, "n": 0, "exact": 0.0, "gen": 0})
                    d["loss"] += l
                    d["n"] += 1
            snap = RoleEngine(copy.deepcopy(self.model), device=self.device)
            stop = [self.tok.eos_id, self.tok.token_id(STOP_TOKEN)] + ([self.tok.token_id(EDIT_CLOSE)] if role in ("M3", "M5") else [])
            sample = exs[: self.cfg.training.dev_generation_eval_max]
            for j in range(0, len(sample), 16):
                chunk = sample[j:j + 16]
                gens = snap.generate([e.prompt for e in chunk], self.cfg.roles.get(role).max_output, SamplingSpec(greedy=True),
                                     stop, pad_id=self.tok.pad_id)
                for e, g in zip(chunk, gens):
                    text = self.tok.decode(g.ids)
                    total += 1
                    pr = parse_role_output(role, text, "f00")
                    valid += int(pr.ok)
                    hit = int(any(text.rstrip().replace("<EOS>", "") == t.rstrip() for t in e.target_texts))
                    exact += hit
                    d = by_round.setdefault(e.round, {"loss": 0.0, "n": 0, "exact": 0.0, "gen": 0})
                    d["exact"] += hit
                    d["gen"] += 1
                    # executable development metric for the patch roles
                    if role in ("M3", "M5") and self.exec_eval and e.record_id in self._exec_index and pr.ok and not pr.payload.is_abandon:
                        task, private, state, name = self._exec_index[e.record_id]
                        ex_n += 1
                        try:
                            o, _ = self._collector.verify(task, private, state, name, pr.payload.code, seed=self.step)
                        except Exception:  # noqa: BLE001 - infrastructure trouble during eval: counted as not run
                            continue
                        ex_valid += int(o.get("stage") not in ("scope",))
                        ex_correct += int(o.get("stage") == "timing")
                        ex_accept += int(bool(o.get("accepted")))
                        if o.get("accepted"):
                            ex_logs.append(_m.log(o["speedup"]))
                    elif role in ("M3", "M5") and self.exec_eval and e.record_id in self._exec_index:
                        ex_n += 1       # malformed / abandon counts as a non-accepted proposal
        self.model.train()
        out["dev_loss"] = losses / max(1, n)
        out["dev_valid_rate"] = valid / max(1, total)
        out["dev_exact"] = exact / max(1, total)
        out["dev_by_round"] = {str(r): {"loss": d["loss"] / max(1, d["n"]), "exact": d["exact"] / max(1, d["gen"]), "n": d["n"]}
                               for r, d in sorted(by_round.items())}
        if len(by_round) > 1:
            top = max(by_round)
            earlier = [d for r, d in by_round.items() if r != top]
            cur = sum(d["exact"] for d in earlier) / max(1, sum(d["gen"] for d in earlier))
            if self.earlier_best_exact is None or cur > self.earlier_best_exact:
                self.earlier_best_exact = cur
            out["dev_forgetting_earlier_rounds"] = (self.earlier_best_exact or 0.0) - cur   # drop vs the best seen this run
        if self.exec_eval:
            out["dev_exec_n"] = ex_n
            out["dev_exec_valid_rate"] = ex_valid / max(1, ex_n)
            out["dev_exec_correct_rate"] = ex_correct / max(1, ex_n)
            out["dev_exec_accept_rate"] = ex_accept / max(1, ex_n)
            out["dev_exec_gm_speedup"] = _m.exp(sum(ex_logs) / len(ex_logs)) if ex_logs else 1.0
        return out

    def fit(self) -> str:
        t = self.cfg.training
        per_update = t.microbatch * t.accumulate
        n_train = len(self._epoch_stream(0))                  # the *weighted* stream length (replay / balance weights)
        total = max(1, t.epochs * math.ceil(n_train / per_update))
        if t.max_updates:
            total = min(total, t.max_updates)
        self.model.train()
        t0 = time.time()
        done = self.step >= total
        last_eval_step = -1
        while not done and self.epoch < t.epochs:
            stream = self._epoch_stream(self.epoch)
            for i in range(self.pos, len(stream), per_update):
                window = stream[i:i + per_update]
                self.pos = i + per_update
                if not window:
                    break
                for g in self.opt.param_groups:
                    g["lr"] = lr_at(self.step, total, self.cfg)
                self.opt.zero_grad(set_to_none=True)
                loss_sum, ntok = 0.0, 0
                for j in range(0, len(window), t.microbatch):
                    mb = window[j:j + t.microbatch]
                    l, nt = self._loss(mb)
                    (l.sum() / len(window)).backward()
                    loss_sum += float(l.detach().sum())
                    ntok += nt
                gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), t.gradient_clip)
                self.opt.step()
                self.step += 1
                self.tokens += ntok
                self.log(loss=loss_sum / len(window), lr=lr_at(self.step - 1, total, self.cfg), grad_norm=float(gn),
                         epoch=self.epoch, tokens=self.tokens, elapsed_s=time.time() - t0)
                if t.eval_every and self.step % t.eval_every == 0 and any(self.dev.values()):
                    self._select(self.evaluate())
                    last_eval_step = self.step
                if self.step % t.checkpoint_every == 0:
                    self.save("last.pt")
                if self.step >= total or (t.token_budget and self.tokens >= t.token_budget):
                    done = True
                    break
            if done and self.pos < len(stream):
                break
            self.epoch += 1
            self.pos = 0
            if any(self.dev.values()):
                self._select(self.evaluate())
                last_eval_step = self.step
            self.save("last.pt")
        if any(self.dev.values()) and last_eval_step != self.step:
            self._select(self.evaluate())                      # a step/token cap never skips checkpoint selection
        self.save("last.pt")
        final = self.save("final.pt", with_state=False)
        if not os.path.exists(os.path.join(self.out, "best.pt")):
            self.save("best.pt", with_state=False)
        with open(os.path.join(self.out, "data_stats.json"), "w") as f:
            json.dump(self.stats, f, indent=1)
        return final

    def _select(self, m: Dict[str, Any]) -> None:
        self.log(**m)
        if self.exec_eval and m.get("dev_exec_n"):
            # executable metric first (accepted speedups on development states), then the role metric, then loss
            score = 10.0 * m["dev_exec_accept_rate"] + m["dev_exec_correct_rate"] + 0.1 * (m["dev_exact"] + 0.25 * m["dev_valid_rate"])
        else:
            score = m["dev_exact"] + 0.25 * m["dev_valid_rate"]
        key = (score, m["dev_loss"])
        if key[0] > self.best[0] or (key[0] == self.best[0] and key[1] < self.best[1]):
            self.best = key
            self.save("best.pt", with_state=False)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--role", required=True, choices=list(ALL_ROLES) + ["shared"])
    p.add_argument("--init", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--match-tokens-from", nargs="*", default=[],
                   help="shared control: stop after the total tokens these role runs processed (aggregate-token-matched)")
    p.add_argument("overrides", nargs="*")
    a = p.parse_args(argv)
    from tinyperf_chain.config import pop_overrides
    pop_overrides(a, ["match_tokens_from"])
    cfg = load_chain_config(a.config, a.overrides)
    if a.match_tokens_from:
        cfg.training.token_budget = tokens_of_runs(a.match_tokens_from)
        cfg.training.shared_sampling = "token_matched"
        print(json.dumps({"token_budget": cfg.training.token_budget, "matched_runs": a.match_tokens_from}))
    cfg.validate("train")
    from tinyperf.model.tokenizer import CodeTokenizer
    from tinyperf_chain.serve.engine import resolve_device

    tok = CodeTokenizer(cfg.model.tokenizer)
    init = a.init or cfg.paths.sub("checkpoints", "initial", f"{'M3' if a.role == 'shared' else a.role}.pt")
    out = a.out or cfg.paths.sub("checkpoints", f"{a.role}_sft")
    tr = RoleSFT(cfg, a.role, init, out, resolve_device(cfg.serving.device), tok)
    print(json.dumps(tr.stats))
    print(tr.fit())


if __name__ == "__main__":
    main()
