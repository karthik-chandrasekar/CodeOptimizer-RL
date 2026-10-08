"""Stage 2b - agent SFT on (observation -> action) pairs.

    python -m tinyperf.train.sft --config configs/sft.yaml

Each trajectory step becomes one example ``<BOS> obs action <EOS>`` with the
loss on the action tokens only.  With ``env.feedback=none`` the observations
are re-rendered exactly as the no-feedback environment would show them
(``C0`` + step counter), which produces the no-feedback control policy from the
*same* trajectories.  ``sft.first_step_only=true`` keeps one
``(s_0 -> <EDIT>final code</EDIT>)`` example per trajectory (the SFT-1 / one-shot baseline).
``sft.mask_failed_actions`` (default on) drops the injected/failed edits from the
imitation targets while keeping the recovery step that observes the failure.
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from tinyperf.common.config import Config, dump_config, parse_cli
from tinyperf.common.utils import MetricLogger, get_logger, read_jsonl, resolve_device, seed_everything
from tinyperf.env.protocol import Action, format_action, format_observation
from tinyperf.model.tokenizer import CodeTokenizer
from tinyperf.model.transformer import TinyPerfLM
from tinyperf.train.common import autocast_ctx, build_optimizer, load_model_and_tokenizer, lr_at, set_lr

log = get_logger("sft")


@dataclass
class Example:
    ids: List[int]
    n_prompt: int          # tokens with no loss (BOS + observation)
    task_id: str


def render_obs(step: Dict, row: Dict, feedback: str) -> str:
    if feedback == "none":
        return format_observation(row["source"], {"step": step["step"], "remaining": step["remaining"]}, feedback="none")
    return step["obs"]


_IMITATE_STATUSES = {"ok", "stop"}


def build_examples(rows: Sequence[Dict], tok: CodeTokenizer, seq_len: int, feedback: str, first_step_only: bool,
                   mask_failed_actions: bool = True) -> Tuple[List[Example], Dict[str, int]]:
    ex: List[Example] = []
    stats = {"rows": 0, "steps": 0, "too_long": 0, "failed_actions_skipped": 0}
    for row in rows:
        stats["rows"] += 1
        if first_step_only:
            # SFT-1: one-shot  (s_0 -> <EDIT> final best code </EDIT>)
            final = row.get("final_code")
            if not row["steps"] or not final or row.get("final_ratio", 1.0) >= 1.0:
                continue
            pairs = [(render_obs(row["steps"][0], row, feedback), format_action(Action("edit", code=final)))]
        else:
            pairs = []
            for st in row["steps"]:
                if mask_failed_actions and st.get("status") not in _IMITATE_STATUSES:
                    stats["failed_actions_skipped"] += 1  # the next step still sees this failure in its observation
                    continue
                pairs.append((render_obs(st, row, feedback), st["action"]))
        for obs, action in pairs:
            p = tok.encode(obs, add_bos=True)
            a = tok.encode(action, add_eos=True)
            if len(p) + len(a) > seq_len:
                stats["too_long"] += 1
                continue
            ex.append(Example(p + a, len(p), row["task_id"]))
            stats["steps"] += 1
    return ex, stats


def split_by_task(examples: List[Example], val_frac: float, seed: int = 0) -> Tuple[List[Example], List[Example]]:
    tasks = sorted({e.task_id for e in examples})
    rng = random.Random(seed)
    rng.shuffle(tasks)
    n_val = max(1, int(len(tasks) * val_frac)) if len(tasks) > 1 else 0
    val_ids = set(tasks[:n_val])
    return [e for e in examples if e.task_id not in val_ids], [e for e in examples if e.task_id in val_ids]


def batches(examples: List[Example], batch_size: int, rng: random.Random, shuffle: bool = True) -> Iterator[List[Example]]:
    """Length-bucketed batches (sort inside chunks of 50 batches, then shuffle the batches)."""
    order = list(range(len(examples)))
    if shuffle:
        rng.shuffle(order)
    chunk = batch_size * 50
    out: List[List[Example]] = []
    for i in range(0, len(order), chunk):
        ids = sorted(order[i:i + chunk], key=lambda k: len(examples[k].ids))
        for j in range(0, len(ids), batch_size):
            out.append([examples[k] for k in ids[j:j + batch_size]])
    if shuffle:
        rng.shuffle(out)
    yield from out


def collate(batch: List[Example], pad_id: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    T = max(len(e.ids) for e in batch)
    x = torch.full((len(batch), T), pad_id, dtype=torch.long)
    y = torch.full((len(batch), T), -100, dtype=torch.long)
    for i, e in enumerate(batch):
        x[i, :len(e.ids)] = torch.tensor(e.ids, dtype=torch.long)
        y[i, e.n_prompt:len(e.ids)] = torch.tensor(e.ids[e.n_prompt:], dtype=torch.long)
    # next-token prediction: inputs x[:, :-1] predict y[:, 1:]
    return x[:, :-1].to(device), y[:, 1:].to(device)


def action_loss(model: TinyPerfLM, x: torch.Tensor, y: torch.Tensor, device: str) -> Tuple[torch.Tensor, float]:
    with autocast_ctx(device):
        logits = model(x)
    logits = logits.float()
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
    with torch.no_grad():
        m = y != -100
        acc = float(((logits.argmax(-1) == y) & m).sum() / m.sum().clamp_min(1))
    return loss, acc


@torch.no_grad()
def evaluate(model: TinyPerfLM, examples: List[Example], pad_id: int, batch_size: int, device: str, max_batches: int = 50) -> Dict[str, float]:
    model.eval()
    losses, accs = [], []
    for i, b in enumerate(batches(examples, batch_size, random.Random(0), shuffle=False)):
        if i >= max_batches:
            break
        x, y = collate(b, pad_id, device)
        loss, acc = action_loss(model, x, y, device)
        losses.append(float(loss))
        accs.append(acc)
    model.train()
    return {"val_loss": sum(losses) / max(1, len(losses)), "val_token_acc": sum(accs) / max(1, len(accs))}


def train(cfg: Config) -> str:
    s, o = cfg.sft, cfg.sft.optim
    device = resolve_device(cfg.device)
    seed_everything(0)
    os.makedirs(s.out_dir, exist_ok=True)
    dump_config(cfg, os.path.join(s.out_dir, "config.yaml"))
    model, tok = load_model_and_tokenizer(cfg, s.init_from or None, device)
    rows = list(read_jsonl(s.trajectories, limit=s.max_trajectories))
    if s.methods:
        rows = [r for r in rows if r.get("method") in s.methods]
    examples, stats = build_examples(rows, tok, s.seq_len, cfg.env.feedback, s.first_step_only, s.mask_failed_actions)
    if not examples:
        raise RuntimeError(f"no SFT examples built from {s.trajectories} ({stats})")
    train_ex, val_ex = split_by_task(examples, s.val_frac)
    log.info(f"SFT data: {stats} -> {len(train_ex)} train / {len(val_ex)} val examples; feedback={cfg.env.feedback} "
             f"first_step_only={s.first_step_only}; mean len={sum(len(e.ids) for e in examples)/len(examples):.0f} tokens")
    steps_per_epoch = max(1, len(train_ex) // o.batch_size)
    total = min(o.max_steps, steps_per_epoch * s.epochs)
    o.max_steps = total
    opt = build_optimizer(model, o.lr, o.weight_decay, o.betas)
    ml = MetricLogger(s.out_dir, cfg.wandb_project, f"{cfg.run_name}-sft", config=None)
    rng = random.Random(0)
    step, t0 = 0, time.time()
    model.train()
    while step < total:
        for batch in batches(train_ex, o.batch_size, rng):
            if step >= total:
                break
            set_lr(opt, lr_at(step, o))
            x, y = collate(batch, tok.pad_id, device)
            loss, acc = action_loss(model, x, y, device)
            (loss / o.grad_accum).backward()
            if (step + 1) % o.grad_accum == 0:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), o.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
            else:
                gn = torch.tensor(0.0)
            step += 1
            if step % s.log_every == 0:
                ml.log_metrics(step, {"loss": float(loss), "token_acc": acc, "lr": lr_at(step, o), "grad_norm": float(gn),
                                      "epoch": step / steps_per_epoch, "elapsed_s": time.time() - t0})
            if val_ex and step % s.eval_every == 0:
                ml.log_metrics(step, evaluate(model, val_ex, tok.pad_id, o.batch_size, device))
            if step % s.save_every == 0:
                model.save(os.path.join(s.out_dir, "last.pt"), extra={"step": step, "tokenizer_path": cfg.tokenizer.path, "stage": "sft"})
    if val_ex:
        ml.log_metrics(step, evaluate(model, val_ex, tok.pad_id, o.batch_size, device))
    final = os.path.join(s.out_dir, "final.pt")
    model.save(final, extra={"step": step, "tokenizer_path": cfg.tokenizer.path, "stage": "sft",
                             "feedback": cfg.env.feedback, "first_step_only": s.first_step_only})
    log.info(f"saved {final}")
    return final


def main(argv: Optional[List[str]] = None) -> None:
    train(parse_cli(argv))


if __name__ == "__main__":
    main()
