"""Stage 1 - code-only pretraining.

    python -m tinyperf.train.pretrain --config configs/pretrain.yaml

The corpus is tokenised once into a flat uint16 array (cached under
``pretrain.out_dir/tokens_{train,val}.npy``); batches are random windows of
``seq_len + 1`` tokens.  Documents are separated by ``<EOS>``.
"""
from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple

import numpy as np
import torch

from tinyperf.common.config import Config, dump_config, parse_cli
from tinyperf.common.utils import MetricLogger, get_logger, resolve_device, seed_everything
from tinyperf.model.tokenizer import CodeTokenizer, iter_corpus
from tinyperf.model.transformer import TinyPerfLM, lm_loss
from tinyperf.train.common import autocast_ctx, build_optimizer, lr_at, set_lr

log = get_logger("pretrain")


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #
def build_token_cache(cfg: Config, tok: CodeTokenizer, val_frac: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    p = cfg.pretrain
    tr_path, va_path = os.path.join(p.out_dir, "tokens_train.npy"), os.path.join(p.out_dir, "tokens_val.npy")
    if os.path.exists(tr_path) and os.path.exists(va_path):
        return np.load(tr_path, mmap_mode="r"), np.load(va_path, mmap_mode="r")
    os.makedirs(p.out_dir, exist_ok=True)
    log.info("tokenising corpus ...")
    texts = list(iter_corpus(p.corpus_dirs, p.corpus_jsonl, p.include_stdlib, p.include_tasks, cfg.tokenizer.max_files))
    if not texts:
        raise RuntimeError("empty pretraining corpus - set pretrain.corpus_dirs / corpus_jsonl or include_stdlib: true")
    rng = np.random.default_rng(0)
    rng.shuffle(texts)
    chunks: List[np.ndarray] = []
    B = 256
    for i in range(0, len(texts), B):
        encs = tok.tok.encode_batch(texts[i:i + B], add_special_tokens=False)
        for e in encs:
            chunks.append(np.asarray(e.ids + [tok.eos_id], dtype=np.uint16))
    n_val = max(1, int(len(chunks) * val_frac))
    min_val = 8 * (p.seq_len + 1)
    while n_val < len(chunks) - 1 and sum(int(c.size) for c in chunks[:n_val]) < min_val:
        n_val += 1  # tiny corpora: make sure the val split can serve full-length windows
    val = np.concatenate(chunks[:n_val])
    train = np.concatenate(chunks[n_val:])
    if train.size < min_val:
        raise RuntimeError(f"pretraining corpus too small ({train.size} tokens) for seq_len={p.seq_len}")
    np.save(tr_path, train)
    np.save(va_path, val)
    log.info(f"corpus: {len(texts)} docs, {train.size/1e6:.1f}M train tokens, {val.size/1e6:.2f}M val tokens")
    return np.load(tr_path, mmap_mode="r"), np.load(va_path, mmap_mode="r")


def sample_batch(tokens: np.ndarray, seq_len: int, batch_size: int, rng: np.random.Generator, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    n = tokens.size - seq_len - 1
    if n <= 0:
        raise RuntimeError(f"token array ({tokens.size}) shorter than seq_len+1 ({seq_len + 1})")
    idx = rng.integers(0, n, size=batch_size)
    x = np.stack([tokens[i:i + seq_len] for i in idx]).astype(np.int64)
    y = np.stack([tokens[i + 1:i + seq_len + 1] for i in idx]).astype(np.int64)
    return torch.from_numpy(x).to(device, non_blocking=True), torch.from_numpy(y).to(device, non_blocking=True)


@torch.no_grad()
def evaluate_lm(model: TinyPerfLM, tokens: np.ndarray, seq_len: int, batch_size: int, device: str, n_batches: int = 20) -> float:
    model.eval()
    rng = np.random.default_rng(1234)
    losses = []
    for _ in range(n_batches):
        x, y = sample_batch(tokens, seq_len, batch_size, rng, device)
        with autocast_ctx(device):
            losses.append(float(lm_loss(model(x), y)))
    model.train()
    return float(np.mean(losses))


# --------------------------------------------------------------------------- #
def train(cfg: Config) -> str:
    p, o = cfg.pretrain, cfg.pretrain.optim
    device = resolve_device(cfg.device)
    seed_everything(0)
    os.makedirs(p.out_dir, exist_ok=True)
    dump_config(cfg, os.path.join(p.out_dir, "config.yaml"))

    tok = CodeTokenizer(cfg.tokenizer.path)
    train_tok, val_tok = build_token_cache(cfg, tok)
    cfg.model.vocab_size = max(cfg.model.vocab_size, tok.vocab_size)
    model = TinyPerfLM(cfg.model).to(device)
    log.info(f"model {cfg.model.name}: {model.n_params()/1e6:.2f}M params ({model.n_params(True)/1e6:.2f}M non-embedding); device={device}")
    opt = build_optimizer(model, o.lr, o.weight_decay, o.betas)
    ml = MetricLogger(p.out_dir, cfg.wandb_project, f"{cfg.run_name}-pretrain", config=None)

    step, t0 = 0, time.time()
    resume = os.path.join(p.out_dir, "last.pt")
    if os.path.exists(resume):
        ck = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["extra"]["optimizer"])
        step = int(ck["extra"]["step"])
        log.info(f"resumed from {resume} at step {step}")

    rng = np.random.default_rng(step + 1)
    model.train()
    tokens_seen = step * o.batch_size * o.grad_accum * p.seq_len
    while step < o.max_steps:
        set_lr(opt, lr_at(step, o))
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(o.grad_accum):
            x, y = sample_batch(train_tok, p.seq_len, o.batch_size, rng, device)
            with autocast_ctx(device):
                loss = lm_loss(model(x), y) / o.grad_accum
            loss.backward()
            loss_acc += float(loss)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), o.grad_clip)
        opt.step()
        step += 1
        tokens_seen += o.batch_size * o.grad_accum * p.seq_len
        if step % p.log_every == 0:
            dt = time.time() - t0
            ml.log_metrics(step, {"loss": loss_acc, "lr": lr_at(step, o), "grad_norm": float(gn), "tokens_M": tokens_seen / 1e6, "tok_per_s": tokens_seen / max(dt, 1e-6)})
        if step % p.eval_every == 0:
            ml.log_metrics(step, {"val_loss": evaluate_lm(model, val_tok, p.seq_len, o.batch_size, device)})
        if step % p.save_every == 0 or step == o.max_steps:
            model.save(resume, extra={"step": step, "optimizer": opt.state_dict(), "tokenizer_path": cfg.tokenizer.path})
    final = os.path.join(p.out_dir, "final.pt")
    model.save(final, extra={"step": step, "tokenizer_path": cfg.tokenizer.path, "stage": "pretrain"})
    log.info(f"saved {final}")
    return final


def main(argv: Optional[List[str]] = None) -> None:
    train(parse_cli(argv))


if __name__ == "__main__":
    main()
