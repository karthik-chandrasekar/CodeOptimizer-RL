"""~8K code-specialized byte-level BPE tokenizer (HF `tokenizers`).

    python -m tinyperf.model.tokenizer --config configs/pretrain.yaml

Special tokens are the protocol tokens from :mod:`tinyperf.env.protocol` so that
``<CODE>``, ``<EDIT>``, ``<STOP>`` etc. are single tokens.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

from tinyperf.common.utils import get_logger, read_jsonl
from tinyperf.env.protocol import BOS, EOS, PAD, SPECIAL_TOKENS

log = get_logger("tokenizer")


# --------------------------------------------------------------------------- #
# Corpus iteration (shared with pretraining)
# --------------------------------------------------------------------------- #
def stdlib_files(max_files: int = 20000) -> List[Path]:
    import sysconfig

    root = Path(sysconfig.get_paths()["stdlib"])
    skip = ("test", "tests", "site-packages", "__pycache__", "idlelib", "tkinter", "turtledemo", "lib2to3")
    files = []
    for p in root.rglob("*.py"):
        if any(s in p.parts for s in skip):
            continue
        files.append(p)
        if len(files) >= max_files:
            break
    return files


def iter_corpus(corpus_dirs: Iterable[str], corpus_jsonl: Iterable[str], include_stdlib: bool, include_tasks: Optional[str], max_files: int = 20000) -> Iterator[str]:
    n = 0
    for d in corpus_dirs:
        for p in Path(d).rglob("*.py"):
            try:
                yield p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            n += 1
            if n >= max_files:
                return
    for j in corpus_jsonl:
        for row in read_jsonl(j):
            txt = row.get("content") or row.get("text") or ""
            if txt:
                yield txt
                n += 1
                if n >= max_files:
                    return
    if include_stdlib:
        for p in stdlib_files(max_files):
            try:
                yield p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
    if include_tasks and os.path.exists(include_tasks):
        for row in read_jsonl(include_tasks):
            yield row["source"]
            for c in row.get("chain", []):
                yield c


# --------------------------------------------------------------------------- #
def train_tokenizer(texts: Iterable[str], vocab_size: int, out_path: str) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tok.save(out_path)
    log.info(f"saved tokenizer ({tok.get_vocab_size()} tokens) to {out_path}")
    return tok


class CodeTokenizer:
    """Thin wrapper with the ids the training code needs."""

    def __init__(self, path: str):
        self.tok = Tokenizer.from_file(path)
        self.pad_id = self.tok.token_to_id(PAD)
        self.bos_id = self.tok.token_to_id(BOS)
        self.eos_id = self.tok.token_to_id(EOS)
        self.special_ids = {t: self.tok.token_to_id(t) for t in SPECIAL_TOKENS}
        assert None not in self.special_ids.values(), "tokenizer is missing protocol special tokens"

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = self.tok.encode(text, add_special_tokens=False).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)

    def token_id(self, s: str) -> int:
        i = self.tok.token_to_id(s)
        assert i is not None, s
        return i


def main(argv: Optional[List[str]] = None) -> None:
    from tinyperf.common.config import parse_cli

    cfg = parse_cli(argv)
    t = cfg.tokenizer
    texts = iter_corpus(t.corpus_dirs, t.corpus_jsonl, t.include_stdlib, t.include_tasks, t.max_files)
    train_tokenizer(texts, t.vocab_size, t.path)
    tok = CodeTokenizer(t.path)
    sample = "def contains_duplicate(xs):\n    return len(xs) != len(set(xs))\n"
    ids = tok.encode(sample)
    log.info(f"sample: {len(sample)} chars -> {len(ids)} tokens; roundtrip ok = {tok.decode(ids) == sample}")


if __name__ == "__main__":
    main()
