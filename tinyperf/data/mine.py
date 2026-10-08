"""Mine self-contained Python functions from a code corpus and turn them into TinyPerf seeds.

    python -m tinyperf.data.mine --corpus data/corpus/python.jsonl --out artifacts/mined/mined_seeds.jsonl --workers 60

Stage 1 - static (parallel over files):
  top-level ``def`` with 1-3 positional parameters, no decorators / generators / globals / nested defs / classes,
  4-40 lines, at least one loop or comprehension, and *self-contained*: every name it reads is a parameter, a local,
  a sandbox-safe builtin, or a module from the sandbox whitelist (the import is carried into the seed).  The
  sandbox's own ``static_check`` must pass.  Exact and structural duplicates (identifiers anonymised) are
  removed, as is anything structurally identical to a hand-written seed.

Stage 2 - dynamic (parallel, in rlimited processes, same restricted namespace as the sandbox):
  guess a type spec per parameter from usage (:mod:`tinyperf.data.argspec`) and try the best-ranked
  combinations on generated inputs.  A combination is accepted when the function runs on >= 75% of inputs,
  is deterministic, returns plain data that varies across inputs, and its runtime grows with input size
  (it has work to speed up).

Each accepted function becomes a seed record; ``natural`` lists rewrites from the inverse-rewrite proposer.  The
dataset builder verifies them: if a rewrite is correct and faster, the *original* becomes a genuinely slow,
human-written starting point (family ``natural``).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import multiprocessing as mp
import os
import random
import resource
import signal
import sys
import time
from copy import deepcopy
from pathlib import Path
from statistics import median
from typing import Dict, Iterator, List, Optional, Tuple

from tinyperf.common.config import EnvConfig
from tinyperf.common.utils import get_logger
from tinyperf.data.argspec import gen_value, guess_specs, ranked_combos

log = get_logger("mine")

ALLOWED_MODULES = [m for m in EnvConfig().allowed_modules if m not in ("re",)]  # C-level regex backtracking cannot be interrupted
SIZES = [0, 1, 2, 3, 5, 8, 13, 21, 34]


def _safe_names() -> set:
    from tinyperf.env.worker import SAFE_BUILTIN_NAMES
    return set(SAFE_BUILTIN_NAMES)


# --------------------------------------------------------------------------- #
# Stage 1: static extraction
# --------------------------------------------------------------------------- #
def _module_imports(tree: ast.Module) -> Dict[str, str]:
    """local name -> import statement, for top-level imports of whitelisted modules."""
    out: Dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in ALLOWED_MODULES:
                    out[a.asname or a.name.split(".")[0]] = ast.unparse(ast.Import([a]))
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0 and node.module.split(".")[0] in ALLOWED_MODULES:
            for a in node.names:
                if a.name != "*":
                    out[a.asname or a.name] = ast.unparse(ast.ImportFrom(node.module, [a], 0))
    return out


def _bound(fn: ast.FunctionDef) -> set:
    names = {a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs}
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            names.add(n.id)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            names.add(n.name)
        elif isinstance(n, ast.arg):
            names.add(n.arg)  # lambda arguments
    return names


def canonical_key(fn: ast.FunctionDef) -> str:
    """Hash of the function with identifiers anonymised (structure + constants + attribute names)."""
    mapping: Dict[str, str] = {fn.name: "F"}

    class Anon(ast.NodeTransformer):
        def visit_Name(self, node):
            node.id = mapping.setdefault(node.id, f"v{len(mapping)}") if node.id in bound else node.id
            return node

        def visit_arg(self, node):
            node.arg = mapping.setdefault(node.arg, f"v{len(mapping)}")
            node.annotation = None
            return node

    bound = _bound(fn) | {fn.name}
    f2 = Anon().visit(ast.parse(ast.unparse(fn)).body[0])
    f2.name, f2.returns = "F", None
    if f2.body and isinstance(f2.body[0], ast.Expr) and isinstance(getattr(f2.body[0], "value", None), ast.Constant):
        f2.body = f2.body[1:] or [ast.Pass()]  # docstrings do not make a function different
    return hashlib.sha1(ast.dump(f2, annotate_fields=False).encode()).hexdigest()


def _strip_docstring_and_annotations(fn: ast.FunctionDef) -> ast.FunctionDef:
    fn = ast.parse(ast.unparse(fn)).body[0]  # copy
    if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(getattr(fn.body[0], "value", None), ast.Constant) \
            and isinstance(fn.body[0].value.value, str):
        fn.body = fn.body[1:] or [ast.Pass()]
    for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs:
        a.annotation = None
    fn.returns = None
    return fn


def extract(text: str) -> List[Dict]:
    """All mineable functions in one source file (static filters only)."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return []
    from tinyperf.env.worker import FORBIDDEN_NAMES, StaticCheckError, static_check

    safe = _safe_names()
    imports = _module_imports(tree)
    out = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.decorator_list:
            continue
        a = node.args
        n_params = len(a.posonlyargs) + len(a.args)
        if not (1 <= n_params <= 3) or a.vararg or a.kwarg or a.kwonlyargs:
            continue
        try:
            fn = _strip_docstring_and_annotations(node)
            body_src = ast.unparse(fn)
        except (ValueError, RecursionError, MemoryError):
            continue
        n_lines = body_src.count("\n") + 1
        if not (4 <= n_lines <= 40) or len(body_src) > 1500:
            continue
        kinds = {type(n) for n in ast.walk(fn)}
        if kinds & {ast.Yield, ast.YieldFrom, ast.Await, ast.Global, ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef,
                    ast.Import, ast.ImportFrom, ast.With} or \
                any(isinstance(n, ast.FunctionDef) and n is not fn for n in ast.walk(fn)):
            continue
        if not kinds & {ast.For, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp}:
            continue
        free = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} - _bound(fn) - {fn.name}
        if free & FORBIDDEN_NAMES or any(isinstance(n, ast.Attribute) and n.attr.startswith("__") for n in ast.walk(fn)):
            continue
        needed = sorted(free - safe)
        if any(x not in imports for x in needed):
            continue  # reads a module-level name we cannot carry (helper, constant, class, non-whitelisted module)
        src = "\n".join([imports[x] for x in needed] + [body_src]) + "\n"
        try:
            static_check(ast.parse(src), fn.name, ALLOWED_MODULES)
        except (StaticCheckError, SyntaxError, RecursionError, MemoryError):
            continue
        out.append({"func_name": fn.name, "source": src, "key": canonical_key(fn), "n_lines": n_lines})
    return out


def iter_corpus(corpus: str, max_files: int) -> Iterator[str]:
    n = 0
    if os.path.isdir(corpus):
        for p in sorted(Path(corpus).rglob("*.py")):
            try:
                yield p.read_text(errors="ignore")
            except OSError:
                continue
            n += 1
            if max_files and n >= max_files:
                return
        return
    with open(corpus) as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line).get("content", "")
            n += 1
            if max_files and n >= max_files:
                return


def builtin_seed_keys() -> set:
    from tinyperf.data.seeds import SEEDS

    keys = set()
    for s in SEEDS:
        for src in [s.source] + list(s.slow_variants):
            try:
                fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef))
                keys.add(canonical_key(fn))
            except (StopIteration, SyntaxError):
                pass
    return keys


# --------------------------------------------------------------------------- #
# Stage 2: dynamic validation (runs inside rlimited pool processes)
# --------------------------------------------------------------------------- #
class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _init_worker(mem_mb: int) -> None:
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 2**20, mem_mb * 2**20))
    except (ValueError, OSError):
        pass
    signal.signal(signal.SIGALRM, _alarm)
    sys.setrecursionlimit(2000)
    import io
    sys.stdout = io.StringIO()  # mined code cannot print anyway (print is not a safe builtin), belt and braces


def _call(fn, args, timeout: float):
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return "ok", fn(*args)
    except _Timeout:
        return "timeout", None
    except (MemoryError, RecursionError):
        return "resource", None
    except Exception as e:  # noqa: BLE001 - the function's own exceptions are part of its behaviour
        return "exc", type(e).__name__
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


_PLAIN = (type(None), bool, int, float, complex, str, bytes)


def _plain(v, depth: int = 0) -> bool:
    if isinstance(v, _PLAIN):
        return True
    if depth > 4:
        return False
    if isinstance(v, (list, tuple, set, frozenset)):
        return len(v) < 100000 and all(_plain(x, depth + 1) for x in list(v)[:200])
    if isinstance(v, dict):
        return len(v) < 100000 and all(_plain(k, depth + 1) and _plain(x, depth + 1) for k, x in list(v.items())[:200])
    return False


def _time_at(fn, specs, rng, n: int, reps: int = 3, timeout: float = 1.0) -> Optional[float]:
    ts = []
    for _ in range(reps):
        args = tuple(gen_value(s, rng, n) for s in specs)
        t0 = time.perf_counter()
        st, _ = _call(fn, args, timeout)
        dt = time.perf_counter() - t0
        if st in ("timeout", "resource"):
            return None
        ts.append(dt)
    return median(ts)


def validate(cand: Dict) -> Optional[Dict]:
    try:
        from tinyperf.env.worker import load_function

        src, name = cand["source"], cand["func_name"]
        tree = ast.parse(src)
        fnode = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        fn = load_function(src, name, ALLOWED_MODULES)
        rng = random.Random(int(cand["key"][:8], 16))
        for specs in ranked_combos(guess_specs(fnode), cap=16):
            ok, outs, det, bad = 0, set(), True, False
            for n in SIZES:
                args = tuple(gen_value(s, rng, n) for s in specs)
                st, v = _call(fn, deepcopy(args), 0.2)
                if st in ("timeout", "resource"):
                    bad = True
                    break
                if st == "ok":
                    if not _plain(v):
                        bad = True
                        break
                    st2, v2 = _call(fn, deepcopy(args), 0.2)
                    det = det and st2 == "ok" and repr(v) == repr(v2)
                    ok += 1
                    outs.add(repr(v)[:300])
            if bad or not det or ok < 0.75 * len(SIZES) or len(outs) < 2 or outs == {"None"}:
                continue
            t_small = _time_at(fn, specs, rng, 64)
            t_big = _time_at(fn, specs, rng, 512)
            if t_small is None or t_big is None or t_big < 2e-5 or t_big > 0.3 or t_big / max(t_small, 1e-9) < 2.5:
                continue
            natural = []
            try:
                from tinyperf.data.rewrites import propose

                for fam, rew in propose(src)[:2]:
                    natural.append(rew)
            except Exception:  # noqa: BLE001
                pass
            return {"seed_name": f"mined_{cand['key'][:12]}", "func_name": name, "source": src, "arg_specs": list(specs),
                    "min_n": 0, "natural": natural,
                    "stats": {"n_lines": cand["n_lines"], "ok_frac": ok / len(SIZES), "t64_us": t_small * 1e6,
                              "t512_us": t_big * 1e6, "scaling": t_big / max(t_small, 1e-9)}}
        return None
    except Exception:  # noqa: BLE001 - never let one function kill the pool
        return None


def validate_batch(batch: List[Dict]) -> List[Optional[Dict]]:
    return [validate(c) for c in batch]


def run_validation(cands: List[Dict], workers: int, mem_mb: int, chunk: int = 4, stall_s: float = 120.0) -> Iterator[Optional[Dict]]:
    """Ordered pool map with a watchdog: a batch stuck in C code for `stall_s` is skipped and the pool rebuilt."""
    batches = [cands[i:i + chunk] for i in range(0, len(cands), chunk)]
    while batches:
        pool = mp.Pool(workers, initializer=_init_worker, initargs=(mem_mb,), maxtasksperchild=100)
        it = pool.imap(validate_batch, batches, chunksize=1)   # chunksize=1 keeps next(timeout=...) available
        done = 0
        try:
            while done < len(batches):
                for r in it.next(timeout=stall_s):
                    yield r
                done += 1
            batches = []
            pool.close()
        except mp.TimeoutError:
            log.warning(f"validation stalled; skipping {len(batches[done])} candidate(s) and restarting the pool")
            batches = batches[done + 1:]
            pool.terminate()
        pool.join()


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", required=True, help="jsonl with a `content` field, or a directory of .py files")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--max_candidates", type=int, default=0, help="cap on distinct candidates validated (random sample)")
    p.add_argument("--mem_mb", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t0 = time.time()

    # stage 1
    seen, exclude = set(), builtin_seed_keys()
    cands: List[Dict] = []
    n_files = n_funcs = n_dup = 0
    with mp.Pool(a.workers) as pool:
        for found in pool.imap_unordered(extract, iter_corpus(a.corpus, a.max_files), chunksize=64):
            n_files += 1
            for c in found:
                n_funcs += 1
                if c["key"] in seen or c["key"] in exclude:
                    n_dup += 1
                    continue
                seen.add(c["key"])
                cands.append(c)
            if n_files % 50000 == 0:
                log.info(f"stage 1: {n_files} files, {n_funcs} candidate functions, {len(cands)} distinct")
    log.info(f"stage 1 done in {time.time() - t0:.0f}s: {n_files} files -> {n_funcs} candidate functions -> "
             f"{len(cands)} distinct ({n_dup} duplicates / hand-written look-alikes removed)")
    if a.max_candidates and len(cands) > a.max_candidates:
        cands = random.Random(a.seed).sample(cands, a.max_candidates)

    # stage 2
    t1 = time.time()
    kept = 0
    specs_count: Dict[str, int] = {}
    with open(a.out, "w") as f:
        for i, rec in enumerate(run_validation(cands, a.workers, a.mem_mb)):
            if rec is not None:
                kept += 1
                for s in rec["arg_specs"]:
                    specs_count[s] = specs_count.get(s, 0) + 1
                f.write(json.dumps(rec) + "\n")
            if (i + 1) % 5000 == 0:
                log.info(f"stage 2: {i + 1}/{len(cands)} validated, {kept} kept")
    n_nat = sum(1 for r in map(json.loads, open(a.out)) if r["natural"])
    log.info(f"stage 2 done in {time.time() - t1:.0f}s: {kept}/{len(cands)} functions kept -> {a.out}; "
             f"{n_nat} with natural-rewrite candidates; parameter types: {dict(sorted(specs_count.items(), key=lambda kv: -kv[1]))}")


if __name__ == "__main__":
    main()
