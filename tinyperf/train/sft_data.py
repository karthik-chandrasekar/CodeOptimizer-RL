"""Stage 2a - generate agent-SFT trajectories  (s_t -> a_t).

    python -m tinyperf.train.sft_data --config configs/sft_data.yaml

Three sources of edits (``sft_data.methods``):

* ``programmatic`` (Method A) - replay the known degradation chain in reverse
  (``C_slow -> ... -> C_fast``) and a direct jump (``C_slow -> C_fast``);
* ``search``       (Method C) - beam search with the inverse-rewrite proposer
  (:mod:`tinyperf.data.rewrites`); *execution* selects what is kept;
* ``teacher``      (Method B) - a stronger model proposes candidates from the
  source code alone; execution selects.  Requires ``pip install anthropic`` and
  ``ANTHROPIC_API_KEY``.

Every trajectory is *replayed through the real environment* so the recorded
observations carry genuine runtime feedback, and failed experiments are
injected (``failure_injection_prob``): a verified-incorrect or slower edit
followed by recovery, so the policy learns what to do after feedback says "no".
"""
from __future__ import annotations

import json
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple

from tinyperf.common.config import Config, parse_cli
from tinyperf.common.utils import get_logger
from tinyperf.data.astutil import normalize
from tinyperf.data.rename import mutate_bug, mutate_syntax
from tinyperf.data.rewrites import propose
from tinyperf.env.env import PerfEnv
from tinyperf.env.executor import Executor
from tinyperf.env.protocol import Action, EDIT_CLOSE, EDIT_OPEN, format_action
from tinyperf.env.task import Task, load_tasks

log = get_logger("sft_data")

Proposer = Callable[[str], List[Tuple[str, str]]]  # code -> [(family, candidate)]


@dataclass
class TrajStep:
    obs: str                 # observation actually shown (full feedback)
    action: str              # formatted action text
    status: str              # env status for this action
    ratio: Optional[float]   # T_cand / T_0 when measured
    step: int                # env.step_idx at observation time
    remaining: int
    code_shown: str          # code inside the observation (current best)


def replay(env: PerfEnv, task: Task, edits: List[str], stop: bool = True) -> Optional[Dict]:
    """Replay a list of candidate sources through the env; return a trajectory row."""
    env.reset(task)
    steps: List[TrajStep] = []
    for code in edits:
        if env.done:
            break
        obs = env.observation()
        st = env.step_idx
        act = Action("edit", code=code)
        res = env.step_action(act)
        steps.append(TrajStep(obs, format_action(act), res.record.status, res.record.ratio, st, env.horizon - st, env.best_code))
    if stop and not env.done:
        obs = env.observation()
        st = env.step_idx
        res = env.step_action(Action("stop"))
        steps.append(TrajStep(obs, format_action(Action("stop")), "stop", None, st, env.horizon - st, env.best_code))
    summ = env.summary()
    if not steps:
        return None
    return {
        "task_id": task.task_id,
        "source": task.source,
        "func_name": task.func_name,
        "families": task.families,
        "steps": [asdict(s) for s in steps],
        "final_ratio": summ.best_ratio,
        "final_code": summ.best_code,
        "n_edits": summ.n_edits,
        "n_invalid": summ.n_invalid,
        "recovered": summ.recovered,
        "stopped": summ.stopped,
    }


# --------------------------------------------------------------------------- #
# Failure injection
# --------------------------------------------------------------------------- #
def inject_failure(edits: List[str], rng: random.Random, horizon: int, rejected_pool: Optional[List[str]] = None) -> List[str]:
    """Insert one plausible-but-bad edit before a random edit (if the horizon allows)."""
    if len(edits) + 1 > horizon - 1:
        return edits
    i = rng.randrange(len(edits))
    r = rng.random()
    bad: Optional[str] = None
    if rejected_pool and r < 0.4:
        bad = rng.choice(rejected_pool)
    elif r < 0.85:
        bad = mutate_bug(edits[i], rng)
    else:
        bad = mutate_syntax(edits[i], rng)
    if bad is None or bad in edits:
        return edits
    return edits[:i] + [bad] + edits[i:]


# --------------------------------------------------------------------------- #
# Method A - programmatic
# --------------------------------------------------------------------------- #
def programmatic_edit_lists(task: Task, rng: random.Random, per_task: int, horizon: int, p_fail: float) -> List[List[str]]:
    chain = [normalize(c) for c in task.chain] if task.chain else []
    if len(chain) < 2:
        if task.fast_source:
            chain = [normalize(task.fast_source), normalize(task.source)]
        else:
            return []
    if chain[-1] != normalize(task.source):
        return []
    full = list(reversed(chain[:-1]))          # C_{n-1}, ..., C_fast
    jump = [chain[0]]
    lists: List[List[str]] = []
    if len(full) > 1:
        lists.append(full)
        lists.append(jump)
    else:
        lists.append(jump)
    out = []
    for i in range(per_task):
        base = lists[i % len(lists)]
        if rng.random() < p_fail or (i >= len(lists)):
            base = inject_failure(base, rng, horizon)
        out.append(base[: horizon - 1])
    # de-duplicate identical lists
    uniq, seen = [], set()
    for e in out:
        k = tuple(e)
        if k not in seen:
            seen.add(k)
            uniq.append(e)
    return uniq


# --------------------------------------------------------------------------- #
# Method B/C - execution-selected search with a proposer
# --------------------------------------------------------------------------- #
def search_edit_lists(task: Task, ex: Executor, env_cfg, proposer: Proposer, rng: random.Random, beam: int, n_candidates: int,
                      max_depth: int, p_fail: float) -> Tuple[List[List[str]], Dict]:
    delta = env_cfg.improvement_delta
    src = normalize(task.source)
    frontier: List[Tuple[float, str]] = [(1.0, src)]
    path: Dict[str, List[str]] = {src: []}
    rejected: List[str] = []
    best = (1.0, src)
    n_exec = 0
    seen = {src}
    for depth in range(max_depth):
        new: List[Tuple[float, str]] = []
        for ratio, code in frontier:
            props = [(f, normalize(c)) for f, c in proposer(code)]
            props = [(f, c) for f, c in props if c not in seen]
            rng.shuffle(props)
            for fam, cand in props[:n_candidates]:
                seen.add(cand)
                r = ex.evaluate(cand, task, baseline_code=task.source, timing_seed=n_exec)
                n_exec += 1
                if r.get("status") == "ok" and r.get("ratio") is not None and float(r["ratio"]) < (1 - delta) * ratio:
                    new.append((float(r["ratio"]), cand))
                    path[cand] = path[code] + [cand]
                else:
                    rejected.append(cand)
        if not new:
            break
        new.sort(key=lambda t: t[0])
        frontier = new[:beam]
        if frontier[0][0] < best[0]:
            best = frontier[0]
    stats = {"n_exec": n_exec, "best_ratio": best[0], "depth": len(path[best[1]])}
    if best[1] == src:
        return [], stats
    edits = path[best[1]]
    lists = [edits]
    if len(edits) > 1:
        lists.append([best[1]])  # the direct jump is also a valid demonstration
    if rng.random() < p_fail:
        lists[0] = inject_failure(edits, rng, env_cfg.horizon, rejected_pool=rejected)
    return lists, stats


def teacher_proposer(model: str, n: int) -> Proposer:
    """Anthropic-API proposer: sees only the code, returns candidate rewrites; execution selects."""
    import anthropic  # lazy: optional dependency

    client = anthropic.Anthropic()
    sys_prompt = (
        "You optimise Python functions for runtime. You will be given ONE function. "
        f"Reply with {n} alternative implementations that preserve its exact behaviour (return value, type, exceptions, argument mutation) "
        "but run faster, each in its own <EDIT>...</EDIT> block containing only the complete function definition with the same name and signature. "
        "Vary the strategies (data structures, hoisting, builtins, algorithmic changes). No prose, no markdown fences."
    )
    block = re.compile(re.escape(EDIT_OPEN) + r"\s*(.*?)\s*" + re.escape(EDIT_CLOSE), re.S)

    def _propose(code: str) -> List[Tuple[str, str]]:
        try:
            msg = client.messages.create(model=model, max_tokens=4000, system=sys_prompt, messages=[{"role": "user", "content": code}])
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        except Exception as e:  # noqa: BLE001
            log.warning(f"teacher call failed: {e}")
            return []
        out = []
        for m in block.finditer(text):
            try:
                out.append(("teacher", normalize(m.group(1))))
            except SyntaxError:
                continue
        return out

    return _propose


def rewrite_proposer(code: str) -> List[Tuple[str, str]]:
    return propose(code)


# --------------------------------------------------------------------------- #
def complete_enough(row: Dict, task: Task, min_ref_frac: float, drops: Dict[str, int]) -> bool:
    """Reject demonstrations that stop well short of the known fast version (they teach partial fixes + STOP)."""
    if min_ref_frac <= 0 or not task.reference_speedup:
        return True
    ok = 1.0 / max(row["final_ratio"], 1e-9) >= min_ref_frac * float(task.reference_speedup)
    if not ok:
        drops["partial"] = drops.get("partial", 0) + 1
    return ok


def generate(cfg: Config) -> str:
    sd, env_cfg = cfg.sft_data, cfg.env
    ex = Executor(env_cfg)
    tasks = load_tasks(sd.tasks, limit=sd.n_tasks)
    rng0 = random.Random(cfg.data.seed)
    horizon = env_cfg.horizon
    os.makedirs(os.path.dirname(sd.out_path) or ".", exist_ok=True)
    proposers: List[Tuple[str, Proposer]] = []
    if "search" in sd.methods:
        proposers.append(("search", rewrite_proposer))
    if "teacher" in sd.methods:
        try:
            proposers.append(("teacher", teacher_proposer(sd.teacher_model, sd.teacher_candidates)))
        except Exception as e:  # noqa: BLE001
            log.warning(f"teacher proposer unavailable ({e}); skipping Method B")
    lock = threading.Lock()
    stats = {"rows": 0, "programmatic": 0, "search": 0, "teacher": 0, "with_failure": 0, "search_exec": 0, "skipped": 0}
    drops: Dict[str, int] = {}
    fout = open(sd.out_path, "w")

    def work(i_task: Tuple[int, Task]) -> None:
        i, task = i_task
        rng = random.Random(rng0.random() * 1e9 + i)
        env = PerfEnv(ex, env_cfg, feedback="full", horizon=horizon)
        rows = []
        if "programmatic" in sd.methods:
            for edits in programmatic_edit_lists(task, rng, sd.programmatic_per_task, horizon, sd.failure_injection_prob):
                row = replay(env, task, edits)
                if row and row["final_ratio"] < 1.0 and complete_enough(row, task, sd.min_ref_frac, drops):
                    row["method"] = "programmatic"
                    rows.append(row)
        for name, prop in proposers:
            depth = sd.teacher_max_depth if name == "teacher" else horizon - 1
            n_cand = sd.teacher_candidates if name == "teacher" else sd.search_candidates
            lists, st = search_edit_lists(task, ex, env_cfg, prop, rng, sd.search_beam, n_cand, depth, sd.failure_injection_prob)
            with lock:
                stats["search_exec"] += st["n_exec"]
            for edits in lists:
                row = replay(env, task, edits)
                if row and row["final_ratio"] < 1.0 and complete_enough(row, task, sd.min_ref_frac, drops):
                    row["method"] = name
                    rows.append(row)
        with lock:
            for row in rows:
                fout.write(json.dumps(row) + "\n")
                stats["rows"] += 1
                stats[row["method"]] += 1
                stats["with_failure"] += int(row["n_invalid"] > 0 or any(s["status"] in ("ok",) and not s["ratio"] is None and s["ratio"] >= 1.0 for s in row["steps"]))
            if not rows:
                stats["skipped"] += 1
            fout.flush()
            if (i + 1) % 50 == 0:
                log.info(f"{i+1}/{len(tasks)} tasks | {stats}")

    with ThreadPoolExecutor(max_workers=max(1, env_cfg.max_parallel_envs)) as pool:
        futs = [pool.submit(work, (i, t)) for i, t in enumerate(tasks)]
        for f in as_completed(futs):
            f.result()
    fout.close()
    ex.close()
    stats["dropped_partial"] = drops.get("partial", 0)
    log.info(f"wrote {stats['rows']} trajectories to {sd.out_path} | {stats}")
    with open(os.path.splitext(sd.out_path)[0] + "_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return sd.out_path


def main(argv: Optional[List[str]] = None) -> None:
    generate(parse_cli(argv))


if __name__ == "__main__":
    main()
