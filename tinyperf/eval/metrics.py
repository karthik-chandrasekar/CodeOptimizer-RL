"""Metrics over episode records (see :func:`episode_record` for the schema).

All functions are pure so they can be used by the trainer (periodic eval), the
evaluator, the ablation runner and the tests.

Record schema (one per rollout)::

    {
      "task_id": str, "families": [str], "seed_name": str, "reference_speedup": float|None,
      "best_ratio": float,            # T_best / T_0   (1.0 = returned the original)
      "n_edits": int, "n_invalid": int, "n_correct": int,
      "stopped": bool,                # policy emitted <STOP> (vs. horizon exhausted)
      "recovered": bool,              # an invalid/incorrect edit was later followed by an improvement
      "reward": float, "horizon": int,
      "steps": [{"step": int, "action_kind": str, "status": str, "correct": bool,
                 "ratio": float|None, "improved": bool}],
    }
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

SUCCESS_THRESHOLD = 1.05
THRESHOLDS = (1.05, 1.25, 1.5, 2.0)


def _edited(raw: Optional[str]) -> List[str]:
    """Names of the functions an edit defined (file tasks: which functions the agent chose to change)."""
    try:
        from tinyperf.env.files import edited_functions
        from tinyperf.env.protocol import parse_action

        a = parse_action(raw or "")
        return edited_functions(a.code) if a.kind == "edit" and a.code else []
    except Exception:  # noqa: BLE001 - analysis only
        return []


def episode_record(summary, reward: float, task, horizon: int) -> Dict[str, Any]:
    """Build a record from an :class:`tinyperf.env.env.EpisodeSummary`."""
    return {
        "task_id": task.task_id,
        "families": list(task.families),
        "seed_name": task.seed_name,
        "reference_speedup": task.reference_speedup,
        "best_ratio": float(summary.best_ratio),
        "best_code": summary.best_code,
        "n_edits": summary.n_edits,
        "n_invalid": summary.n_invalid,
        "n_correct": summary.n_correct,
        "stopped": bool(summary.stopped),
        "recovered": bool(summary.recovered),
        "reward": float(reward),
        "horizon": int(horizon),
        "slow_functions": (task.meta or {}).get("slow_functions"),
        "steps": [{"step": s.step, "action_kind": s.action_kind, "status": s.status, "correct": bool(s.correct),
                   "ratio": s.ratio, "improved": bool(s.improved),
                   "edited": _edited(getattr(s, "raw_action", None)) if s.action_kind == "edit" else []} for s in summary.steps],
    }


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return float(sum(xs) / len(xs)) if xs else float("nan")


def speedup_of(rec: Dict[str, Any]) -> float:
    return 1.0 / max(float(rec["best_ratio"]), 1e-9)


def steps_to_best(rec: Dict[str, Any]) -> Optional[int]:
    """Index (1-based edit count) at which the final best was reached; None if never improved."""
    last = None
    for s in rec["steps"]:
        if s["action_kind"] == "edit" and s.get("improved"):
            last = s["step"]
    return last


def compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics for a list of episode records."""
    n = len(records)
    if n == 0:
        return {"n_episodes": 0}
    speedups = [speedup_of(r) for r in records]
    log_s = [math.log(s) for s in speedups]
    edits = [r["steps"] for r in records]
    # every non-STOP action is an "edit attempt" (malformed outputs count as invalid edits, as in the reward)
    all_edits = [s for st in edits for s in st if s["action_kind"] != "stop"]
    first_edits = [st[0] for st in edits if st and st[0]["action_kind"] != "stop"]
    with_failure = [r for r in records if r["n_invalid"] > 0]
    successes = [r for r in records if speedup_of(r) >= SUCCESS_THRESHOLD]
    with_ref = [r for r in records if r.get("reference_speedup") and r["reference_speedup"] > 1.0]

    m: Dict[str, Any] = {
        "n_episodes": n,
        "n_tasks": len({r["task_id"] for r in records}),
        # behaviour preservation: the returned program is always C0 or a verified-equivalent candidate (rollback),
        # so we report it explicitly together with the rate at which *proposed* edits were correct.
        "behavior_preservation": 1.0 if all(r["best_ratio"] <= 1.0 + 1e-12 or r["n_correct"] > 0 for r in records) else 0.0,
        "edit_correct_rate": _mean(float(s["correct"]) for s in all_edits) if all_edits else float("nan"),
        "first_edit_correct_rate": _mean(float(s["correct"]) for s in first_edits) if first_edits else float("nan"),
        "success_rate": len(successes) / n,
        "gm_speedup": math.exp(_mean(log_s)),
        "median_speedup": sorted(speedups)[n // 2],
        "mean_log_speedup": _mean(log_s),
        "regression_rate": _mean(float(r["best_ratio"] > 1.0 + 1e-9) for r in records),
        "slower_edit_rate": _mean(float(s["correct"] and s["ratio"] is not None and s["ratio"] > 1.0) for s in all_edits) if all_edits else float("nan"),
        "mean_edits": _mean(r["n_edits"] for r in records),
        "mean_invalid": _mean(r["n_invalid"] for r in records),
        "invalid_edit_rate": _mean(float(not s["correct"]) for s in all_edits) if all_edits else float("nan"),
        "malformed_rate": _mean(float(s["status"] == "malformed") for s in all_edits) if all_edits else float("nan"),
        "stop_rate": _mean(float(r["stopped"]) for r in records),
        "mean_reward": _mean(r["reward"] for r in records),
        "zero_edit_rate": _mean(float(r["n_edits"] == 0) for r in records),
    }
    for t in THRESHOLDS:
        m[f"p_speedup_ge_{t}"] = _mean(float(s >= t) for s in speedups)
    # search efficiency
    stb = [steps_to_best(r) for r in successes]
    m["mean_edits_to_best"] = _mean(x for x in stb if x is not None) if any(x is not None for x in stb) else float("nan")
    m["wasted_edits_after_best"] = _mean(r["n_edits"] - x for r, x in zip(successes, stb) if x is not None) if any(x is not None for x in stb) else float("nan")
    m["edits_per_success"] = (sum(r["n_edits"] for r in records) / len(successes)) if successes else float("inf")
    # recovery
    m["failure_episode_rate"] = len(with_failure) / n
    m["recovery_rate"] = _mean(float(r["recovered"]) for r in with_failure) if with_failure else float("nan")
    # fraction of the known reference speedup achieved (in log space, clipped to [0, 1+])
    if with_ref:
        m["reference_fraction"] = _mean(max(0.0, math.log(speedup_of(r)) / math.log(r["reference_speedup"])) for r in with_ref)
        m["matched_reference_rate"] = _mean(float(speedup_of(r) >= 0.9 * r["reference_speedup"]) for r in with_ref)
    # status histogram
    hist: Dict[str, int] = defaultdict(int)
    for s in all_edits:
        hist[s["status"]] += 1
    m["status_counts"] = dict(sorted(hist.items()))
    return m


def per_family_metrics(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Success / GM speedup broken down by degradation family (a record counts for each family it contains)."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        for f in r.get("families") or ["unknown"]:
            groups[f].append(r)
    out = {}
    for f, rs in sorted(groups.items()):
        mm = compute_metrics(rs)
        out[f] = {k: mm[k] for k in ("n_episodes", "success_rate", "gm_speedup", "mean_edits", "invalid_edit_rate", "recovery_rate") if k in mm}
    return out


def per_depth_metrics(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups[len(r.get("families") or [])].append(r)
    return {str(d): {k: v for k, v in compute_metrics(rs).items() if k in ("n_episodes", "success_rate", "gm_speedup", "mean_edits")}
            for d, rs in sorted(groups.items())}


def compare(a: Dict[str, Any], b: Dict[str, Any], keys=("success_rate", "gm_speedup", "recovery_rate", "mean_edits", "invalid_edit_rate")) -> Dict[str, float]:
    """Difference a - b on the headline metrics (e.g. feedback gain = full-feedback minus no-feedback)."""
    out = {}
    for k in keys:
        if k in a and k in b and isinstance(a[k], (int, float)) and isinstance(b[k], (int, float)):
            out[f"delta_{k}"] = float(a[k]) - float(b[k])
    if "gm_speedup" in a and "gm_speedup" in b:
        out["ratio_gm_speedup"] = float(a["gm_speedup"]) / max(float(b["gm_speedup"]), 1e-9)
    return out


def bootstrap_ci(records: List[Dict[str, Any]], metric: str = "success_rate", n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
                 cluster: str = "seed_name"):
    """Cluster bootstrap CI: resamples whole seed functions (renamed copies of one function are not independent)."""
    import random

    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_task[r.get(cluster) or r["task_id"]].append(r)
    tasks = list(by_task)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample = [r for t in (rng.choice(tasks) for _ in tasks) for r in by_task[t]]
        vals.append(compute_metrics(sample)[metric])
    vals.sort()
    return {"lo": vals[int(alpha / 2 * n_boot)], "hi": vals[int((1 - alpha / 2) * n_boot) - 1], "point": compute_metrics(records)[metric]}


def format_table(rows: Dict[str, Dict[str, Any]], keys: List[str]) -> str:
    """Plain-text table: row name + selected metric columns."""
    head = f"{'':28s}" + "".join(f"{k:>18s}" for k in keys)
    lines = [head, "-" * len(head)]
    for name, m in rows.items():
        cells = []
        for k in keys:
            v = m.get(k, float("nan"))
            cells.append(f"{v:18.3f}" if isinstance(v, float) else f"{str(v):>18s}")
        lines.append(f"{name:28s}" + "".join(cells))
    return "\n".join(lines)
