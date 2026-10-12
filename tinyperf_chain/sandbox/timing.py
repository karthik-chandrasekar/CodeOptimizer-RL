"""Paired-timing statistics and acceptance decisions (trusted, pure functions).

For block ``b`` with reference (current best, or the original in the audit) time ``t_s,b`` and candidate time
``t_c,b``::

    l_b = log(t_s,b / t_c,b)          S = exp(median_b l_b)

``paired_log_median_ci_lower`` (the registered rule): bootstrap the median of the ``l_b`` (percentile interval);
accept when the lower endpoint is >= log(threshold), label ``no_gain`` when the upper endpoint is below it, and
``uncertain`` otherwise (rejected).  With few blocks this interval is a stability heuristic, not a calibrated
coverage guarantee.  Threshold 1.03 means a 1.03-fold speedup (not a 3 % time reduction).

``legacy_ratio`` reproduces the TinyPerf-RL rule: ratio of medians, accept when ``ratio < 1 - delta``.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence


def median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        raise ValueError("median of empty sequence")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def paired_log_ratios(ref_ns: Sequence[float], cand_ns: Sequence[float]) -> List[float]:
    if len(ref_ns) != len(cand_ns):
        raise ValueError("paired blocks must have equal length")
    return [math.log(max(r, 1.0) / max(c, 1.0)) for r, c in zip(ref_ns, cand_ns)]


def bootstrap_median_interval(ls: Sequence[float], draws: int = 1000, level: float = 0.95, seed: int = 0):
    if not ls:
        raise ValueError("no blocks")
    if len(ls) == 1:
        return ls[0], ls[0]
    rng = random.Random(seed)
    n = len(ls)
    meds = sorted(median([ls[rng.randrange(n)] for _ in range(n)]) for _ in range(draws))
    lo_i = int(math.floor((1 - level) / 2 * draws))
    hi_i = min(draws - 1, int(math.ceil((1 + level) / 2 * draws)) - 1)
    return meds[lo_i], meds[hi_i]


@dataclass
class TimingVerdict:
    decision: str                 # "accept" | "no_gain" | "uncertain" | "slower"
    speedup: float                # exp(median l_b)
    lo: Optional[float]           # speedup-scale interval (None for rules without an interval)
    hi: Optional[float]
    n_blocks: int
    rule: str
    threshold: float
    log_ratios: List[float]

    @property
    def accepted(self) -> bool:
        return self.decision == "accept"

    def to_dict(self) -> Dict:
        return asdict(self)


def timing_verdict(ref_ns: Sequence[float], cand_ns: Sequence[float], *, threshold: float = 1.03,
                   rule: str = "paired_log_median_ci_lower", draws: int = 1000, level: float = 0.95, seed: int = 0,
                   legacy_delta: float = 0.03) -> TimingVerdict:
    ls = paired_log_ratios(ref_ns, cand_ns)
    s = math.exp(median(ls))
    lt = math.log(threshold)
    if rule == "legacy_ratio":
        ratio = median(cand_ns) / max(1.0, median(ref_ns))
        dec = "accept" if ratio < (1.0 - legacy_delta) else ("slower" if ratio > 1.0 else "no_gain")
        return TimingVerdict(dec, 1.0 / ratio if ratio > 0 else float("inf"), None, None, len(ls), rule, threshold, ls)
    if rule == "point_estimate":
        dec = "accept" if median(ls) >= lt else ("slower" if s < 1.0 else "no_gain")
        return TimingVerdict(dec, s, None, None, len(ls), rule, threshold, ls)
    if rule != "paired_log_median_ci_lower":
        raise ValueError(f"unknown timing rule {rule!r}")
    lo, hi = bootstrap_median_interval(ls, draws, level, seed)
    if lo >= lt:
        dec = "accept"
    elif hi < lt:
        dec = "slower" if hi < 0.0 else "no_gain"
    else:
        dec = "uncertain"
    return TimingVerdict(dec, s, math.exp(lo), math.exp(hi), len(ls), rule, threshold, ls)
