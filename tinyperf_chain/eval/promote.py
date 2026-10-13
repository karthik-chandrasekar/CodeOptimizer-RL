"""Development promotion gate: publish a refreshed bundle only after an integrated development comparison (10.4,
11.4 'publish a bundle only after integrated development evaluation').

    python -m tinyperf_chain.eval.promote --report artifacts/chain/runs/dev/report.json \\
        --candidate B4_r2 --incumbent B4 [--role-report roles_B4_r2.json --role-baseline roles_B4.json] \\
        [--min-success-diff-lo -0.02] [--out artifacts/chain/promotions/B4_r2.json]

The report is the paired cluster-bootstrap output of ``eval.aggregate --paired --reference <incumbent>``.  The gate
passes when the candidate's whole-system success is not worse than the incumbent beyond ``min_success_diff_lo``
(the lower interval endpoint of the paired difference) *and* its effective GM speedup is not worse beyond
``min_log_speedup_lo``, and - when role reports are given - the refreshed role's local metric did not regress by
more than ``max_role_regression``.  A role that improves its local metric but worsens the integrated chain is
exactly the case the gate exists for (10.4 'whole-system behavior as well as the role's local metric').
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Optional

ROLE_METRICS = {"m1": ("bundle", "mean_regret", -1), "m2": ("bundle", "useful_plan_rate", 1),
                "m2m3": ("actual", "accepted", 1), "m4": ("bundle", "detection_given_discrepancy", 1),
                "m5": (None, "correct_and_faster_rate", 1)}


def _role_value(rep: Dict[str, Any], check: str) -> Optional[float]:
    sub, key, _ = ROLE_METRICS[check]
    d = rep.get(check)
    if not isinstance(d, dict):
        return None
    d = d.get(sub) if sub else d
    if not isinstance(d, dict) or key not in d:
        return None
    return float(d[key])


def promotion_decision(report: Dict[str, Any], candidate: str, incumbent: str, *, min_success_diff_lo: float = -0.02,
                       min_log_speedup_lo: float = -0.02, role_report: Optional[Dict[str, Any]] = None,
                       role_baseline: Optional[Dict[str, Any]] = None, max_role_regression: float = 0.0) -> Dict[str, Any]:
    paired = report.get(f"paired_vs_{incumbent}", {})
    rows = {k: v for k, v in paired.items() if k.split("@")[0] == candidate}
    out: Dict[str, Any] = {"candidate": candidate, "incumbent": incumbent, "checked_at": time.time(), "checks": [], "promote": False}
    if not rows:
        out["reason"] = f"no paired comparison of {candidate} against {incumbent} in the report (run eval.aggregate --paired --reference {incumbent})"
        return out
    ok = True
    for name, r in sorted(rows.items()):
        s, g = r.get("success", {}), r.get("log_speedup", {})
        c1 = {"system": name, "metric": "success_diff_lo", "value": s.get("lo"), "threshold": min_success_diff_lo,
              "n_clusters": s.get("n_clusters"), "pass": s.get("lo") is not None and s["lo"] >= min_success_diff_lo}
        c2 = {"system": name, "metric": "log_speedup_diff_lo", "value": g.get("lo"), "threshold": min_log_speedup_lo,
              "pass": g.get("lo") is not None and g["lo"] >= min_log_speedup_lo}
        out["checks"] += [c1, c2]
        ok = ok and c1["pass"] and c2["pass"]
    if role_report is not None and role_baseline is not None:
        for check in ROLE_METRICS:
            a, b = _role_value(role_report, check), _role_value(role_baseline, check)
            if a is None or b is None:
                continue
            sign = ROLE_METRICS[check][2]
            delta = sign * (a - b)
            c = {"role_check": check, "candidate": a, "incumbent": b, "improvement": delta, "pass": delta >= -max_role_regression}
            out["checks"].append(c)
            ok = ok and c["pass"]
    out["promote"] = ok
    out["reason"] = "all gates passed" if ok else "a gate failed: keep the incumbent bundle and improve data (16.3)"
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--incumbent", required=True)
    p.add_argument("--role-report", default=None)
    p.add_argument("--role-baseline", default=None)
    p.add_argument("--min-success-diff-lo", type=float, default=-0.02)
    p.add_argument("--min-log-speedup-lo", type=float, default=-0.02)
    p.add_argument("--max-role-regression", type=float, default=0.0)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    with open(a.report) as f:
        report = json.load(f)
    rr = rb = None
    if a.role_report and a.role_baseline:
        with open(a.role_report) as f:
            rr = json.load(f)
        with open(a.role_baseline) as f:
            rb = json.load(f)
    d = promotion_decision(report, a.candidate, a.incumbent, min_success_diff_lo=a.min_success_diff_lo,
                           min_log_speedup_lo=a.min_log_speedup_lo, role_report=rr, role_baseline=rb,
                           max_role_regression=a.max_role_regression)
    print(json.dumps(d, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(d, f, indent=1)
    raise SystemExit(0 if d["promote"] else 3)


if __name__ == "__main__":
    main()
