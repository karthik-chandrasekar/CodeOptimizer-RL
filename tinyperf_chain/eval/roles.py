"""Role-level development diagnostics (supportive; the primary endpoint is whole-system final-audited success).

    python -m tinyperf_chain.eval.roles --config configs/chain/eval_dev.yaml --bundle bundles/B4.json --split dev \\
        --checks m1 m2m3 m4 m5 [--limit 50] [--out roles_report.json]

m1    attainable regret against the measured target utilities (``target_label`` records): bundle M1 vs hottest-self,
      hottest-cumulative, random and opportunity rules on the same public profile; valid-id rate; consistency of the
      choice under a different catalog order / id permutation (position or name shortcuts show up here).
m2m3  counterfactual plan tests for the writer: actual plan, plan removed, a different applicable plan substituted,
      operands scrambled; acceptance per condition, plan adherence (does the patch instantiate the requested
      mechanism/operands) and how often the implementation changes with the plan.  A writer that ignores M2 shows
      no difference across conditions.
m4    discrepancy detection at a fixed probe budget on real and near-miss failing candidates (bundle M4 vs random,
      fixed edge suite and transformation-specific rules), and mean selection size on valid candidates.
m5    repairs that become correct, correct and faster (accepted), and abandonment precision / recall.
"""
from __future__ import annotations

import argparse
import ast
import json
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

from tinyperf_chain import registry as R
from tinyperf_chain.adapters.base import RoleAdapter
from tinyperf_chain.adapters.rules import RuleProbeSelector, RuleTargetSelector
from tinyperf_chain.config import load_chain_config
from tinyperf_chain.context import (CatalogEntry, M1View, M3View, M4View, M5View, calls_bucket, compact_diff,
                                    share_bucket)
from tinyperf_chain.data.collect import Collector
from tinyperf_chain.program import Program, dependency_closure
from tinyperf_chain.schemas import Plan, RequestEnvelope, SCHEMA_VERSIONS, format_plan, parse_plan


def _env(role: str, rid: str, max_new: int) -> RequestEnvelope:
    return RequestEnvelope(role, "diag", rid, "", "diag", SCHEMA_VERSIONS[role], max_new, float("inf"))


def _call(adapter: RoleAdapter, role: str, view, ctx, rid: str):
    prep = adapter.prepare(role, view, ctx)
    return adapter.execute(prep, _env(role, rid, prep.max_new_tokens))


class RoleDiagnostics:
    def __init__(self, cfg, router, split: str = "dev", seed: int = 0):
        from tinyperf_chain.adapters.verifier import Verifier

        self.cfg, self.router = cfg, router
        self.col = Collector(cfg, Verifier(cfg, cache_dir=cfg.paths.cache), split, seed)
        self.tasks = self.col._task_index()
        self.trials = [r for r in self.col.ledger.read() if r.get("split") == split]
        self.rng = random.Random(seed)

    def _state(self, r):
        task, private = self.tasks[r["task_id"]]
        return task, private, Program(self.col.objects.get_text(r["state_ref"]), tuple(task.functions))

    # ------------------------------------------------------------------ #
    def m1(self, limit: int = 0) -> Dict[str, Any]:
        recs = [r for r in self.trials if r["kind"] == "target_label" and r["task_id"] in self.tasks]
        if limit:
            recs = recs[:limit]
        selectors: Dict[str, RoleAdapter] = {"bundle": self.router["M1"]}
        for s in ("hottest_self", "hottest_cumulative", "random", "opportunity"):
            selectors[s] = RuleTargetSelector(s, seed=7)
        acc = {k: defaultdict(float) for k in selectors}
        for i, r in enumerate(recs):
            task, _, state = self._state(r)
            umax = max((v["U"] for v in r["utilities"].values()), default=0.0)
            picks = []
            for perm in range(2):
                rng = random.Random(f"{r['trial_id']}|{perm}")
                ids = list(range(len(task.functions)))
                rng.shuffle(ids)
                disp = {n: f"f{ids[k]:02d}" for k, n in enumerate(task.functions)}
                order = list(task.functions)
                rng.shuffle(order)
                entries, cat = [], []
                for n in order:
                    fn = state.function_node(n)
                    src = state.function_source(n)
                    p = r["profile"].get(n, {"ncalls": 0, "self_share": 0.0, "cum_share": 0.0})
                    entries.append(CatalogEntry(disp[n], n, src, share_bucket(p["self_share"]), share_bucket(p["cum_share"]),
                                                calls_bucket(p["ncalls"]), len(src.splitlines()), len(R.anchors(fn)), 0,
                                                summary=src.splitlines()[0]))
                    cat.append({"display_id": disp[n], "name": n, "self_share": p["self_share"], "cum_share": p["cum_share"],
                                "ncalls": p["ncalls"], "has_plan": bool(R.static_plans(fn, task.contract.function(n), 1))})
                inv = {v: k for k, v in disp.items()}
                view = M1View(entries, [], self.cfg.budget.patch_slots)
                for name, ad in selectors.items():
                    resp = _call(ad, "M1", view, {"catalog": cat}, f"m1|{i}|{perm}")
                    a = acc[name]
                    a["n"] += 1
                    if not resp.ok:
                        a["invalid"] += 1
                        a["regret"] += umax
                        if perm == 0 and name == "bundle":
                            picks.append(None)
                        continue
                    fid = resp.parsed.payload.function_id
                    chosen = inv.get(fid) if fid else None
                    if fid and chosen is None:
                        a["invalid"] += 1
                    u = r["utilities"].get(chosen, {"U": 0.0})["U"] if chosen else 0.0
                    a["regret"] += umax - u
                    a["hit"] += float((chosen in r["label"]) if r["label"] else chosen is None)
                    if name == "bundle":
                        picks.append(chosen)
            if len(picks) == 2:
                acc["bundle"]["consistent"] += float(picks[0] == picks[1])
                acc["bundle"]["pairs"] += 1
        out = {}
        for k, a in acc.items():
            n = max(1.0, a["n"])
            out[k] = {"n": int(a["n"]), "mean_regret": a["regret"] / n, "hit_rate": a["hit"] / n,
                      "valid_id_rate": 1 - a["invalid"] / n}
        if acc["bundle"]["pairs"]:
            out["bundle"]["order_consistency"] = acc["bundle"]["consistent"] / acc["bundle"]["pairs"]
        return out

    # ------------------------------------------------------------------ #
    def m2m3(self, limit: int = 0) -> Dict[str, Any]:
        recs = [r for r in self.trials if r["kind"] == "plan_set" and r["task_id"] in self.tasks
                and any(p["useful"] for p in r["plans"])]
        if limit:
            recs = recs[:limit]
        res: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        changed = 0.0
        for i, r in enumerate(recs):
            task, private, state = self._state(r)
            name = r["function"]
            fn = state.function_node(name)
            fsrc = state.function_source(name)
            deps = [state.function_source(f) for f in dependency_closure(state, name).functions]
            actual = parse_plan(next(p["plan"] for p in r["plans"] if p["useful"]), "f00").payload
            others = [p for p in R.static_plans(fn, task.contract.function(name)) if p.key() != actual.key()]
            names = [n for n in R.operand_names(fn) if n != actual.operand("src")]
            conds = {"actual": format_plan(actual), "removed": "none"}
            if others:
                conds["substituted"] = format_plan(others[0])
            if names and actual.operand("src"):
                ops = tuple(f"src={self.rng.choice(names)}" if o.startswith("src=") else o for o in actual.operands)
                conds["scrambled"] = format_plan(Plan("f00", actual.transform_id, ops, actual.preserve))
            outs = {}
            for c, ptxt in conds.items():
                v = M3View("f00", fsrc, deps, task.contract.render(name), ptxt, R.render_anchors(fn))
                pr = parse_plan(ptxt, "f00")
                plan = pr.payload if pr.ok else Plan("f00", None)
                resp = _call(self.router["M3"], "M3", v, {"function_id": "f00", "function_name": name, "fn_source": fsrc,
                                                          "plan": plan, "file_source": state.source,
                                                          "legacy_state": {"best_runtime": 1.0, "last_runtime": "n/a",
                                                                           "last_correct": -1, "last_status": "start",
                                                                           "step": 0, "remaining": 6}}, f"m3|{i}|{c}")
                d = res[c]
                d["n"] += 1
                if not resp.ok or resp.parsed.payload.is_abandon:
                    d["abandon_or_invalid"] += 1
                    outs[c] = None
                    continue
                code = resp.parsed.payload.code
                outs[c] = code.strip()
                o, _ = self.col.verify(task, private, state, name, code, seed=i)
                d["accepted"] += float(bool(o.get("accepted")))
                d["correct"] += float(o.get("stage") == "timing")
                if plan.transform_id in R.MECHANISMS:
                    got = R.extract_plan(fsrc, code, plan.transform_id)
                    d["adherent"] += float(got is not None and (plan.operand("src") is None or got.operand("src") == plan.operand("src")))
            if outs.get("actual") is not None and any(outs.get(c) is not None and outs[c] != outs["actual"] for c in conds if c != "actual"):
                changed += 1
        out = {c: {k: (v / max(1.0, d["n"]) if k != "n" else int(v)) for k, v in d.items()} for c, d in res.items()}
        out["implementation_changes_with_plan"] = changed / max(1, len(recs))
        return out

    # ------------------------------------------------------------------ #
    def m4(self, limit: int = 0, budget: int = 3) -> Dict[str, Any]:
        recs = [r for r in self.trials if r["kind"] == "probe_matrix" and r["task_id"] in self.tasks]
        if limit:
            recs = recs[:limit]
        sels: Dict[str, RoleAdapter] = {"bundle": self.router["M4"], "random": RuleProbeSelector("random", budget, 3),
                                        "fixed_edge": RuleProbeSelector("fixed_edge", budget),
                                        "transform_specific": RuleProbeSelector("transform_specific", budget)}
        acc = {k: defaultdict(float) for k in sels}
        for i, r in enumerate(recs):
            task, _, state = self._state(r)
            name = r["function"]
            failing = any(r["outcomes"].values())
            cur = state.function_source(name)
            cand = self.col.objects.get_text(r["candidate_fn_ref"])
            menu_ids = [p for p, _ in r["menu"]]
            pr = parse_plan(r.get("plan") or "", "f00")
            plan = pr.payload if pr.ok else None
            v = M4View("f00", cur, cand, task.contract.render(name), r.get("plan") or "none",
                       ",".join(f"{p}:{n}" for p, n in r["menu"]) or "none", compact_diff(cur, cand))
            for k, ad in sels.items():
                resp = _call(ad, "M4", v, {"function_id": "f00", "menu_ids": menu_ids, "plan": plan}, f"m4|{i}")
                a = acc[k]
                ids = list(resp.parsed.payload.probe_ids) if resp.ok else []
                ok_sel = resp.ok and len(set(ids)) == len(ids) <= budget and all(p in menu_ids for p in ids)
                if not ok_sel:
                    a["invalid"] += 1
                    ids = []
                if failing:
                    a["n_failing"] += 1
                    a["detected"] += float(any(r["outcomes"].get(p) for p in ids))
                    a["probes_failing"] += len(ids)
                else:
                    a["n_valid"] += 1
                    a["probes_valid"] += len(ids)
        out = {}
        for k, a in acc.items():
            out[k] = {"detection_given_discrepancy": a["detected"] / max(1.0, a["n_failing"]),
                      "mean_probes_failing": a["probes_failing"] / max(1.0, a["n_failing"]),
                      "mean_probes_valid": a["probes_valid"] / max(1.0, a["n_valid"]),
                      "invalid_rate": a["invalid"] / max(1.0, a["n_failing"] + a["n_valid"]),
                      "n_failing": int(a["n_failing"]), "n_valid": int(a["n_valid"])}
        return out

    # ------------------------------------------------------------------ #
    def m5(self, limit: int = 0) -> Dict[str, Any]:
        recs = [r for r in self.trials if r["kind"] == "repair" and r["task_id"] in self.tasks]
        if limit:
            recs = recs[:limit]
        a = defaultdict(float)
        for i, r in enumerate(recs):
            task, private, state = self._state(r)
            name = r["function"]
            failed = self.col.objects.get_text(r["failed_fn_ref"])
            deps = [state.function_source(f) for f in dependency_closure(state, name).functions]
            pr = parse_plan(r.get("plan") or "", "f00")
            plan = pr.payload if pr.ok else Plan("f00", None)
            fb = r.get("feedback") or {"failure_kind": r.get("failure_kind") or "unknown"}
            v = M5View("f00", state.function_source(name), failed, deps, task.contract.render(name), r.get("plan") or "none", fb)
            resp = _call(self.router["M5"], "M5", v, {"function_id": "f00", "fn_source": state.function_source(name),
                                                      "plan": plan, "failed_source": failed, "feedback": fb}, f"m5|{i}")
            label_abandon = r["target"]["action"] == "abandon"
            a["n"] += 1
            a["label_abandon"] += float(label_abandon)
            if not resp.ok:
                a["invalid"] += 1
                continue
            if resp.parsed.payload.is_abandon:
                a["abandon"] += 1
                a["abandon_correct"] += float(label_abandon)
                continue
            o, _ = self.col.verify(task, private, state, name, resp.parsed.payload.code, seed=i)
            a["correct"] += float(o.get("stage") == "timing")
            a["accepted"] += float(bool(o.get("accepted")))
            a["reverted_to_current"] += float(resp.parsed.payload.code.strip() == state.function_source(name).strip())
        n = max(1.0, a["n"])
        return {"n": int(a["n"]), "correct_rate": a["correct"] / n, "correct_and_faster_rate": a["accepted"] / n,
                "reverted_rate": a["reverted_to_current"] / n, "invalid_rate": a["invalid"] / n,
                "abandon_rate": a["abandon"] / n, "abandon_precision": a["abandon_correct"] / max(1.0, a["abandon"]),
                "abandon_recall": a["abandon_correct"] / max(1.0, a["label_abandon"])}


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--bundle", required=True)
    p.add_argument("--split", default="dev")
    p.add_argument("--checks", nargs="+", default=["m1", "m2m3", "m4", "m5"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("overrides", nargs="*")
    a = p.parse_args(argv)
    from tinyperf_chain.config import pop_overrides

    pop_overrides(a, ["checks"])
    cfg = load_chain_config(a.config, a.overrides)
    cfg.validate(f"collect:{a.split}")
    from tinyperf_chain.bundle import build_router, load_bundle_json
    from tinyperf_chain.serve.engine import resolve_device

    router, _ = build_router(load_bundle_json(a.bundle), cfg, device=resolve_device(cfg.serving.device))
    d = RoleDiagnostics(cfg, router, a.split)
    rep = {c: getattr(d, c)(a.limit) for c in a.checks}
    rep["bundle"] = router.bundle_hash
    print(json.dumps(rep, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
