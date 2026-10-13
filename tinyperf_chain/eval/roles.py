"""Role-level development diagnostics (supportive; the primary endpoint is whole-system final-audited success).

    python -m tinyperf_chain.eval.roles --config configs/chain/eval_dev.yaml --bundle bundles/B4.json --split dev \\
        --checks m1 m2m3 m4 m5 [--limit 50] [--out roles_report.json]

m1    attainable regret against the measured target utilities (``target_label`` records of one downstream bundle -
      ``data.label_bundle`` - never a mixture): bundle M1 vs hottest-self, hottest-cumulative, random and opportunity
      rules on the same public profile; valid-id rate; consistency of the choice under a different catalog order /
      id permutation *and* under renamed, unrelated function names (position or name shortcuts show up here).
m2    the planner itself against the measured plan sets: useful-plan rate (the emitted plan is in the measured
      useful set), wrong-operand rate, rejected-precondition rate, no-site rate, NO_PLAN precision/recall, versus
      the rule planner on the same states.
m2m3  counterfactual plan tests for the writer: actual plan, plan removed, a different applicable plan substituted,
      operands scrambled; acceptance per condition, plan adherence (does the patch instantiate the requested
      mechanism/operands) and how often the implementation changes with the plan.  A writer that ignores M2 shows
      no difference across conditions.
m4    discrepancy detection at a fixed probe budget *and* wall-time on real and near-miss failing candidates (bundle
      M4 vs random, fixed edge suite and transformation-specific rules), false alarms (a probe flagged on a
      candidate that passes the mandatory suite - a comparator or constructor problem, not the model's), detection
      on held-out mutation families (``--holdout-families``, read from the restricted private store) and the
      downstream repairs the detections enable (``additional_repairs_enabled``).
m5    repairs that become correct, correct and faster (accepted), reversions, and abandonment precision / recall.
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
import time

from tinyperf_chain.adapters.rules import RulePlanner, RuleProbeSelector, RuleTargetSelector
from tinyperf_chain.config import load_chain_config
from tinyperf_chain.context import (CatalogEntry, M1View, M2View, M3View, M4View, M5View, calls_bucket, compact_diff,
                                    rename_functions, share_bucket)
from tinyperf_chain.data.collect import Collector
from tinyperf_chain.program import Program, dependency_closure, dependency_sources
from tinyperf_chain.schemas import Plan, RequestEnvelope, SCHEMA_VERSIONS, format_plan, parse_plan


def _env(role: str, rid: str, max_new: int) -> RequestEnvelope:
    return RequestEnvelope(role, "diag", rid, "", "diag", SCHEMA_VERSIONS[role], max_new, float("inf"))


def _call(adapter: RoleAdapter, role: str, view, ctx, rid: str):
    prep = adapter.prepare(role, view, ctx)
    return adapter.execute(prep, _env(role, rid, prep.max_new_tokens))


_UNRELATED = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet", "kilo", "lima"]


class RoleDiagnostics:
    def __init__(self, cfg, router, split: str = "dev", seed: int = 0, holdout_families: Optional[Sequence[str]] = None):
        from tinyperf_chain.adapters.verifier import Verifier

        self.cfg, self.router = cfg, router
        self.col = Collector(cfg, Verifier(cfg, cache_dir=cfg.paths.cache), split, seed)
        self.tasks = self.col._task_index()
        self.trials = [r for r in self.col.ledger.read() if r.get("split") == split]
        self.rng = random.Random(seed)
        self.holdout = set(holdout_families or [])
        self.families = self.col.private_families() if self.holdout else {}   # evaluator-side access only

    def _state(self, r):
        task, private = self.tasks[r["task_id"]]
        return task, private, Program(self.col.objects.get_text(r["state_ref"]), tuple(task.functions))

    def _label_bundle(self, recs: List[Dict[str, Any]]) -> Optional[str]:
        want = self.cfg.data.label_bundle
        if want == "all" or not recs:
            return None
        if want != "latest":
            return want
        top = max(int(r.get("round", 0) or 0) for r in recs)
        from collections import Counter
        return Counter(r.get("downstream_bundle") for r in recs if int(r.get("round", 0) or 0) == top).most_common(1)[0][0]

    # ------------------------------------------------------------------ #
    def m1(self, limit: int = 0) -> Dict[str, Any]:
        recs = [r for r in self.trials if r["kind"] == "target_label" and r["task_id"] in self.tasks
                and r.get("history_variant", "fresh") == "fresh"]
        bundle = self._label_bundle(recs)
        stale = sum(1 for r in recs if bundle is not None and r.get("downstream_bundle") != bundle)
        recs = [r for r in recs if bundle is None or r.get("downstream_bundle") == bundle]
        if limit:
            recs = recs[:limit]
        selectors: Dict[str, RoleAdapter] = {"bundle": self.router["M1"]}
        for s in ("hottest_self", "hottest_cumulative", "random", "opportunity"):
            selectors[s] = RuleTargetSelector(s, seed=7)
        acc = {k: defaultdict(float) for k in selectors}
        # perm 0/1: shuffled ids + order; perm 2: ids + order of perm 0 but every real function name replaced by an
        # unrelated word (5.3: performance must not collapse when unrelated function names change)
        for i, r in enumerate(recs):
            task, _, state = self._state(r)
            umax = max((v["U"] for v in r["utilities"].values()), default=0.0)
            picks: Dict[int, Optional[str]] = {}
            for perm in range(3):
                rng = random.Random(f"{r['trial_id']}|{perm if perm < 2 else 0}")
                ids = list(range(len(task.functions)))
                rng.shuffle(ids)
                disp = {n: f"f{ids[k]:02d}" for k, n in enumerate(task.functions)}
                order = list(task.functions)
                rng.shuffle(order)
                renamed = {}
                if perm == 2:
                    words = list(_UNRELATED)
                    random.Random(f"{r['trial_id']}|names").shuffle(words)
                    renamed = {n: f"{words[k % len(words)]}_{k}" for k, n in enumerate(task.functions)}
                entries, cat = [], []
                for n in order:
                    fn = state.function_node(n)
                    src = state.function_source(n)
                    if renamed:
                        src = rename_functions(src, renamed)
                    p = r["profile"].get(n, {"ncalls": 0, "self_share": 0.0, "cum_share": 0.0})
                    entries.append(CatalogEntry(disp[n], n, src, share_bucket(p["self_share"]), share_bucket(p["cum_share"]),
                                                calls_bucket(p["ncalls"]), len(src.splitlines()), len(R.anchors(fn)), 0,
                                                summary=src.splitlines()[0]))
                    cat.append({"display_id": disp[n], "name": n, "self_share": p["self_share"], "cum_share": p["cum_share"],
                                "ncalls": p["ncalls"], "has_plan": bool(R.static_plans(fn, task.contract.function(n), 1))})
                inv = {v: k for k, v in disp.items()}
                view = M1View(entries, [], self.cfg.budget.patch_slots, rename={} if renamed else dict(disp))
                for name, ad in selectors.items():
                    if perm == 2 and name != "bundle":
                        continue
                    resp = _call(ad, "M1", view, {"catalog": cat}, f"m1|{i}|{perm}")
                    a = acc[name]
                    if perm < 2:
                        a["n"] += 1
                    if not resp.ok:
                        if perm < 2:
                            a["invalid"] += 1
                            a["regret"] += umax
                        if name == "bundle":
                            picks[perm] = None
                        continue
                    fid = resp.parsed.payload.function_id
                    chosen = inv.get(fid) if fid else None
                    if perm < 2:
                        if fid and chosen is None:
                            a["invalid"] += 1
                        u = r["utilities"].get(chosen, {"U": 0.0})["U"] if chosen else 0.0
                        a["regret"] += umax - u
                        a["hit"] += float((chosen in r["label"]) if r["label"] else chosen is None)
                    if name == "bundle":
                        picks[perm] = chosen
                        if perm == 2:
                            a["renamed_hit"] += float((chosen in r["label"]) if r["label"] else chosen is None)
                            a["renamed_n"] += 1
            if 0 in picks and 1 in picks:
                acc["bundle"]["consistent"] += float(picks[0] == picks[1])
                acc["bundle"]["pairs"] += 1
            if 0 in picks and 2 in picks:
                acc["bundle"]["name_consistent"] += float(picks[0] == picks[2])
                acc["bundle"]["name_pairs"] += 1
        out: Dict[str, Any] = {"label_bundle": bundle, "stale_labels_excluded": stale}
        for k, a in acc.items():
            n = max(1.0, a["n"])
            out[k] = {"n": int(a["n"]), "mean_regret": a["regret"] / n, "hit_rate": a["hit"] / n,
                      "valid_id_rate": 1 - a["invalid"] / n}
        if acc["bundle"]["pairs"]:
            out["bundle"]["order_consistency"] = acc["bundle"]["consistent"] / acc["bundle"]["pairs"]
        if acc["bundle"]["name_pairs"]:
            out["bundle"]["renamed_consistency"] = acc["bundle"]["name_consistent"] / acc["bundle"]["name_pairs"]
            out["bundle"]["renamed_hit_rate"] = acc["bundle"]["renamed_hit"] / acc["bundle"]["name_pairs"]
        return out

    # ------------------------------------------------------------------ #
    def m2(self, limit: int = 0) -> Dict[str, Any]:
        """The planner against measured plan sets (6.3): useful-plan rate, wrong-operand / rejected-precondition /
        no-site rates, NO_PLAN precision and recall; the rule planner on the same states for comparison."""
        recs = [r for r in self.trials if r["kind"] == "plan_set" and r["task_id"] in self.tasks]
        if limit:
            recs = recs[:limit]
        planners: Dict[str, RoleAdapter] = {"bundle": self.router["M2"], "rule": RulePlanner()}
        acc = {k: defaultdict(float) for k in planners}
        for i, r in enumerate(recs):
            task, _, state = self._state(r)
            name = r["function"]
            fn = state.function_node(name)
            fc = task.contract.function(name)
            fsrc = state.function_source(name)
            deps = dependency_sources(state, dependency_closure(state, name))
            useful = {parse_plan(p["plan"], "f00").payload.key() for p in r["plans"] if p["useful"] and parse_plan(p["plan"], "f00").ok}
            searched_all = all(p.get("searched", True) for p in r["plans"])
            label_no_plan = not useful and (not r["plans"] or searched_all)
            v = M2View("f00", fsrc, deps, task.contract.render(name), R.render_anchors(fn), R.operand_names(fn),
                       R.applicable_mechanisms(fn), [])
            for k, ad in planners.items():
                a = acc[k]
                a["n"] += 1
                resp = _call(ad, "M2", v, {"function_id": "f00", "fn_source": fsrc, "fcontract": fc, "attempted": []}, f"m2|{i}|{k}")
                if not resp.ok:
                    a["malformed"] += 1
                    continue
                plan = resp.parsed.payload
                if plan.is_no_plan:
                    a["no_plan"] += 1
                    a["no_plan_correct"] += float(label_no_plan)
                    continue
                chk = R.validate_plan(plan, fn, fc)
                if not chk.ok:
                    a[chk.kind] += 1          # wrong_operand | precondition_rejected | no_site | unknown_transform | inapplicable
                    a["invalid_plan"] += 1
                    continue
                a["valid_plan"] += 1
                a["useful"] += float(plan.key() in useful)
                a["right_mechanism"] += float(any(plan.transform_id == parse_plan(p["plan"], "f00").payload.transform_id
                                                  for p in r["plans"] if p["useful"] and parse_plan(p["plan"], "f00").ok))
            for a in acc.values():
                a["label_no_plan"] += float(label_no_plan)
        out = {}
        for k, a in acc.items():
            n = max(1.0, a["n"])
            out[k] = {"n": int(a["n"]), "useful_plan_rate": a["useful"] / n, "right_mechanism_wrong_operands_rate": (a["right_mechanism"] - a["useful"]) / n,
                      "wrong_operand_rate": a["wrong_operand"] / n, "rejected_precondition_rate": a["precondition_rejected"] / n,
                      "no_site_rate": (a["no_site"] + a["inapplicable"]) / n, "malformed_rate": a["malformed"] / n,
                      "no_plan_rate": a["no_plan"] / n,
                      "no_plan_precision": a["no_plan_correct"] / max(1.0, a["no_plan"]),
                      "no_plan_recall": a["no_plan_correct"] / max(1.0, a["label_no_plan"])}
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
            deps = dependency_sources(state, dependency_closure(state, name))
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
        repairs = {r["source_trial"]: r for r in self.trials if r["kind"] == "repair"}
        sels: Dict[str, RoleAdapter] = {"bundle": self.router["M4"], "random": RuleProbeSelector("random", budget, 3),
                                        "fixed_edge": RuleProbeSelector("fixed_edge", budget),
                                        "transform_specific": RuleProbeSelector("transform_specific", budget)}
        acc = {k: defaultdict(float) for k in sels}
        big = 1 << 60
        for i, r in enumerate(recs):
            task, _, state = self._state(r)
            name = r["function"]
            failing = any(r["outcomes"].values())
            # a candidate that passed the mandatory suite yet has a flagged probe is a comparator / constructor false
            # alarm, not a model finding; never credited as a detection (8.3)
            false_alarm = failing and bool(r.get("candidate_valid"))
            family = self.families.get(r["trial_id"])
            held_out = family in self.holdout if family else False
            cur = state.function_source(name)
            cand = self.col.objects.get_text(r["candidate_fn_ref"])
            menu_ids = [p for p, _ in r["menu"]]
            costs = {p: (c if c is not None else big) for p, c in (r.get("costs") or {}).items()}
            pr = parse_plan(r.get("plan") or "", "f00")
            plan = pr.payload if pr.ok else None
            v = M4View("f00", cur, cand, task.contract.render(name), r.get("plan") or "none",
                       ",".join(f"{p}:{n}" for p, n in r["menu"]) or "none", compact_diff(cur, cand))
            rep = repairs.get(r.get("source_trial"))
            for k, ad in sels.items():
                t0 = time.perf_counter_ns()
                resp = _call(ad, "M4", v, {"function_id": "f00", "menu_ids": menu_ids, "plan": plan}, f"m4|{i}")
                a = acc[k]
                a["select_ns"] += time.perf_counter_ns() - t0
                ids = list(resp.parsed.payload.probe_ids) if resp.ok else []
                ok_sel = resp.ok and len(set(ids)) == len(ids) <= budget and all(p in menu_ids for p in ids)
                if not ok_sel:
                    a["invalid"] += 1
                    ids = []
                a["probe_cost_ns"] += sum(costs.get(p, 0) for p in ids if costs.get(p, big) != big)
                if false_alarm:
                    a["n_false_alarm_cands"] += 1
                    a["false_alarms"] += float(any(r["outcomes"].get(p) for p in ids))
                    continue
                if failing:
                    a["n_failing"] += 1
                    det = float(any(r["outcomes"].get(p) for p in ids))
                    a["detected"] += det
                    a["probes_failing"] += len(ids)
                    if held_out:
                        a["n_heldout"] += 1
                        a["detected_heldout"] += det
                    elif family:
                        a["n_seen_family"] += 1
                        a["detected_seen_family"] += det
                    if rep is not None and rep["target"]["action"] == "edit":
                        a["repairs_enabled"] += det      # a detection that led to a verified repair in the ledger
                else:
                    a["n_valid"] += 1
                    a["probes_valid"] += len(ids)
        out: Dict[str, Any] = {"holdout_families": sorted(self.holdout), "budget_probes": budget}
        for k, a in acc.items():
            n_all = max(1.0, a["n_failing"] + a["n_valid"] + a["n_false_alarm_cands"])
            out[k] = {"detection_given_discrepancy": a["detected"] / max(1.0, a["n_failing"]),
                      "false_alarm_rate": a["false_alarms"] / max(1.0, a["n_false_alarm_cands"]),
                      "n_false_alarm_candidates": int(a["n_false_alarm_cands"]),
                      "detection_heldout_families": (a["detected_heldout"] / a["n_heldout"]) if a["n_heldout"] else None,
                      "detection_seen_families": (a["detected_seen_family"] / a["n_seen_family"]) if a["n_seen_family"] else None,
                      "additional_repairs_enabled": a["repairs_enabled"] / max(1.0, a["n_failing"]),
                      "mean_probes_failing": a["probes_failing"] / max(1.0, a["n_failing"]),
                      "mean_probes_valid": a["probes_valid"] / max(1.0, a["n_valid"]),
                      "mean_probe_cost_ms": a["probe_cost_ns"] / 1e6 / n_all, "mean_select_ms": a["select_ns"] / 1e6 / n_all,
                      "invalid_rate": a["invalid"] / n_all,
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
            deps = dependency_sources(state, dependency_closure(state, name))
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
    p.add_argument("--checks", nargs="+", default=["m1", "m2", "m2m3", "m4", "m5"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--holdout-families", nargs="*", default=[], help="m4: mutation families held out of training")
    p.add_argument("--out", default=None)
    p.add_argument("overrides", nargs="*")
    a = p.parse_args(argv)
    from tinyperf_chain.config import pop_overrides

    pop_overrides(a, ["checks", "holdout_families"])
    cfg = load_chain_config(a.config, a.overrides)
    cfg.validate(f"collect:{a.split}")
    from tinyperf_chain.bundle import build_router, load_bundle_json
    from tinyperf_chain.serve.engine import resolve_device

    router, _ = build_router(load_bundle_json(a.bundle), cfg, device=resolve_device(cfg.serving.device))
    d = RoleDiagnostics(cfg, router, a.split, holdout_families=a.holdout_families)
    rep = {c: getattr(d, c)(a.limit) for c in a.checks}
    rep["bundle"] = router.bundle_hash
    print(json.dumps(rep, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
