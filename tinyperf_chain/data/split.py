"""Source-function clusters and train / dev / confirmation assignment (before any mutation or composition).

    python -m tinyperf_chain.data.split --config configs/chain/pilot.yaml --out artifacts/chain/split_manifest.json \\
        [--mined artifacts/mined/mined_seeds.jsonl] [--fail-on-overlap]
    python -m tinyperf_chain.data.split --check artifacts/chain/split_manifest.json --fail-on-overlap \\
        [--tasks-root artifacts/chain/tasks]

A *unit* is one seed function (fast reference + hand-written slow variants).  Units are merged into clusters by
union-find over: declared group (mined natural variants share their function's group), identical source hashes,
identical normalized-AST fingerprints and identical alpha-normalized fingerprints of any of their sources, and
declared ``relatives``.  Alpha fingerprints are used only for leakage detection.  Clusters are assigned to
partitions by a salted hash order (exact requested counts when enough clusters exist; otherwise the split is
labelled a pilot and proportions are kept).  ``--check`` verifies that no exact / structural / lineage overlap
exists across partitions and that every derived task inherits its cluster's partition.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tinyperf.data.astutil import normalize
from tinyperf.data.seeds import SEEDS, Seed
from tinyperf_chain.hashing import fingerprints, hash_json, sha256_text

PARTITIONS = ("train", "dev", "confirm")
MANIFEST_SCHEMA = "split_manifest_v1"


class _UF:
    def __init__(self) -> None:
        self.p: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.p[rb] = ra
            else:
                self.p[ra] = rb


def _sources(s: Seed) -> List[str]:
    out = []
    for src in [s.source] + list(s.slow_variants):
        try:
            out.append(normalize(src))
        except SyntaxError:
            out.append(src)
    return out


def all_units(mined: Optional[str] = None) -> List[Seed]:
    units = list(SEEDS)
    if mined:
        from tinyperf.data.build import load_mined_seeds

        units += load_mined_seeds(mined)
    seen, out = set(), []
    for s in units:
        if s.name in seen:
            continue
        seen.add(s.name)
        out.append(s)
    return out


def build_clusters(units: Sequence[Seed], relatives: Optional[Dict[str, List[str]]] = None) -> Dict[str, Dict[str, Any]]:
    uf = _UF()
    owner: Dict[Tuple[str, str], str] = {}
    fps: Dict[str, Dict[str, List[str]]] = {}
    for s in units:
        uf.find(s.name)
        if s.group and s.group != s.name:
            uf.union(s.name, s.group)
        fp: Dict[str, List[str]] = {"exact": [], "ast": [], "alpha": []}
        for src in _sources(s):
            for kind, h in fingerprints(src).items():
                if h is None:
                    continue
                fp[kind].append(h)
                k = (kind, h)
                if k in owner:
                    uf.union(s.name, owner[k])
                else:
                    owner[k] = s.name
        fps[s.name] = fp
    for a, rels in (relatives or {}).items():
        for b in rels:
            if b in fps and a in fps:
                uf.union(a, b)
    clusters: Dict[str, Dict[str, Any]] = {}
    for s in units:
        root = uf.find(s.name)
        cid = "c_" + sha256_text(root)[:12]
        c = clusters.setdefault(cid, {"members": [], "fingerprints": {"exact": [], "ast": [], "alpha": []}})
        c["members"].append(s.name)
        for k, v in fps[s.name].items():
            c["fingerprints"][k].extend(v)
    for c in clusters.values():
        c["members"].sort()
        for k in c["fingerprints"]:
            c["fingerprints"][k] = sorted(set(c["fingerprints"][k]))
    return clusters


def assign(clusters: Dict[str, Dict[str, Any]], n_train: int, n_dev: int, n_confirm: int, salt: int) -> Dict[str, str]:
    order = sorted(clusters, key=lambda c: sha256_text(f"split|{salt}|{c}"))
    total = len(order)
    want = n_train + n_dev + n_confirm
    if total >= want:
        nc, nd = n_confirm, n_dev
    else:  # pilot: keep proportions, never fabricate independence
        nc = max(1, round(total * n_confirm / want))
        nd = max(1, round(total * n_dev / want))
    out = {}
    for i, c in enumerate(order):
        out[c] = "confirm" if i < nc else ("dev" if i < nc + nd else "train")
    return out


def make_manifest(units: Sequence[Seed], n_train: int, n_dev: int, n_confirm: int, salt: int,
                  relatives: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    clusters = build_clusters(units, relatives)
    split_of = assign(clusters, n_train, n_dev, n_confirm, salt)
    for cid, c in clusters.items():
        c["split"] = split_of[cid]
    counts = {p: sum(1 for c in clusters.values() if c["split"] == p) for p in PARTITIONS}
    m = {
        "schema": MANIFEST_SCHEMA, "salt": salt, "requested": {"train": n_train, "dev": n_dev, "confirm": n_confirm},
        "pilot": len(clusters) < n_train + n_dev + n_confirm, "n_units": len(units), "n_clusters": len(clusters),
        "counts": counts, "duplication_ratio": len(units) / max(1, len(clusters)),
        "clusters": clusters,
        "unit_to_cluster": {m_: cid for cid, c in clusters.items() for m_ in c["members"]},
    }
    m["manifest_sha256"] = hash_json({k: v for k, v in m.items() if k != "manifest_sha256"})
    return m


def check_manifest(m: Dict[str, Any]) -> List[str]:
    """Exact / structural / lineage overlap between partitions (empty list = clean)."""
    errs = []
    by_part: Dict[str, Dict[str, set]] = {p: defaultdict(set) for p in PARTITIONS}
    for cid, c in m["clusters"].items():
        part = c["split"]
        for kind, hs in c["fingerprints"].items():
            by_part[part][kind].update(hs)
        by_part[part]["members"].update(c["members"])
    for i, a in enumerate(PARTITIONS):
        for b in PARTITIONS[i + 1:]:
            for kind in ("exact", "ast", "alpha", "members"):
                inter = by_part[a][kind] & by_part[b][kind]
                if inter:
                    errs.append(f"{kind} overlap between {a} and {b}: {len(inter)} (e.g. {sorted(inter)[0][:24]})")
    for name, cid in m["unit_to_cluster"].items():
        if name not in m["clusters"][cid]["members"]:
            errs.append(f"unit {name} maps to a cluster that does not list it")
    return errs


def check_tasks(m: Dict[str, Any], tasks_root: str) -> List[str]:
    """Every task (and every component of a file task) must sit in its clusters' partition."""
    errs = []
    if not os.path.isdir(tasks_root):
        return errs
    from tinyperf_chain.tasks import PublicTask

    for fn in sorted(os.listdir(tasks_root)):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(tasks_root, fn)) as f:
            for line in f:
                if not line.strip():
                    continue
                t = PublicTask.from_row(json.loads(line))
                for unit in t.meta.get("component_units", []):
                    cid = m["unit_to_cluster"].get(unit)
                    if cid is None:
                        errs.append(f"{t.task_id}: unit {unit} not in manifest")
                    elif m["clusters"][cid]["split"] != t.split:
                        errs.append(f"{t.task_id}: unit {unit} belongs to {m['clusters'][cid]['split']}, task is {t.split}")
    return errs


def load_manifest(path: str) -> Dict[str, Any]:
    with open(path) as f:
        m = json.load(f)
    if m.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"{path}: not a split manifest")
    want = hash_json({k: v for k, v in m.items() if k != "manifest_sha256"})
    if m.get("manifest_sha256") != want:
        raise ValueError(f"{path}: manifest hash mismatch (edited after creation?)")
    return m


def main(argv: Optional[List[str]] = None) -> None:
    from tinyperf_chain.config import load_chain_config

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--mined", default=None)
    p.add_argument("--relatives", default=None, help="json {unit: [related units]} (e.g. same repository)")
    p.add_argument("--check", default=None, help="check an existing manifest instead of building one")
    p.add_argument("--manifest", default=None, help="alias of --check")
    p.add_argument("--tasks-root", default=None)
    p.add_argument("--fail-on-overlap", action="store_true")
    p.add_argument("overrides", nargs="*")
    a = p.parse_args(argv)
    path = a.check or a.manifest
    if path:
        m = load_manifest(path)
        errs = check_manifest(m)
        if a.tasks_root:
            errs += check_tasks(m, a.tasks_root)
        print(json.dumps({"manifest": path, "counts": m["counts"], "pilot": m["pilot"], "errors": errs}, indent=2))
        if errs and a.fail_on_overlap:
            sys.exit(2)
        return
    cfg = load_chain_config(a.config, a.overrides)
    rel = json.load(open(a.relatives)) if a.relatives else None
    units = all_units(a.mined)
    m = make_manifest(units, cfg.data.n_train_clusters, cfg.data.n_dev_clusters, cfg.data.n_confirm_clusters,
                      cfg.data.split_salt, rel)
    errs = check_manifest(m)
    out = a.out or cfg.paths.sub("split_manifest.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(m, f, indent=1)
    print(json.dumps({"out": out, "units": m["n_units"], "clusters": m["n_clusters"], "counts": m["counts"],
                      "pilot": m["pilot"], "duplication_ratio": round(m["duplication_ratio"], 3), "errors": errs}, indent=2))
    if errs and a.fail_on_overlap:
        sys.exit(2)


if __name__ == "__main__":
    main()
