"""T03 split integrity: no exact / structural / lineage overlap; derived records inherit their split."""
import json

import pytest

from tinyperf.data.seeds import SEEDS, Seed
from tinyperf_chain.data.split import build_clusters, check_manifest, make_manifest


def test_t03_manifest_has_no_overlap():
    m = make_manifest(SEEDS, 300, 60, 100, 0)
    assert check_manifest(m) == []
    assert sum(m["counts"].values()) == m["n_clusters"]
    assert m["pilot"] is True                                  # fewer independent clusters than requested


def test_t03_near_duplicates_share_a_cluster():
    gen = SEEDS[0].gen
    a = Seed("dup_a", "def f(xs):\n    return len(set(xs)) != len(xs)\n", gen)
    b = Seed("dup_b", "def g(ys):\n    return len(set(ys)) != len(ys)\n", gen)       # alpha-equivalent
    c = Seed("other", "def h(xs):\n    return sorted(xs)\n", gen)
    cl = build_clusters([a, b, c])
    owner = {m: cid for cid, v in cl.items() for m in v["members"]}
    assert owner["dup_a"] == owner["dup_b"] != owner["other"]


def test_t03_detects_injected_overlap():
    m = make_manifest(SEEDS[:40], 20, 10, 10, 0)
    cids = {c["split"]: cid for cid, c in m["clusters"].items()}
    a, b = cids["train"], cids["dev"]
    m["clusters"][b]["fingerprints"]["alpha"].append(m["clusters"][a]["fingerprints"]["alpha"][0])
    assert any("alpha overlap" in e for e in check_manifest(m))


def test_t03_projection_rejects_split_mismatch(cfg, tok):
    from tinyperf_chain.data.project import Projector
    from tinyperf_chain.tasks import PublicTask
    from tinyperf_chain.contracts import CONTRACT_VERSION, TaskContract

    pj = Projector(cfg, tok)
    t = PublicTask("x", "c1", "dev", "def f(x):\n    return x\n", ["f"], TaskContract(CONTRACT_VERSION, []), [])
    with pytest.raises(ValueError):
        pj._check_split({"split": "train", "cluster_id": "c1", "trial_id": "t"}, t)
