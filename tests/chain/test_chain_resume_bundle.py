"""T25 resume and provenance: SFT resumes to an identical state; locked bundles detect changes."""
import json
import os

import pytest
import torch

from tinyperf.model.transformer import TinyPerfLM
from tinyperf_chain.bundle import BundleError, load_bundle_json, lock_bundle, verify_lock
from tinyperf_chain.ledger import ObjectStore


def _dataset(cfg, role="M1"):
    objs = ObjectStore(cfg.paths.objects)
    for split in ("train", "dev"):
        path = os.path.join(cfg.paths.datasets, role, f"{split}.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for i in range(12):
                p = objs.put_text(f"role=M1 v=chain_v1\n<CODE>\n# f0{i % 3}\ndef f(x):\n    return x + {i}\n</CODE>\n<STATE>\ncatalog=f00\n</STATE>\n")
                t = objs.put_text(f"TARGET f0{i % 3}")
                f.write(json.dumps({"record_id": f"r{split}{i}", "split": split, "prompt_artifact": p, "target_artifacts": [t],
                                    "weight": 1.0}) + "\n")


def test_t25_sft_resume_is_exact(tmp_path, cfg, tok, tok_path, tiny_model_cfg):
    from tinyperf_chain.train.sft import RoleSFT
    import copy

    c = copy.deepcopy(cfg)
    c.paths.root = str(tmp_path / "root")
    c.model.tokenizer = tok_path
    c.training.microbatch, c.training.accumulate, c.training.checkpoint_every = 2, 2, 1
    c.training.eval_every, c.training.precision, c.training.lr = 0, "fp32", 1e-3
    c.training.dev_generation_eval_max = 2
    _dataset(c)
    torch.manual_seed(0)
    init = str(tmp_path / "init.pt")
    TinyPerfLM(tiny_model_cfg).save(init)
    c.training.max_updates = 4
    full = RoleSFT(c, "M1", init, str(tmp_path / "full"), "cpu", tok)
    full.fit()
    c.training.max_updates = 2
    part = RoleSFT(c, "M1", init, str(tmp_path / "part"), "cpu", tok)
    part.fit()
    c.training.max_updates = 4
    resumed = RoleSFT(c, "M1", init, str(tmp_path / "part"), "cpu", tok)
    assert resumed.step == 2
    resumed.fit()
    for (k, a), (_, b) in zip(full.model.state_dict().items(), resumed.model.state_dict().items()):
        assert torch.allclose(a, b, atol=1e-6), k


def test_t25_locked_bundle_detects_changes(tmp_path, cfg, tok_path, tiny_model_cfg):
    ck = str(tmp_path / "m.pt")
    TinyPerfLM(tiny_model_cfg).save(ck)
    b = {"schema_version": "bundle_v1", "system_id": "X", "tokenizer": tok_path,
         "roles": {r: {"adapter": "neural", "checkpoint": ck} for r in ("M1", "M2", "M3", "M4", "M5")}}
    bp = str(tmp_path / "b.json")
    json.dump(b, open(bp, "w"))
    lock_bundle(bp, str(tmp_path / "b.lock.json"), cfg)
    lb = load_bundle_json(str(tmp_path / "b.lock.json"))
    verify_lock(lb, cfg)
    m = TinyPerfLM.load(ck)
    with torch.no_grad():
        m.wte.weight[0, 0] += 1.0
    m.save(ck)
    with pytest.raises(BundleError):
        verify_lock(lb, cfg)
