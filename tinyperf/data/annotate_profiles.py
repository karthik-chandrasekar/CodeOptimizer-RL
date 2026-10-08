"""Add the step-0 profile (each function's share of the slow file's runtime) to existing file tasks.

    python -m tinyperf.data.annotate_profiles --config configs/data.yaml \\
        --src artifacts/data_r2/files_test.jsonl --out artifacts/data_r2/files_test_prof.jsonl

Tasks, inputs and ids are unchanged, so results stay comparable with runs on the original file.
"""
from __future__ import annotations

import argparse
from typing import List, Optional

from tinyperf.common.config import load_config
from tinyperf.common.utils import get_logger
from tinyperf.env.executor import Executor
from tinyperf.env.files import ENTRY, profile_shares, with_driver
from tinyperf.env.task import Task, load_tasks, save_tasks

log = get_logger("annotate_profiles")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("overrides", nargs="*")
    a = p.parse_args(argv)
    cfg = load_config(a.config, a.overrides)
    tasks = load_tasks(a.src)
    ex = Executor(cfg.env)

    def annotate(t: Task) -> bool:
        b = ex.baseline(with_driver(t.source, t.meta["driver"]), ENTRY, t.correct_inputs, t.perf_inputs, group_by_first_arg=True)
        prof = profile_shares(b.get("candidate_group_ns"), t.meta["functions"]) if b.get("status") == "ok" else None
        t.meta["profile0"] = prof
        return prof is not None

    ok = ex.map(annotate, tasks)
    ex.close()
    save_tasks(a.out, tasks)
    log.info(f"annotated {sum(ok)}/{len(tasks)} tasks -> {a.out}")


if __name__ == "__main__":
    main()
