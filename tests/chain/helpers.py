"""Hand-made tasks with declared contracts for verifier tests."""
from typing import List, Sequence, Tuple

from tinyperf_chain.contracts import CONTRACT_VERSION, FunctionContract, ParamSpec, TaskContract
from tinyperf_chain.program import Program
from tinyperf_chain.tasks import PrivateSuite, PublicTask


def make_task(src: str, functions: Sequence[str], params: Sequence[Sequence[Tuple[str, str]]], correctness: List[tuple],
              timing: List[tuple], task_id: str = "t_manual", cluster: str = "c_manual"):
    fcs = [FunctionContract(f, [ParamSpec(n, t, max_len=100000, int_min=-10**9, int_max=10**9) for n, t in ps])
           for f, ps in zip(functions, params)]
    contract = TaskContract(CONTRACT_VERSION, fcs, permitted_imports=["math", "collections", "itertools"])
    p0 = Program.from_source(src, functions)
    pub = PublicTask(task_id, cluster, "dev", p0.source, list(functions), contract, list(timing),
                     {f: [c[1:] for c in correctness if c[0] == i][:2] for i, f in enumerate(functions)})
    priv = PrivateSuite(task_id, list(correctness), list(timing), list(correctness), list(timing))
    return pub, priv
