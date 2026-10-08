"""Sandbox worker.

Runs as a *fresh subprocess* per candidate (``python -m tinyperf.env.worker``),
reads one pickled request from stdin and writes one pickled response to stdout.

Responsibilities
----------------
1. Static safety checks on the candidate AST (no forbidden imports, no dunder
   attribute access, no async, ...).
2. Restricted execution namespace (whitelisted builtins and modules).
3. Differential correctness against the *original* implementation's recorded
   behaviour (return value, type, exception type, argument mutation).
4. Robust timing: warm-up, repeated runs, median; optionally *paired* with the
   baseline implementation in the same process (interleaved A/B/B/A order) so the
   reported ratio is insensitive to machine load.  The candidate namespace is
   re-executed before every timing repeat, so ``lru_cache``/module-level caches
   cannot survive across repeats.
5. OS resource limits (address space, CPU time, no forking) — best effort.

This is defence in depth against a *tiny RL policy* hacking its reward, not a
security boundary against adversarial humans.
"""
from __future__ import annotations

import ast
import builtins as _builtins
import gc
import math
import pickle
import random
import signal
import sys
import time
import traceback
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Static checks
# --------------------------------------------------------------------------- #
FORBIDDEN_NAMES = {
    "__import__", "__builtins__", "__loader__", "__spec__", "exec", "eval", "compile",
    "open", "input", "globals", "locals", "vars", "breakpoint", "exit", "quit", "help",
    "memoryview", "super", "type", "object", "classmethod", "staticmethod", "property",
}

SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes", "callable", "chr",
    "complex", "dict", "divmod", "enumerate", "filter", "float", "format", "frozenset",
    "hash", "hex", "id", "int", "isinstance", "issubclass", "iter", "len", "list", "map",
    "max", "min", "next", "oct", "ord", "pow", "range", "repr", "reversed", "round", "set",
    "slice", "sorted", "str", "sum", "tuple", "zip", "getattr", "hasattr",
    "True", "False", "None", "NotImplemented", "Ellipsis",
    # exceptions
    "Exception", "BaseException", "ArithmeticError", "AssertionError", "AttributeError",
    "IndexError", "KeyError", "LookupError", "NameError", "NotImplementedError",
    "OverflowError", "RecursionError", "RuntimeError", "StopIteration", "TypeError",
    "ValueError", "ZeroDivisionError", "FloatingPointError", "UnicodeError",
]


class StaticCheckError(Exception):
    pass


def static_check(tree: ast.AST, func_name: str, allowed_modules: List[str]) -> None:
    allowed = set(allowed_modules)
    found_target = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            else:
                if node.level:
                    raise StaticCheckError("relative imports are not allowed")
                names = [(node.module or "").split(".")[0]]
            for n in names:
                if n not in allowed:
                    raise StaticCheckError(f"import of module {n!r} is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                raise StaticCheckError(f"dunder attribute access {node.attr!r} is not allowed")
            if node.attr.startswith("_") and not node.attr.startswith("__"):
                pass  # single underscore is fine
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES:
                raise StaticCheckError(f"use of {node.id!r} is not allowed")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("getattr", "hasattr", "setattr", "delattr"):
            for arg in node.args[1:2]:
                if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)) or arg.value.startswith("_"):
                    raise StaticCheckError("dynamic attribute access must use a literal, non-underscore name")
        elif isinstance(node, (ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith, ast.Await)):
            raise StaticCheckError("async constructs are not allowed")
        elif isinstance(node, ast.Global):
            raise StaticCheckError("global statements are not allowed")
        elif isinstance(node, ast.ClassDef):
            raise StaticCheckError("class definitions are not allowed in V0")
    for node in tree.body:  # type: ignore[attr-defined]
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            found_target = True
        elif isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)):
            continue
        elif isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
            continue  # docstring
        else:
            raise StaticCheckError(f"top-level statement of type {type(node).__name__} is not allowed")
    if not found_target:
        raise StaticCheckError(f"candidate must define a function named {func_name!r}")


# --------------------------------------------------------------------------- #
# Restricted namespace
# --------------------------------------------------------------------------- #
def make_namespace(allowed_modules: List[str]) -> Dict[str, Any]:
    allowed = set(allowed_modules)
    real_import = _builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if level != 0 or root not in allowed:
            raise ImportError(f"import of {name!r} is not allowed in the sandbox")
        return real_import(name, globals, locals, fromlist, level)

    safe = {n: getattr(_builtins, n) for n in SAFE_BUILTIN_NAMES if hasattr(_builtins, n)}
    safe["__import__"] = guarded_import
    ns: Dict[str, Any] = {"__builtins__": safe, "__name__": "candidate"}
    return ns


def load_function(code: str, func_name: str, allowed_modules: List[str]):
    ns = make_namespace(allowed_modules)
    exec(compile(code, "<candidate>", "exec"), ns)  # noqa: S102 - this is the sandbox
    fn = ns[func_name]
    if not callable(fn):
        raise TypeError(f"{func_name} is not callable")
    return fn


# --------------------------------------------------------------------------- #
# Timeouts
# --------------------------------------------------------------------------- #
class CallTimeout(Exception):
    pass


def _alarm_handler(signum, frame):  # pragma: no cover - signal path
    raise CallTimeout()


class time_limit:
    def __init__(self, seconds: float):
        self.seconds = seconds

    def __enter__(self):
        if hasattr(signal, "setitimer"):
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def __exit__(self, *exc):
        if hasattr(signal, "setitimer"):
            signal.setitimer(signal.ITIMER_REAL, 0)
        return False


# --------------------------------------------------------------------------- #
# Behaviour capture & comparison
# --------------------------------------------------------------------------- #
def _normalize(v: Any) -> Any:
    """Make return values comparable/picklable: materialize iterators."""
    if isinstance(v, (str, bytes, bytearray, int, float, bool, complex, type(None))):
        return v
    if isinstance(v, (list, tuple)):
        return type(v)(_normalize(x) for x in v)
    if isinstance(v, (set, frozenset)):
        try:
            return type(v)(_normalize(x) for x in v)
        except TypeError:
            return v
    if isinstance(v, dict):
        return {k: _normalize(x) for k, x in v.items()}
    if hasattr(v, "__next__") and hasattr(v, "__iter__"):
        return ("__iterator__", [_normalize(x) for x in v])
    try:
        pickle.dumps(v)
        return v
    except Exception:
        return ("__repr__", type(v).__name__, repr(v))


def run_capture(fn, args: Tuple, timeout_s: float) -> Dict[str, Any]:
    """Call fn(*args) on a deep copy of args; record outcome + mutated args."""
    a = deepcopy(args)
    rec: Dict[str, Any] = {"ret": None, "exc": None, "args_after": None}
    try:
        with time_limit(timeout_s):
            out = fn(*a)
        rec["ret"] = _normalize(out)
    except CallTimeout:
        rec["exc"] = "__timeout__"
    except RecursionError:
        rec["exc"] = "RecursionError"
    except MemoryError:
        rec["exc"] = "MemoryError"
    except Exception as e:  # noqa: BLE001
        rec["exc"] = type(e).__name__
    rec["args_after"] = _normalize(a)
    return rec


def values_equal(a: Any, b: Any, rel: float, abs_: float) -> bool:
    if type(a) is not type(b):
        # allow bool/int distinction to stand; everything else must match exactly
        return False
    if isinstance(a, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=rel, abs_tol=abs_)
    if isinstance(a, complex):
        return values_equal(a.real, b.real, rel, abs_) and values_equal(a.imag, b.imag, rel, abs_)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(values_equal(x, y, rel, abs_) for x, y in zip(a, b))
    if isinstance(a, dict):
        if len(a) != len(b) or set(a.keys()) != set(b.keys()):
            return False
        return all(values_equal(a[k], b[k], rel, abs_) for k in a)
    if isinstance(a, (set, frozenset)):
        return a == b
    try:
        return bool(a == b)
    except Exception:
        return False


def compare_records(expected: Dict[str, Any], got: Dict[str, Any], rel: float, abs_: float) -> Optional[str]:
    if expected["exc"] != got["exc"]:
        return f"exception mismatch: expected {expected['exc']}, got {got['exc']}"
    if expected["exc"] is None and not values_equal(expected["ret"], got["ret"], rel, abs_):
        return f"return mismatch: expected {_short(expected['ret'])}, got {_short(got['ret'])}"
    if not values_equal(expected["args_after"], got["args_after"], rel, abs_):
        return "argument mutation mismatch"
    return None


def _short(v: Any, n: int = 80) -> str:
    s = repr(v)
    return s if len(s) <= n else s[: n - 3] + "..."


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
class WorkloadTimeout(Exception):
    pass


def time_workload(code: str, func_name: str, allowed: List[str], perf_inputs: List[Tuple], timeout_s: float,
                  by_group: bool = False):
    """Re-exec the code (fresh namespace/caches) and time the full workload once. Returns ns.

    With ``by_group`` (multi-function files, whose inputs are ``(k, *args)`` grouped by k) it returns
    ``(total_ns, {k: ns})``: the clock is read at every group boundary, so this costs nothing extra."""
    fn = load_function(code, func_name, allowed)
    inputs = [deepcopy(a) for a in perf_inputs]  # fresh copies in case the function mutates
    per: Dict[Any, int] = {}
    gc.collect()
    gc_was = gc.isenabled()
    gc.disable()
    try:
        with time_limit(timeout_s):
            t0 = time.perf_counter_ns()
            if not by_group:
                for a in inputs:
                    fn(*a)
            else:
                cur, g0 = None, t0
                for a in inputs:
                    if a[0] != cur:
                        now = time.perf_counter_ns()
                        if cur is not None:
                            per[cur] = per.get(cur, 0) + now - g0
                        cur, g0 = a[0], now
                    fn(*a)
            t1 = time.perf_counter_ns()
            if by_group and cur is not None:
                per[cur] = per.get(cur, 0) + t1 - g0
    except CallTimeout:
        raise WorkloadTimeout()
    finally:
        if gc_was:
            gc.enable()
    return (t1 - t0, per) if by_group else t1 - t0


def median(xs: List[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


# --------------------------------------------------------------------------- #
# Main protocol
# --------------------------------------------------------------------------- #
def set_limits(limits: Dict[str, Any]) -> None:
    try:
        import resource

        mem = int(limits.get("memory_mb", 512)) * 1024 * 1024
        for r in ("RLIMIT_AS",):
            if hasattr(resource, r):
                try:
                    resource.setrlimit(getattr(resource, r), (mem, mem))
                except (ValueError, OSError):
                    pass
        cpu = int(limits.get("cpu_seconds", 20))
        if hasattr(resource, "RLIMIT_CPU"):
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 2))
            except (ValueError, OSError):
                pass
        if hasattr(resource, "RLIMIT_NPROC") and sys.platform.startswith("linux"):
            try:
                resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))  # no fork / no threads
            except (ValueError, OSError):
                pass
        if hasattr(resource, "RLIMIT_FSIZE"):
            try:
                resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
            except (ValueError, OSError):
                pass
    except ImportError:  # pragma: no cover
        pass


def handle(req: Dict[str, Any]) -> Dict[str, Any]:
    code: str = req["code"]
    func_name: str = req["func_name"]
    allowed: List[str] = req.get("allowed_modules", [])
    limits = req.get("limits", {})
    timing = req.get("timing", {})
    mode = req.get("mode", "evaluate")  # "baseline" | "evaluate" | "correctness_only"
    rel = float(req.get("float_rel_tol", 1e-9))
    abs_ = float(req.get("float_abs_tol", 1e-12))
    resp: Dict[str, Any] = {"status": "ok", "message": ""}

    # 1. parse + static checks
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"status": "syntax_error", "message": f"SyntaxError: {e.msg} (line {e.lineno})"}
    try:
        static_check(tree, func_name, allowed)
    except StaticCheckError as e:
        return {"status": "static_error", "message": str(e)}

    # 2. load
    try:
        with time_limit(float(limits.get("call_timeout_s", 2.0))):
            fn = load_function(code, func_name, allowed)
    except CallTimeout:
        return {"status": "timeout", "message": "timeout while loading candidate"}
    except Exception as e:  # noqa: BLE001
        return {"status": "load_error", "message": f"{type(e).__name__}: {e}"}

    # 3. correctness
    correct_inputs: List[Tuple] = req.get("correct_inputs", [])
    call_timeout = float(limits.get("call_timeout_s", 2.0))
    records = []
    for a in correct_inputs:
        records.append(run_capture(fn, a, call_timeout))
    if mode == "baseline":
        resp["expected"] = records
        if any(r["exc"] == "__timeout__" for r in records):
            resp["status"] = "timeout"
            resp["message"] = "baseline timed out on a correctness input"
            return resp
    else:
        expected = req.get("expected") or []
        n_pass = 0
        first_failure = None
        for i, (e, g) in enumerate(zip(expected, records)):
            msg = compare_records(e, g, rel, abs_)
            if msg is None:
                n_pass += 1
            elif first_failure is None:
                first_failure = f"input #{i}: {msg}"
        resp["n_pass"], resp["n_total"] = n_pass, len(expected)
        if n_pass != len(expected):
            resp["status"] = "incorrect"
            resp["message"] = first_failure or "correctness failure"
            if any(g["exc"] == "__timeout__" for g in records):
                resp["status"] = "timeout"
            return resp
    if mode == "correctness_only":
        return resp

    # 4. timing
    perf_inputs: List[Tuple] = req.get("perf_inputs", [])
    warmup = int(timing.get("warmup", 1))
    repeats = int(timing.get("repeats", 7))
    paired = bool(timing.get("paired", True)) and req.get("baseline_code") is not None
    wl_timeout = float(limits.get("workload_timeout_s", 8.0))
    max_ratio = float(timing.get("max_candidate_ratio", 20.0))
    try:
        cand_ns: List[int] = []
        base_ns: List[int] = []
        by_group = bool(req.get("group_by_first_arg"))
        cand_groups: List[Dict[Any, int]] = []
        for _ in range(warmup):
            time_workload(code, func_name, allowed, perf_inputs, wl_timeout)
            if paired:
                time_workload(req["baseline_code"], func_name, allowed, perf_inputs, wl_timeout)
        rng = random.Random(req.get("timing_seed", 0))
        for i in range(repeats):
            order = ["c", "b"] if (i % 2 == 0) == (rng.random() < 0.5) else ["b", "c"]
            for which in order:
                if which == "c":
                    if by_group:
                        tot, per = time_workload(code, func_name, allowed, perf_inputs, wl_timeout, by_group=True)
                        cand_ns.append(tot)
                        cand_groups.append(per)
                    else:
                        cand_ns.append(time_workload(code, func_name, allowed, perf_inputs, wl_timeout))
                elif paired:
                    base_ns.append(time_workload(req["baseline_code"], func_name, allowed, perf_inputs, wl_timeout))
            # early exit for hopeless candidates
            if paired and len(base_ns) >= 2 and len(cand_ns) >= 2 and median(cand_ns) > max_ratio * median(base_ns):
                break
    except WorkloadTimeout:
        return {**resp, "status": "timeout", "message": "workload timed out during benchmarking"}
    except MemoryError:
        return {**resp, "status": "resource", "message": "memory limit exceeded during benchmarking"}
    except Exception as e:  # noqa: BLE001
        return {**resp, "status": "runtime_error", "message": f"{type(e).__name__}: {e} during benchmarking"}
    resp["candidate_ns"] = cand_ns
    resp["candidate_median_ns"] = median(cand_ns) if cand_ns else None
    if cand_groups:  # per-function profile of the candidate: median over repeats of each group's time
        keys = sorted({k for g in cand_groups for k in g})
        resp["candidate_group_ns"] = {k: median([g.get(k, 0) for g in cand_groups]) for k in keys}
    if paired:
        resp["baseline_ns"] = base_ns
        resp["baseline_median_ns"] = median(base_ns) if base_ns else None
        if base_ns and cand_ns:
            resp["ratio"] = median(cand_ns) / max(1.0, median(base_ns))
    return resp


def main() -> None:  # pragma: no cover - exercised via subprocess in tests
    import io

    data = sys.stdin.buffer.read()
    req = pickle.loads(data)
    real_stdout = sys.stdout
    sys.stdout = io.StringIO()  # candidate output must never reach the protocol channel
    set_limits(req.get("limits", {}))
    sys.setrecursionlimit(int(req.get("limits", {}).get("recursion_limit", 3000)))
    try:
        resp = handle(req)
    except MemoryError:
        resp = {"status": "resource", "message": "memory limit exceeded"}
    except Exception:  # noqa: BLE001
        resp = {"status": "worker_error", "message": traceback.format_exc()[-2000:]}
    out = pickle.dumps(resp, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stdout = real_stdout
    sys.stdout.buffer.write(out)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
