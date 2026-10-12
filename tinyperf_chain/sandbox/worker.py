"""Chain sandbox worker (child side).  Runs in a fresh, isolated interpreter per request.

It is executed by path (``python -P -s worker.py`` with a scrubbed environment, or ``python -I`` on 3.10), reads
one pickled request written by the trusted parent from stdin, and writes one **JSON** response to stdout.  It never
returns pickles: the parent parses candidate-influenced output only as size-checked JSON (see
:mod:`tinyperf_chain.sandbox.runner`).

The static AST checks, restricted builtins, import guard, call timeouts and OS rlimits are the legacy TinyPerf
sandbox rules, loaded from ``tinyperf/env/worker.py`` by file path so the two sandboxes cannot drift apart.

Modes
-----
``run``         call the entry point on each input (fresh deep copy each call) and record, per input, the exception
                type (or None) and the plain-value encoding of ``(args_after_call, return_value)``; with
                ``output="digest"`` only a SHA-256 of each encoding is returned (large timing workloads).
``time_pair``   paired timing blocks: in each block the two programs run in a randomised order, each from a fresh
                namespace, ``reps`` repetitions of the full workload, every repetition on fresh deep copies built
                outside the timed region, GC disabled inside it.  Returns raw per-block nanoseconds.
``calibrate``   time one workload pass of a program and derive a fixed repetition count for a target block length.
``profile``     cProfile one pass of the public workload; per-function call counts, self and cumulative time.
``selftest``    report which forbidden operations the OS allows (isolation self-test; trusted code, not candidate).
"""
from __future__ import annotations

import gc
import importlib.util
import io
import json
import os
import pickle
import random
import sys
import time
import traceback
from copy import deepcopy
from typing import Any, Dict, List


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class _Ctx:
    def __init__(self, req: Dict[str, Any]):
        self.req = req
        self.legacy = _load(req["legacy_worker_path"], "tp_legacy_worker")
        self.codec = _load(req["codec_path"], "tp_chain_codec")
        self.allowed: List[str] = list(req.get("allowed_modules", []))
        lim = req.get("limits", {})
        self.call_timeout = float(lim.get("call_timeout_s", 2.0))
        self.workload_timeout = float(lim.get("workload_timeout_s", 10.0))
        self.dict_order = req.get("dict_order", "ignore")

    # ---------------------------------------------------------------- #
    def check_and_load(self, code: str, entry: str):
        import ast

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return None, {"status": "syntax_error", "message": f"SyntaxError: {e.msg} (line {e.lineno})"}
        try:
            self.legacy.static_check(tree, entry, self.allowed)
        except self.legacy.StaticCheckError as e:
            return None, {"status": "static_error", "message": str(e)}
        try:
            with self.legacy.time_limit(self.call_timeout):
                fn = self.legacy.load_function(code, entry, self.allowed)
        except self.legacy.CallTimeout:
            return None, {"status": "timeout", "message": "timeout while loading program"}
        except Exception as e:  # noqa: BLE001
            return None, {"status": "load_error", "message": f"{type(e).__name__}: {e}"}
        return fn, None

    def fresh(self, code: str, entry: str):
        return self.legacy.load_function(code, entry, self.allowed)


# --------------------------------------------------------------------------- #
def mode_run(ctx: _Ctx) -> Dict[str, Any]:
    req = ctx.req
    fn, err = ctx.check_and_load(req["code"], req["entry"])
    if err:
        return err
    output = req.get("output", "full")
    max_nodes = int(req.get("max_nodes_per_record", 200_000))
    max_chars = int(req.get("max_chars_per_record", 2_000_000))
    records = []
    for args in req["inputs"]:
        a = deepcopy(args)
        exc = None
        ret = None
        try:
            with ctx.legacy.time_limit(ctx.call_timeout):
                ret = fn(*a)
        except ctx.legacy.CallTimeout:
            exc = "__timeout__"
        except RecursionError:
            exc = "RecursionError"
        except MemoryError:
            exc = "MemoryError"
        except Exception as e:  # noqa: BLE001
            exc = type(e).__name__
        try:
            enc = ctx.codec.encode((a, ret), dict_order=ctx.dict_order, max_nodes=max_nodes, max_chars=max_chars)
            rec: Dict[str, Any] = {"exc": exc}
            if output == "digest":
                rec["digest"] = ctx.codec.digest(enc)
            else:
                rec["enc"] = enc
        except ctx.codec.CodecError as e:
            rec = {"exc": exc, "unencodable": str(e)}
        except RecursionError:
            rec = {"exc": exc, "unencodable": "value too deep"}
        records.append(rec)
        if exc == "__timeout__" and req.get("stop_on_timeout", True):
            break
    return {"status": "ok", "records": records}


def _time_once(ctx: _Ctx, fn, workload) -> int:
    copies = deepcopy(workload)                     # built outside the timed region
    gc.collect()
    was = gc.isenabled()
    gc.disable()
    try:
        with ctx.legacy.time_limit(ctx.workload_timeout):
            t0 = time.perf_counter_ns()
            for a in copies:
                fn(*a)
            t1 = time.perf_counter_ns()
    finally:
        if was:
            gc.enable()
    return t1 - t0


def mode_time_pair(ctx: _Ctx) -> Dict[str, Any]:
    req = ctx.req
    codes = [req["code_a"], req["code_b"]]
    entry = req["entry"]
    for c in codes:
        _, err = ctx.check_and_load(c, entry)
        if err:
            return err
    workload = req["workload"]
    blocks, reps = int(req["blocks"]), max(1, int(req["reps"]))
    rng = random.Random(int(req.get("seed", 0)))
    out = {"a_ns": [], "b_ns": [], "orders": [], "reps": reps}
    try:
        for _ in range(int(req.get("warmup", 1))):
            for c in codes:
                _time_once(ctx, ctx.fresh(c, entry), workload)
        for _b in range(blocks):
            order = [0, 1]
            rng.shuffle(order)
            got = [0, 0]
            for which in order:
                fn = ctx.fresh(codes[which], entry)        # fresh namespace: no cache survives between blocks
                tot = 0
                for _r in range(reps):
                    tot += _time_once(ctx, fn, workload)
                got[which] = tot
            out["a_ns"].append(got[0])
            out["b_ns"].append(got[1])
            out["orders"].append("ab" if order == [0, 1] else "ba")
    except ctx.legacy.CallTimeout:
        return {"status": "timeout", "message": "workload timed out during timing", **out}
    except MemoryError:
        return {"status": "resource", "message": "memory limit exceeded during timing", **out}
    except Exception as e:  # noqa: BLE001
        return {"status": "runtime_error", "message": f"{type(e).__name__}: {e} during timing", **out}
    out["status"] = "ok"
    return out


def mode_calibrate(ctx: _Ctx) -> Dict[str, Any]:
    req = ctx.req
    fn, err = ctx.check_and_load(req["code"], req["entry"])
    if err:
        return err
    try:
        _time_once(ctx, fn, req["workload"])
        samples = [_time_once(ctx, ctx.fresh(req["code"], req["entry"]), req["workload"]) for _ in range(3)]
    except ctx.legacy.CallTimeout:
        return {"status": "timeout", "message": "workload timed out during calibration"}
    except Exception as e:  # noqa: BLE001
        return {"status": "runtime_error", "message": f"{type(e).__name__}: {e}"}
    samples.sort()
    t = max(1, samples[1])
    target = float(req.get("target_ms", 50.0)) * 1e6
    reps = int(max(1, min(int(req.get("max_reps", 1000)), round(target / t))))
    return {"status": "ok", "t_once_ns": t, "reps": reps}


def mode_profile(ctx: _Ctx) -> Dict[str, Any]:
    import cProfile
    import pstats

    req = ctx.req
    fn, err = ctx.check_and_load(req["code"], req["entry"])
    if err:
        return err
    names = set(req.get("function_names", []))
    workload = deepcopy(req["workload"])
    prof = cProfile.Profile()
    try:
        with ctx.legacy.time_limit(ctx.workload_timeout * 3):
            t0 = time.perf_counter_ns()
            prof.enable()
            for a in workload:
                fn(*a)
            prof.disable()
            t1 = time.perf_counter_ns()
    except ctx.legacy.CallTimeout:
        prof.disable()
        return {"status": "timeout", "message": "workload timed out during profiling"}
    except Exception as e:  # noqa: BLE001
        prof.disable()
        return {"status": "runtime_error", "message": f"{type(e).__name__}: {e}"}
    st = pstats.Stats(prof)
    funcs: Dict[str, Dict[str, float]] = {}
    for (filename, _line, fname), (cc, nc, tt, ct, _callers) in st.stats.items():  # type: ignore[attr-defined]
        if filename == "<candidate>" and fname in names:
            d = funcs.setdefault(fname, {"ncalls": 0, "primitive_calls": 0, "self_ns": 0.0, "cum_ns": 0.0})
            d["ncalls"] += int(nc)
            d["primitive_calls"] += int(cc)
            d["self_ns"] += tt * 1e9
            d["cum_ns"] = max(d["cum_ns"], ct * 1e9)
    return {"status": "ok", "functions": funcs, "profiled_wall_ns": t1 - t0}


def mode_selftest(ctx: _Ctx) -> Dict[str, Any]:  # pragma: no cover - environment dependent
    res: Dict[str, Any] = {"status": "ok", "uid": os.getuid() if hasattr(os, "getuid") else None}
    try:
        with open("/tmp/_tp_chain_selftest_write", "w") as f:
            f.write("x")
        os.unlink("/tmp/_tp_chain_selftest_write")
        res["fs_write_tmp"] = True
    except OSError:
        res["fs_write_tmp"] = False
    try:
        home = os.path.expanduser("~")
        res["home_listable"] = bool(os.listdir(home)) if os.path.isdir(home) else False
    except OSError:
        res["home_listable"] = False
    try:
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.0)
        try:
            s.connect(("1.1.1.1", 53))
            res["network"] = True
        except OSError:
            res["network"] = False
        finally:
            s.close()
    except Exception:  # noqa: BLE001
        res["network"] = False
    try:
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        os.waitpid(pid, 0)
        res["fork"] = True
    except (OSError, AttributeError):
        res["fork"] = False
    return res


MODES = {"run": mode_run, "time_pair": mode_time_pair, "calibrate": mode_calibrate, "profile": mode_profile,
         "selftest": mode_selftest}


def main() -> None:  # pragma: no cover - exercised via subprocess
    data = sys.stdin.buffer.read()
    req = pickle.loads(data)   # written by the trusted parent only
    real_stdout = sys.stdout
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    resp: Dict[str, Any]
    try:
        ctx = _Ctx(req)
        lim = req.get("limits", {})
        if req.get("mode") != "selftest":
            ctx.legacy.set_limits(lim)
        sys.setrecursionlimit(int(lim.get("recursion_limit", 3000)))
        resp = MODES[req["mode"]](ctx)
    except MemoryError:
        resp = {"status": "resource", "message": "memory limit exceeded"}
    except Exception:  # noqa: BLE001
        resp = {"status": "worker_error", "message": traceback.format_exc()[-2000:]}
    try:
        resp["child_cpu_s"] = time.process_time()
    except Exception:  # noqa: BLE001
        pass
    try:
        out = json.dumps(resp, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, RecursionError) as e:
        out = json.dumps({"status": "worker_error", "message": f"response not serialisable: {e}"}).encode()
    real_stdout.buffer.write(out)
    real_stdout.buffer.flush()


if __name__ == "__main__":
    main()
