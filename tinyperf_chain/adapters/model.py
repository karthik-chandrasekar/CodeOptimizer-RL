"""Neural role adapters (role specialists, shared-role model, dense controls) and the legacy-writer adapter.

``NeuralRoleAdapter`` serves one role from one checkpoint through a :class:`RoleEngine` (directly, or through the
resident-role :class:`RoleScheduler`).  ``prompt_role`` lets a checkpoint answer another role's prompt format: a
shared 2M model serves all five roles, or M3's weights answer the M5 prompt in the "M3 repairs with the same
feedback" ablation.

``LegacyWriterAdapter`` connects the existing TinyPerf-RL optimizer, unchanged, as a writer: it sees the legacy
whole-file ``<CODE>/<STATE>`` observation and replies with ``<EDIT>`` or ``<STOP>``.  Its output is mapped to the
chain payload by one explicit, logged policy (``legacy_extract=target_only``): keep the selected function's ``def``
from the edit; an edit without it is malformed; ``<STOP>`` becomes ``ABANDON``.  This is the first-milestone
writer and is not a cleanup of a chain role's output.
"""
from __future__ import annotations

import ast
import time
from typing import Any, Dict, List, Optional

from tinyperf.env.protocol import EDIT_CLOSE, STOP_TOKEN, format_observation, parse_action
from tinyperf_chain.adapters.base import Prepared, RoleAdapter
from tinyperf_chain.config import RoleDecoding
from tinyperf_chain.context import ContextOverflow, PromptBuilder
from tinyperf_chain.hashing import sha256_text
from tinyperf_chain.schemas import (PATCH_ROLES, PatchAction, RequestEnvelope, RoleResponse, format_patch,
                                    parse_role_output)
from tinyperf_chain.serve.engine import EngineOverflow, RoleEngine, SamplingSpec


def sampling_spec(d: RoleDecoding) -> SamplingSpec:
    return SamplingSpec(greedy=d.mode == "greedy", temperature=d.temperature if d.mode == "sample" else 1.0,
                        top_k=d.top_k, top_p=d.top_p)


def request_seed(request_id: str, salt: int = 0) -> int:
    return int(sha256_text(f"{salt}|{request_id}")[:15], 16)


class NeuralRoleAdapter(RoleAdapter):
    kind = "neural"

    def __init__(self, engine: RoleEngine, tok, builder: PromptBuilder, decoding: RoleDecoding,
                 prompt_role: Optional[str] = None, scheduler=None, engine_key: str = "", seed: int = 0):
        self.engine = engine
        self.tok = tok
        self.builder = builder
        self.decoding = decoding
        self.prompt_role = prompt_role
        self.scheduler = scheduler
        self.engine_key = engine_key
        self.seed = seed
        self.stop_common = [tok.eos_id, tok.token_id(STOP_TOKEN)]

    def describe(self) -> Dict[str, Any]:
        return {"kind": self.kind, "checkpoint_sha256": self.engine.checkpoint_sha256, "prompt_role": self.prompt_role,
                "decoding": self.decoding.__dict__}

    def prepare(self, role: str, view: Any, ctx: Dict[str, Any]) -> Prepared:
        pr = self.prompt_role or role
        built = self.builder.build(pr, view)
        max_new = self.decoding.max_output
        if built.n_tokens + max_new > self.engine.max_seq_len:
            raise ContextOverflow(role, built.n_tokens + max_new, self.engine.max_seq_len)
        return Prepared(role, pr, built.ids, built.text, max_new, view, ctx, built.dropped)

    def execute(self, prep: Prepared, env: RequestEnvelope) -> RoleResponse:
        t0 = time.perf_counter_ns()
        stop = list(self.stop_common)
        if prep.role in PATCH_ROLES:
            stop.append(self.tok.token_id(EDIT_CLOSE))
        spec = sampling_spec(self.decoding)
        seed = request_seed(env.request_id, self.seed)
        deadline = env.deadline_monotonic if env.deadline_monotonic != float("inf") else None
        try:
            if self.scheduler is not None:
                out = self.scheduler.generate(self.engine_key, prep.prompt_ids, prep.max_new_tokens, spec, stop, seed,
                                              self.tok.pad_id, deadline)
            else:
                out = self.engine.generate([prep.prompt_ids], prep.max_new_tokens, spec, stop, pad_id=self.tok.pad_id,
                                           seeds=[seed], deadline=deadline)[0]
        except EngineOverflow as e:
            r = RoleResponse(prep.role, env.request_id, "", parse_role_output(prep.role, "", prep.ctx.get("function_id", "")),
                             self.kind, prep.prompt_tokens, 0, False, error="context_overflow")
            r.meta["detail"] = str(e)
            return r
        text = self.tok.decode(out.ids)
        parsed = parse_role_output(prep.role, text, prep.ctx.get("function_id", ""))
        r = RoleResponse(prep.role, env.request_id, text, parsed, self.kind, prep.prompt_tokens, len(out.ids), out.finished,
                         token_ids=list(out.ids), prompt_ids=list(prep.prompt_ids), logprobs=list(out.logprobs),
                         latency_ns=time.perf_counter_ns() - t0,
                         meta={"prefill_ns": out.prefill_ns, "decode_ns": out.decode_ns, "batch": out.batch_size,
                               "sampling": spec.__dict__, "seed": seed})
        if not out.finished and deadline is not None and time.monotonic() > deadline:
            r.error = "timeout"
        return r


class LegacyWriterAdapter(RoleAdapter):
    """The unchanged TinyPerf-RL optimizer used as M3 (whole-file legacy observation)."""

    kind = "legacy"

    def __init__(self, engine: RoleEngine, tok, max_new: int = 768, temperature: float = 0.6, seed: int = 0):
        self.engine, self.tok = engine, tok
        self.max_new = max_new
        self.spec = SamplingSpec(greedy=False, temperature=temperature)
        self.seed = seed
        self.stop = [tok.token_id(EDIT_CLOSE), tok.token_id(STOP_TOKEN), tok.eos_id]

    def describe(self) -> Dict[str, Any]:
        return {"kind": self.kind, "checkpoint_sha256": self.engine.checkpoint_sha256, "policy": "legacy_extract=target_only"}

    def prepare(self, role: str, view: Any, ctx: Dict[str, Any]) -> Prepared:
        obs = format_observation(ctx["file_source"], ctx["legacy_state"], feedback="full")
        ids = self.tok.encode(obs, add_bos=True)
        if len(ids) + self.max_new > self.engine.max_seq_len:
            raise ContextOverflow(role, len(ids) + self.max_new, self.engine.max_seq_len)
        return Prepared(role, "legacy", ids, obs, self.max_new, view, ctx)

    def execute(self, prep: Prepared, env: RequestEnvelope) -> RoleResponse:
        t0 = time.perf_counter_ns()
        out = self.engine.generate([prep.prompt_ids], self.max_new, self.spec, self.stop, pad_id=self.tok.pad_id,
                                   seeds=[request_seed(env.request_id, self.seed)])[0]
        raw = self.tok.decode(out.ids)
        act = parse_action(raw)
        target = prep.ctx.get("function_name", "")
        meta: Dict[str, Any] = {"legacy_raw": raw[:4000], "policy": "legacy_extract=target_only"}
        if act.kind == "stop":
            text = format_patch(PatchAction("abandon"))
        elif act.kind == "edit" and act.code:
            fn_src = None
            try:
                for n in ast.parse(act.code).body:
                    if isinstance(n, ast.FunctionDef) and n.name == target:
                        fn_src = ast.unparse(n)
            except SyntaxError:
                fn_src = None
            meta["extracted"] = fn_src is not None
            text = format_patch(PatchAction("edit", fn_src)) if fn_src else raw
        else:
            text = raw
        parsed = parse_role_output(prep.role, text, prep.ctx.get("function_id", ""))
        return RoleResponse(prep.role, env.request_id, text, parsed, self.kind, prep.prompt_tokens, len(out.ids),
                            out.finished, token_ids=list(out.ids), prompt_ids=list(prep.prompt_ids),
                            logprobs=list(out.logprobs), latency_ns=time.perf_counter_ns() - t0, meta=meta)
