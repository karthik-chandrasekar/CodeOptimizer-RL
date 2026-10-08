"""Evaluate an off-the-shelf Hugging Face chat model (e.g. Qwen/Qwen3.5-0.8B) on TinyPerf tasks.

    python -m tinyperf.eval.evaluate_hf --model Qwen/Qwen3.5-0.8B --tasks artifacts/data_mined/files_mined_test.jsonl \\
        --n_samples 4 --shots_from artifacts/data_mined/files_msft_rl.jsonl --out artifacts/qwen_baseline/eval/qwen_real.json \\
        env.horizon=6

The model plays the same episodes as the TinyPerf models: same sandbox, hidden tests, timing, best-version tracking,
horizon and test files, and the records it writes are identical in format, so tinyperf.eval.budget,
scripts/best_of_k.py and scripts/analyze_files.py work unchanged.  Each turn it sees the current best code and state
(exactly the observation our models get), wrapped in its chat template with a system prompt describing the protocol and
two worked examples built from a TRAINING file (an edit, then STOP).  Replies are parsed leniently - an <EDIT> block, a
<STOP>, or failing both a fenced code block is taken as an edit - and how each reply was read is counted, so format
failures are visible separately from optimization failures.  Thinking mode is set explicitly (--thinking).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from tinyperf.common.config import load_config
from tinyperf.common.utils import get_logger, resolve_device, write_jsonl
from tinyperf.env.env import PerfEnv, compute_reward
from tinyperf.env.executor import Executor
from tinyperf.env.task import Task, load_tasks
from tinyperf.eval.metrics import compute_metrics, episode_record

log = get_logger("evaluate_hf")

SYSTEM_PROMPT = """You make Python code faster without changing what it does.

Each turn you see the current best version of a Python file inside <CODE> ... </CODE>, and a <STATE> block: best_runtime is the file's runtime relative to the original (1.000 = no speedup yet), last_status says what happened to your previous action, remaining is how many actions you have left.

Reply with exactly one action and nothing else:
<EDIT>
one or more complete function definitions; each replaces the function with the same name
</EDIT>
or
<STOP>
when you cannot make the code faster.

Every edit is run against hidden tests. An edit that changes any function's behaviour (return values, exceptions, mutation of arguments) or does not run is rejected and costs you an action. Only edits that keep behaviour identical and make the file faster are kept.

Use <STOP> only after your edits have already made the code faster and you see nothing more to improve. If best_runtime is still 1.000, make an edit. Do not explain your answer."""

COMPLETION_HEADER = ("# Task: rewrite functions in the Python file below so the file runs faster.\n"
                     "# Every function must keep exactly the same behaviour (return values, exceptions, argument mutation).\n"
                     "# Write only the rewritten function definitions after '# --- faster ---' and finish with '# --- end ---'.\n\n")
COMPLETION_STOPS = ["# --- end ---", "# --- original ---"]

FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
THINK = re.compile(r"<think>.*?</think>", re.S)


def parse_reply(text: str) -> Tuple[str, str]:
    """Map a free-form reply to protocol text for the environment, and say how it was read."""
    t = THINK.sub("", text or "")
    t = re.split(r"<\|im_end\|>|<\|endoftext\|>", t)[0].strip()
    i_edit, i_stop = t.find("<EDIT>"), t.find("<STOP>")
    if i_edit >= 0 and (i_stop < 0 or i_edit < i_stop):
        j = t.find("</EDIT>", i_edit)
        body = t[i_edit + len("<EDIT>"):j if j >= 0 else len(t)]
        m = FENCE.search(body)
        body = (m.group(1) if m else body).strip("\n")
        return f"<EDIT>\n{body}\n</EDIT>", "edit" if j >= 0 else "edit_unclosed"
    if i_stop >= 0:
        return "<STOP>", "stop"
    m = FENCE.search(t)
    if m:
        return f"<EDIT>\n{m.group(1).strip()}\n</EDIT>", "fenced_edit"
    try:                                         # bare code: accept only if it parses and defines a function
        tree = ast.parse(t)
        if any(isinstance(n, ast.FunctionDef) for n in tree.body):
            return f"<EDIT>\n{t}\n</EDIT>", "bare_code_edit"
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        pass
    return t, "unreadable"


def _code_and_state(obs: str) -> Tuple[str, str]:
    a, b = obs.find("<CODE>"), obs.find("</CODE>")
    code = obs[a + 6:b].strip("\n") if a >= 0 and b > a else obs
    c, d = obs.find("<STATE>"), obs.find("</STATE>")
    state = " ".join(obs[c + 7:d].split()) if c >= 0 and d > c else ""
    return code, state


def completion_prompt(obs: str, shots: Sequence[Tuple[str, str]]) -> str:
    """The episode as plain code for a completion-only model: a worked example, then the current file."""
    parts = [COMPLETION_HEADER]
    for o, a in shots[:1]:
        code, _ = _code_and_state(o)
        fix = a.split("<EDIT>", 1)[-1].split("</EDIT>", 1)[0].strip("\n")
        parts.append(f"# --- original ---\n{code}\n# --- faster ---\n{fix}\n# --- end ---\n\n")
    code, state = _code_and_state(obs)
    parts.append(f"# --- original ---\n{code}\n# state: {state}\n# --- faster ---\n")
    return "".join(parts)


def make_shots(cfg, train_tasks_path: str, n_candidates: int = 300) -> List[Tuple[str, str]]:
    """Two worked examples from a small TRAINING file: its first fix (hottest slow function), then the final STOP."""
    from tinyperf.train.sft_data import replay
    tasks = [t for t in load_tasks(train_tasks_path, limit=n_candidates) if t.fast_source and (t.meta or {}).get("slow_functions")]
    tasks.sort(key=lambda t: (len(t.meta["slow_functions"]) != 1, len(t.source)))     # prefer one slow function, small files
    ex = Executor(cfg.env)
    h = max(cfg.env.horizon, 6)            # examples need room for the fixes and a STOP, whatever the evaluation horizon
    try:
        for t in tasks[:20]:
            fast = {n.name: ast.unparse(n) for n in ast.parse(t.fast_source).body if isinstance(n, ast.FunctionDef)}
            prof = t.meta.get("profile0") or {}
            slow = sorted((f for f in t.meta["slow_functions"] if f in fast), key=lambda f: -prof.get(f, 0.0))
            if not slow or len(slow) > h - 1:
                continue
            row = replay(PerfEnv(ex, cfg.env, feedback="full", horizon=h), t, [fast[f] for f in slow])
            if row and all(s["status"] in ("ok", "stop") for s in row["steps"]) and row["steps"][-1]["status"] == "stop":
                return [(row["steps"][0]["obs"], row["steps"][0]["action"]), (row["steps"][-1]["obs"], row["steps"][-1]["action"])]
    finally:
        ex.close()
    raise RuntimeError(f"no usable worked example in {train_tasks_path}")


def _remote_code_options(trust_remote_code: str) -> List[bool]:
    return {"auto": [False, True], "true": [True], "false": [False]}[trust_remote_code]


def _param_count_from_index(model_id: str) -> Optional[int]:
    """Total parameters from the checkpoint's safetensors index (bf16/fp16 = 2 bytes each), without loading weights."""
    try:
        from huggingface_hub import hf_hub_download
        path = model_id if os.path.isdir(model_id) else None
        f = os.path.join(path, "model.safetensors.index.json") if path else hf_hub_download(model_id, "model.safetensors.index.json")
        return int(json.load(open(f))["metadata"]["total_size"] // 2)
    except Exception:  # noqa: BLE001
        return None


class HFPolicy:
    def __init__(self, model_id: str, device: str, thinking: bool, shots: Sequence[Tuple[str, str]], max_new_tokens: int,
                 temperature: float, top_p: float, top_k: int, batch_size: int, dtype: str = "bfloat16",
                 backend: str = "hf", trust_remote_code: str = "auto", seed: int = 0):
        self.device, self.thinking, self.shots, self.backend = device, thinking, list(shots), backend
        self.max_new_tokens, self.temperature, self.top_p, self.top_k, self.batch_size = max_new_tokens, temperature, top_p, top_k, batch_size
        if backend == "vllm":
            os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
            try:
                from vllm import LLM
            except ImportError as e:
                raise RuntimeError("backend 'vllm' needs vLLM installed in this Python environment (see scripts/run_hf_baselines.sh)") from e
            errors = []
            for trc in _remote_code_options(trust_remote_code):
                try:
                    self.llm = LLM(model=model_id, dtype=dtype, trust_remote_code=trc, seed=seed, max_model_len=16384,
                                   gpu_memory_utilization=float(os.environ.get("VLLM_GPU_UTIL", "0.6")))
                    break
                except Exception as e:  # noqa: BLE001
                    errors.append(f"trust_remote_code={trc}: {e}")
            else:
                raise RuntimeError("vLLM could not load " + model_id + ":\n" + "\n".join(errors))
            self.tok = self.llm.get_tokenizer()
            self.n_params = _param_count_from_index(model_id) or 0
            return
        from transformers import AutoTokenizer
        for trc in _remote_code_options(trust_remote_code):
            try:
                self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trc)
                break
            except Exception:  # noqa: BLE001
                if trc == _remote_code_options(trust_remote_code)[-1]:
                    raise
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = self._load(model_id, getattr(torch, dtype), trust_remote_code).to(device).eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())

    @staticmethod
    def _load(model_id: str, dtype, trust_remote_code: str = "auto"):
        import transformers
        errors = []
        kw = {"dtype": dtype} if int(transformers.__version__.split(".")[0]) >= 5 else {"torch_dtype": dtype}
        for trc in _remote_code_options(trust_remote_code):            # native implementation first, remote code if needed
            for name in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModelForMultimodalLM", "AutoModelForVision2Seq"):
                cls = getattr(transformers, name, None)
                if cls is None:
                    continue
                try:
                    return cls.from_pretrained(model_id, trust_remote_code=trc, **kw)
                except Exception as e:  # noqa: BLE001 - wrong auto class, or needs remote code
                    errors.append(f"{name} (trust_remote_code={trc}): {str(e)[:300]}")
        raise RuntimeError("could not load " + model_id + ":\n" + "\n".join(errors))

    def set_format(self, fmt: str) -> str:
        self.format = ("chat" if getattr(self.tok, "chat_template", None) else "completion") if fmt == "auto" else fmt
        return self.format

    def prompt(self, obs: str) -> str:
        if getattr(self, "format", "chat") == "completion":
            return completion_prompt(obs, self.shots)
        turns = []
        for o, a in self.shots:
            turns += [{"role": "user", "content": o}, {"role": "assistant", "content": a}]
        turns.append({"role": "user", "content": obs})
        try:
            return self.tok.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT}] + turns, tokenize=False,
                                                add_generation_prompt=True, enable_thinking=self.thinking)
        except Exception:  # noqa: BLE001 - templates without a system role: fold it into the first user turn
            turns[0] = {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + turns[0]["content"]}
            return self.tok.apply_chat_template(turns, tokenize=False, add_generation_prompt=True, enable_thinking=self.thinking)

    def _generate_vllm(self, observations: Sequence[str]) -> List[Tuple[str, bool]]:
        from vllm import SamplingParams
        completion = getattr(self, "format", "chat") == "completion"
        sp = SamplingParams(temperature=self.temperature, top_p=self.top_p, top_k=self.top_k if self.top_k > 0 else -1,
                            max_tokens=self.max_new_tokens, stop=COMPLETION_STOPS if completion else None)
        outs = self.llm.generate([self.prompt(o) for o in observations], sp, use_tqdm=False)
        return [(o.outputs[0].text, o.outputs[0].finish_reason == "stop") for o in outs]

    @torch.no_grad()
    def generate(self, observations: Sequence[str]) -> List[Tuple[str, bool]]:
        if self.backend == "vllm":
            return self._generate_vllm(observations)
        out: List[Tuple[str, bool]] = []
        eos = {self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")} - {None, self.tok.unk_token_id}
        for b in range(0, len(observations), self.batch_size):
            completion = getattr(self, "format", "chat") == "completion"
            enc = self.tok([self.prompt(o) for o in observations[b:b + self.batch_size]], return_tensors="pt",
                           padding=True, add_special_tokens=completion).to(self.device)   # chat templates add their own BOS
            kw = dict(max_new_tokens=self.max_new_tokens, pad_token_id=self.tok.pad_token_id, eos_token_id=sorted(eos))
            if completion:
                kw.update(stop_strings=COMPLETION_STOPS, tokenizer=self.tok)
            if self.temperature > 0:
                kw.update(do_sample=True, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k)
            else:
                kw.update(do_sample=False)
            gen = self.model.generate(**enc, **kw)[:, enc["input_ids"].shape[1]:]
            for row in gen.tolist():
                text = self.tok.decode(row, skip_special_tokens=completion)
                finished = any(t in eos for t in row) or (completion and any(x in text for x in COMPLETION_STOPS))
                out.append((text, finished))
        return out


def run(cfg, policy: HFPolicy, tasks: Sequence[Task], n_samples: int, seed: int, raw_path: Optional[str]) -> Tuple[List[Dict], Counter, int]:
    torch.manual_seed(seed)
    ex = Executor(cfg.env)
    envs, meta = [], []
    for task in tasks:
        for g in range(n_samples):
            env = PerfEnv(ex, cfg.env, feedback=cfg.env.feedback, horizon=cfg.env.horizon)
            env.reset(task)
            envs.append(env)
            meta.append((task, g))
    reads: Counter = Counter()
    truncated, raw = 0, []
    active = list(range(len(envs)))
    t0 = time.time()
    while active:
        obs = [envs[i].observation() for i in active]
        replies = policy.generate(obs)
        actions = []
        for i, (text, finished) in zip(active, replies):
            if getattr(policy, "format", "chat") == "completion":
                text = re.split("|".join(map(re.escape, COMPLETION_STOPS)), text)[0]
            act, how = parse_reply(text)
            reads[how] += 1
            truncated += not finished
            actions.append((i, act))
            if raw_path and len(raw) < 200:
                raw.append({"task_id": meta[i][0].task_id, "how": how, "finished": finished, "reply": text[:4000]})
        ex.map(lambda ia: envs[ia[0]].step(ia[1]), actions)
        active = [i for i in active if not envs[i].done]
        log.info(f"turn done | {len(active)} episodes still active | reads {dict(reads)} | {time.time() - t0:.0f}s")
    ex.close()
    records = []
    for i, env in enumerate(envs):
        summ = env.summary()
        task, g = meta[i]
        r = episode_record(summ, compute_reward(summ, cfg.reward), task, cfg.env.horizon)
        r["group"] = g
        records.append(r)
    if raw_path:
        write_jsonl(raw_path, raw)
    return records, reads, truncated


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--model", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--n_tasks", type=int, default=0)
    p.add_argument("--n_samples", type=int, default=4)
    p.add_argument("--out", required=True)
    p.add_argument("--shots_from", required=True, help="TRAINING file tasks for the two worked examples")
    p.add_argument("--shots", choices=["edit", "edit+stop"], default="edit", help="worked examples: an edit, or an edit then a STOP")
    p.add_argument("--format", choices=["auto", "chat", "completion"], default="auto",
                   help="chat for instruction-tuned models; completion for base code models (auto: chat if it has a chat template)")
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.6, help="0.6, top_p 1, top_k 0 = the TinyPerf models' settings")
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--backend", choices=["hf", "vllm"], default="hf", help="vllm: much faster for mixture-of-experts models")
    p.add_argument("--trust_remote_code", choices=["auto", "true", "false"], default="auto",
                   help="auto: use the native transformers implementation, the checkpoint's own code only if needed")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("overrides", nargs="*")
    a = p.parse_args(argv)
    cfg = load_config(a.config, a.overrides)
    if a.backend == "vllm":
        # vLLM starts its engine in a child process; touching CUDA here first (even torch.cuda.is_available) breaks a
        # forked child, so ask for spawn and leave device selection to vLLM
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        device = "cuda"
    else:
        device = resolve_device(cfg.device)
    shots = make_shots(cfg, a.shots_from)
    if a.shots == "edit":
        shots = shots[:1]          # a small model copies the last example; a STOP example makes it stop at once
    policy = HFPolicy(a.model, device, a.thinking, shots, a.max_new_tokens, a.temperature, a.top_p, a.top_k, a.batch_size, a.dtype,
                      a.backend, a.trust_remote_code, a.seed)
    fmt = policy.set_format(a.format)
    log.info(f"{a.model}: {policy.n_params / 1e6:.0f}M parameters, {fmt} prompt, thinking={a.thinking}, {len(shots)} worked examples")
    tasks = load_tasks(a.tasks, limit=a.n_tasks)
    records, reads, truncated = run(cfg, policy, tasks, a.n_samples, a.seed, os.path.splitext(a.out)[0] + "_replies.jsonl")
    m = compute_metrics(records)
    n_turns = max(1, sum(reads.values()))
    result = {"model": a.model, "n_params": policy.n_params, "thinking": a.thinking, "shots": a.shots, "format": fmt, "backend": a.backend, "tasks": a.tasks, "n_tasks": len(tasks),
              "n_samples": a.n_samples, "env": {"horizon": cfg.env.horizon, "feedback": cfg.env.feedback},
              "generation": {"temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k, "max_new_tokens": a.max_new_tokens},
              "metrics": m, "reply_reading": {k: v / n_turns for k, v in reads.items()}, "truncated_replies": truncated / n_turns}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)
    write_jsonl(os.path.splitext(a.out)[0] + "_trajectories.jsonl", records)
    log.info(f"eval: final {os.path.basename(a.model)} on {os.path.basename(a.tasks)} (H={cfg.env.horizon}, thinking={a.thinking}): "
             f"success={m['success_rate']:.3f} gm_speedup={m['gm_speedup']:.3f} p>=1.5x={m['p_speedup_ge_1.5']:.3f} "
             f"edits={m['mean_edits']:.2f} invalid={m['invalid_edit_rate']:.3f} | replies read as {dict(reads)} | truncated {truncated}")


if __name__ == "__main__":
    main()
