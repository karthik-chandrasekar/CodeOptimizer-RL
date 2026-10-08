"""Reading free-form chat replies as TinyPerf actions."""
from tinyperf.eval.evaluate_hf import parse_reply

FN = "def f(xs):\n    return sorted(set(xs))"


def test_protocol_replies():
    assert parse_reply(f"<EDIT>\n{FN}\n</EDIT>") == (f"<EDIT>\n{FN}\n</EDIT>", "edit")
    assert parse_reply("<STOP>") == ("<STOP>", "stop")
    assert parse_reply(f"<EDIT>\n```python\n{FN}\n```\n</EDIT><|im_end|>") == (f"<EDIT>\n{FN}\n</EDIT>", "edit")
    assert parse_reply(f"<EDIT>\n{FN}\n")[1] == "edit_unclosed"


def test_lenient_replies_and_thinking():
    assert parse_reply(f"Here is a faster version:\n```python\n{FN}\n```") == (f"<EDIT>\n{FN}\n</EDIT>", "fenced_edit")
    assert parse_reply(FN) == (f"<EDIT>\n{FN}\n</EDIT>", "bare_code_edit")
    assert parse_reply(f"<think>maybe use a set\n<EDIT>\nnot this\n</EDIT></think>\n<EDIT>\n{FN}\n</EDIT>")[0] == f"<EDIT>\n{FN}\n</EDIT>"
    assert parse_reply("I cannot improve this.")[1] == "unreadable"
    assert parse_reply("No change needed. <STOP> <EDIT>\nx\n</EDIT>")[1] == "stop"


def test_completion_prompt_for_base_code_models():
    from tinyperf.eval.evaluate_hf import COMPLETION_STOPS, completion_prompt
    shot_obs = "<CODE>\ndef g(xs):\n    return sorted(list(set(xs)))\n</CODE>\n<STATE>\nbest_runtime=1.000\n</STATE>"
    shot_act = "<EDIT>\ndef g(xs):\n    return sorted(set(xs))\n</EDIT>"
    obs = "<CODE>\ndef h(n):\n    return [i for i in range(n)]\n</CODE>\n<STATE>\nbest_runtime=1.000\nlast_status=none\n</STATE>"
    p = completion_prompt(obs, [(shot_obs, shot_act), ("ignored", "<STOP>")])
    assert p.count("# --- original ---\n") == 2 and p.count("\n# --- end ---\n") == 1     # one worked example only
    assert "def g(xs):\n    return sorted(set(xs))\n# --- end ---" in p
    assert p.endswith("def h(n):\n    return [i for i in range(n)]\n# state: best_runtime=1.000 last_status=none\n# --- faster ---\n")
    reply = "def h(n):\n    return list(range(n))\n" + COMPLETION_STOPS[0] + "\n\n# --- original ---\ndef x(): pass"
    import re
    text = re.split("|".join(map(re.escape, COMPLETION_STOPS)), reply)[0]
    assert parse_reply(text) == ("<EDIT>\ndef h(n):\n    return list(range(n))\n</EDIT>", "bare_code_edit")
