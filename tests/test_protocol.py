from tinyperf.env.protocol import EDIT_CLOSE, EDIT_OPEN, STOP_TOKEN, Action, format_action, format_observation, parse_action


def test_parse_edit():
    a = parse_action(f"{EDIT_OPEN}\ndef f(x):\n    return x\n{EDIT_CLOSE}")
    assert a.kind == "edit" and a.code == "def f(x):\n    return x"


def test_parse_edit_without_close_tag_is_accepted():
    a = parse_action(f"{EDIT_OPEN}\ndef f(x):\n    return x\n")
    assert a.kind == "edit" and "return x" in a.code


def test_parse_stop_and_precedence():
    assert parse_action(STOP_TOKEN).kind == "stop"
    assert parse_action(f"  {STOP_TOKEN} {EDIT_OPEN} junk {EDIT_CLOSE}").kind == "stop"
    assert parse_action(f"{EDIT_OPEN}\nx=1\n{EDIT_CLOSE}{STOP_TOKEN}").kind == "edit"


def test_parse_malformed():
    assert parse_action("def f(): pass").kind == "malformed"
    assert parse_action(f"{EDIT_OPEN}\n\n{EDIT_CLOSE}").kind == "malformed"
    assert parse_action("").kind == "malformed"


def test_format_roundtrip():
    code = "def g(a):\n    return a + 1"
    assert parse_action(format_action(Action("edit", code=code))).code == code
    assert parse_action(format_action(Action("stop"))).kind == "stop"


def test_observation_feedback_modes():
    state = {"best_runtime": 0.5, "last_runtime": "0.750", "last_correct": 1, "last_status": "slower", "step": 2, "remaining": 4}
    full = format_observation("def f(): pass", state, "full")
    none = format_observation("def f(): pass", state, "none")
    assert "best_runtime=0.500" in full and "last_status=slower" in full
    assert "best_runtime" not in none and "step=2" in none and "remaining=4" in none
    assert full.startswith("<CODE>\n") and "</STATE>\n" in full
