"""T04 context capacity: optional sections drop first, required content is never truncated."""
import pytest

from tinyperf_chain.config import RolesCfg
from tinyperf_chain.context import ContextOverflow, M2View, M3View, M5View, PromptBuilder


def test_t04_drop_optional_then_overflow(tok):
    roles = RolesCfg()
    roles.M2.max_input = 260
    b = PromptBuilder(tok, roles, 2048)
    src = "def f(xs):\n    out = []\n    for x in xs:\n        out.append(x)\n    return out\n"
    v = M2View("f00", src, [], "xs:list[int]", "n00:L3:for", ["xs", "out", "x"], ["sequence_builder"],
               history=[f"membership_index(src=xs;loop=n00):write:private_mismatch{i}" for i in range(30)])
    p = b.build("M2", v)
    assert "history" in p.dropped and p.n_tokens <= 260
    assert "def f(xs):" in p.text and "contract=xs:list[int]" in p.text and "menu=sequence_builder" in p.text
    roles.M2.max_input = 40
    with pytest.raises(ContextOverflow):
        b.build("M2", v)


def test_t04_reserved_output_and_no_truncation(tok):
    roles = RolesCfg()
    b = PromptBuilder(tok, roles, 2048)
    for r in ("M1", "M2", "M3", "M4", "M5"):
        d = roles.get(r)
        assert d.max_input + d.max_output <= 2048
    long_src = "def f(xs):\n" + "".join(f"    v{i} = len(xs) + {i}\n" for i in range(400)) + "    return xs\n"
    with pytest.raises(ContextOverflow):
        b.build("M3", M3View("f00", long_src, [], "xs:list[int]", "PLAN loop_invariant_hoist; src=xs"))
    p = b.build("M5", M5View("f00", "def f(xs):\n    return xs\n", "def f(xs):\n    return xs[::-1]\n", [], "xs:list[int]",
                             "PLAN x", {"failure_kind": "return_mismatch", "probe_id": "p03", "input": "([2, 1],)",
                                        "expected": "[2, 1]", "got": "[1, 2]"}))
    assert "failure=return_mismatch" in p.text and "expected=[2, 1]" in p.text
