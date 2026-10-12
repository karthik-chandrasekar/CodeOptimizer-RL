"""T05 role grammar; source-field escaping."""
import random

import pytest

from tinyperf_chain.schemas import (PatchAction, Plan, ProbeSelection, TargetChoice, escape_field, format_patch,
                                    format_plan, format_probes, format_target, parse_patch, parse_plan, parse_probes,
                                    parse_target, unescape_field)


def test_t05_target_grammar():
    assert parse_target("TARGET f02").payload.function_id == "f02"
    assert parse_target("TARGET f02<EOS>").ok
    assert parse_target("NO_TARGET").payload.is_no_target
    for bad in ("TARGET  f02", "TARGET f02 extra", "target f02", " TARGET f02", "TARGET fx", "TARGET f02\n\n", "f02"):
        assert not parse_target(bad).ok, bad


def test_t05_plan_grammar_roundtrip():
    p = Plan("f01", "membership_index", ("src=allowed", "loop=n04"), ("order", "duplicates"))
    txt = format_plan(p)
    assert txt == "PLAN membership_index; src=allowed; loop=n04; preserve=order,duplicates"
    q = parse_plan(txt, "f01")
    assert q.ok and q.payload == p
    assert parse_plan("NO_PLAN", "f01").payload.is_no_plan
    for bad in ("PLAN membership_index;src=a", "PLAN Membership; src=a", "PLAN m; src=a; src=b", "PLAN m; src=a b",
                "PLAN m; preserve=order; preserve=ties", "please use a set"):
        assert not parse_plan(bad, "f01").ok, bad


def test_t05_patch_and_probe_grammar():
    code = "def f(xs):\n    return '<CODE>' + str(len(xs))\n"
    txt = format_patch(PatchAction("edit", code))
    r = parse_patch(txt)
    assert r.ok and r.payload.code.rstrip() == code.rstrip()
    assert parse_patch("ABANDON").payload.is_abandon
    for bad in ("ABANDON please", "<EDIT>\ndef f(): pass", "def f(): pass", "<EDIT>\n\n</EDIT>", "x<EDIT>\ndef f(): pass\n</EDIT>"):
        assert not parse_patch(bad).ok, bad
    assert parse_probes("PROBES p01,p04,p09").payload.probe_ids == ("p01", "p04", "p09")
    assert parse_probes("PROBES none").payload.probe_ids == ()
    for bad in ("PROBES", "PROBES p1", "PROBES p01, p02", "PROBES p01,"):
        assert not parse_probes(bad).ok, bad
    assert format_probes(ProbeSelection(())) == "PROBES none"
    assert format_target(TargetChoice(None)) == "NO_TARGET"


def test_escaping_is_reversible():
    rng = random.Random(0)
    alphabet = ["<", ">", "~", "/", "CODE", "EDIT", "STATE", "EOS", "x", "\n", "<~", "<CODE>", "</EDIT>"]
    for _ in range(2000):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
        e = escape_field(s)
        for tag in ("<CODE>", "</CODE>", "<EDIT>", "</EDIT>", "<STATE>", "<EOS>", "<STOP>"):
            assert tag not in e
        assert unescape_field(e) == s
