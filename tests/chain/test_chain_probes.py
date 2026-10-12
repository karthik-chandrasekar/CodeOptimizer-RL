"""T11 probe validity: menus contain only contract-valid recipes; selections are validated."""
from tinyperf_chain import registry as R
from tinyperf_chain.contracts import CONTRACT_VERSION, FunctionContract, ParamSpec, TaskContract, validate_args


def _contract(params):
    return TaskContract(CONTRACT_VERSION, [FunctionContract("f", [ParamSpec(**p) for p in params])])


def test_t11_menu_inputs_are_contract_valid():
    tc = _contract([{"name": "items", "type": "list[str]", "max_len": 50, "max_str_len": 4, "alphabet": "abcdef "},
                    {"name": "allowed", "type": "list[str]", "max_len": 50, "max_str_len": 4, "alphabet": "abcdef "}])
    m = R.probe_menu(tc, "t1", "f")
    assert 1 <= len(m.entries) <= 12
    for pid in m.ids():
        assert validate_args(tc, "f", m.inputs[pid]) is None
    assert "p08" in m.ids()                       # missing keys: queried items absent from the lookup list
    a, b = m.inputs["p08"]
    assert not set(a) & set(b)


def test_t11_out_of_contract_recipes_never_offered():
    tc = _contract([{"name": "n", "type": "int", "int_min": 0, "int_max": 100}])
    m = R.probe_menu(tc, "t1", "f")
    assert "p07" not in m.ids()                   # mixed sign impossible when ints are non-negative
    assert "p11" not in m.ids() and "p03" not in m.ids()
    for pid in m.ids():
        assert validate_args(tc, "f", m.inputs[pid]) is None
    tc2 = _contract([{"name": "xs", "type": "list[float]", "max_len": 10}])
    assert R.probe_menu(tc2, "t", "f").ids() in ([], ["p01"])   # floats deferred: only the empty input is constructible


def test_probe_inputs_deterministic():
    tc = _contract([{"name": "xs", "type": "list[int]", "max_len": 20, "int_min": -50, "int_max": 50}])
    assert R.probe_menu(tc, "t", "f").inputs == R.probe_menu(tc, "t", "f").inputs
