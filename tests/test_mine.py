"""Function-mining filters, canonical dedup and type guessing."""
import ast

from tinyperf.data.argspec import gen_value, guess_specs, make_gen, ranked_combos
from tinyperf.data.mine import canonical_key, extract, validate

GOOD = '''
import math

def norm_all(xs):
    """docstring is dropped"""
    out = []
    for x in xs:
        out.append(math.sqrt(abs(x)))
    return out
'''


def test_extract_keeps_self_contained_functions_and_carries_imports():
    got = extract(GOOD)
    assert len(got) == 1 and got[0]["func_name"] == "norm_all"
    assert got[0]["source"].startswith("import math") and "docstring" not in got[0]["source"]


def test_extract_rejects_unsafe_or_unmineable_functions():
    bad = [
        "HELPER = 3\ndef f(xs):\n    out = []\n    for x in xs:\n        out.append(x * HELPER)\n    return out\n",   # module global
        "def f(path):\n    out = []\n    for line in open(path):\n        out.append(line)\n    return out\n",        # I/O
        "def f(xs):\n    for x in xs:\n        yield x\n    return\n    pass\n",                                        # generator
        "def f(xs):\n    return xs[0]\n",                                                                          # no loop, too short
        "class A:\n    def f(self, xs):\n        out = []\n        for x in xs:\n            out.append(x)\n        return out\n",  # method
        "import os\ndef f(xs):\n    out = []\n    for x in xs:\n        out.append(os.path.join(x))\n    return out\n",  # module not allowed
    ]
    for src in bad:
        assert extract(src) == [], src


def test_canonical_key_ignores_names_and_docstrings():
    a = ast.parse("def f(xs):\n    t = 0\n    for x in xs:\n        t += x\n    return t\n").body[0]
    b = ast.parse("def total(values):\n    '''sum'''\n    acc = 0\n    for v in values:\n        acc += v\n    return acc\n").body[0]
    c = ast.parse("def f(xs):\n    t = 1\n    for x in xs:\n        t *= x\n    return t\n").body[0]
    assert canonical_key(a) == canonical_key(b) != canonical_key(c)


def test_type_guesses_and_generators():
    fn = ast.parse("def f(words, k):\n    out = []\n    for i in range(k):\n        out.append(words[i].lower())\n    return out\n").body[0]
    specs = guess_specs(fn)
    assert "int_size" == specs[1][0] and specs[0][0] in ("list_int", "list_str")
    combos = ranked_combos(specs, cap=5)
    assert len(combos) == 5 and len(combos[0]) == 2
    import random
    args = make_gen(["list_str", "int_size"])(random.Random(0), 10)
    assert len(args[0]) == 10 and 5 <= args[1] <= 10 and isinstance(gen_value("dict_str_int", random.Random(0), 4), dict)


def test_validate_accepts_a_real_function_with_a_sensible_type():
    rec = validate(extract(GOOD)[0])
    assert rec is not None and rec["arg_specs"] in (["list_int"], ["list_float"]) and rec["stats"]["scaling"] > 2
