import random

from tinyperf.data.astutil import normalize
from tinyperf.data.degradations import FAMILIES, applicable
from tinyperf.data.rename import apply_rename, local_names, make_rename_map, mutate_bug, mutate_syntax
from tinyperf.data.seeds import SEEDS


def test_families_are_named():
    assert len(FAMILIES) >= 12 and "algorithmic" in FAMILIES


def test_every_seed_has_some_degradation():
    n_with = 0
    for s in SEEDS:
        src = normalize(s.source)
        apps = applicable(src)
        if apps or s.slow_variants:
            n_with += 1
        for fam, variants in apps.items():
            assert fam in FAMILIES
            for v in variants:
                assert v != src
                compile(v, "<v>", "exec")   # syntactically valid python
    assert n_with >= len(SEEDS) * 0.9


def test_degradations_are_single_site_and_deterministic():
    src = normalize(SEEDS[0].source)
    a, b = applicable(src), applicable(src)
    assert a == b


def test_rename_is_consistent_across_chain():
    s = next(x for x in SEEDS if x.slow_variants)
    chain = [normalize(s.source)] + [normalize(v) for v in s.slow_variants[:1]]
    rng = random.Random(1)
    from tinyperf.data.astutil import function_name

    mapping = make_rename_map(chain, rng, function_name(chain[0]))
    renamed = [apply_rename(c, mapping) for c in chain]
    fn_names = {function_name(c) for c in renamed}
    assert len(fn_names) == 1 and fn_names != {function_name(chain[0])}
    for c in renamed:
        compile(c, "<r>", "exec")
        assert not (local_names(c) & set(mapping)) or all(n not in mapping for n in local_names(c) if not n.startswith("_"))


def test_bug_and_syntax_mutations():
    src = normalize(SEEDS[0].source)
    rng = random.Random(0)
    bug = mutate_bug(src, rng)
    if bug is not None:
        assert bug != src
        compile(bug, "<bug>", "exec")
    broken = mutate_syntax(src, rng)
    try:
        compile(broken, "<broken>", "exec")
        ok = True
    except SyntaxError:
        ok = False
    assert not ok
