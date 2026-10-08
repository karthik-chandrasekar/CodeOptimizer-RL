"""Additional seed functions (imported by :mod:`tinyperf.data.seeds`).

Each seed is written so that *several* degradation operators apply to it
(a hoistable invariant, a set membership, a comprehension, a join, ...), so a
single seed yields a small tree of verified chains instead of one.  Function
names are deliberately varied, since the policy sees them.
"""
from __future__ import annotations

import random

from tinyperf.data.seeds import floats, ints, seed, text, word, words


def pairs(rng: random.Random, n: int, vocab: int = 300):
    ks = [word(rng, 6) for _ in range(max(1, vocab))]
    return [(rng.choice(ks), rng.randint(0, 1000)) for _ in range(n)]


def int_pairs(rng: random.Random, n: int, lo: int = 0, hi: int = 500):
    return [(rng.randint(lo, hi), rng.randint(lo, hi)) for _ in range(n)]


def lists_of_ints(rng: random.Random, n: int, k: int = 6):
    return [ints(rng, rng.randint(0, k), 0, 100) for _ in range(n)]


def sentence(rng: random.Random, n: int):
    return " ".join(words(rng, n, vocab=150, k=7))


# --------------------------------------------------------------------------- #
# A. membership against a parameter collection (set_to_list_membership + unhoist_convert)
# --------------------------------------------------------------------------- #
seed("count_hits", """
def count_hits(queries, table):
    lookup = set(table)
    return sum(1 for q in queries if q in lookup)
""", lambda rng, n: (ints(rng, n, 0, 5000), ints(rng, n, 0, 5000)))

seed("filter_known", """
def filter_known(items, known):
    allowed = set(known)
    return [x for x in items if x in allowed]
""", lambda rng, n: (words(rng, n, 400), words(rng, n // 2 + 1, 400)))

seed("drop_banned", """
def drop_banned(tokens, banned):
    blocked = set(banned)
    return [t for t in tokens if t not in blocked]
""", lambda rng, n: (words(rng, n, 300), words(rng, n // 4 + 1, 300)))

seed("first_missing", """
def first_missing(candidates, present):
    have = set(present)
    for c in candidates:
        if c not in have:
            return c
    return None
""", lambda rng, n: (ints(rng, n, 0, 3 * n + 1), ints(rng, n, 0, 3 * n + 1)), edge_inputs=[([], []), ([1], [1])])

seed("all_covered", """
def all_covered(required, available):
    avail = set(available)
    return all(r in avail for r in required)
""", lambda rng, n: (ints(rng, n, 0, 2 * n + 1), ints(rng, 2 * n, 0, 2 * n + 1)), edge_inputs=[([], []), ([1], [])])

seed("overlap_count", """
def overlap_count(a, b):
    b_set = set(b)
    return sum(1 for x in a if x in b_set)
""", lambda rng, n: (ints(rng, n, 0, 2000), ints(rng, n, 0, 2000)))

seed("mark_seen", """
def mark_seen(events, watched):
    watch = set(watched)
    return [(e, e in watch) for e in events]
""", lambda rng, n: (words(rng, n, 250), words(rng, n // 3 + 1, 250)))

seed("split_by_membership", """
def split_by_membership(xs, chosen):
    chosen_set = set(chosen)
    inside = [x for x in xs if x in chosen_set]
    outside = [x for x in xs if x not in chosen_set]
    return inside, outside
""", lambda rng, n: (ints(rng, n, 0, 1000), ints(rng, n // 2 + 1, 0, 1000)))

seed("count_new_ids", """
def count_new_ids(ids):
    seen = set()
    fresh = 0
    for i in ids:
        if i not in seen:
            fresh += 1
            seen.add(i)
    return fresh
""", lambda rng, n: (ints(rng, n, 0, n // 2 + 1),))

seed("first_repeat", """
def first_repeat(xs):
    seen = set()
    for x in xs:
        if x in seen:
            return x
        seen.add(x)
    return None
""", lambda rng, n: (ints(rng, n, 0, n + 1),), edge_inputs=[([],), ([3],), ([1, 1],)])

seed("longest_unique_prefix", """
def longest_unique_prefix(xs):
    seen = set()
    count = 0
    for x in xs:
        if x in seen:
            break
        seen.add(x)
        count += 1
    return count
""", lambda rng, n: (ints(rng, n, 0, n * 3 + 1),), edge_inputs=[([],), ([1, 1],)])

seed("symmetric_difference_size", """
def symmetric_difference_size(a, b):
    sa = set(a)
    sb = set(b)
    return sum(1 for x in a if x not in sb) + sum(1 for x in b if x not in sa)
""", lambda rng, n: (ints(rng, n, 0, 3 * n + 1), ints(rng, n, 0, 3 * n + 1)))

seed("unknown_words", """
def unknown_words(text_tokens, vocabulary):
    vocab = set(vocabulary)
    out = []
    for tok in text_tokens:
        if tok not in vocab:
            out.append(tok)
    return out
""", lambda rng, n: (words(rng, n, 500), words(rng, n, 500)))

seed("intersect_ordered", """
def intersect_ordered(a, b):
    in_b = set(b)
    result = []
    for x in a:
        if x in in_b:
            result.append(x)
    return result
""", lambda rng, n: (ints(rng, n, 0, 2 * n + 1), ints(rng, n, 0, 2 * n + 1)))

seed("count_pairs_in_table", """
def count_pairs_in_table(pairs_list, table):
    keys = set(table)
    return sum(1 for a, b in pairs_list if a in keys and b in keys)
""", lambda rng, n: (int_pairs(rng, n, 0, 300), ints(rng, n, 0, 300)))

# --------------------------------------------------------------------------- #
# B. loop-invariant computations (unhoist_call / unhoist_sort / unhoist_arith / unhoist_alloc)
# --------------------------------------------------------------------------- #
seed("fraction_of_total", """
def fraction_of_total(values):
    total = sum(values)
    return [v / total for v in values]
""", lambda rng, n: (ints(rng, n, 1, 100),), min_n=1)

seed("deviation_from_max", """
def deviation_from_max(xs):
    top = max(xs)
    return [top - x for x in xs]
""", lambda rng, n: (ints(rng, n, 0, 1000),), min_n=1)

seed("relative_to_min", """
def relative_to_min(xs):
    low = min(xs)
    return [x - low for x in xs]
""", lambda rng, n: (ints(rng, n, -500, 500),), min_n=1)

seed("center_values", """
def center_values(xs):
    mean = sum(xs) / len(xs)
    return [x - mean for x in xs]
""", lambda rng, n: (floats(rng, n),), min_n=1)

seed("count_above_half_max", """
def count_above_half_max(xs):
    limit = max(xs) / 2
    count = 0
    for x in xs:
        if x > limit:
            count += 1
    return count
""", lambda rng, n: (ints(rng, n, 0, 1000),), min_n=1)

seed("index_in_sorted", """
def index_in_sorted(xs, queries):
    ordered = sorted(xs)
    return [ordered.index(q) for q in queries]
""", lambda rng, n: (lambda base: (base, [rng.choice(base) for _ in range(n)]))(ints(rng, n, 0, 100)), min_n=1)

seed("rank_all", """
def rank_all(scores):
    ordered = sorted(scores)
    return [ordered.index(s) for s in scores]
""", lambda rng, n: (ints(rng, n, 0, 50),), min_n=1)

seed("scale_to_unit", """
def scale_to_unit(xs):
    lo = min(xs)
    span = max(xs) - lo
    if span == 0:
        return [0.0 for x in xs]
    return [(x - lo) / span for x in xs]
""", lambda rng, n: (floats(rng, n),), min_n=1, edge_inputs=[([2.0, 2.0],), ([1.0],)])

seed("above_threshold_ratio", """
def above_threshold_ratio(xs, k):
    threshold = k * len(xs)
    return [x > threshold for x in xs]
""", lambda rng, n: (ints(rng, n, 0, 10 * n + 1), rng.randint(1, 5)), min_n=1)

seed("last_n_sum", """
def last_n_sum(xs, k):
    tail = xs[len(xs) - k:]
    return [x + sum(tail) for x in xs]
""", lambda rng, n: (ints(rng, n, 0, 100), rng.randint(1, 8)), min_n=1)

seed("bucket_index", """
def bucket_index(xs, buckets):
    width = max(xs) / buckets + 1
    return [int(x / width) for x in xs]
""", lambda rng, n: (ints(rng, n, 0, 1000), rng.randint(2, 10)), min_n=1)

seed("compare_to_median", """
def compare_to_median(xs):
    ordered = sorted(xs)
    mid = ordered[len(ordered) // 2]
    return [x > mid for x in xs]
""", lambda rng, n: (ints(rng, n, 0, 1000),), min_n=1)

seed("zero_padded", """
def zero_padded(rows, width):
    pad = [0] * width
    return [r + pad[len(r):] for r in rows]
""", lambda rng, n: (lists_of_ints(rng, n, 6), 6), min_n=1)

seed("rows_above_mean_len", """
def rows_above_mean_len(rows):
    mean_len = sum(len(r) for r in rows) / len(rows)
    return [r for r in rows if len(r) > mean_len]
""", lambda rng, n: (lists_of_ints(rng, n, 8),), min_n=1)

seed("tokens_longer_than_avg", """
def tokens_longer_than_avg(tokens):
    avg = sum(len(t) for t in tokens) / len(tokens)
    return [t for t in tokens if len(t) > avg]
""", lambda rng, n: (words(rng, n, 200, 9),), min_n=1)

seed("percent_of_max", """
def percent_of_max(xs):
    top = max(xs)
    return [round(100 * x / top) for x in xs]
""", lambda rng, n: (ints(rng, n, 1, 1000),), min_n=1)

seed("distance_to_nearest_end", """
def distance_to_nearest_end(xs):
    n = len(xs)
    return [min(i, n - 1 - i) for i in range(n)]
""", lambda rng, n: (ints(rng, n),))

seed("windowed_flags", """
def windowed_flags(xs, k):
    n = len(xs)
    return [i + k < n for i in range(n)]
""", lambda rng, n: (ints(rng, n), rng.randint(1, 10)))

seed("mask_by_sorted_cutoff", """
def mask_by_sorted_cutoff(xs, k):
    cutoff = sorted(xs)[k]
    return [x >= cutoff for x in xs]
""", lambda rng, n: (lambda xs: (xs, rng.randint(0, len(xs) - 1)))(ints(rng, max(n, 1), 0, 1000)), min_n=1)

seed("normalize_by_first", """
def normalize_by_first(xs):
    base = abs(xs[0]) + 1
    return [x / base for x in xs]
""", lambda rng, n: (ints(rng, n, -100, 100),), min_n=1)

seed("count_in_top_half", """
def count_in_top_half(xs):
    ordered = sorted(xs)
    cutoff = ordered[len(ordered) // 2]
    count = 0
    for x in xs:
        if x >= cutoff:
            count += 1
    return count
""", lambda rng, n: (ints(rng, n, 0, 1000),), min_n=1)

seed("label_by_quantile", """
def label_by_quantile(xs):
    ordered = sorted(xs)
    q1 = ordered[len(ordered) // 4]
    q3 = ordered[3 * len(ordered) // 4]
    return ['low' if x < q1 else 'high' if x > q3 else 'mid' for x in xs]
""", lambda rng, n: (ints(rng, n, 0, 1000),), min_n=1)

# --------------------------------------------------------------------------- #
# C. lookup tables (dict_linear_search + unhoist_convert)
# --------------------------------------------------------------------------- #
seed("resolve_ids", """
def resolve_ids(ids, mapping_pairs):
    mapping = dict(mapping_pairs)
    return [mapping.get(i, -1) for i in ids]
""", lambda rng, n: (ints(rng, n, 0, 400), int_pairs(rng, n, 0, 400)))

seed("price_total", """
def price_total(cart, catalog_pairs):
    prices = dict(catalog_pairs)
    return sum(prices.get(item, 0) for item in cart)
""", lambda rng, n: (words(rng, n, 200), pairs(rng, n, 200)))

seed("known_ids_count", """
def known_ids_count(ids, registry_pairs):
    registry = dict(registry_pairs)
    return sum(1 for i in ids if i in registry)
""", lambda rng, n: (ints(rng, n, 0, 500), int_pairs(rng, n, 0, 500)))

seed("translate_tokens", """
def translate_tokens(tokens, dictionary_pairs):
    dictionary = dict(dictionary_pairs)
    return [dictionary.get(t, t) for t in tokens]
""", lambda rng, n: (words(rng, n, 200), [(w, w.upper()) for w in words(rng, n, 200)]))

seed("missing_keys", """
def missing_keys(keys, entries):
    table = dict(entries)
    return [k for k in keys if k not in table]
""", lambda rng, n: (words(rng, n, 300), pairs(rng, n, 300)))

seed("weighted_total", """
def weighted_total(items, weight_pairs):
    weights = dict(weight_pairs)
    return sum(weights.get(i, 1) * 2 for i in items)
""", lambda rng, n: (ints(rng, n, 0, 300), int_pairs(rng, n, 0, 300)))

seed("relabel", """
def relabel(labels, rename_pairs):
    renames = dict(rename_pairs)
    out = []
    for lab in labels:
        out.append(renames.get(lab, lab))
    return out
""", lambda rng, n: (words(rng, n, 150), [(w, w[::-1]) for w in words(rng, n, 150)]))

seed("score_lookup_max", """
def score_lookup_max(names, score_pairs):
    scores = dict(score_pairs)
    return max(scores.get(nm, 0) for nm in names)
""", lambda rng, n: (words(rng, n, 200), pairs(rng, n, 200)), min_n=1)

seed("lookup_pairs_both", """
def lookup_pairs_both(edges, weight_pairs):
    weights = dict(weight_pairs)
    return [weights.get(a, 0) + weights.get(b, 0) for a, b in edges]
""", lambda rng, n: (int_pairs(rng, n, 0, 300), int_pairs(rng, n, 0, 300)))

seed("has_all_keys", """
def has_all_keys(needed, entries):
    table = dict(entries)
    return all(k in table for k in needed)
""", lambda rng, n: (ints(rng, n, 0, 2 * n + 1), int_pairs(rng, 3 * n, 0, 2 * n + 1)), edge_inputs=[([], []), ([1], [])])

# --------------------------------------------------------------------------- #
# D. string building (join_to_concat) - often combined with a comprehension / hoist
# --------------------------------------------------------------------------- #
seed("comma_list", """
def comma_list(items):
    return ', '.join(items)
""", lambda rng, n: (words(rng, n, 200),))

seed("render_row", """
def render_row(cells):
    return '|' + '|'.join(str(c) for c in cells) + '|'
""", lambda rng, n: (ints(rng, n, 0, 1000),), slow_variants=["""
def render_row(cells):
    parts = []
    for c in cells:
        parts = parts + [str(c)]
    return '|' + '|'.join(parts) + '|'
"""])

seed("slugify_tokens", """
def slugify_tokens(tokens):
    return '-'.join(t.lower() for t in tokens)
""", lambda rng, n: ([w.upper() if rng.random() < 0.5 else w for w in words(rng, n, 200)],))

seed("initials", """
def initials(names):
    return ''.join(nm[0] for nm in names)
""", lambda rng, n: (words(rng, n, 200),))

seed("key_value_line", """
def key_value_line(entries):
    return ';'.join(k + '=' + str(v) for k, v in entries)
""", lambda rng, n: (pairs(rng, n, 200),))

seed("build_path", """
def build_path(parts):
    cleaned = [p.strip('/') for p in parts]
    return '/'.join(cleaned)
""", lambda rng, n: (['/' + w for w in words(rng, n, 200)],))

seed("wrap_words", """
def wrap_words(tokens, width):
    lines = []
    current = []
    used = 0
    for t in tokens:
        if used + len(t) > width and current:
            lines.append(' '.join(current))
            current = []
            used = 0
        current.append(t)
        used += len(t) + 1
    if current:
        lines.append(' '.join(current))
    return '\\n'.join(lines)
""", lambda rng, n: (words(rng, n, 200, 8), rng.randint(10, 40)))

seed("repeat_joined", """
def repeat_joined(token, count, sep):
    return sep.join(token for _ in range(count))
""", lambda rng, n: (word(rng, 6), n, rng.choice([',', ' ', '-'])), slow_variants=["""
def repeat_joined(token, count, sep):
    result = ''
    for i in range(count):
        result = sep.join([result, token]) if i else token
    return result
"""])

seed("reverse_each", """
def reverse_each(tokens):
    return ' '.join(t[::-1] for t in tokens)
""", lambda rng, n: (words(rng, n, 200),))

seed("csv_rows", """
def csv_rows(rows):
    return '\\n'.join(','.join(str(x) for x in r) for r in rows)
""", lambda rng, n: (lists_of_ints(rng, n, 5),))

seed("capitalize_sentence", """
def capitalize_sentence(tokens):
    fixed = [t[:1].upper() + t[1:] for t in tokens]
    return ' '.join(fixed)
""", lambda rng, n: (words(rng, n, 200),))

seed("bracketed", """
def bracketed(items):
    return ''.join('[' + str(i) + ']' for i in items)
""", lambda rng, n: (ints(rng, n, 0, 100),))

seed("hex_string", """
def hex_string(values):
    return ''.join(format(v, '02x') for v in values)
""", lambda rng, n: (ints(rng, n, 0, 255),))

seed("tag_tokens", """
def tag_tokens(tokens, tag):
    prefix = '<' + tag + '>'
    suffix = '</' + tag + '>'
    return ''.join(prefix + t + suffix for t in tokens)
""", lambda rng, n: (words(rng, n, 200), word(rng, 4)))

seed("dedupe_join", """
def dedupe_join(tokens):
    seen = set()
    kept = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            kept.append(t)
    return ' '.join(kept)
""", lambda rng, n: (words(rng, n, 100),))

# --------------------------------------------------------------------------- #
# E. selection builtins (worse_builtins) - max/min with and without keys
# --------------------------------------------------------------------------- #
seed("longest_row", """
def longest_row(rows):
    return max(rows, key=len)
""", lambda rng, n: (lists_of_ints(rng, n, 10),), min_n=1)

seed("shortest_word", """
def shortest_word(tokens):
    return min(tokens, key=len)
""", lambda rng, n: (words(rng, n, 300, 9),), min_n=1)

seed("value_span", """
def value_span(xs):
    return max(xs) - min(xs)
""", lambda rng, n: (ints(rng, n),), min_n=1)

seed("peak_pair", """
def peak_pair(pairs_list):
    return max(pairs_list, key=lambda p: p[1])
""", lambda rng, n: (int_pairs(rng, n),), min_n=1)

seed("closest_to_zero", """
def closest_to_zero(xs):
    return min(xs, key=abs)
""", lambda rng, n: (ints(rng, n, -1000, 1000),), min_n=1)

seed("row_maxima", """
def row_maxima(rows):
    return [max(r) for r in rows if r]
""", lambda rng, n: (lists_of_ints(rng, n, 8),))

seed("row_minima", """
def row_minima(rows):
    return [min(r) for r in rows if r]
""", lambda rng, n: (lists_of_ints(rng, n, 8),))

seed("best_score_name", """
def best_score_name(entries):
    return max(entries, key=lambda e: e[1])[0]
""", lambda rng, n: (pairs(rng, n, 300),), min_n=1)

seed("clamp_all", """
def clamp_all(xs, lo, hi):
    return [max(lo, min(hi, x)) for x in xs]
""", lambda rng, n: (ints(rng, n, -100, 100), -50, 50))

seed("extreme_gap", """
def extreme_gap(xs):
    hi = max(xs)
    lo = min(xs)
    return [hi - x if x - lo > hi - x else x - lo for x in xs]
""", lambda rng, n: (ints(rng, n, 0, 1000),), min_n=1)

seed("longest_streak_value", """
def longest_streak_value(xs):
    best = xs[0]
    best_len = 0
    current = xs[0]
    length = 0
    for x in xs:
        if x == current:
            length += 1
        else:
            current = x
            length = 1
        if length > best_len:
            best_len = length
            best = current
    return best
""", lambda rng, n: (ints(rng, n, 0, 5),), min_n=1, slow_variants=["""
def longest_streak_value(xs):
    best = xs[0]
    best_len = 0
    for i in range(len(xs)):
        length = 0
        for j in range(i, len(xs)):
            if xs[j] == xs[i]:
                length += 1
            else:
                break
        if length > best_len:
            best_len = length
            best = xs[i]
    return best
"""])

seed("widest_range_row", """
def widest_range_row(rows):
    return max(rows, key=lambda r: max(r) - min(r))
""", lambda rng, n: ([ints(rng, rng.randint(1, 6), 0, 100) for _ in range(n)],), min_n=1)

# --------------------------------------------------------------------------- #
# F. lazy builtins over generators (materialize)
# --------------------------------------------------------------------------- #
seed("has_long_token", """
def has_long_token(tokens, limit):
    return any(len(t) > limit for t in tokens)
""", lambda rng, n: (words(rng, n, 200, 8), rng.randint(6, 9)), edge_inputs=[([], 3), (["abcd"], 3)])

seed("all_positive", """
def all_positive(xs):
    return all(x > 0 for x in xs)
""", lambda rng, n: (ints(rng, n, 1, 1000) + ([0] if rng.random() < 0.3 else []),), edge_inputs=[([],), ([0],)])

seed("count_even", """
def count_even(xs):
    return sum(1 for x in xs if x % 2 == 0)
""", lambda rng, n: (ints(rng, n),))

seed("sum_of_squares_odd", """
def sum_of_squares_odd(xs):
    return sum(x * x for x in xs if x % 2 == 1)
""", lambda rng, n: (ints(rng, n, 0, 100),))

seed("first_negative", """
def first_negative(xs):
    return next((x for x in xs if x < 0), 0)
""", lambda rng, n: (ints(rng, n, 0, 1000) + ([rng.randint(-9, -1)] if rng.random() < 0.5 else []),), edge_inputs=[([],), ([-1],)])

seed("first_long_word", """
def first_long_word(tokens, limit):
    return next((t for t in tokens if len(t) >= limit), '')
""", lambda rng, n: (words(rng, n, 200, 8), rng.randint(5, 9)), edge_inputs=[([], 3), (["abcdefgh"], 3)])

seed("any_pair_sums_to", """
def any_pair_sums_to(pairs_list, target):
    return any(a + b == target for a, b in pairs_list)
""", lambda rng, n: (int_pairs(rng, n, 0, 100), rng.randint(0, 200)), edge_inputs=[([], 5)])

seed("mean_length", """
def mean_length(tokens):
    return sum(len(t) for t in tokens) / len(tokens)
""", lambda rng, n: (words(rng, n, 200, 8),), min_n=1)

seed("count_matching_prefix", """
def count_matching_prefix(tokens, prefix):
    return sum(1 for t in tokens if t.startswith(prefix))
""", lambda rng, n: (words(rng, n, 200), rng.choice("abcdefghij")))

seed("all_sorted_pairs", """
def all_sorted_pairs(pairs_list):
    return all(a <= b for a, b in pairs_list)
""", lambda rng, n: ([(a, max(a, b)) if rng.random() < 0.9 else (a, b) for a, b in int_pairs(rng, n)],), edge_inputs=[([],), ([(2, 1)],)])

seed("contains_negative_product", """
def contains_negative_product(pairs_list):
    return any(a * b < 0 for a, b in pairs_list)
""", lambda rng, n: (int_pairs(rng, n, -100, 100),), edge_inputs=[([],)])

seed("first_over_budget", """
def first_over_budget(costs, budget):
    running = 0
    for i, c in enumerate(costs):
        running += c
        if running > budget:
            return i
    return -1
""", lambda rng, n: (ints(rng, n, 0, 100), rng.randint(0, 50 * n + 1)), slow_variants=["""
def first_over_budget(costs, budget):
    for i in range(len(costs)):
        if sum(costs[:i + 1]) > budget:
            return i
    return -1
"""])

seed("total_positive", """
def total_positive(xs):
    return sum(x for x in xs if x > 0)
""", lambda rng, n: (ints(rng, n),))

seed("weighted_sum", """
def weighted_sum(values, weights):
    return sum(v * w for v, w in zip(values, weights))
""", lambda rng, n: (ints(rng, n, -50, 50), ints(rng, n, -50, 50)))

# --------------------------------------------------------------------------- #
# G. comprehensions (comprehension_to_loop, then append_to_concat at depth 2)
# --------------------------------------------------------------------------- #
seed("squares_of_evens", """
def squares_of_evens(xs):
    return [x * x for x in xs if x % 2 == 0]
""", lambda rng, n: (ints(rng, n),))

seed("lengths_of", """
def lengths_of(tokens):
    return [len(t) for t in tokens]
""", lambda rng, n: (words(rng, n, 200, 9),))

seed("pair_sums", """
def pair_sums(pairs_list):
    return [a + b for a, b in pairs_list]
""", lambda rng, n: (int_pairs(rng, n),))

seed("nonempty_rows", """
def nonempty_rows(rows):
    return [r for r in rows if len(r) > 0]
""", lambda rng, n: (lists_of_ints(rng, n, 4),))

seed("first_elements", """
def first_elements(rows):
    return [r[0] for r in rows if r]
""", lambda rng, n: (lists_of_ints(rng, n, 5),))

seed("scaled_ints", """
def scaled_ints(xs, factor):
    return [x * factor for x in xs]
""", lambda rng, n: (ints(rng, n), rng.randint(2, 9)))

seed("uppercase_short", """
def uppercase_short(tokens, limit):
    return [t.upper() for t in tokens if len(t) <= limit]
""", lambda rng, n: (words(rng, n, 200, 8), rng.randint(2, 6)))

seed("signs", """
def signs(xs):
    return [1 if x > 0 else -1 if x < 0 else 0 for x in xs]
""", lambda rng, n: (ints(rng, n, -10, 10),))

seed("indexed_tokens", """
def indexed_tokens(tokens):
    return [str(i) + ':' + t for i, t in enumerate(tokens)]
""", lambda rng, n: (words(rng, n, 200),))

seed("absolute_values", """
def absolute_values(xs):
    return [abs(x) for x in xs]
""", lambda rng, n: (ints(rng, n),))

seed("row_sums", """
def row_sums(rows):
    return [sum(r) for r in rows]
""", lambda rng, n: (lists_of_ints(rng, n, 8),))

seed("diffs_from_previous", """
def diffs_from_previous(xs):
    return [xs[i] - xs[i - 1] for i in range(1, len(xs))]
""", lambda rng, n: (ints(rng, n),), edge_inputs=[([],), ([5],)])

seed("threshold_flags", """
def threshold_flags(xs, limit):
    return [x >= limit for x in xs]
""", lambda rng, n: (ints(rng, n, 0, 100), rng.randint(0, 100)))

seed("strip_all", """
def strip_all(tokens):
    return [t.strip() for t in tokens]
""", lambda rng, n: ([' ' * rng.randint(0, 3) + w + ' ' * rng.randint(0, 3) for w in words(rng, n, 200)],))

seed("nonzero_positions", """
def nonzero_positions(xs):
    return [i for i, x in enumerate(xs) if x != 0]
""", lambda rng, n: (ints(rng, n, -2, 2),))

seed("flatten_pairs", """
def flatten_pairs(pairs_list):
    out = []
    for a, b in pairs_list:
        out.append(a)
        out.append(b)
    return out
""", lambda rng, n: (int_pairs(rng, n),), slow_variants=["""
def flatten_pairs(pairs_list):
    out = []
    for a, b in pairs_list:
        out = out + [a, b]
    return out
"""])

seed("interleave", """
def interleave(a, b):
    out = []
    for x, y in zip(a, b):
        out.append(x)
        out.append(y)
    return out
""", lambda rng, n: (ints(rng, n), ints(rng, n)), slow_variants=["""
def interleave(a, b):
    out = []
    for i in range(len(a)):
        out = out + [a[i]] + [b[i]]
    return out
"""])

seed("cumulative_max", """
def cumulative_max(xs):
    out = []
    best = None
    for x in xs:
        if best is None or x > best:
            best = x
        out.append(best)
    return out
""", lambda rng, n: (ints(rng, n),), slow_variants=["""
def cumulative_max(xs):
    out = []
    for i in range(len(xs)):
        out.append(max(xs[:i + 1]))
    return out
"""])

seed("running_total", """
def running_total(xs):
    out = []
    total = 0
    for x in xs:
        total += x
        out.append(total)
    return out
""", lambda rng, n: (ints(rng, n),), slow_variants=["""
def running_total(xs):
    out = []
    for i in range(len(xs)):
        out.append(sum(xs[:i + 1]))
    return out
"""])

seed("even_odd_split", """
def even_odd_split(xs):
    evens = [x for x in xs if x % 2 == 0]
    odds = [x for x in xs if x % 2 != 0]
    return evens, odds
""", lambda rng, n: (ints(rng, n),))

# --------------------------------------------------------------------------- #
# H. dict building and counting (dict_build, repeated_scan)
# --------------------------------------------------------------------------- #
seed("index_by_first_letter", """
def index_by_first_letter(tokens):
    groups = {}
    for t in tokens:
        groups.setdefault(t[0], []).append(t)
    return groups
""", lambda rng, n: (words(rng, n, 300),))

seed("positions_of", """
def positions_of(xs):
    where = {}
    for i, x in enumerate(xs):
        where.setdefault(x, []).append(i)
    return where
""", lambda rng, n: (ints(rng, n, 0, 30),))

seed("group_pairs", """
def group_pairs(pairs_list):
    grouped = {}
    for k, v in pairs_list:
        grouped.setdefault(k, []).append(v)
    return grouped
""", lambda rng, n: (int_pairs(rng, n, 0, 40),))

seed("group_by_parity", """
def group_by_parity(xs):
    groups = {}
    for x in xs:
        groups.setdefault(x % 2, []).append(x)
    return groups
""", lambda rng, n: (ints(rng, n),))

seed("count_tokens", """
def count_tokens(tokens):
    counts = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    return counts
""", lambda rng, n: (words(rng, n, 80),))

seed("count_values", """
def count_values(xs):
    counts = {}
    for x in xs:
        counts[x] = counts.get(x, 0) + 1
    return counts
""", lambda rng, n: (ints(rng, n, 0, 20),))

seed("histogram_buckets", """
def histogram_buckets(xs, width):
    bins = [x // width for x in xs]
    counts = {}
    for b in bins:
        counts[b] = counts.get(b, 0) + 1
    return counts
""", lambda rng, n: (ints(rng, n, 0, 1000), rng.randint(5, 50)))

seed("mode_value", """
def mode_value(xs):
    counts = {}
    for x in xs:
        counts[x] = counts.get(x, 0) + 1
    return max(counts, key=counts.get)
""", lambda rng, n: (ints(rng, n, 0, 15),), min_n=1)

seed("duplicates_only", """
def duplicates_only(xs):
    counts = {}
    for x in xs:
        counts[x] = counts.get(x, 0) + 1
    return [x for x in counts if counts[x] > 1]
""", lambda rng, n: (ints(rng, n, 0, n // 2 + 1),))

seed("token_lengths_by_letter", """
def token_lengths_by_letter(tokens):
    by_letter = {}
    for t in tokens:
        by_letter.setdefault(t[0], []).append(len(t))
    return by_letter
""", lambda rng, n: (words(rng, n, 300, 8),))

seed("sum_by_key", """
def sum_by_key(pairs_list):
    totals = {}
    for k, v in pairs_list:
        totals[k] = totals.get(k, 0) + v
    return totals
""", lambda rng, n: (int_pairs(rng, n, 0, 30),), slow_variants=["""
def sum_by_key(pairs_list):
    totals = {}
    for k, v in pairs_list:
        if k not in totals:
            totals[k] = sum(v2 for k2, v2 in pairs_list if k2 == k)
    return totals
"""])

seed("count_pairs_by_first", """
def count_pairs_by_first(pairs_list):
    counts = {}
    for a, b in pairs_list:
        counts[a] = counts.get(a, 0) + 1
    return counts
""", lambda rng, n: (int_pairs(rng, n, 0, 30),), slow_variants=["""
def count_pairs_by_first(pairs_list):
    counts = {}
    for a, b in pairs_list:
        if a not in counts:
            counts[a] = sum(1 for a2, b2 in pairs_list if a2 == a)
    return counts
"""])

seed("unique_per_key", """
def unique_per_key(pairs_list):
    seen = {}
    for k, v in pairs_list:
        seen.setdefault(k, set()).add(v)
    return {k: len(vs) for k, vs in seen.items()}
""", lambda rng, n: (int_pairs(rng, n, 0, 30),), slow_variants=["""
def unique_per_key(pairs_list):
    out = {}
    for k, v in pairs_list:
        if k not in out:
            vals = []
            for k2, v2 in pairs_list:
                if k2 == k and v2 not in vals:
                    vals.append(v2)
            out[k] = len(vals)
    return out
"""])

# --------------------------------------------------------------------------- #
# I. algorithmic regressions (hand-written slow variants)
# --------------------------------------------------------------------------- #
seed("has_pair_with_diff", """
def has_pair_with_diff(xs, d):
    seen = set()
    for x in xs:
        if x - d in seen or x + d in seen:
            return True
        seen.add(x)
    return False
""", lambda rng, n: (ints(rng, n, 0, 10 * n + 1), rng.randint(1, 20)), edge_inputs=[([], 1), ([1, 3], 2)], slow_variants=["""
def has_pair_with_diff(xs, d):
    for i in range(len(xs)):
        for j in range(i):
            if abs(xs[i] - xs[j]) == d:
                return True
    return False
"""])

seed("count_distinct_pairs_sum", """
def count_distinct_pairs_sum(xs, target):
    counts = {}
    for x in xs:
        counts[x] = counts.get(x, 0) + 1
    total = 0
    for x, c in counts.items():
        y = target - x
        if y in counts and x < y:
            total += 1
    return total
""", lambda rng, n: (ints(rng, n, 0, 100), rng.randint(0, 200)), slow_variants=["""
def count_distinct_pairs_sum(xs, target):
    found = set()
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if xs[i] + xs[j] == target and xs[i] != xs[j]:
                found.add((min(xs[i], xs[j]), max(xs[i], xs[j])))
    return len(found)
"""])

seed("remove_duplicates_sorted_output", """
def remove_duplicates_sorted_output(xs):
    return sorted(set(xs))
""", lambda rng, n: (ints(rng, n, 0, n // 2 + 1),), slow_variants=["""
def remove_duplicates_sorted_output(xs):
    out = []
    for x in xs:
        if x not in out:
            out.append(x)
    out.sort()
    return out
""", """
def remove_duplicates_sorted_output(xs):
    out = []
    for x in sorted(xs):
        if not out or out[-1] != x:
            out.append(x)
    return out
"""])

seed("majority_element", """
def majority_element(xs):
    counts = {}
    for x in xs:
        counts[x] = counts.get(x, 0) + 1
    best = max(counts, key=counts.get)
    return best if counts[best] * 2 > len(xs) else None
""", lambda rng, n: ([rng.choice([7, 7, 7, 1, 2, 3]) for _ in range(n)],), min_n=1, slow_variants=["""
def majority_element(xs):
    for x in xs:
        if xs.count(x) * 2 > len(xs):
            return x
    return None
"""])

seed("longest_common_prefix", """
def longest_common_prefix(tokens):
    if not tokens:
        return ''
    shortest = min(tokens, key=len)
    for i, ch in enumerate(shortest):
        for t in tokens:
            if t[i] != ch:
                return shortest[:i]
    return shortest
""", lambda rng, n: ([word(rng, 3) + word(rng, 6) for _ in range(n)],), edge_inputs=[([],), (['ab', 'ab'],)], slow_variants=["""
def longest_common_prefix(tokens):
    if not tokens:
        return ''
    prefix = tokens[0]
    for t in tokens:
        while not t.startswith(prefix):
            prefix = prefix[:-1]
    return prefix
"""])

seed("kth_smallest", """
def kth_smallest(xs, k):
    return sorted(xs)[k]
""", lambda rng, n: (lambda xs: (xs, rng.randint(0, len(xs) - 1)))(ints(rng, max(n, 1), 0, 1000)), min_n=1, slow_variants=["""
def kth_smallest(xs, k):
    remaining = list(xs)
    for _ in range(k):
        remaining.remove(min(remaining))
    return min(remaining)
"""])

seed("count_less_than_each", """
def count_less_than_each(xs):
    ordered = sorted(xs)
    counts = {}
    for i, x in enumerate(ordered):
        if x not in counts:
            counts[x] = i
    return [counts[x] for x in xs]
""", lambda rng, n: (ints(rng, n, 0, 100),), slow_variants=["""
def count_less_than_each(xs):
    return [sum(1 for y in xs if y < x) for x in xs]
"""])

seed("merge_sorted", """
def merge_sorted(a, b):
    out = []
    i = 0
    j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out
""", lambda rng, n: (sorted(ints(rng, n)), sorted(ints(rng, n))), slow_variants=["""
def merge_sorted(a, b):
    out = list(a)
    for x in b:
        pos = 0
        while pos < len(out) and out[pos] <= x:
            pos += 1
        out.insert(pos, x)
    return out
"""])

seed("two_sum_indices", """
def two_sum_indices(xs, target):
    seen = {}
    for i, x in enumerate(xs):
        if target - x in seen:
            return seen[target - x], i
        if x not in seen:
            seen[x] = i
    return None
""", lambda rng, n: (ints(rng, n, 0, 5 * n + 1), rng.randint(0, 10 * n + 1)), edge_inputs=[([], 0), ([1, 2], 3)], slow_variants=["""
def two_sum_indices(xs, target):
    for j in range(len(xs)):
        for i in range(j):
            if xs[i] + xs[j] == target:
                return i, j
    return None
"""])

seed("min_window_sum", """
def min_window_sum(xs, k):
    if k > len(xs):
        return None
    window = sum(xs[:k])
    best = window
    for i in range(k, len(xs)):
        window += xs[i] - xs[i - k]
        if window < best:
            best = window
    return best
""", lambda rng, n: (ints(rng, n), rng.randint(1, 6)), edge_inputs=[([1, 2], 3), ([5], 1)], slow_variants=["""
def min_window_sum(xs, k):
    if k > len(xs):
        return None
    return min(sum(xs[i:i + k]) for i in range(len(xs) - k + 1))
"""])

seed("intersect_count_multi", """
def intersect_count_multi(rows):
    if not rows:
        return 0
    common = set(rows[0])
    for r in rows[1:]:
        common &= set(r)
    return len(common)
""", lambda rng, n: ([ints(rng, 20, 0, 40) for _ in range(max(1, n // 10))],), edge_inputs=[([],), ([[1, 2]],)], slow_variants=["""
def intersect_count_multi(rows):
    if not rows:
        return 0
    count = 0
    for x in set(rows[0]):
        if all(x in r for r in rows[1:]):
            count += 1
    return count
"""])

seed("is_anagram_pair", """
def is_anagram_pair(a, b):
    return sorted(a) == sorted(b)
""", lambda rng, n: (lambda t: (t, ''.join(rng.sample(t, len(t))) if rng.random() < 0.5 else t + 'x'))(text(rng, n)), slow_variants=["""
def is_anagram_pair(a, b):
    if len(a) != len(b):
        return False
    remaining = list(b)
    for ch in a:
        if ch not in remaining:
            return False
        remaining.remove(ch)
    return True
"""])

seed("longest_zero_run", """
def longest_zero_run(xs):
    best = 0
    current = 0
    for x in xs:
        if x == 0:
            current += 1
            if current > best:
                best = current
        else:
            current = 0
    return best
""", lambda rng, n: (ints(rng, n, 0, 2),), slow_variants=["""
def longest_zero_run(xs):
    best = 0
    for i in range(len(xs)):
        for j in range(i, len(xs) + 1):
            if all(x == 0 for x in xs[i:j]) and j - i > best:
                best = j - i
    return best
"""])

seed("count_subarrays_with_sum", """
def count_subarrays_with_sum(xs, target):
    seen = {0: 1}
    total = 0
    running = 0
    for x in xs:
        running += x
        total += seen.get(running - target, 0)
        seen[running] = seen.get(running, 0) + 1
    return total
""", lambda rng, n: (ints(rng, n, -3, 3), rng.randint(-2, 2)), slow_variants=["""
def count_subarrays_with_sum(xs, target):
    total = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs) + 1):
            if sum(xs[i:j]) == target:
                total += 1
    return total
"""])

seed("pair_with_max_product", """
def pair_with_max_product(xs):
    ordered = sorted(xs)
    return max(ordered[0] * ordered[1], ordered[-1] * ordered[-2])
""", lambda rng, n: (ints(rng, max(n, 2), -100, 100),), min_n=2, slow_variants=["""
def pair_with_max_product(xs):
    best = None
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            p = xs[i] * xs[j]
            if best is None or p > best:
                best = p
    return best
"""])

# --------------------------------------------------------------------------- #
# J. multi-site functions (several families in one function -> deeper chains)
# --------------------------------------------------------------------------- #
seed("report_known_ratio", """
def report_known_ratio(items, known):
    allowed = set(known)
    total = len(items)
    hits = sum(1 for x in items if x in allowed)
    return str(hits) + '/' + str(total)
""", lambda rng, n: (ints(rng, n, 0, 500), ints(rng, n, 0, 500)))

seed("top_tokens_line", """
def top_tokens_line(tokens, k):
    counts = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    ordered = sorted(counts, key=counts.get, reverse=True)
    return ','.join(ordered[:k])
""", lambda rng, n: (words(rng, n, 60), rng.randint(1, 5)))

seed("normalized_names", """
def normalized_names(names, banned):
    blocked = set(banned)
    cleaned = [nm.strip().lower() for nm in names]
    kept = [nm for nm in cleaned if nm not in blocked]
    return ' '.join(kept)
""", lambda rng, n: ([' ' + w.upper() for w in words(rng, n, 200)], words(rng, n // 4 + 1, 200)))

seed("row_fractions", """
def row_fractions(rows):
    grand = sum(sum(r) for r in rows)
    return [[x / grand for x in r] for r in rows]
""", lambda rng, n: ([ints(rng, rng.randint(1, 5), 1, 50) for _ in range(n)],), min_n=1)

seed("above_mean_names", """
def above_mean_names(entries):
    mean = sum(v for nm, v in entries) / len(entries)
    return [nm for nm, v in entries if v > mean]
""", lambda rng, n: (pairs(rng, n, 300),), min_n=1)

seed("dedupe_keep_max", """
def dedupe_keep_max(pairs_list):
    best = {}
    for k, v in pairs_list:
        if k not in best or v > best[k]:
            best[k] = v
    return sorted(best.items())
""", lambda rng, n: (int_pairs(rng, n, 0, 50),), slow_variants=["""
def dedupe_keep_max(pairs_list):
    keys = []
    for k, v in pairs_list:
        if k not in keys:
            keys.append(k)
    return sorted((k, max(v for k2, v in pairs_list if k2 == k)) for k in keys)
"""])

seed("flag_frequent", """
def flag_frequent(xs, limit):
    counts = {}
    for x in xs:
        counts[x] = counts.get(x, 0) + 1
    return [counts[x] >= limit for x in xs]
""", lambda rng, n: (ints(rng, n, 0, 30), rng.randint(2, 5)))

seed("format_scores", """
def format_scores(names, score_pairs):
    scores = dict(score_pairs)
    lines = [nm + ': ' + str(scores.get(nm, 0)) for nm in names]
    return '\\n'.join(lines)
""", lambda rng, n: (words(rng, n, 200), pairs(rng, n, 200)))

seed("count_shared_letters", """
def count_shared_letters(a, b):
    letters = set(b)
    return sum(1 for ch in a if ch in letters)
""", lambda rng, n: (text(rng, n), text(rng, n)))

seed("filter_and_scale", """
def filter_and_scale(xs, keep):
    keep_set = set(keep)
    top = max(xs)
    return [x / top for x in xs if x in keep_set]
""", lambda rng, n: (ints(rng, n, 1, 300), ints(rng, n, 1, 300)), min_n=1)

seed("rank_tokens", """
def rank_tokens(tokens):
    counts = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    ordered = sorted(counts, key=counts.get, reverse=True)
    rank = {t: i for i, t in enumerate(ordered)}
    return [rank[t] for t in tokens]
""", lambda rng, n: (words(rng, n, 50),))

seed("outlier_flags", """
def outlier_flags(xs):
    mean = sum(xs) / len(xs)
    spread = max(abs(x - mean) for x in xs) / 2
    return [abs(x - mean) > spread for x in xs]
""", lambda rng, n: (floats(rng, n),), min_n=1)

seed("pair_lookup_sum", """
def pair_lookup_sum(queries, table_pairs):
    table = dict(table_pairs)
    hits = [table.get(q, 0) for q in queries]
    return sum(hits)
""", lambda rng, n: (ints(rng, n, 0, 300), int_pairs(rng, n, 0, 300)))

seed("distinct_sorted_line", """
def distinct_sorted_line(tokens):
    return ' '.join(sorted(set(tokens)))
""", lambda rng, n: (words(rng, n, 100),))

seed("longest_tokens", """
def longest_tokens(tokens):
    top = max(len(t) for t in tokens)
    return [t for t in tokens if len(t) == top]
""", lambda rng, n: (words(rng, n, 200, 8),), min_n=1)

seed("missing_in_range", """
def missing_in_range(xs):
    present = set(xs)
    hi = max(xs)
    return [i for i in range(hi) if i not in present]
""", lambda rng, n: (ints(rng, n, 0, 2 * n + 1),), min_n=1)

seed("common_prefix_count", """
def common_prefix_count(tokens, prefix):
    k = len(prefix)
    return sum(1 for t in tokens if t[:k] == prefix)
""", lambda rng, n: (words(rng, n, 200), word(rng, 2)))

seed("bounded_diffs", """
def bounded_diffs(xs, limit):
    n = len(xs)
    return [min(limit, abs(xs[i] - xs[i - 1])) for i in range(1, n)]
""", lambda rng, n: (ints(rng, n), rng.randint(1, 50)), edge_inputs=[([], 3), ([1], 3)])

seed("group_sizes_line", """
def group_sizes_line(pairs_list):
    groups = {}
    for k, v in pairs_list:
        groups.setdefault(k, []).append(v)
    return ' '.join(str(k) + ':' + str(len(vs)) for k, vs in sorted(groups.items()))
""", lambda rng, n: (int_pairs(rng, n, 0, 20),))

seed("percent_known", """
def percent_known(tokens, vocab):
    known = set(vocab)
    hits = [t for t in tokens if t in known]
    return round(100 * len(hits) / len(tokens))
""", lambda rng, n: (words(rng, n, 300), words(rng, n, 300)), min_n=1)

seed("max_row_by_total", """
def max_row_by_total(rows):
    return max(rows, key=sum)
""", lambda rng, n: ([ints(rng, rng.randint(1, 6)) for _ in range(n)],), min_n=1)

seed("center_and_round", """
def center_and_round(xs, digits):
    mean = sum(xs) / len(xs)
    return [round(x - mean, digits) for x in xs]
""", lambda rng, n: (floats(rng, n), rng.randint(0, 3)), min_n=1)

seed("nested_membership", """
def nested_membership(rows, wanted):
    targets = set(wanted)
    return [sum(1 for x in r if x in targets) for r in rows]
""", lambda rng, n: (lists_of_ints(rng, n, 8), ints(rng, 20, 0, 100)))

seed("shifted_by_min", """
def shifted_by_min(rows):
    low = min(min(r) for r in rows)
    return [[x - low for x in r] for r in rows]
""", lambda rng, n: ([ints(rng, rng.randint(1, 6)) for _ in range(n)],), min_n=1)

seed("summary_line", """
def summary_line(xs):
    return 'min=' + str(min(xs)) + ' max=' + str(max(xs)) + ' n=' + str(len(xs))
""", lambda rng, n: (ints(rng, n),), min_n=1)
