"""Seed functions.

Each :class:`Seed` holds an *efficient* reference implementation, an input
generator producing hidden correctness / performance workloads, and optional
hand-written *algorithmic* slow variants (complexity-level inefficiencies that
are awkward to produce with AST rewrites).  The AST degradation operators in
:mod:`tinyperf.data.degradations` are applied on top of any of these sources.

Every generated pair (fast, slow) is *verified by execution* at dataset-build
time; nothing here is trusted to be behaviour-preserving without a differential
test, and nothing is trusted to be slower without a benchmark.
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

Args = Tuple


@dataclass
class Seed:
    name: str
    source: str
    gen: Callable[[random.Random, int], Args]     # (rng, n) -> one argument tuple of size ~n
    slow_variants: List[str] = field(default_factory=list)
    edge_inputs: List[Args] = field(default_factory=list)  # always included in correctness inputs
    min_n: int = 0                                  # generator's minimum meaningful size
    slow_family: str = "algorithmic"                # family label of slow_variants ("natural" for mined slow code)
    only_slow_variants: bool = False                # skip the degradation operators (natural seeds)
    group: str = ""                                 # identity for splits / CIs (a natural seed shares its function's group)


# --------------------------------------------------------------------------- #
# input helpers
# --------------------------------------------------------------------------- #
def ints(rng: random.Random, n: int, lo: int = -1000, hi: int = 1000) -> List[int]:
    return [rng.randint(lo, hi) for _ in range(n)]


def floats(rng: random.Random, n: int, lo: float = -100.0, hi: float = 100.0) -> List[float]:
    return [rng.uniform(lo, hi) for _ in range(n)]


def word(rng: random.Random, k: int = 5) -> str:
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(1, k)))


def words(rng: random.Random, n: int, vocab: int = 200, k: int = 6) -> List[str]:
    v = [word(rng, k) for _ in range(max(1, vocab))]
    return [rng.choice(v) for _ in range(n)]


def text(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(string.ascii_lowercase + "   ") for _ in range(n))


SEEDS: List[Seed] = []


def seed(name, source, gen, slow_variants=(), edge_inputs=(), min_n=0):
    s = Seed(name, source.strip("\n") + "\n", gen, [v.strip("\n") + "\n" for v in slow_variants], list(edge_inputs), min_n)
    SEEDS.append(s)
    return s


# --------------------------------------------------------------------------- #
# 1. membership / set-vs-list
# --------------------------------------------------------------------------- #
seed(
    "contains_duplicate",
    """
def contains_duplicate(xs):
    return len(xs) != len(set(xs))
""",
    lambda rng, n: (ints(rng, n, 0, max(1, 10 * n)),),
    slow_variants=[
        """
def contains_duplicate(xs):
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if xs[i] == xs[j]:
                return True
    return False
""",
        """
def contains_duplicate(xs):
    seen = []
    for x in xs:
        if x in seen:
            return True
        seen.append(x)
    return False
""",
    ],
    edge_inputs=[([],), ([7],), ([3, 3],)],
)

seed(
    "count_common",
    """
def count_common(queries, values):
    lookup = set(values)
    count = 0
    for q in queries:
        if q in lookup:
            count += 1
    return count
""",
    lambda rng, n: (ints(rng, n, 0, 4 * n), ints(rng, n, 0, 4 * n)),
    edge_inputs=[([], []), ([1], []), ([1, 2], [2])],
)

seed(
    "remove_stopwords",
    """
def remove_stopwords(tokens, stopwords):
    banned = set(stopwords)
    return [t for t in tokens if t not in banned]
""",
    lambda rng, n: (words(rng, n, 300), words(rng, max(1, n // 4), 120)),
    edge_inputs=[([], []), (["a"], ["a"]), (["a", "b"], [])],
)

seed(
    "two_sum_exists",
    """
def two_sum_exists(xs, target):
    seen = set()
    for x in xs:
        if target - x in seen:
            return True
        seen.add(x)
    return False
""",
    lambda rng, n: (ints(rng, n, -10 * n, 10 * n), rng.randint(-20 * n, 20 * n)),
    slow_variants=[
        """
def two_sum_exists(xs, target):
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if xs[i] + xs[j] == target:
                return True
    return False
""",
    ],
    edge_inputs=[([], 0), ([5], 10), ([5, 5], 10), ([1, 2], 3)],
)

seed(
    "unique_preserve_order",
    """
def unique_preserve_order(xs):
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
""",
    lambda rng, n: (ints(rng, n, 0, n),),
    slow_variants=[
        """
def unique_preserve_order(xs):
    out = []
    for x in xs:
        if x not in out:
            out.append(x)
    return out
""",
    ],
    edge_inputs=[([],), ([1, 1, 1],), ([1, 2, 3],)],
)

seed(
    "intersection_size",
    """
def intersection_size(a, b):
    return len(set(a) & set(b))
""",
    lambda rng, n: (ints(rng, n, 0, 3 * n), ints(rng, n, 0, 3 * n)),
    slow_variants=[
        """
def intersection_size(a, b):
    common = []
    for x in a:
        if x in b and x not in common:
            common.append(x)
    return len(common)
""",
    ],
    edge_inputs=[([], []), ([1], [1]), ([1, 1, 2], [2, 2, 1])],
)

# --------------------------------------------------------------------------- #
# 2. counting / grouping / dict building
# --------------------------------------------------------------------------- #
seed(
    "word_frequencies",
    """
def word_frequencies(tokens):
    counts = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    return counts
""",
    lambda rng, n: (words(rng, n, 150),),
    slow_variants=[
        """
def word_frequencies(tokens):
    counts = {}
    for t in tokens:
        counts[t] = tokens.count(t)
    return counts
""",
    ],
    edge_inputs=[([],), (["a"],), (["a", "b", "a"],)],
)

seed(
    "group_by_key",
    """
def group_by_key(pairs):
    groups = {}
    for key, value in pairs:
        groups.setdefault(key, []).append(value)
    return groups
""",
    lambda rng, n: ([(rng.randint(0, 40), rng.randint(0, 1000)) for _ in range(n)],),
    slow_variants=[
        """
def group_by_key(pairs):
    groups = {}
    for key, value in pairs:
        if key in groups:
            groups[key] = groups[key] + [value]
        else:
            groups[key] = [value]
    return groups
""",
    ],
    edge_inputs=[([],), ([(1, 2)],), ([(1, 2), (1, 3), (2, 4)],)],
)

seed(
    "bucket_by_length",
    """
def bucket_by_length(tokens):
    buckets = {}
    for t in tokens:
        buckets.setdefault(len(t), []).append(t)
    return buckets
""",
    lambda rng, n: (words(rng, n, 400, 9),),
    slow_variants=[
        """
def bucket_by_length(tokens):
    lengths = []
    for t in tokens:
        if len(t) not in lengths:
            lengths.append(len(t))
    buckets = {}
    for n in lengths:
        buckets[n] = [t for t in tokens if len(t) == n]
    return buckets
""",
    ],
    edge_inputs=[([],), (["ab", "cd", "e"],)],
)

seed(
    "anagram_groups",
    """
def anagram_groups(tokens):
    groups = {}
    for t in tokens:
        key = "".join(sorted(t))
        groups.setdefault(key, []).append(t)
    return list(groups.values())
""",
    lambda rng, n: ([("".join(rng.sample(w, len(w)))) for w in words(rng, n, 60, 6)],),
    slow_variants=[
        """
def anagram_groups(tokens):
    groups = []
    for t in tokens:
        placed = False
        for g in groups:
            if sorted(g[0]) == sorted(t):
                g.append(t)
                placed = True
                break
        if not placed:
            groups.append([t])
    return groups
""",
    ],
    edge_inputs=[([],), (["ab", "ba", "c"],)],
)

seed(
    "char_histogram",
    """
def char_histogram(s):
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    return counts
""",
    lambda rng, n: (text(rng, n),),
    slow_variants=[
        """
def char_histogram(s):
    counts = {}
    for ch in s:
        counts[ch] = s.count(ch)
    return counts
""",
    ],
    edge_inputs=[("",), ("aaa",), ("abc",)],
)

seed(
    "most_frequent",
    """
def most_frequent(xs):
    counts = {}
    for x in xs:
        counts[x] = counts.get(x, 0) + 1
    best = xs[0]
    for x in xs:
        if counts[x] > counts[best]:
            best = x
    return best
""",
    lambda rng, n: (ints(rng, n, 0, max(1, n // 3)),),
    slow_variants=[
        """
def most_frequent(xs):
    best = xs[0]
    for x in xs:
        if xs.count(x) > xs.count(best):
            best = x
    return best
""",
    ],
    edge_inputs=[([1],), ([1, 2, 2],), ([3, 3, 1, 1],)],
    min_n=1,
)

# --------------------------------------------------------------------------- #
# 3. hoisting / repeated computation
# --------------------------------------------------------------------------- #
seed(
    "normalize_scores",
    """
def normalize_scores(scores):
    total = sum(scores)
    return [s / total for s in scores]
""",
    lambda rng, n: (floats(rng, n, 1.0, 50.0),),
    edge_inputs=[([2.0],), ([1.0, 3.0],)],
    min_n=1,
)

seed(
    "scale_by_max",
    """
def scale_by_max(xs):
    m = max(xs)
    return [x / m for x in xs]
""",
    lambda rng, n: (floats(rng, n, 1.0, 100.0),),
    edge_inputs=[([1.0],), ([2.0, 4.0],)],
    min_n=1,
)

seed(
    "index_pairs_in_range",
    """
def index_pairs_in_range(xs, lo, hi):
    n = len(xs)
    out = []
    for i in range(n):
        if lo <= xs[i] <= hi:
            out.append(i)
    return out
""",
    lambda rng, n: (ints(rng, n), -300, 300),
    edge_inputs=[([], 0, 1), ([5], 0, 10), ([-5, 5], 0, 10)],
)

seed(
    "count_above_mean",
    """
def count_above_mean(xs):
    mean = sum(xs) / len(xs)
    count = 0
    for x in xs:
        if x > mean:
            count += 1
    return count
""",
    lambda rng, n: (floats(rng, n),),
    edge_inputs=[([1.0],), ([1.0, 2.0, 3.0],)],
    min_n=1,
)

seed(
    "clip_to_bounds",
    """
def clip_to_bounds(xs, bounds):
    lo = min(bounds)
    hi = max(bounds)
    return [lo if x < lo else hi if x > hi else x for x in xs]
""",
    lambda rng, n: (ints(rng, n), ints(rng, 8, -200, 200)),
    edge_inputs=[([], [0]), ([5], [0, 1]), ([-9, 9], [-1, 1])],
)

seed(
    "rank_by_sorted_position",
    """
def rank_by_sorted_position(xs):
    order = sorted(xs)
    return [order.index(x) for x in xs]
""",
    lambda rng, n: (ints(rng, min(n, 400), 0, 10**6),),
    edge_inputs=[([],), ([3],), ([2, 1, 2],)],
)

seed(
    "matches_sorted_prefix",
    """
def matches_sorted_prefix(xs, ys):
    prefix = sorted(xs)[:5]
    return [y for y in ys if y in prefix]
""",
    lambda rng, n: (ints(rng, n, 0, 100), ints(rng, n, 0, 100)),
    edge_inputs=[([], []), ([1], [1]), ([3, 1, 2], [2, 9])],
)

seed(
    "distances_to_center",
    """
def distances_to_center(points):
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return [abs(p[0] - cx) + abs(p[1] - cy) for p in points]
""",
    lambda rng, n: ([(rng.uniform(-10, 10), rng.uniform(-10, 10)) for _ in range(n)],),
    edge_inputs=[([(0.0, 0.0)],), ([(1.0, 1.0), (3.0, 3.0)],)],
    min_n=1,
)

# --------------------------------------------------------------------------- #
# 4. string building
# --------------------------------------------------------------------------- #
seed(
    "join_tokens",
    """
def join_tokens(tokens):
    return " ".join(tokens)
""",
    lambda rng, n: (words(rng, n, 500),),
    edge_inputs=[([],), (["a"],), (["a", "b"],)],
)

seed(
    "csv_line",
    """
def csv_line(fields):
    return ",".join(str(f) for f in fields)
""",
    lambda rng, n: (ints(rng, n),),
    edge_inputs=[([],), ([1],), ([1, -2, 3],)],
)

seed(
    "repeat_pattern",
    """
def repeat_pattern(unit, count):
    return "".join(unit for _ in range(count))
""",
    lambda rng, n: (word(rng, 4), n),
    edge_inputs=[("ab", 0), ("ab", 1), ("x", 3)],
)

seed(
    "reverse_words",
    """
def reverse_words(s):
    return " ".join(s.split()[::-1])
""",
    lambda rng, n: (text(rng, n),),
    slow_variants=[
        """
def reverse_words(s):
    parts = s.split()
    out = ""
    for i in range(len(parts) - 1, -1, -1):
        out = out + parts[i]
        if i > 0:
            out = out + " "
    return out
""",
    ],
    edge_inputs=[("",), ("a",), ("a b",), ("  a  b ",)],
)

seed(
    "is_palindrome",
    """
def is_palindrome(s):
    return s == s[::-1]
""",
    lambda rng, n: (rng.choice([text(rng, n), (lambda t: t + t[::-1])(text(rng, n // 2))]),),
    slow_variants=[
        """
def is_palindrome(s):
    for i in range(len(s)):
        if s[i] != s[len(s) - 1 - i]:
            return False
    return True
""",
    ],
    edge_inputs=[("",), ("a",), ("ab",), ("aba",)],
)

# --------------------------------------------------------------------------- #
# 5. numeric / prefix / windows
# --------------------------------------------------------------------------- #
seed(
    "prefix_sums",
    """
def prefix_sums(xs):
    out = []
    running = 0
    for x in xs:
        running += x
        out.append(running)
    return out
""",
    lambda rng, n: (ints(rng, n),),
    slow_variants=[
        """
def prefix_sums(xs):
    return [sum(xs[: i + 1]) for i in range(len(xs))]
""",
    ],
    edge_inputs=[([],), ([5],), ([1, -1, 2],)],
)

seed(
    "moving_average",
    """
def moving_average(xs, k):
    if k <= 0 or k > len(xs):
        return []
    window = sum(xs[:k])
    out = [window / k]
    for i in range(k, len(xs)):
        window += xs[i] - xs[i - k]
        out.append(window / k)
    return out
""",
    lambda rng, n: (ints(rng, n), rng.randint(1, max(1, n // 20))),
    slow_variants=[
        """
def moving_average(xs, k):
    if k <= 0 or k > len(xs):
        return []
    return [sum(xs[i : i + k]) / k for i in range(len(xs) - k + 1)]
""",
    ],
    edge_inputs=[([], 1), ([1], 1), ([1, 2, 3], 2), ([1, 2], 5), ([1, 2], 0)],
)

seed(
    "max_subarray",
    """
def max_subarray(xs):
    best = xs[0]
    current = xs[0]
    for x in xs[1:]:
        current = x if current < 0 else current + x
        if current > best:
            best = current
    return best
""",
    lambda rng, n: (ints(rng, min(n, 700), -50, 50),),
    slow_variants=[
        """
def max_subarray(xs):
    best = xs[0]
    for i in range(len(xs)):
        total = 0
        for j in range(i, len(xs)):
            total += xs[j]
            if total > best:
                best = total
    return best
""",
    ],
    edge_inputs=[([1],), ([-1],), ([-2, 1, -3, 4, -1, 2, 1, -5, 4],)],
    min_n=1,
)

seed(
    "dot_product",
    """
def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))
""",
    lambda rng, n: (ints(rng, n, -50, 50), ints(rng, n, -50, 50)),
    slow_variants=[
        """
def dot_product(a, b):
    total = 0
    for i in range(len(a)):
        total = total + a[i] * b[i]
    return total
""",
    ],
    edge_inputs=[([], []), ([2], [3]), ([1, 2], [3, 4])],
)

seed(
    "count_pairs_with_sum",
    """
def count_pairs_with_sum(xs, target):
    seen = {}
    count = 0
    for x in xs:
        count += seen.get(target - x, 0)
        seen[x] = seen.get(x, 0) + 1
    return count
""",
    lambda rng, n: (ints(rng, min(n, 800), 0, 50), rng.randint(0, 100)),
    slow_variants=[
        """
def count_pairs_with_sum(xs, target):
    count = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if xs[i] + xs[j] == target:
                count += 1
    return count
""",
    ],
    edge_inputs=[([], 0), ([1, 1], 2), ([1, 2, 3], 4)],
)

seed(
    "count_inversions",
    """
def count_inversions(xs):
    def merge_count(a):
        if len(a) <= 1:
            return a, 0
        mid = len(a) // 2
        left, cl = merge_count(a[:mid])
        right, cr = merge_count(a[mid:])
        merged = []
        i = j = 0
        inv = cl + cr
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                inv += len(left) - i
                j += 1
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged, inv
    return merge_count(list(xs))[1]
""",
    lambda rng, n: (ints(rng, min(n, 600)),),
    slow_variants=[
        """
def count_inversions(xs):
    count = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if xs[i] > xs[j]:
                count += 1
    return count
""",
    ],
    edge_inputs=[([],), ([1],), ([2, 1],), ([3, 1, 2],)],
)

seed(
    "squares_sum",
    """
def squares_sum(n):
    return sum(i * i for i in range(n))
""",
    lambda rng, n: (rng.randint(n, 2 * n),),
    slow_variants=[
        """
def squares_sum(n):
    total = 0
    for i in range(n):
        total = total + i ** 2
    return total
""",
    ],
    edge_inputs=[(0,), (1,), (5,)],
)

seed(
    "running_min",
    """
def running_min(xs):
    out = []
    current = None
    for x in xs:
        if current is None or x < current:
            current = x
        out.append(current)
    return out
""",
    lambda rng, n: (ints(rng, n),),
    slow_variants=[
        """
def running_min(xs):
    return [min(xs[: i + 1]) for i in range(len(xs))]
""",
    ],
    edge_inputs=[([],), ([3],), ([3, 1, 2],)],
)

# --------------------------------------------------------------------------- #
# 6. selection / builtins
# --------------------------------------------------------------------------- #
seed(
    "longest_token",
    """
def longest_token(tokens):
    return max(tokens, key=len)
""",
    lambda rng, n: (words(rng, n, 500, 9),),
    slow_variants=[
        """
def longest_token(tokens):
    return sorted(tokens, key=len, reverse=True)[0]
""",
    ],
    edge_inputs=[(["a"],), (["ab", "c", "ab"],)],
    min_n=1,
)

seed(
    "value_range",
    """
def value_range(xs):
    return max(xs) - min(xs)
""",
    lambda rng, n: (ints(rng, n),),
    edge_inputs=[([1],), ([1, 5],)],
    min_n=1,
)

seed(
    "top_k",
    """
import heapq

def top_k(xs, k):
    return heapq.nlargest(k, xs)
""",
    lambda rng, n: (ints(rng, n, 0, 10**6), rng.randint(1, 5)),
    slow_variants=[
        """
def top_k(xs, k):
    return sorted(xs, reverse=True)[:k]
""",
    ],
    edge_inputs=[([], 3), ([1], 1), ([5, 1, 9, 3], 2), ([5, 1], 9)],
)

seed(
    "has_negative",
    """
def has_negative(xs):
    return any(x < 0 for x in xs)
""",
    lambda rng, n: (ints(rng, n, 0, 1000) + ([rng.randint(-5, -1)] if rng.random() < 0.3 else []),),
    edge_inputs=[([],), ([-1],), ([1, 2],)],
)

seed(
    "count_matching",
    """
def count_matching(a, b):
    return sum(1 for x, y in zip(a, b) if x == y)
""",
    lambda rng, n: (ints(rng, n, 0, 5), ints(rng, n, 0, 5)),
    slow_variants=[
        """
def count_matching(a, b):
    count = 0
    for i in range(min(len(a), len(b))):
        if a[i] == b[i]:
            count += 1
    return count
""",
    ],
    edge_inputs=[([], []), ([1], [1]), ([1, 2], [2, 2])],
)

seed(
    "first_above",
    """
def first_above(xs, threshold):
    return next((x for x in xs if x > threshold), None)
""",
    lambda rng, n: (ints(rng, n, 0, 1000), rng.randint(900, 1100)),
    edge_inputs=[([], 0), ([1], 0), ([1, 2], 5)],
)

# --------------------------------------------------------------------------- #
# 7. lookup tables
# --------------------------------------------------------------------------- #
seed(
    "lookup_prices",
    """
def lookup_prices(items, catalog):
    prices = dict(catalog)
    return [prices.get(item, 0) for item in items]
""",
    lambda rng, n: (words(rng, n, 150), [(w, rng.randint(1, 100)) for w in set(words(rng, 200, 150))]),
    edge_inputs=[([], []), (["a"], []), (["a", "b"], [("a", 1)])],
)

seed(
    "map_ids",
    """
def map_ids(ids, names):
    lookup = {i: nm for i, nm in names}
    return [lookup[i] for i in ids]
""",
    lambda rng, n: ([rng.randint(0, 199) for _ in range(n)], [(i, word(rng)) for i in range(200)]),
    slow_variants=[
        """
def map_ids(ids, names):
    out = []
    for i in ids:
        for j, nm in names:
            if j == i:
                out.append(nm)
                break
    return out
""",
    ],
    edge_inputs=[([], []), ([0], [(0, "a")]), ([1, 0], [(0, "a"), (1, "b")])],
)

seed(
    "member_flags",
    """
def member_flags(xs, allowed):
    allowed_set = set(allowed)
    return [x in allowed_set for x in xs]
""",
    lambda rng, n: (ints(rng, n, 0, 2000), ints(rng, n // 2, 0, 2000)),
    edge_inputs=[([], []), ([1], [1]), ([1, 2], [2])],
)

# --------------------------------------------------------------------------- #
# 8. list building
# --------------------------------------------------------------------------- #
seed(
    "flatten",
    """
def flatten(lists):
    return [x for sub in lists for x in sub]
""",
    lambda rng, n: ([ints(rng, rng.randint(0, 12)) for _ in range(max(1, n // 6))],),
    slow_variants=[
        """
def flatten(lists):
    out = []
    for sub in lists:
        out = out + list(sub)
    return out
""",
    ],
    edge_inputs=[([],), ([[]],), ([[1], [2, 3]],)],
)

seed(
    "filter_in_range",
    """
def filter_in_range(xs, lo, hi):
    return [x for x in xs if lo <= x <= hi]
""",
    lambda rng, n: (ints(rng, n), -250, 250),
    slow_variants=[
        """
def filter_in_range(xs, lo, hi):
    out = []
    for x in xs:
        if lo <= x <= hi:
            out = out + [x]
    return out
""",
    ],
    edge_inputs=[([], 0, 1), ([5], 0, 10), ([-5, 5], 0, 10)],
)

seed(
    "pairwise_diffs",
    """
def pairwise_diffs(xs):
    return [b - a for a, b in zip(xs, xs[1:])]
""",
    lambda rng, n: (ints(rng, n),),
    slow_variants=[
        """
def pairwise_diffs(xs):
    out = []
    for i in range(len(xs) - 1):
        out = out + [xs[i + 1] - xs[i]]
    return out
""",
    ],
    edge_inputs=[([],), ([1],), ([1, 4, 2],)],
)

seed(
    "dedupe_sorted",
    """
def dedupe_sorted(xs):
    out = []
    for x in xs:
        if not out or out[-1] != x:
            out.append(x)
    return out
""",
    lambda rng, n: (sorted(ints(rng, n, 0, n // 2)),),
    slow_variants=[
        """
def dedupe_sorted(xs):
    out = []
    for x in xs:
        if x not in out:
            out.append(x)
    return out
""",
    ],
    edge_inputs=[([],), ([1],), ([1, 1, 2],)],
)

seed(
    "chunk_sums",
    """
def chunk_sums(xs, size):
    return [sum(xs[i : i + size]) for i in range(0, len(xs), size)]
""",
    lambda rng, n: (ints(rng, n), rng.randint(1, 16)),
    slow_variants=[
        """
def chunk_sums(xs, size):
    out = []
    for i in range(0, len(xs), size):
        total = 0
        for j in range(i, min(i + size, len(xs))):
            total = total + xs[j]
        out = out + [total]
    return out
""",
    ],
    edge_inputs=[([], 2), ([1], 1), ([1, 2, 3], 2)],
)


SEED_BY_NAME = {s.name: s for s in SEEDS}


def make_inputs(s: Seed, rng: random.Random, perf_scale: int, n_correct: int = 10, n_perf: int = 3) -> Tuple[List[Args], List[Args]]:
    """Hidden correctness inputs (edge cases + small/medium random) and perf workloads."""
    correct: List[Args] = list(s.edge_inputs)
    sizes = [max(s.min_n, k) for k in (1, 2, 3, 5, 8, 13, 21, 34, 60, 120)][:n_correct]
    for k in sizes:
        correct.append(s.gen(rng, k))
    perf = [s.gen(rng, max(s.min_n, int(perf_scale * rng.uniform(0.7, 1.3)))) for _ in range(n_perf)]
    return correct, perf


# extended library (appends to SEEDS)
from tinyperf.data import seeds_extra  # noqa: E402,F401
