"""Plain-value codec: a JSON-compatible, tagged encoding of supported Python values *including object identity*.

This module is stdlib-only on purpose: the sandbox worker loads it by file path inside an isolated interpreter,
and the trusted parent imports it normally.  The child encodes what the candidate produced; the parent only ever
parses JSON (never pickle) from a candidate-controlled process and compares encodings in trusted code.

Encoding (``enc``), one list per node, first element is the tag::

    ["N"]                      None
    ["b", true]                bool
    ["i", 123]                 int (exact type; subclasses are unsupported)
    ["f", "1.5"]               float (repr, so nan / inf / -0.0 survive)
    ["c", "1.0", "-2.0"]       complex
    ["s", "text"]              str
    ["y", "00ff"]              bytes (hex)
    ["T", [enc, ...]]          tuple
    ["F", [enc, ...]]          frozenset (elements in canonical order)
    ["L", label, [enc, ...]]   list        \
    ["D", label, [[k, v], ...]]  dict        | mutable containers carry a label;
    ["S", label, [enc, ...]]   set         | a second reference to the same object
    ["B", label, "00ff"]       bytearray  /  is encoded as ["r", label]

Labels are assigned in traversal order.  With ``dict_order="ignore"`` dict items are traversed in canonical key
order (and sets are always canonical), so two values that are equal *and have the same aliasing structure*
produce byte-identical encodings.  :func:`compare` adds float tolerance on top of that.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Tuple

CODEC_VERSION = "plain_value_v1"   # part of every cache key; bump when the encoding or comparison changes


class CodecError(Exception):
    """Value cannot be encoded under the plain-value contract (unsupported type, too large, too deep)."""


def _cj(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class _Encoder:
    def __init__(self, dict_order: str, max_nodes: int, max_chars: int, max_depth: int):
        self.dict_order = dict_order
        self.max_nodes = max_nodes
        self.max_chars = max_chars
        self.max_depth = max_depth
        self.nodes = 0
        self.chars = 0
        self.labels: Dict[int, int] = {}
        self.keep: List[Any] = []  # keep encoded objects alive so ids are not reused mid-encoding

    def _tick(self, chars: int = 0) -> None:
        self.nodes += 1
        self.chars += chars
        if self.nodes > self.max_nodes:
            raise CodecError("value too large (node budget)")
        if self.chars > self.max_chars:
            raise CodecError("value too large (character budget)")

    def _label(self, v: Any) -> Tuple[bool, int]:
        k = id(v)
        if k in self.labels:
            return True, self.labels[k]
        lab = len(self.labels)
        self.labels[k] = lab
        self.keep.append(v)
        return False, lab

    def enc(self, v: Any, depth: int = 0) -> Any:
        if depth > self.max_depth:
            raise CodecError("value too deep")
        t = type(v)
        if v is None:
            self._tick()
            return ["N"]
        if t is bool:
            self._tick()
            return ["b", v]
        if t is int:
            self._tick(len(str(v)) if abs(v) > 10**15 else 1)
            return ["i", v]
        if t is float:
            self._tick()
            return ["f", repr(v)]
        if t is complex:
            self._tick()
            return ["c", repr(v.real), repr(v.imag)]
        if t is str:
            self._tick(len(v))
            return ["s", v]
        if t is bytes:
            self._tick(2 * len(v))
            return ["y", v.hex()]
        if t is tuple:
            self._tick()
            return ["T", [self.enc(x, depth + 1) for x in v]]
        if t is frozenset:
            self._tick()
            return ["F", sorted((self.enc(x, depth + 1) for x in v), key=_cj)]
        if t in (list, dict, set, bytearray):
            seen, lab = self._label(v)
            self._tick()
            if seen:
                return ["r", lab]
            if t is list:
                return ["L", lab, [self.enc(x, depth + 1) for x in v]]
            if t is bytearray:
                self.chars += 2 * len(v)
                return ["B", lab, bytes(v).hex()]
            if t is set:
                return ["S", lab, sorted((self.enc(x, depth + 1) for x in v), key=_cj)]
            # dict
            items = list(v.items())
            if self.dict_order == "ignore":
                keyed = [(self.enc(k, depth + 1), val) for k, val in items]
                keyed.sort(key=lambda kv: _cj(kv[0]))
                return ["D", lab, [[k, self.enc(val, depth + 1)] for k, val in keyed]]
            return ["D", lab, [[self.enc(k, depth + 1), self.enc(val, depth + 1)] for k, val in items]]
        raise CodecError(f"unsupported value type {t.__name__}")


def encode(value: Any, dict_order: str = "ignore", max_nodes: int = 500_000, max_chars: int = 4_000_000,
           max_depth: int = 200) -> Any:
    return _Encoder(dict_order, max_nodes, max_chars, max_depth).enc(value)


def digest(enc: Any) -> str:
    return hashlib.sha256(_cj(enc).encode("utf-8")).hexdigest()


def has_aliasing(enc: Any) -> bool:
    """True when the encoding contains a back-reference (shared or cyclic mutable object)."""
    stack = [enc]
    while stack:
        x = stack.pop()
        if not isinstance(x, list) or not x:
            continue
        tag = x[0]
        if tag == "r":
            return True
        if tag in ("T", "F"):
            stack.extend(x[1])
        elif tag in ("L", "S"):
            stack.extend(x[2])
        elif tag == "D":
            for k, v in x[2]:
                stack.append(k)
                stack.append(v)
    return False


# --------------------------------------------------------------------------- #
# Decoding (trusted side; reconstructs object graphs, aliasing included)
# --------------------------------------------------------------------------- #
def decode(enc: Any) -> Any:
    labels: Dict[int, Any] = {}

    def dec(x: Any) -> Any:
        if not isinstance(x, list) or not x:
            raise CodecError("malformed encoding")
        tag = x[0]
        if tag == "N":
            return None
        if tag == "b":
            return bool(x[1])
        if tag == "i":
            if type(x[1]) is not int:
                raise CodecError("malformed int")
            return x[1]
        if tag == "f":
            return float(x[1])
        if tag == "c":
            return complex(float(x[1]), float(x[2]))
        if tag == "s":
            if type(x[1]) is not str:
                raise CodecError("malformed str")
            return x[1]
        if tag == "y":
            return bytes.fromhex(x[1])
        if tag == "T":
            return tuple(dec(e) for e in x[1])
        if tag == "F":
            return frozenset(dec(e) for e in x[1])
        if tag == "r":
            if x[1] not in labels:
                raise CodecError("dangling reference")
            return labels[x[1]]
        if tag == "L":
            out: List[Any] = []
            labels[x[1]] = out
            out.extend(dec(e) for e in x[2])
            return out
        if tag == "S":
            s: set = set()
            labels[x[1]] = s
            for e in x[2]:
                s.add(dec(e))
            return s
        if tag == "B":
            b = bytearray(bytes.fromhex(x[2]))
            labels[x[1]] = b
            return b
        if tag == "D":
            d: Dict[Any, Any] = {}
            labels[x[1]] = d
            for k, v in x[2]:
                d[dec(k)] = dec(v)
            return d
        raise CodecError(f"unknown tag {tag!r}")

    return dec(enc)


# --------------------------------------------------------------------------- #
# Comparison (trusted side)
# --------------------------------------------------------------------------- #
def _float_eq(a: str, b: str, rel: float, abs_: float) -> bool:
    if a == b:
        return True
    fa, fb = float(a), float(b)
    if math.isnan(fa) or math.isnan(fb):
        return math.isnan(fa) and math.isnan(fb)
    if rel <= 0 and abs_ <= 0:
        return False  # exact policy: repr must match (distinguishes -0.0 from 0.0)
    return math.isclose(fa, fb, rel_tol=rel, abs_tol=abs_)


def compare(a: Any, b: Any, float_rel: float = 0.0, float_abs: float = 0.0) -> Optional[str]:
    """None when equal (values, exact types, aliasing structure); otherwise a short reason."""
    if float_rel <= 0 and float_abs <= 0:
        return None if a == b else _first_difference(a, b)
    return _cmp(a, b, float_rel, float_abs, "$")


def _first_difference(a: Any, b: Any) -> str:
    return _cmp(a, b, 0.0, 0.0, "$") or "encodings differ"


def _cmp(a: Any, b: Any, rel: float, abs_: float, path: str) -> Optional[str]:
    if not (isinstance(a, list) and isinstance(b, list) and a and b):
        return None if a == b else f"{path}: malformed"
    ta, tb = a[0], b[0]
    if ta != tb:
        if "r" in (ta, tb):
            return f"{path}: aliasing differs (shared object vs distinct object)"
        return f"{path}: type {_tag_name(ta)} vs {_tag_name(tb)}"
    if ta == "f":
        return None if _float_eq(a[1], b[1], rel, abs_) else f"{path}: float {a[1]} vs {b[1]}"
    if ta == "c":
        ok = _float_eq(a[1], b[1], rel, abs_) and _float_eq(a[2], b[2], rel, abs_)
        return None if ok else f"{path}: complex differs"
    if ta in ("N", "b", "i", "s", "y"):
        return None if a == b else f"{path}: value differs"
    if ta == "r":
        return None if a[1] == b[1] else f"{path}: aliasing differs"
    if ta in ("L", "S", "D", "B") and a[1] != b[1]:
        return f"{path}: aliasing differs"
    if ta == "B":
        return None if a[2] == b[2] else f"{path}: bytearray differs"
    ea = a[1] if ta in ("T", "F") else a[2]
    eb = b[1] if ta in ("T", "F") else b[2]
    if len(ea) != len(eb):
        return f"{path}: length {len(ea)} vs {len(eb)}"
    for i, (x, y) in enumerate(zip(ea, eb)):
        if ta == "D":
            r = _cmp(x[0], y[0], rel, abs_, f"{path}.key{i}") or _cmp(x[1], y[1], rel, abs_, f"{path}[{i}]")
        else:
            r = _cmp(x, y, rel, abs_, f"{path}[{i}]")
        if r:
            return r
    return None


_TAGS = {"N": "None", "b": "bool", "i": "int", "f": "float", "c": "complex", "s": "str", "y": "bytes", "T": "tuple",
         "F": "frozenset", "L": "list", "D": "dict", "S": "set", "B": "bytearray", "r": "alias"}


def _tag_name(t: str) -> str:
    return _TAGS.get(t, t)


def preview(enc: Any, limit: int = 120) -> str:
    """Short human-readable rendering of an encoding (for public feedback lines)."""
    try:
        s = repr(decode(enc))
    except CodecError:
        s = "<undecodable>"
    return s if len(s) <= limit else s[: limit - 3] + "..."
