"""Pull a generated value back onto the cell the context actually holds.

The reader writes `51.8` where the cell says `0.518`, `47.0` where it says `53.0`,
`primera a` where it says `categoria primera a`. In every one of those the answer
is sitting in the context and the reader rewrote it on the way out. This maps the
written value back to the stored one -- using ONLY the prediction and the retrieved
lines, never the gold answer, so it is a pipeline stage and not a scorer change.

A snap happens only when exactly one context cell explains the prediction. Ties
and misses are left alone: a wrong answer stays wrong rather than becoming a
differently wrong one.
"""
from __future__ import annotations
import re

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")

#: Pre-registered transform families (PREREG-2026-09-10-value-snap.md). A cell
#: value v "explains" a written number p when some f here maps v to p.
_T = {
    "x100": lambda v: v * 100,
    "/100": lambda v: v / 100,
    "x1000": lambda v: v * 1000,
    "/1000": lambda v: v / 1000,
    "100-x": lambda v: 100 - v,
    "1-x": lambda v: 1 - v,
    "neg": lambda v: -v,
}
SETS = {
    "S1": ("x100", "/100"),
    "S2": ("x100", "/100", "x1000", "/1000", "neg"),
    "S3": ("x100", "/100", "x1000", "/1000", "neg", "100-x", "1-x"),
}
#: S3 also enables the substring rule for non-numeric predictions.
_STRING_SET = "S3"


def _f(s):
    try:
        return float(str(s).replace(",", ""))
    except ValueError:
        return None


def _norm(s):
    return " ".join(str(s).strip().lower().rstrip(".").split())


def _close(a, b):
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def snap(pred, cell_values, rule="S3"):
    """(new prediction, audit). `cell_values` are the stored values of the context."""
    pred = str(pred)
    vals = [v for v in cell_values if v is not None]
    numeric = [(v, _f(v)) for v in vals]
    numeric = [(v, x) for v, x in numeric if x is not None]

    # 0. already a cell value verbatim -> never move it
    if any(_norm(pred) == _norm(v) for v in vals):
        return pred, {"snapped": False, "reason": "already_a_cell_value"}

    found = _NUM.findall(pred)
    if not found:
        if rule != _STRING_SET:
            return pred, {"snapped": False, "reason": "no_number"}
        p = _norm(pred)
        if not p:
            return pred, {"snapped": False, "reason": "empty"}
        hits = [v for v in vals if p and (p in _norm(v) or _norm(v) in p) and _norm(v)]
        uniq = {_norm(v) for v in hits}
        if len(uniq) != 1:
            return pred, {"snapped": False,
                          "reason": "string_ambiguous" if uniq else "string_no_match"}
        return hits[0], {"snapped": True, "kind": "substring", "to": hits[0]}

    out, used, moved = pred, [], 0
    for tok in found:
        p = _f(tok)
        if p is None:
            continue
        if any(_close(p, x) for _, x in numeric):
            continue                                  # this number is already a cell
        cands = {}
        for v, x in numeric:
            for name in SETS[rule]:
                if _close(_T[name](x), p):
                    cands.setdefault(_norm(v), (v, name))
        if len(cands) != 1:
            continue                                  # 0 or >1 explanations: leave it
        v, name = next(iter(cands.values()))
        out = out.replace(tok, v, 1)
        used.append({"from": tok, "to": v, "via": name})
        moved += 1
    if not moved:
        return pred, {"snapped": False, "reason": "no_unique_transform"}
    return out, {"snapped": True, "kind": "numeric", "moves": used}
