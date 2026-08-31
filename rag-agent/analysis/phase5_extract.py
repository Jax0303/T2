#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Phase 5 -- the extract condition: the reader names the operands and the
operator, python does the arithmetic.

Operator set is closed over what HiTab's ``aggregation`` field actually holds
on the arithmetic population (see phase5_taskB.py's inventory): div, sum,
opposite, diff, average, range. No count/counta -- they do not occur.
"""
from __future__ import annotations

import json
import re

OPS = ("sum", "diff", "div", "average", "range", "opposite")

SYS_EXTRACT = (
    "Extract the numbers needed and the operation. "
    'Output only JSON: {"values": [<numbers>], "op": "<one of '
    'sum|diff|div|average|range|opposite>"}. No explanation.')

# sum: add all. diff: values[0]-values[1]. div: values[0]/values[1].
# average: mean. range: max-min. opposite: -values[0].
_JSON = re.compile(r"\{.*?\}", re.S)


def compute(values, op):
    v = [float(x) for x in values]
    if op == "sum":
        return sum(v)
    if op == "average":
        return sum(v) / len(v)
    if op == "range":
        return max(v) - min(v)
    if op == "opposite":
        return -v[0]
    if op == "diff":
        return v[0] - v[1]
    if op == "div":
        return v[0] / v[1]
    raise ValueError(f"unknown op {op!r}")


def parse(raw):
    """-> (value_str, op, reason). value_str is '' when parsing failed."""
    m = _JSON.search(raw or "")
    if not m:
        return "", None, "no_json_object"
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return "", None, f"json_decode: {e.msg}"
    if not isinstance(d, dict) or "values" not in d or "op" not in d:
        return "", None, "missing_key"
    op = str(d["op"]).strip().lower()
    if op not in OPS:
        return "", op, "unknown_op"
    vals = d["values"]
    if not isinstance(vals, list) or not vals:
        return "", op, "values_not_list"
    try:
        vals = [float(str(x).replace(",", "").replace("%", "").replace("$", ""))
                for x in vals]
    except (TypeError, ValueError):
        return "", op, "value_not_numeric"
    need = {"diff": 2, "div": 2, "opposite": 1}.get(op)
    if need and len(vals) < need:
        return "", op, f"needs_{need}_values"
    if op == "range" and len(vals) < 2:
        return "", op, "needs_2_values"
    try:
        out = compute(vals, op)
    except ZeroDivisionError:
        return "", op, "zero_division"
    s = f"{out:.6f}".rstrip("0").rstrip(".") if out != int(out) else str(int(out))
    return s, op, "ok"
