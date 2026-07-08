"""Query router — decide between alpha (full table dump), beta (sub-table
extraction), and gamma (pandas code generation).

Inference-time router uses ONLY the query text (no formula). The training-time
ground-truth comes from HiTab's ``answer_formulas`` field and is computed by
``gold_route_from_formula`` — used to evaluate router accuracy.
"""
from __future__ import annotations

import re
from typing import Literal, Optional


Route = Literal["alpha", "beta", "gamma"]


# ---------------------------------------------------------------------------
# Ground-truth derivation from HiTab annotation
# ---------------------------------------------------------------------------

def _formula_uses_arithmetic(formulas) -> bool:
    if not formulas:
        return False
    body = (formulas[0][1:] if formulas[0].startswith("=") else formulas[0]).upper()
    if any(op in body for op in ("+", "-", "*", "/")):
        return True
    if any(fn in body for fn in ("SUM", "AVERAGE", "AVG", "COUNT", "MAX", "MIN", "IF")):
        return True
    if ":" in body:  # range like B2:B10
        return True
    return False


def _formula_is_list(formulas) -> bool:
    if not formulas:
        return False
    f = formulas[0]
    if f.startswith("="):
        return False
    # Cases like 'A23,A21,A18,...' — non-equal pure cell list
    return "," in f


def gold_route_from_formula(formulas, answer) -> Route:
    """Derive the oracle route given HiTab's answer_formulas + answer.

    - Multi-cell arithmetic / SUM / AVG / ranges       -> gamma
    - List of cell references (no '=', comma-separated) -> beta (list answer)
    - Single-cell lookup (=A1)                          -> beta (or alpha if free-form)
    - Missing formula but multi-element answer          -> beta
    - Otherwise                                         -> beta
    """
    if _formula_uses_arithmetic(formulas):
        return "gamma"
    if _formula_is_list(formulas):
        return "beta"
    if formulas:
        return "beta"  # single-cell lookup
    # No formula at all (rare) — fall back on answer shape.
    if isinstance(answer, list) and len(answer) > 1:
        return "beta"
    return "beta"


# ---------------------------------------------------------------------------
# Inference-time router (uses query text only)
# ---------------------------------------------------------------------------

# STRICT arithmetic patterns. Most HiTab tables already contain pre-computed
# percent / share / inflation-adjusted columns, so a bare "percent" or "after
# adjusting" doesn't imply arithmetic is needed. Route to gamma only when the
# query *names* an explicit multi-cell operation.
_ARITH_PATTERNS = [
    # explicit ratio / multiple between two named things
    r"\bratio\s+of\s+\w+.*\bto\s+\w+",
    r"\bmultiple\s+(of|relationship)\b",
    r"\b(times|fold)\s+(more|less|of|the|as)\b",
    r"\bfor\s+every\s+\w+",                   # "how many X for every Y"
    # differences between two named things
    r"\bdifference\s+(between|of)\s+\w+\s+and\s+\w+",
    # explicit aggregation across multiple items
    r"\btotal\s+(amount|number|sum|count)\s+of\s+all\b",
    r"\bsum\s+of\s+(all\s+)?\w+\s+and\s+\w+",
    r"\baverage\s+of\s+(all\s+)?\w+",
    r"\bmean\s+of\s+\w+",
    r"\baggregate\s+of\b",
    r"\bcombined\s+(total|amount|number)\b",
    # range
    r"\brange\s+of\b",
    # superlative comparisons requiring scan: argmax over many rows
    r"\bwhich\s+\w+\s+(had|has|have|is|was|were)\s+the\s+(largest|smallest|highest|lowest|most|least|biggest|greatest)\b",
    r"\bcompare\s+\w+\s+(to|with|between|and)\b",
]

_LIST_PATTERNS = [
    r"\blist\b",
    r"\btop\s+\w+\b",          # top six / top three
    r"\bwhich\s+(are|were|industries|countries|cities|sectors|groups|places)\b",
    r"\bwhat\s+are\s+the\b",
]

_COMPARE_PATTERNS = [
    r"\bwhich\s+\w+\s+(had|has|have)\s+(the\s+)?(largest|smallest|highest|lowest|most|least|greater|fewer|more)\b",
    r"\bwhich\s+\w+\s+(is|was|were)\s+(the\s+)?(largest|smallest|highest|lowest|most|least|greater|fewer|more)\b",
    r"\b(higher|lower|larger|smaller|greater|fewer|more)\b.*\bthan\b",
]


def _matches_any(text: str, patterns) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def route_query(query: str) -> Route:
    q = query.strip().lower()
    # Comparisons + arithmetic always go to gamma.
    if _matches_any(q, _COMPARE_PATTERNS):
        return "gamma"
    if _matches_any(q, _ARITH_PATTERNS):
        return "gamma"
    # Lists go to beta (single cell-list lookup).
    if _matches_any(q, _LIST_PATTERNS):
        return "beta"
    # Default: gamma. The code path falls back to beta on sandbox failure,
    # and simple lookups become trivial pandas expressions (df.loc[...]).
    return "gamma"


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_router(samples) -> dict:
    """Compute confusion matrix between rule-based router and HiTab gold."""
    from collections import Counter
    confusion = Counter()
    n_correct = 0
    n = 0
    for s in samples:
        gold = gold_route_from_formula(s.get("answer_formulas"), s.get("answer"))
        pred = route_query(s.get("question", ""))
        confusion[(gold, pred)] += 1
        n_correct += int(gold == pred)
        n += 1
    return {
        "n": n,
        "accuracy": n_correct / n if n else 0.0,
        "confusion": {f"gold={g}_pred={p}": v for (g, p), v in confusion.items()},
    }


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    path = sys.argv[1] if len(sys.argv) > 1 else \
        "/mnt/d/hart_data/hitab/HiTab/data/dev_samples.jsonl"
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    res = evaluate_router(samples)
    print(f"Router (rule-only) on {res['n']} samples")
    print(f"  accuracy = {res['accuracy']:.3f}")
    print("  confusion:")
    # Pretty table
    routes = ("alpha", "beta", "gamma")
    print(f"    {'pred->':<8}", " ".join(f"{r:>8s}" for r in routes), "  total")
    for g in routes:
        row = []
        total = 0
        for p in routes:
            v = res["confusion"].get(f"gold={g}_pred={p}", 0)
            row.append(v)
            total += v
        print(f"    gold={g:<5}", " ".join(f"{v:>8d}" for v in row), f"  {total}")
