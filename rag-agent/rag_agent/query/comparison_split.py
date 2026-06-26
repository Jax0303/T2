# SPDX-License-Identifier: MIT
"""Named-pair decomposition for comparison / ratio queries (row- and column-axis).

The residual header-path failures on both axes share a structure: the query compares
two *named* entities ("aboriginal **vs** non-aboriginal", "difference **between** A
**and** B", "how many times A **than** B"), and a single ranked match catches only
one of the two. We split such a query into its two entity sub-spans so each can be
resolved independently and the two scopes unioned — the same fix serves the row
"named-pair" (cross-parent) and the column two-column-comparison cases.

LLM-free, pattern-based. ``split_comparison(q)`` returns the list of entity spans
(``[q]`` when no comparison structure is found).
"""
from __future__ import annotations

import re
from typing import List

# comparison/ratio aggregation types whose operands are two named entities
COMPARISON_AGG = {"diff", "div", "range", "opposite"}

# split connectives, tried in priority order; each captures the two sides
_PATTERNS = [
    re.compile(r"\bbetween\s+(?P<a>.+?)\s+and\s+(?P<b>.+?)(?:\?|$)", re.I),
    re.compile(r"(?P<a>.+?)\s+(?:vs\.?|versus)\s+(?P<b>.+?)(?:\?|$)", re.I),
    re.compile(r"(?P<a>.+?)\s+compared\s+(?:to|with)\s+(?P<b>.+?)(?:\?|$)", re.I),
    re.compile(r"(?P<a>.+?)\s+(?:more|less|higher|lower|greater)\s+.*?\bthan\s+(?P<b>.+?)(?:\?|$)", re.I),
    re.compile(r"(?P<a>.+?)\s+(?:for every|per|to)\s+(?P<b>.+?)(?:\?|$)", re.I),
]

_RATIO_CUE = re.compile(r"\b(times|ratio|relationship|for every|per\b|compared)\b", re.I)


def is_comparison(question: str, aggregation: str | None = None) -> bool:
    """True if the query compares/relates two entities (by agg type or cue words)."""
    if aggregation and aggregation in COMPARISON_AGG:
        return True
    q = question or ""
    return bool(_RATIO_CUE.search(q) or re.search(r"\b(vs\.?|versus|between)\b", q, re.I))


def _clean(span: str) -> str:
    span = span.strip(" ,.?")
    # drop a leading interrogative/aux clause so the entity phrase remains
    span = re.sub(r"^(what|how|the|is|was|are|were|did|do|of)\b\s*", "", span, flags=re.I)
    return span.strip()


def split_comparison(question: str) -> List[str]:
    """Split a comparison query into its two entity sub-spans, else ``[question]``."""
    if not question:
        return [question]
    for pat in _PATTERNS:
        m = pat.search(question)
        if m:
            a, b = _clean(m.group("a")), _clean(m.group("b"))
            if a and b and a.lower() != b.lower():
                return [a, b]
    return [question]
