"""Rule-based query classifier.

Outputs a coarse intent label aligned to the HiTab paper's appendix
taxonomy (the same categories the existing hard-query eval uses):

  simple_lookup        — read a single cell directly
  single_arg           — argmax / argmin / max / min on a column
  comparison_or_count  — greater/less/opposite/counta
  arithmetic_agg       — sum / diff / div / average / range
  multi_op_formula     — ≥2 arithmetic ops
  reasoning_only       — no table needed (rare in HiTab; supported for skip)

The classifier is INTENTIONALLY simple keyword + regex matching: this is
the same family of heuristics used to derive HiTab's gold supervision
labels from `aggregation` and `answer_formulas`, just applied at the
question level (we don't have the gold tags at inference time).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class QueryType(str, Enum):
    REASONING_ONLY = "reasoning_only"
    SIMPLE_LOOKUP = "simple_lookup"
    SINGLE_ARG = "single_arg"
    COMPARISON_OR_COUNT = "comparison_or_count"
    ARITHMETIC_AGG = "arithmetic_agg"
    MULTI_OP_FORMULA = "multi_op_formula"


@dataclass
class QueryIntent:
    qtype: QueryType
    needs_table: bool
    needs_symbolic: bool      # → cell extraction + pandas eval path
    keywords: list            # signal terms that triggered the label (for trace)


_ARG_PAT = re.compile(
    r"\b(highest|lowest|largest|smallest|maximum|minimum|most|least|biggest|"
    r"top|bottom|best|worst|peak|higher|lower|more|fewer|greater)\b", re.IGNORECASE
)
_PAIR_PAT = re.compile(r"\b(which|who|or)\b.*\b(or)\b", re.IGNORECASE)
# Entity-answer cue: questions starting with "who/which X" usually want a name,
# not a number — even when an arithmetic noun appears later. Catches the
# "who had a higher proportion of ..." class that was being misrouted.
_ENTITY_QUESTION_PAT = re.compile(
    r"^\s*(?:who|which|what|where|in what|in which)\b", re.IGNORECASE
)
_CMP_PAT = re.compile(
    r"\b(greater|less|more|fewer|higher|lower|exceed|exceeds|exceeded|"
    r"bigger|smaller|above|below|than|compared|opposite|sign)\b", re.IGNORECASE
)
_COUNT_PAT = re.compile(r"\b(how many|count|number of)\b", re.IGNORECASE)
_ARITH_PAT = re.compile(
    r"\b(sum of|combined|altogether|together|average|mean|"
    r"difference|differ|differen|gap|change|increase|decrease|grew|drop|dropped|"
    r"ratio|fraction|proportion|percentage|percent|per cent|share of|out of|"
    r"divided by|multiplied|product|range|spread|"
    r"how much (?:more|less|higher|lower))\b",
    re.IGNORECASE,
)
# "total" alone is ambiguous — only counts as aggregation when NOT modifying
# a header (i.e. not "total row/column/of <header>").
_TOTAL_AGG_PAT = re.compile(
    r"\btotal\b(?!\s+(?:row|column|col|cell|of\s+the))", re.IGNORECASE
)
# Two or more separate arithmetic-trigger words → multi-op
_ARITH_TRIGGERS = re.compile(
    r"\b(sum of|average|mean|difference|change|ratio|percentage|"
    r"divided|times|gap|increase|decrease)\b", re.IGNORECASE
)
# Explicit math symbols are the strongest multi-op signal.
_MATH_SYMBOL_RE = re.compile(r"[+\-*/]")
_MULTI_PAT = re.compile(r"\b(and|plus)\b.*\b(divided|over|minus|less|times)\b", re.IGNORECASE)

# Heuristic: questions with NO numbers / NO table tokens, asking about a
# definition or generic concept, are "reasoning_only". Very rare in HiTab;
# the policy short-circuits retrieval if matched.
_REASONING_PAT = re.compile(
    r"^\s*(what is|what are|define|explain|why does|describe|how does)\b",
    re.IGNORECASE,
)
_TABLE_HINT_PAT = re.compile(
    r"\b(table|row|column|cell|value|figure|chart|data)\b", re.IGNORECASE
)


def classify_query(q: str) -> QueryIntent:
    q = (q or "").strip()
    sig: list = []

    # Strongest multi-op signal: ≥2 distinct math symbols in the question.
    math_syms = _MATH_SYMBOL_RE.findall(q)
    if len(math_syms) >= 2:
        sig.append("math-symbols")
        return QueryIntent(QueryType.MULTI_OP_FORMULA, True, True, sig)

    arith_hits = _ARITH_TRIGGERS.findall(q)
    if len(set(s.lower() for s in arith_hits)) >= 2 or _MULTI_PAT.search(q):
        sig.append("multi-op")
        return QueryIntent(QueryType.MULTI_OP_FORMULA, True, True, sig)

    # Entity-cue questions ("who/which had higher proportion") want a name back,
    # not a computed value — route to single_arg even if an arithmetic noun appears.
    if _ENTITY_QUESTION_PAT.match(q) and (_ARG_PAT.search(q) or _PAIR_PAT.search(q)):
        sig.append("entity-question")
        return QueryIntent(QueryType.SINGLE_ARG, True, False, sig)

    if _ARITH_PAT.search(q) or _TOTAL_AGG_PAT.search(q):
        sig.append("arith")
        return QueryIntent(QueryType.ARITHMETIC_AGG, True, True, sig)

    if _PAIR_PAT.search(q) or _ARG_PAT.search(q):
        sig.append("arg")
        return QueryIntent(QueryType.SINGLE_ARG, True, False, sig)

    if _CMP_PAT.search(q) or _COUNT_PAT.search(q):
        sig.append("cmp")
        return QueryIntent(QueryType.COMPARISON_OR_COUNT, True, False, sig)

    if _REASONING_PAT.match(q) and not _TABLE_HINT_PAT.search(q):
        # only treat as reasoning-only if no table-ish hint at all
        sig.append("definition-like")
        return QueryIntent(QueryType.REASONING_ONLY, False, False, sig)

    return QueryIntent(QueryType.SIMPLE_LOOKUP, True, False, sig)
