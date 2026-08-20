# SPDX-License-Identifier: MIT
"""Which decomposition matcher to use, per benchmark — locked to measured evidence.

The query decomposer (:mod:`rag_agent.query.operand_decomposer`) ranks a table's
header paths against the question with one of three matchers — ``fuzzy`` (lexical
token overlap), ``embedding`` (sentence-embedding cosine), ``hybrid`` (weighted
sum). Which one is best is **not** a free choice: it was measured as the
decomposition ceiling ``header_path_match_accuracy`` and committed.

Source: ``results/operand_rag/<bench>/summary.json``, key ``ceiling`` (seed 42).
These are the *ceiling* on operand-targeted retrieval — the fraction of questions
whose gold header paths the matcher ranks into the top slots — so a matcher that
loses here caps everything downstream.

| bench   | split      | n_op | fuzzy  | embedding | hybrid | best      |
|---------|------------|------|--------|-----------|--------|-----------|
| hitab   | dev        | 241  | .3029  | **.4855** | .4090  | embedding |
| finqa   | validation | 238  | .4943  | **.5716** | .5437  | embedding |
| wikisql | validation | 300  | .7413  | .7112     | **.7874** | hybrid |

Reading: on the two *hierarchical* corpora the embedding matcher is decisively
ahead (+18.3pp HiTab, +7.7pp FinQA over fuzzy); on *flat* WikiSQL, where headers
are short exact strings, lexical signal helps and hybrid wins while embedding
alone falls below fuzzy. So there is no single winner — the matcher is a property
of the corpus, which is why it lives here as a sourced policy rather than a
hard-coded default buried in the decomposer.

This module deliberately imports nothing heavy (no torch, no embedder), so the
policy — and the test that locks it to the committed numbers
(``tests/test_decomposer_matcher_policy.py``) — runs without the experiment venv.
"""
from __future__ import annotations

from typing import Dict

# Committed decomposition-ceiling accuracy per (bench, matcher).
# DO NOT edit a value without re-running scripts/operand_rag_eval.py for that
# bench and pointing at the new results/operand_rag/<bench>/summary.json. The
# repo rule is: every number carries the file it came from.
BENCH_CEILING: Dict[str, Dict[str, float]] = {
    "hitab":   {"fuzzy": 0.3029, "embedding": 0.4855, "hybrid": 0.4090},
    "finqa":   {"fuzzy": 0.4943, "embedding": 0.5716, "hybrid": 0.5437},
    "wikisql": {"fuzzy": 0.7413, "embedding": 0.7112, "hybrid": 0.7874},
}

# Prior for a corpus with no committed ceiling of its own. Both MultiHiertt and
# RealHiTBench are hierarchical like HiTab/FinQA, where embedding won, so the
# hierarchical prior is ``embedding`` — but it is a *prior*, not a measurement,
# and must be replaced by a real ceiling run before any number is quoted for it.
_HIERARCHICAL_PRIOR = "embedding"
_UNMEASURED = {"multihiertt", "realhitbench"}


def best_matcher(bench: str) -> str:
    """The matcher with the highest committed decomposition ceiling on ``bench``.

    Falls back to the hierarchical prior (``embedding``) for a corpus that has no
    committed ceiling yet; the caller should treat that as provisional.
    """
    key = (bench or "").strip().lower()
    ceil = BENCH_CEILING.get(key)
    if ceil:
        return max(ceil, key=ceil.get)
    return _HIERARCHICAL_PRIOR


def is_measured(bench: str) -> bool:
    """True iff ``bench`` has a committed ceiling backing its ``best_matcher``."""
    return (bench or "").strip().lower() in BENCH_CEILING
