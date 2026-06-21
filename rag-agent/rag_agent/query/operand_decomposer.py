# SPDX-License-Identifier: MIT
"""Operand decomposition + header-path matching (the decomposition ceiling).

A question is decomposed into the *operands* it needs — each a header path the
answer cell sits at — and every candidate header path in the gold table is scored
against the question. ``header_path_match_accuracy`` is whether the gold
operands' header paths are ranked into the top slots: the **ceiling** on
operand-targeted retrieval, since a path you cannot name+match you cannot
retrieve.

Three matchers, mirroring the spec:
  * ``fuzzy``     — token overlap + sequence ratio (no model, deterministic).
  * ``embedding`` — cosine of a sentence embedder over the HPIR-expanded query
    vs each header-path string (needs an embedder).
  * ``hybrid``    — weighted sum ``w*fuzzy + (1-w)*embedding``.

Builds on ``header_path_resolver`` (term extraction / retrieval expansion); kept
benchmark-agnostic by operating on :class:`~rag_agent.bench.schema.BenchTable`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .header_path_resolver import extract_target_terms, expand_for_retrieval
from ..bench.schema import BenchTable, GoldOperand

MATCHERS = ("fuzzy", "embedding", "hybrid")


@dataclass
class Operand:
    """A decomposed operand: the header path it targets + its value type."""

    header_path: List[str]
    value_type: str = "number"
    score: float = 0.0

    def path_str(self) -> str:
        return " > ".join(self.header_path)


# ---------------------------------------------------------------------------
# Candidate header paths over a table
# ---------------------------------------------------------------------------

def candidate_paths(table: BenchTable) -> List[Tuple[str, List[int], List[int]]]:
    """Distinct full header paths in ``table`` with the cells each one covers.

    Returns ``(path_str, rows, cols)`` so a matched path maps straight back to
    the cells operand-targeted retrieval must surface.
    """
    by_path: Dict[str, Tuple[List[int], List[int]]] = {}
    for r in range(table.n_rows):
        for c in range(table.n_cols):
            path = table.full_path(r, c)
            if not path:
                continue
            key = " > ".join(path)
            rows, cols = by_path.setdefault(key, ([], []))
            if r not in rows:
                rows.append(r)
            if c not in cols:
                cols.append(c)
    return [(k, v[0], v[1]) for k, v in by_path.items()]


# ---------------------------------------------------------------------------
# Matchers
# ---------------------------------------------------------------------------

def _tokens(s: str) -> set:
    return {t for t in "".join(ch if ch.isalnum() else " " for ch in s.lower()).split() if t}


def fuzzy_score(query_terms: Sequence[str], path_str: str) -> float:
    """Token overlap (Jaccard-ish) blended with a sequence-similarity ratio."""
    q = set(t.lower() for t in query_terms)
    p = _tokens(path_str)
    if not q or not p:
        return 0.0
    overlap = len(q & p) / len(p)                 # how much of the path the query covers
    ratio = SequenceMatcher(None, " ".join(sorted(q)), " ".join(sorted(p))).ratio()
    return 0.7 * overlap + 0.3 * ratio


class Embedder:
    """Lazy sentence-transformer wrapper for the embedding matcher."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str]):
        import numpy as np  # noqa: F401
        return self.model.encode(list(texts), normalize_embeddings=True,
                                 show_progress_bar=False)


def _embedding_scores(expanded_query: str, path_strs: Sequence[str], embedder: Embedder) -> List[float]:
    import numpy as np
    vecs = embedder.encode([expanded_query] + list(path_strs))
    q = vecs[0]
    return [float(np.dot(q, vecs[i + 1])) for i in range(len(path_strs))]


# ---------------------------------------------------------------------------
# Ranking + ceiling metric
# ---------------------------------------------------------------------------

def rank_paths(
    query: str,
    table: BenchTable,
    matcher: str = "fuzzy",
    embedder: Optional[Embedder] = None,
    hybrid_w: float = 0.5,
) -> List[Tuple[str, float, List[int], List[int]]]:
    """Score every candidate header path against ``query``; return ranked desc."""
    if matcher not in MATCHERS:
        raise ValueError(f"unknown matcher {matcher!r}; expected one of {MATCHERS}")
    cands = candidate_paths(table)
    if not cands:
        return []
    terms = extract_target_terms(query)
    path_strs = [c[0] for c in cands]

    fuzzy = [fuzzy_score(terms, ps) for ps in path_strs]
    if matcher == "fuzzy":
        scores = fuzzy
    else:
        if embedder is None:
            raise ValueError(f"matcher {matcher!r} needs an embedder")
        emb = _embedding_scores(expand_for_retrieval(query), path_strs, embedder)
        if matcher == "embedding":
            scores = emb
        else:
            scores = [hybrid_w * f + (1 - hybrid_w) * e for f, e in zip(fuzzy, emb)]

    ranked = sorted(
        ((cands[i][0], scores[i], cands[i][1], cands[i][2]) for i in range(len(cands))),
        key=lambda t: -t[1],
    )
    return ranked


def decompose(
    query: str,
    table: BenchTable,
    top_k: int = 4,
    matcher: str = "fuzzy",
    embedder: Optional[Embedder] = None,
) -> List[Operand]:
    """Top-``k`` operands (header paths) the query is predicted to need."""
    ranked = rank_paths(query, table, matcher, embedder)
    return [Operand(header_path=ps.split(" > "), score=sc) for ps, sc, _, _ in ranked[:top_k]]


def header_path_match_accuracy(
    query: str,
    table: BenchTable,
    gold_operands: Sequence[GoldOperand],
    matcher: str = "fuzzy",
    embedder: Optional[Embedder] = None,
    k: Optional[int] = None,
) -> Optional[float]:
    """Fraction of distinct gold header paths ranked into the top slots.

    ``k`` defaults to the number of distinct gold paths (the natural budget): a
    perfect decomposer puts exactly the gold paths on top → 1.0. Returns ``None``
    when there are no gold paths (excluded from any average).
    """
    gold_paths = {op.path_str() for op in gold_operands if op.header_path}
    if not gold_paths:
        return None
    budget = k if k is not None else len(gold_paths)
    ranked = rank_paths(query, table, matcher, embedder)
    topk = {ps for ps, _, _, _ in ranked[:budget]}
    return len(gold_paths & topk) / len(gold_paths)
