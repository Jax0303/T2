# SPDX-License-Identifier: MIT
"""Operand-targeted retrieval over serialized row-chunks + ``operand_recall@k``.

Each table is serialized to row chunks (S2 by default); a hybrid retriever scores
chunks with BM25 + dense cosine fused by reciprocal-rank fusion (RRF, k=60 — the
Cormack et al. 2009 default, no tuning). Dense uses an in-memory normalized
embedding matrix (table chunk counts are small, so no faiss/Chroma is needed).

Retrieval modes:
  * ``operand``  — decompose the query into operand header paths, search each
    one (top-k), union the hits. The method under test.
  * ``oracle``   — same but queries are the *gold* operand header paths: the
    retrieval ceiling given perfect decomposition.
  * ``plain``    — a single search on the raw question (the BM25/dense baseline).

``operand_recall@k`` is the fraction of gold operands whose covering chunk is in
the retrieved set — the completeness signal the pipeline is built around.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ..bench.schema import BenchTable, GoldOperand, Chunk
from ..serialize import serialize_table, S2
from ..query.operand_decomposer import decompose, Operand, Embedder

_TOK_RE = re.compile(r"[a-z0-9]+")
RRF_K = 60


def _tok(text: str) -> List[str]:
    return _TOK_RE.findall(text.lower())


class HybridRetriever:
    """BM25 + (optional) dense retriever over a fixed list of chunks, RRF-fused."""

    def __init__(self, chunks: Sequence[Chunk], embedder: Optional[Embedder] = None,
                 rrf_k: int = RRF_K):
        from rank_bm25 import BM25Okapi
        self.chunks = list(chunks)
        self.rrf_k = rrf_k
        self._bm25 = BM25Okapi([_tok(c.text) for c in self.chunks] or [[""]])
        self._emb = None
        if embedder is not None and self.chunks:
            self._emb = embedder.encode([c.text for c in self.chunks])
            self._embedder = embedder

    def _rank(self, scores) -> List[int]:
        return sorted(range(len(scores)), key=lambda i: -scores[i])

    def search(self, query: str, k: int = 5, use_dense: bool = True) -> List[int]:
        """Return chunk indices for ``query``, best first (≤ k)."""
        if not self.chunks:
            return []
        bm25_rank = self._rank(self._bm25.get_scores(_tok(query)))
        if not (use_dense and self._emb is not None):
            return bm25_rank[:k]
        import numpy as np
        qv = self._embedder.encode([query])[0]
        dense_rank = self._rank(self._emb @ qv)
        # reciprocal-rank fusion
        fused = {}
        for rank, i in enumerate(bm25_rank):
            fused[i] = fused.get(i, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, i in enumerate(dense_rank):
            fused[i] = fused.get(i, 0.0) + 1.0 / (self.rrf_k + rank)
        return sorted(fused, key=lambda i: -fused[i])[:k]


@dataclass
class RetrievalResult:
    retrieved: List[Chunk]
    operands: List[Operand]          # the operands that drove retrieval
    mode: str


def _gather(retriever: HybridRetriever, queries: Sequence[str], k: int,
            use_dense: bool) -> List[Chunk]:
    seen, out = set(), []
    for q in queries:
        for i in retriever.search(q, k, use_dense):
            if i not in seen:
                seen.add(i)
                out.append(retriever.chunks[i])
    return out


def retrieve(
    query: str,
    table: BenchTable,
    gold_operands: Optional[Sequence[GoldOperand]] = None,
    mode: str = "operand",
    k: int = 5,
    scheme: str = S2,
    matcher: str = "fuzzy",
    embedder: Optional[Embedder] = None,
    top_operands: int = 4,
    retriever: Optional[HybridRetriever] = None,
) -> RetrievalResult:
    """Run operand-targeted (or oracle/plain) retrieval over ``table``'s chunks."""
    if retriever is None:
        retriever = HybridRetriever(serialize_table(table, scheme), embedder)
    use_dense = embedder is not None

    if mode == "plain":
        chunks = _gather(retriever, [query], k, use_dense)
        return RetrievalResult(chunks, [], mode)

    if mode == "oracle":
        ops = [Operand(header_path=o.header_path, value_type=o.value_type)
               for o in (gold_operands or [])]
    else:  # operand
        ops = decompose(query, table, top_k=top_operands, matcher=matcher, embedder=embedder)

    queries = [o.path_str() for o in ops] or [query]
    chunks = _gather(retriever, queries, k, use_dense)
    return RetrievalResult(chunks, ops, mode)


def operand_recall(retrieved: Sequence[Chunk], gold_operands: Sequence[GoldOperand]) -> Optional[float]:
    """Fraction of gold operands whose covering chunk is in ``retrieved``."""
    if not gold_operands:
        return None
    covered = sum(
        1 for op in gold_operands
        if any(ch.covers(op.row, op.col) for ch in retrieved)
    )
    return covered / len(gold_operands)
