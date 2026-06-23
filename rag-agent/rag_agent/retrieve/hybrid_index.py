"""Hybrid BM25 + dense index over serialized :class:`Chunk` objects.

This is the index the operand-targeted retriever queries. It holds the S2 chunks
of a table (or a small candidate set) and scores a query with a weighted
combination of a lexical (BM25) and a dense (cosine) signal. Scores from each
backend are min-max normalized per query before combining, so the ``alpha``
weight is meaningful regardless of the backends' raw score scales.

FAISS is used for the dense search when available; otherwise a NumPy matrix
product is used (table-scale corpora make this a non-issue). The encoder is held
once and used for both chunks and queries, enforcing embedding consistency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..serialization.base import Chunk
from .encoders import Encoder, _tokenize, default_encoder


def _minmax(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    bm25: float
    dense: float


class HybridIndex:
    """Weighted BM25 + dense retriever over a fixed chunk set.

    Parameters
    ----------
    chunks:
        The corpus. Order is preserved; results reference these objects.
    encoder:
        Dense encoder; defaults to :func:`default_encoder` (real model if its
        deps import, else the hashing fallback).
    alpha:
        Final score = ``alpha * dense + (1 - alpha) * bm25`` after per-query
        min-max normalization. ``alpha=0`` is BM25-only, ``alpha=1`` dense-only.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        encoder: Optional[Encoder] = None,
        alpha: float = 0.5,
    ) -> None:
        from rank_bm25 import BM25Okapi

        self.chunks: List[Chunk] = list(chunks)
        self.alpha = alpha
        self.encoder = encoder or default_encoder()

        texts = [c.text for c in self.chunks]
        self._tokens = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(self._tokens) if self._tokens else None
        self._emb = (
            self.encoder.encode(texts) if texts else np.zeros((0, 1), dtype=np.float32)
        )
        self._faiss = self._try_build_faiss(self._emb)

    @staticmethod
    def _try_build_faiss(emb: np.ndarray):
        if emb.shape[0] == 0:
            return None
        try:
            import faiss

            index = faiss.IndexFlatIP(emb.shape[1])
            index.add(np.ascontiguousarray(emb))
            return index
        except Exception:
            return None

    def _dense_scores(self, query: str) -> np.ndarray:
        if self._emb.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        q = self.encoder.encode([query])[0].astype(np.float32)
        # cosine == dot product since rows are L2-normalized
        return self._emb @ q

    def _bm25_scores(self, query: str) -> np.ndarray:
        if self._bm25 is None:
            return np.zeros(len(self.chunks), dtype=np.float32)
        return np.asarray(self._bm25.get_scores(_tokenize(query)), dtype=np.float32)

    def search(self, query: str, k: int = 5) -> List[RetrievedChunk]:
        if not self.chunks:
            return []
        bm = self._bm25_scores(query)
        dn = self._dense_scores(query)
        combined = self.alpha * _minmax(dn) + (1.0 - self.alpha) * _minmax(bm)
        k = min(k, len(self.chunks))
        # argpartition for top-k, then sort that slice
        top = np.argpartition(-combined, k - 1)[:k]
        top = top[np.argsort(-combined[top])]
        return [
            RetrievedChunk(
                chunk=self.chunks[i],
                score=float(combined[i]),
                bm25=float(bm[i]),
                dense=float(dn[i]),
            )
            for i in top
        ]
