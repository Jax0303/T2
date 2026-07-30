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


def _rrf(scores: np.ndarray, rrf_k: int) -> np.ndarray:
    """Reciprocal-rank contribution ``1 / (rrf_k + rank)``, rank 1-based.

    Rank fusion is scale-free: it reads only the ORDER each backend produces, so
    a backend whose raw scores are bunched together cannot dominate the sum the
    way it can under min-max normalization (which stretches any spread to [0,1]).
    """
    if scores.size == 0:
        return scores
    ranks = np.empty(scores.size, dtype=np.int64)
    ranks[np.argsort(-scores, kind="stable")] = np.arange(1, scores.size + 1)
    return 1.0 / (rrf_k + ranks)


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
        Weight on the dense signal. Under ``fusion="weighted"`` the final score
        is ``alpha * dense + (1 - alpha) * bm25`` after per-query min-max
        normalization; under ``fusion="rrf"`` it weights the two reciprocal-rank
        contributions instead. ``alpha=0`` is BM25-only, ``alpha=1`` dense-only.
    fusion:
        How the lexical and dense signals are combined — the two embodiments of
        the similarity-search component:

        * ``"weighted"`` (default) — per-query min-max normalization then
          weighted sum. Uses score magnitudes, so a confident backend can carry
          a query outright.
        * ``"rrf"`` — reciprocal rank fusion (Cormack et al. 2009). Uses only
          each backend's ordering, which makes it robust when the two score
          scales are not comparable.

        Default stays ``"weighted"`` so existing measurements are unaffected.
    rrf_k:
        Rank-fusion damping constant; ignored unless ``fusion="rrf"``. 60 is the
        value from the original paper and the one the repo's scripts use.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        encoder: Optional[Encoder] = None,
        alpha: float = 0.5,
        fusion: str = "weighted",
        rrf_k: int = 60,
    ) -> None:
        from rank_bm25 import BM25Okapi

        if fusion not in ("weighted", "rrf"):
            raise ValueError(f"fusion must be 'weighted' or 'rrf', got {fusion!r}")
        self.chunks: List[Chunk] = list(chunks)
        self.alpha = alpha
        self.fusion = fusion
        self.rrf_k = rrf_k
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
        if self.fusion == "rrf":
            combined = (self.alpha * _rrf(dn, self.rrf_k)
                        + (1.0 - self.alpha) * _rrf(bm, self.rrf_k))
        else:
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
