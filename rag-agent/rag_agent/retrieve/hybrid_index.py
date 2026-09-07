"""Hybrid BM25 + dense index over serialized :class:`Chunk` objects.

This is the index the operand-targeted retriever queries. It holds the S2 chunks
of a table (or a small candidate set) and scores a query with a weighted
combination of a lexical (BM25) and a dense (cosine) signal. Scores from each
backend are min-max normalized per query before combining, so the ``alpha``
weight is meaningful regardless of the backends' raw score scales.

The dense half is a vector index, not an ad-hoc matrix product: cell sentences
are embedded once and searched by inner product from a FAISS ``IndexFlatIP``.
The index is *exact* (it scans every vector), which is the whole reason for
choosing it over an ANN store — approximate search would silently reorder
results and invalidate every retrieval number already on disk. ``vector_backend
="numpy"`` selects the equivalent matmul, kept because it is the reference the
FAISS path is checked against (``tests/test_vector_backend.py``) and because
CPU-only checkouts without faiss must still run.

The index is in-memory and per-run by construction: it is built from the chunks
handed to the constructor and dropped with them, so no run inherits state from
the previous one. :meth:`HybridIndex.close` makes that explicit where a script
builds many indexes in a loop. The encoder is held once and used for both chunks
and queries, enforcing embedding consistency.
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


DENSE_BACKENDS = ("faiss", "numpy")


def resolve_dense_backend(name: str = "auto") -> str:
    """Which vector backend a run would use. ``"auto"`` prefers faiss.

    An explicit ``"faiss"`` lets the ImportError through instead of degrading to
    numpy: a result file that says it used the vector index must not have been
    produced by something else. ``"auto"`` may degrade, but the choice it made is
    recorded (:func:`rag_agent.runenv.run_env` splices it into ``env``), so the
    fallback is never silent the way an unrecorded one would be.
    """
    if name not in DENSE_BACKENDS + ("auto",):
        raise ValueError(f"vector_backend must be one of "
                         f"{DENSE_BACKENDS + ('auto',)}, got {name!r}")
    if name == "numpy":
        return name
    try:
        import faiss  # noqa: F401
    except ImportError:
        if name == "faiss":
            raise
        return "numpy"
    return "faiss"


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
    vector_backend:
        ``"faiss"`` (exact ``IndexFlatIP``), ``"numpy"`` (equivalent matmul), or
        ``"auto"`` — faiss when importable. The two produce the same ranking;
        see :func:`resolve_dense_backend`. The resolved name is on
        ``self.vector_backend``.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        encoder: Optional[Encoder] = None,
        alpha: float = 0.5,
        fusion: str = "weighted",
        rrf_k: int = 60,
        vector_backend: str = "auto",
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
        self.vector_backend = resolve_dense_backend(vector_backend)
        self._index = (self._build_index(self._emb)
                       if self.vector_backend == "faiss" else None)

    @staticmethod
    def _build_index(emb: np.ndarray):
        if emb.shape[0] == 0:
            return None
        import faiss

        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(np.ascontiguousarray(emb, dtype=np.float32))
        return index

    def close(self) -> None:
        """Drop the vector index. Nothing was persisted, so this leaves no state
        for a later run to pick up; call it in loops that build many indexes."""
        self._index = None

    def _dense_scores(self, query: str) -> np.ndarray:
        if self._emb.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        # encode_query, not encode: BGE/E5 mark the QUERY side with the
        # instruction they were trained with, and encoding a question as if it
        # were a passage quietly gives up the asymmetry the model learned.
        # Encoders without the distinction fall back to the plain path.
        enc_q = getattr(self.encoder, "encode_query", self.encoder.encode)
        q = enc_q([query])[0].astype(np.float32)
        # cosine == dot product since rows are L2-normalized
        if self._index is None:
            return self._emb @ q
        # ntotal, not k: weighted fusion min-max-normalizes over the FULL dense
        # score vector, so a truncated search would change the normalizer and
        # with it the ranking. IndexFlatIP is exhaustive, so this is the same
        # arithmetic as the matmul, just executed by the index.
        sims, ids = self._index.search(np.ascontiguousarray(q.reshape(1, -1)),
                                       self._index.ntotal)
        out = np.empty(self._emb.shape[0], dtype=np.float32)
        out[ids[0]] = sims[0]
        return out

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
