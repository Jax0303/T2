# SPDX-License-Identifier: MIT
"""The FAISS vector index must rank identically to the NumPy reference.

Every retrieval number on disk was produced by the matmul path. Routing the
dense stage through an index is only safe because ``IndexFlatIP`` is exhaustive;
if that ever stops holding (a swap to an ANN index, a normalization change),
these tests fail rather than the results quietly moving.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from rag_agent.retrieve.encoders import HashingEncoder  # noqa: E402
from rag_agent.retrieve.hybrid_index import (  # noqa: E402
    HybridIndex,
    resolve_dense_backend,
)
from rag_agent.serialization.base import Chunk  # noqa: E402

faiss = pytest.importorskip("faiss")

WORDS = ["revenue", "cost", "2023", "2024", "total", "north", "south", "units"]


def _chunks(n=400):
    rng = np.random.default_rng(0)
    return [
        Chunk(chunk_id=f"c{i}", text=" ".join(rng.choice(WORDS, size=5)),
              table_id="t1", scheme="S2", row_index=i // 20, col_index=i % 20)
        for i in range(n)
    ]


def _both(chunks, **kw):
    enc = HashingEncoder(dim=256)
    return (HybridIndex(chunks, encoder=enc, vector_backend="faiss", **kw),
            HybridIndex(chunks, encoder=enc, vector_backend="numpy", **kw))


@pytest.mark.parametrize("alpha,fusion", [(0.5, "weighted"), (1.0, "weighted"),
                                          (0.5, "rrf")])
def test_faiss_and_numpy_rank_identically(alpha, fusion):
    chunks = _chunks()
    fx, npx = _both(chunks, alpha=alpha, fusion=fusion)
    assert fx.vector_backend == "faiss" and npx.vector_backend == "numpy"
    for q in ("total revenue 2023", "cost north units", "south"):
        a = [h.chunk.chunk_id for h in fx.search(q, k=25)]
        b = [h.chunk.chunk_id for h in npx.search(q, k=25)]
        assert a == b, f"ranking diverged on {q!r} (alpha={alpha}, {fusion})"


def test_dense_scores_match_the_matmul():
    chunks = _chunks()
    fx, npx = _both(chunks)
    got = fx._dense_scores("total revenue 2023")
    want = npx._dense_scores("total revenue 2023")
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, atol=1e-5)


def test_close_drops_the_index_and_falls_back_cleanly():
    fx, npx = _both(_chunks(50))
    before = fx.search("total", k=5)
    fx.close()
    assert fx._index is None
    # the embeddings outlive the index, so the same query still ranks the same
    assert [h.chunk.chunk_id for h in fx.search("total", k=5)] == \
           [h.chunk.chunk_id for h in before]


def test_empty_corpus_needs_no_index():
    idx = HybridIndex([], encoder=HashingEncoder(dim=256))
    assert idx.search("anything", k=5) == []


def test_explicit_faiss_never_degrades_silently():
    assert resolve_dense_backend("faiss") == "faiss"
    assert resolve_dense_backend("numpy") == "numpy"
    assert resolve_dense_backend("auto") in ("faiss", "numpy")
    with pytest.raises(ValueError, match="vector_backend"):
        resolve_dense_backend("chroma")
