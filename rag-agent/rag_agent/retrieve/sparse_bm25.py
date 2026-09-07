# SPDX-License-Identifier: MIT
"""BM25 as one sparse matrix, so a corpus-wide index fits in memory.

``rank_bm25`` keeps a Python dict of term frequencies per document. At 468,466
cell sentences — every data cell HiTab ships — that is several GB before a
single vector is embedded, and the box this thesis runs on has 7 GB. The same
scoring is a CSR matrix of the per-term document weights and one sparse
column-sum per query, which holds the whole store in ~200 MB and scores a query
in a millisecond.

Same formula as ``BM25Okapi`` (Robertson/Sparck Jones, k1=1.5, b=0.75, the
``rank_bm25`` defaults), so a run under either backend ranks the same way. The
equivalence is checked in ``tests/test_sparse_bm25.py``.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
from scipy.sparse import csr_matrix


class SparseBM25:
    def __init__(self, corpus_tokens: Iterable[Sequence[str]],
                 k1: float = 1.5, b: float = 0.75) -> None:
        # an ITERABLE, not a sequence: the corpus-wide index tokenizes 468k cell
        # sentences, and holding every token list alive at once costs more than
        # the embedding matrix it sits next to. Streamed, only the CSR survives.
        self.k1, self.b = k1, b
        vocab: dict = {}
        indptr, indices, tf, lens = [0], [], [], []
        for toks in corpus_tokens:
            counts: dict = {}
            for t in toks:
                i = vocab.get(t)
                if i is None:
                    i = vocab[t] = len(vocab)
                counts[i] = counts.get(i, 0) + 1
            indices.extend(counts.keys())
            tf.extend(counts.values())
            indptr.append(len(indices))
            lens.append(len(toks))
        n_docs = len(lens)
        lens = np.asarray(lens, dtype=np.float32)
        self.vocab = vocab
        avgdl = float(lens.mean()) if n_docs else 0.0
        tf_a = np.asarray(tf, dtype=np.float32)
        # rank_bm25's BM25Okapi idf, including its floor on negative values
        m = csr_matrix((np.ones_like(tf_a), np.asarray(indices, dtype=np.int32),
                        np.asarray(indptr, dtype=np.int64)),
                       shape=(n_docs, len(vocab)))
        df = np.asarray(m.sum(axis=0)).ravel()
        idf = np.log(n_docs - df + 0.5) - np.log(df + 0.5)
        # BM25Okapi floors a negative idf (a term in more than half the corpus)
        # at epsilon * the MEAN idf, negatives included -- match it exactly or
        # the two backends disagree on common words.
        eps = 0.25 * (float(idf.mean()) if idf.size else 0.0)
        idf = np.where(idf < 0, eps, idf).astype(np.float32)
        self.idf = idf
        denom = tf_a + self.k1 * (1 - self.b + self.b
                                  * np.repeat(lens, np.diff(indptr)) / (avgdl or 1.0))
        w = (idf[np.asarray(indices, dtype=np.int32)] * tf_a * (self.k1 + 1)) / denom
        self.m = csr_matrix((w.astype(np.float32),
                             np.asarray(indices, dtype=np.int32),
                             np.asarray(indptr, dtype=np.int64)),
                            shape=(n_docs, len(vocab)))
        self.n_docs = n_docs

    def get_scores(self, query_tokens: Iterable[str]) -> np.ndarray:
        cols: List[int] = [self.vocab[t] for t in query_tokens if t in self.vocab]
        if not cols:
            return np.zeros(self.n_docs, dtype=np.float32)
        q = np.zeros(self.m.shape[1], dtype=np.float32)
        np.add.at(q, np.asarray(cols, dtype=np.int32), 1.0)
        return self.m @ q
