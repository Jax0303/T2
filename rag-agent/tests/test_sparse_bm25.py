# SPDX-License-Identifier: MIT
"""SparseBM25 must rank exactly as rank_bm25's BM25Okapi does."""
import numpy as np
from rank_bm25 import BM25Okapi

from rag_agent.retrieve.encoders import _tokenize
from rag_agent.retrieve.sparse_bm25 import SparseBM25

CORPUS = [
    "In the table 'labour force survey', among ontario, the value of employment rate is 61.2.",
    "In the table 'labour force survey', among quebec, the value of employment rate is 59.8.",
    "In the table 'labour force survey', among ontario, the value of unemployment rate is 5.4.",
    "among 35 to 44 years, the value of average volunteer hours in 2013 is 122.",
    "the value of total is 4.",
]
QUERIES = ["what was the employment rate in ontario",
           "average volunteer hours for 35 to 44 year olds in 2013",
           "total", "a term that appears nowhere"]


def test_scores_match_rank_bm25():
    toks = [_tokenize(t) for t in CORPUS]
    ref, ours = BM25Okapi(toks), SparseBM25(toks)
    for q in QUERIES:
        a = np.asarray(ref.get_scores(_tokenize(q)), dtype=np.float32)
        b = ours.get_scores(_tokenize(q))
        assert np.allclose(a, b, atol=1e-4), (q, a, b)


if __name__ == "__main__":
    test_scores_match_rank_bm25()
    print("ok")
