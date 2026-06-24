# SPDX-License-Identifier: MIT
"""Unit tests for the embedding tree-node resolver (fake embedder, no model)."""
import numpy as np

from rag_agent.query.header_embed_resolver import EmbedResolver, _node_candidates


def test_node_candidates_include_parents():
    cands = _node_candidates([["industry", "construction"], ["industry", "mining"]])
    assert ["industry"] in cands            # parent node is a candidate
    assert ["industry", "construction"] in cands
    assert ["industry", "mining"] in cands
    # parent deduped to one entry
    assert sum(c == ["industry"] for c in cands) == 1


class FakeEmbedder:
    """Encode text to a 3-d keyword indicator vector, L2-normalized (cosine-ready)."""

    KEYS = ("construction", "mining", "2014")

    def encode(self, texts):
        out = []
        for t in texts:
            tl = t.lower()
            v = np.array([1.0 if k in tl else 0.0 for k in self.KEYS])
            n = np.linalg.norm(v)
            out.append(v / n if n else v)
        return np.array(out)


class FakeTable:
    table_id = "t1"

    def __init__(self):
        self.left = {0: ["industry", "construction"], 1: ["industry", "mining"]}
        self.top = {0: ["fy", "2013"], 1: ["fy", "2014"]}

    @property
    def n_rows(self):
        return 2

    @property
    def n_cols(self):
        return 2

    def row_path(self, r):
        return self.left[r]

    def col_path(self, c):
        return self.top[c]


def test_embed_resolver_picks_semantic_match():
    r = EmbedResolver(FakeEmbedder(), top_n_rows=1, top_n_cols=1)
    intent = r.resolve("construction revenue in 2014", FakeTable())
    # row: should rank the construction node top (not mining)
    assert intent.row_paths[0][-1] == "construction"
    # col: should rank the 2014 column node top (not 2013)
    assert intent.col_paths[0][-1] == "2014"
    assert intent.source == "embed"


def test_cache_reused_across_queries():
    r = EmbedResolver(FakeEmbedder())
    t = FakeTable()
    r.resolve("construction", t)
    assert t.table_id in r._cache
    # second query on same table must not rebuild (same cached tuple object)
    cached = r._cache[t.table_id]
    r.resolve("mining", t)
    assert r._cache[t.table_id] is cached
