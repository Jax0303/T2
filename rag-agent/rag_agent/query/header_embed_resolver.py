# SPDX-License-Identifier: MIT
"""Embedding-based tree-node header resolver (idea: semantic match over the tree).

The deterministic ``resolve_against_table`` ranks header paths by *lexical* overlap
(``_fuzzy_score``), so a query that names a header with different words ("building
sector" vs "construction") scores 0 and the scope is missed — the measured
row-axis bottleneck. This resolver instead ranks header **tree nodes** by
*semantic* (embedding) similarity to the query, and treats every prefix of a leaf
path as a candidate node so a query can bind at any level of the tree (a parent
node then enumerates to all its children via ``find_rows_by_header``).

Drop-in: returns the same :class:`HeaderPathIntent` the enumeration path consumes.
Candidate embeddings are cached per table.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from ..router.query_classifier import classify_query
from .header_path_resolver import (
    HeaderPathIntent, _distinct_paths, _rank_paths, extract_target_terms,
)


def _node_candidates(paths: Sequence[Sequence[str]], cap: int = 80) -> List[List[str]]:
    """Every prefix (tree node) of every leaf path, deduped — match at any level."""
    seen: set = set()
    out: List[List[str]] = []
    for p in paths:
        for j in range(1, len(p) + 1):
            pref = tuple(p[:j])
            if pref and pref not in seen:
                seen.add(pref)
                out.append(list(pref))
                if len(out) >= cap:
                    return out
    return out


class EmbedResolver:
    """Semantic tree-node resolver with per-table candidate-embedding cache."""

    def __init__(self, embedder, top_n_cols: int = 3, top_n_rows: int = 4,
                 include_parents: bool = True,
                 row_mode: str = "embed", col_mode: str = "embed"):
        """``row_mode``/``col_mode`` in {"embed","lexical"}. E3 showed row entities
        benefit from semantic (embed) matching while column codes/years match
        better lexically; the hybrid is row_mode="embed", col_mode="lexical"."""
        self.embedder = embedder
        self.top_n_cols = top_n_cols
        self.top_n_rows = top_n_rows
        self.include_parents = include_parents
        self.row_mode = row_mode
        self.col_mode = col_mode
        # table_id -> (col_cands, col_mat, row_cands, row_mat)
        self._cache: Dict[str, Tuple] = {}

    def _cands(self, table, axis: str) -> List[List[str]]:
        leaves = _distinct_paths(table, axis)
        return _node_candidates(leaves) if self.include_parents else leaves

    def _prep(self, table):
        key = table.table_id
        if key in self._cache:
            return self._cache[key]
        import numpy as np
        col_c = self._cands(table, "col")
        row_c = self._cands(table, "row")
        col_m = self.embedder.encode([" > ".join(c) for c in col_c]) if col_c else np.zeros((0, 1))
        row_m = self.embedder.encode([" > ".join(c) for c in row_c]) if row_c else np.zeros((0, 1))
        self._cache[key] = (col_c, np.asarray(col_m), row_c, np.asarray(row_m))
        return self._cache[key]

    @staticmethod
    def _topn(cands, mat, qv, n):
        if not cands:
            return []
        import numpy as np
        scores = mat @ qv  # embeddings are pre-normalized -> cosine
        order = np.argsort(-scores)[:n]
        return [cands[i] for i in order]

    def resolve(self, query: str, table) -> HeaderPathIntent:
        import numpy as np
        col_c, col_m, row_c, row_m = self._prep(table)
        qv = np.asarray(self.embedder.encode([query])[0])
        terms = extract_target_terms(query)
        qintent = classify_query(query)

        def axis(mode, cands, mat, top_n, axis_name):
            if mode == "lexical":
                return _rank_paths(table, terms, axis_name, top_n)
            return self._topn(cands, mat, qv, top_n)

        row_paths = axis(self.row_mode, row_c, row_m, self.top_n_rows, "row")
        col_paths = axis(self.col_mode, col_c, col_m, self.top_n_cols, "col")
        src = "embed" if self.row_mode == self.col_mode == "embed" else \
              f"hybrid(row={self.row_mode},col={self.col_mode})"
        return HeaderPathIntent(
            operation=qintent.qtype.value,
            needs_symbolic=qintent.needs_symbolic,
            target_terms=terms,
            expansion="",
            col_paths=col_paths,
            row_paths=row_paths,
            source=src,
        )
