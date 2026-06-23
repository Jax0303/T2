# SPDX-License-Identifier: MIT
"""Unit tests for header-tree scope enumeration (no dataset needed)."""
from rag_agent.retrieve.header_enum import enumerate_scope


class FakeTable:
    """Minimal OriginalTable stand-in: 3 rows x 2 cols, one non-numeric cell."""

    def __init__(self):
        self.left = {0: ["industry", "construction"],
                     1: ["industry", "mining"],
                     2: ["total"]}
        self.top = {0: ["2013"], 1: ["2014"]}
        self.values = {(0, 0): 10.0, (0, 1): 12.0,
                       (1, 0): 5.0, (1, 1): None,  # missing/non-numeric
                       (2, 0): 15.0, (2, 1): 12.0}

    @property
    def n_rows(self):
        return 3

    @property
    def n_cols(self):
        return 2

    def cell_num(self, r, c):
        return self.values.get((r, c))

    def _match(self, token, path):
        toks = [t.strip().lower() for t in token.replace(">", " ").split()]
        joined = " ".join(path).lower()
        return all(t in joined for t in toks)

    def find_rows_by_header(self, token):
        return [r for r in range(self.n_rows) if self._match(token, self.left[r])]

    def find_cols_by_header(self, token):
        return [c for c in range(self.n_cols) if self._match(token, self.top[c])]


def test_enumerate_single_row_across_cols():
    t = FakeTable()
    # "construction" row across all year cols (no col predicate -> fallback all)
    e = enumerate_scope(t, row_paths=[["construction"]], col_paths=[])
    assert e.rows == {0}
    assert e.col_fallback is True and e.cols == {0, 1}
    assert e.cells == {(0, 0), (0, 1)}  # both numeric


def test_drops_non_numeric_cells():
    t = FakeTable()
    e = enumerate_scope(t, row_paths=[["mining"]], col_paths=[])
    # (1,1) is None -> excluded; only (1,0) numeric
    assert e.cells == {(1, 0)}


def test_parent_node_enumerates_children():
    t = FakeTable()
    # parent token "industry" should match both construction and mining rows
    e = enumerate_scope(t, row_paths=[["industry"]], col_paths=[["2014"]])
    assert e.rows == {0, 1}
    assert e.cols == {1}
    assert e.cells == {(0, 1)}  # (1,1) is None -> dropped


def test_full_fallback_when_no_predicate():
    t = FakeTable()
    e = enumerate_scope(t, row_paths=[], col_paths=[])
    assert e.row_fallback and e.col_fallback
    # all numeric cells (5 of 6)
    assert len(e.cells) == 5
