# SPDX-License-Identifier: MIT
"""Unit tests for header-tree scope enumeration (no dataset needed)."""
from rag_agent.retrieve.header_enum import (
    enumerate_scope, expand_sibling_groups, is_ratio_query, total_like_rows,
)


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

    def row_path(self, r):
        return self.left[r]


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


def test_total_like_rows_detects_total():
    t = FakeTable()
    assert total_like_rows(t) == {2}  # row 2 = ["total"], both cols numeric


def test_add_total_rows_augments_matched_scope():
    t = FakeTable()
    # "construction" matches row 0 only; total augmentation adds the total row (2)
    base = enumerate_scope(t, row_paths=[["construction"]], col_paths=[])
    aug = enumerate_scope(t, row_paths=[["construction"]], col_paths=[],
                          add_total_rows=True)
    assert base.rows == {0}
    assert aug.rows == {0, 2}
    assert (2, 0) in aug.cells and (2, 1) in aug.cells


def test_add_total_rows_noop_on_fallback():
    t = FakeTable()
    # no row predicate -> whole-axis fallback; augmentation must not fire
    e = enumerate_scope(t, row_paths=[], col_paths=[["2014"]], add_total_rows=True)
    assert e.row_fallback is True


def test_expand_sibling_groups():
    t = FakeTable()
    # matching only "construction" (row 0) should expand to its sibling "mining" (1)
    assert expand_sibling_groups(t, {0}) == {0, 1}
    # the parent-less total row (depth 1) has no siblings to expand
    assert expand_sibling_groups(t, {2}) == {2}


def test_expand_siblings_via_enumerate():
    t = FakeTable()
    e = enumerate_scope(t, row_paths=[["construction"]], col_paths=[["2013"]],
                        expand_siblings=True)
    assert e.rows == {0, 1}  # construction + sibling mining
    assert e.cells == {(0, 0), (1, 0)}


def test_last_numeric_col_fallback():
    from rag_agent.retrieve.header_enum import last_numeric_col
    t = FakeTable()
    # col 1 has numeric in rows 0 and 2 -> rightmost data col is 1
    assert last_numeric_col(t, {0, 1, 2}) == {1}
    # unpinned column with mode="last" keeps only that column (not the whole axis)
    e = enumerate_scope(t, row_paths=[["construction"]], col_paths=[],
                        col_fallback_mode="last")
    assert e.col_fallback is True and e.cols == {1}
    assert e.cells == {(0, 1)}              # vs {(0,0),(0,1)} under mode "all"


def test_is_ratio_query():
    assert is_ratio_query("what percentage of total r&d ...")
    assert is_ratio_query("how many times more likely ...")
    assert not is_ratio_query("what is the sum of construction and mining")
