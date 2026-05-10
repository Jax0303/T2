# SPDX-License-Identifier: MIT
"""Unit tests for the four structural metrics."""

from __future__ import annotations

import pytest

from src.io.table_schema import Cell, HeaderNode, Table
from src.metrics.cell_coord_preserve import cell_coord_preservation
from src.metrics.header_path_acc import header_path_accuracy
from src.metrics.merged_cell_recovery import merged_cell_recovery
from src.metrics.teds import teds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _table_a() -> Table:
    """Simple 2×3 table with one merge (row 0, cols 0-1)."""
    return Table(
        cells=[
            [
                Cell("Header", row_span=1, col_span=2, is_header=True),
                Cell("", row_span=0, col_span=0),
                Cell("C", is_header=True),
            ],
            [Cell("1"), Cell("2"), Cell("3")],
        ],
        merged_cells=[(0, 0, 0, 1)],
        top_header_tree=HeaderNode(
            name="<TOP>", span_start=-1, span_end=-1,
            children=[
                HeaderNode(name="Header", span_start=0, span_end=1),
                HeaderNode(name="C", span_start=2, span_end=2),
            ],
        ),
        left_header_tree=HeaderNode(name="<LEFT>", span_start=-1, span_end=-1),
        metadata={"top_header_rows_num": 1, "left_header_columns_num": 0},
    )


def _table_a_identical() -> Table:
    """Exact copy of _table_a."""
    return _table_a()


def _table_a_partial() -> Table:
    """Variant with merge lost and one value changed."""
    return Table(
        cells=[
            [
                Cell("Header", is_header=True),
                Cell("Header", is_header=True),  # merge expanded
                Cell("C", is_header=True),
            ],
            [Cell("1"), Cell("WRONG"), Cell("3")],
        ],
        merged_cells=[],
        top_header_tree=HeaderNode(name="<ROOT>", span_start=-1, span_end=-1),
        left_header_tree=HeaderNode(name="<ROOT>", span_start=-1, span_end=-1),
        metadata={"top_header_rows_num": 1, "left_header_columns_num": 0},
    )


# ---------------------------------------------------------------------------
# TEDS
# ---------------------------------------------------------------------------

class TestTEDS:
    def test_identical_tables(self) -> None:
        score = teds(_table_a(), _table_a_identical())
        assert score == pytest.approx(1.0)

    def test_different_tables(self) -> None:
        score = teds(_table_a(), _table_a_partial())
        assert 0.0 < score < 1.0

    def test_empty_tables(self) -> None:
        empty = Table(cells=[])
        assert teds(empty, empty) == pytest.approx(1.0)

    def test_symmetry(self) -> None:
        s1 = teds(_table_a(), _table_a_partial())
        s2 = teds(_table_a_partial(), _table_a())
        assert s1 == pytest.approx(s2)


# ---------------------------------------------------------------------------
# Cell Coordinate Preservation
# ---------------------------------------------------------------------------

class TestCellCoordPreservation:
    def test_identical(self) -> None:
        assert cell_coord_preservation(_table_a(), _table_a_identical()) == pytest.approx(1.0)

    def test_one_wrong(self) -> None:
        # _table_a has 4 non-covered cells: (0,0)Header, (0,2)C, (1,0)1, (1,1)2, (1,2)3 = 5
        # _table_a_partial has same grid but cell (1,1) is "WRONG" and (0,1) is now non-covered.
        # Origin cells in _table_a: (0,0), (0,2), (1,0), (1,1), (1,2) = 5
        # Match: (0,0)Header=Header, (0,2)C=C, (1,0)1=1, (1,1)2≠WRONG, (1,2)3=3 → 4/5
        score = cell_coord_preservation(_table_a(), _table_a_partial())
        assert score == pytest.approx(4.0 / 5.0)

    def test_empty(self) -> None:
        empty = Table(cells=[])
        assert cell_coord_preservation(empty, empty) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Merged Cell Recovery
# ---------------------------------------------------------------------------

class TestMergedCellRecovery:
    def test_identical(self) -> None:
        assert merged_cell_recovery(_table_a(), _table_a_identical()) == pytest.approx(1.0)

    def test_merge_lost(self) -> None:
        assert merged_cell_recovery(_table_a(), _table_a_partial()) == pytest.approx(0.0)

    def test_no_merges(self) -> None:
        flat = Table(cells=[[Cell("a"), Cell("b")]])
        assert merged_cell_recovery(flat, flat) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Header Path Accuracy
# ---------------------------------------------------------------------------

class TestHeaderPathAccuracy:
    def test_identical(self) -> None:
        assert header_path_accuracy(_table_a(), _table_a_identical()) == pytest.approx(1.0)

    def test_tree_lost(self) -> None:
        """When recovered table has no tree, paths are empty → mismatch."""
        score = header_path_accuracy(_table_a(), _table_a_partial())
        # Original has top chains for data cells; partial has none → 0.
        assert score < 1.0

    def test_no_data_cells(self) -> None:
        """Table with only header cells → vacuously 1.0."""
        hdr_only = Table(
            cells=[[Cell("H1", is_header=True), Cell("H2", is_header=True)]],
            metadata={"top_header_rows_num": 1, "left_header_columns_num": 0},
        )
        assert header_path_accuracy(hdr_only, hdr_only) == pytest.approx(1.0)
