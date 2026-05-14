# SPDX-License-Identifier: MIT
"""Round-trip tests for all five serializers."""

from __future__ import annotations

import pytest

from src.io.table_schema import Cell, HeaderNode, Table
from src.serializers.csv_ser import CsvSerializer
from src.serializers.html_ser import HtmlSerializer
from src.serializers.json_tree_ser import JsonTreeSerializer
from src.serializers.markdown_ser import MarkdownSerializer
from src.serializers.otsl_ser import OtslSerializer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _simple_table() -> Table:
    """2×3 flat table, no merges."""
    return Table(
        cells=[
            [Cell("A", is_header=True), Cell("B", is_header=True), Cell("C", is_header=True)],
            [Cell("1"), Cell("2"), Cell("3")],
        ],
    )


def _merged_table() -> Table:
    """Table with merged cells.

    Layout (4 rows × 5 cols):
        Row 0: ["",       "2020"(cs=2), covered, "2021"(cs=2), covered]
        Row 1: ["",       "Q1",         "Q2",    "Q1",         "Q2"   ]
        Row 2: ["Apple",  "100",        "150",   "200",        "250"  ]
        Row 3: ["Banana", "80",         "90",    "110",        "120"  ]
    """
    return Table(
        cells=[
            [
                Cell("", is_header=True),
                Cell("2020", row_span=1, col_span=2, is_header=True),
                Cell("", row_span=0, col_span=0, is_header=False),
                Cell("2021", row_span=1, col_span=2, is_header=True),
                Cell("", row_span=0, col_span=0, is_header=False),
            ],
            [
                Cell("", is_header=True),
                Cell("Q1", is_header=True),
                Cell("Q2", is_header=True),
                Cell("Q1", is_header=True),
                Cell("Q2", is_header=True),
            ],
            [
                Cell("Apple", is_header=True),
                Cell("100"), Cell("150"), Cell("200"), Cell("250"),
            ],
            [
                Cell("Banana", is_header=True),
                Cell("80"), Cell("90"), Cell("110"), Cell("120"),
            ],
        ],
        merged_cells=[(0, 1, 0, 2), (0, 3, 0, 4)],
        top_header_tree=HeaderNode(
            name="<TOP>", span_start=-1, span_end=-1,
            children=[
                HeaderNode(name="2020", span_start=1, span_end=2, children=[
                    HeaderNode(name="Q1", span_start=1, span_end=1),
                    HeaderNode(name="Q2", span_start=2, span_end=2),
                ]),
                HeaderNode(name="2021", span_start=3, span_end=4, children=[
                    HeaderNode(name="Q1", span_start=3, span_end=3),
                    HeaderNode(name="Q2", span_start=4, span_end=4),
                ]),
            ],
        ),
        left_header_tree=HeaderNode(
            name="<LEFT>", span_start=-1, span_end=-1,
            children=[
                HeaderNode(name="Apple", span_start=2, span_end=2),
                HeaderNode(name="Banana", span_start=3, span_end=3),
            ],
        ),
    )


# ---------------------------------------------------------------------------
# HTML Serializer
# ---------------------------------------------------------------------------

class TestHtmlSerializer:
    ser = HtmlSerializer()

    def test_roundtrip_simple(self) -> None:
        t = _simple_table()
        text = self.ser.serialize(t)
        assert "<table>" in text
        assert "<th>" in text
        rt = self.ser.parse(text)
        assert rt.n_rows == t.n_rows
        assert rt.n_cols == t.n_cols
        for r in range(t.n_rows):
            for c in range(t.n_cols):
                assert rt.cells[r][c].value == t.cells[r][c].value

    def test_roundtrip_merged(self) -> None:
        t = _merged_table()
        text = self.ser.serialize(t)
        assert 'colspan="2"' in text
        rt = self.ser.parse(text)
        # Origin cell should have col_span=2.
        assert rt.cells[0][1].col_span == 2
        assert rt.cells[0][1].value == "2020"
        # Covered cell.
        assert rt.cells[0][2].col_span == 0
        # Merged cells list recovered.
        assert (0, 1, 0, 2) in rt.merged_cells

    def test_header_tag(self) -> None:
        t = _simple_table()
        text = self.ser.serialize(t)
        rt = self.ser.parse(text)
        assert rt.cells[0][0].is_header
        assert not rt.cells[1][0].is_header


# ---------------------------------------------------------------------------
# Markdown Serializer (lossy)
# ---------------------------------------------------------------------------

class TestMarkdownSerializer:
    ser = MarkdownSerializer()

    def test_roundtrip_simple_values(self) -> None:
        t = _simple_table()
        text = self.ser.serialize(t)
        assert "| A | B | C |" in text
        rt = self.ser.parse(text)
        assert rt.n_rows == 2
        assert rt.cells[1][0].value == "1"

    def test_lossy_no_merges(self) -> None:
        t = _merged_table()
        text = self.ser.serialize(t)
        rt = self.ser.parse(text)
        # Markdown cannot recover merge info.
        assert rt.merged_cells == []

    def test_header_only_first_row(self) -> None:
        t = _merged_table()
        text = self.ser.serialize(t)
        rt = self.ser.parse(text)
        assert rt.cells[0][0].is_header
        assert not rt.cells[1][0].is_header  # 2nd row not header in MD


# ---------------------------------------------------------------------------
# CSV Serializer (lossy)
# ---------------------------------------------------------------------------

class TestCsvSerializer:
    ser = CsvSerializer()

    def test_roundtrip_simple(self) -> None:
        t = _simple_table()
        text = self.ser.serialize(t)
        rt = self.ser.parse(text)
        assert rt.n_rows == 2
        assert rt.n_cols == 3
        assert rt.cells[0][0].value == "A"

    def test_lossy_no_header_flag(self) -> None:
        t = _simple_table()
        text = self.ser.serialize(t)
        rt = self.ser.parse(text)
        # CSV has no header concept.
        assert not rt.cells[0][0].is_header

    def test_lossy_no_merges(self) -> None:
        t = _merged_table()
        text = self.ser.serialize(t)
        rt = self.ser.parse(text)
        assert rt.merged_cells == []


# ---------------------------------------------------------------------------
# JSON-tree Serializer (lossless)
# ---------------------------------------------------------------------------

class TestJsonTreeSerializer:
    ser = JsonTreeSerializer()

    def test_roundtrip_lossless_cells(self) -> None:
        t = _merged_table()
        text = self.ser.serialize(t)
        rt = self.ser.parse(text)
        assert rt.n_rows == t.n_rows
        assert rt.n_cols == t.n_cols
        for r in range(t.n_rows):
            for c in range(t.n_cols):
                assert rt.cells[r][c].value == t.cells[r][c].value
                assert rt.cells[r][c].row_span == t.cells[r][c].row_span
                assert rt.cells[r][c].col_span == t.cells[r][c].col_span
                assert rt.cells[r][c].is_header == t.cells[r][c].is_header

    def test_roundtrip_lossless_merges(self) -> None:
        t = _merged_table()
        text = self.ser.serialize(t)
        rt = self.ser.parse(text)
        assert set(rt.merged_cells) == set(t.merged_cells)

    def test_roundtrip_lossless_header_tree(self) -> None:
        t = _merged_table()
        text = self.ser.serialize(t)
        rt = self.ser.parse(text)
        # Top header leaves.
        orig_leaves = [n.name for n in t.top_header_tree.leaves()]
        rt_leaves = [n.name for n in rt.top_header_tree.leaves()]
        assert orig_leaves == rt_leaves
        # Left header leaves.
        orig_left = [n.name for n in t.left_header_tree.leaves()]
        rt_left = [n.name for n in rt.left_header_tree.leaves()]
        assert orig_left == rt_left


# ---------------------------------------------------------------------------
# OTSL Serializer
# ---------------------------------------------------------------------------

class TestOtslSerializer:
    ser = OtslSerializer()

    def test_roundtrip_simple(self) -> None:
        t = _simple_table()
        text = self.ser.serialize(t)
        assert "C\tA" in text
        assert "NL" in text
        rt = self.ser.parse(text)
        assert rt.n_rows == 2
        assert rt.n_cols == 3
        assert rt.cells[0][0].value == "A"
        assert rt.cells[1][2].value == "3"

    def test_roundtrip_merged_spans(self) -> None:
        t = _merged_table()
        text = self.ser.serialize(t)
        assert "\tL\t" in text  # colspan continuation
        rt = self.ser.parse(text)
        assert rt.cells[0][1].col_span == 2
        assert rt.cells[0][1].value == "2020"
        # Covered cell.
        assert rt.cells[0][2].col_span == 0

    def test_no_header_tree(self) -> None:
        """OTSL does not preserve header trees."""
        t = _merged_table()
        text = self.ser.serialize(t)
        rt = self.ser.parse(text)
        assert rt.top_header_tree.name == "<ROOT>"
        assert rt.top_header_tree.children == []
