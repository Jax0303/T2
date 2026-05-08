# SPDX-License-Identifier: MIT
"""Tests for HiTab loader and Table schema."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.io.hitab_loader import load_table, load_tables
from src.io.table_schema import Cell, HeaderNode, Table

# ---------------------------------------------------------------------------
# Fixtures: minimal HiTab-format JSON
# ---------------------------------------------------------------------------

SAMPLE_HITAB_JSON: dict = {
    "title": "test_table_001",
    "top_header_rows_num": 2,
    "left_header_columns_num": 1,
    "texts": [
        ["",       "2020",  "2020",  "2021",  "2021"],
        ["",       "Q1",    "Q2",    "Q1",    "Q2"],
        ["Apple",  "100",   "150",   "200",   "250"],
        ["Banana", "80",    "90",    "110",   "120"],
    ],
    "merged_regions": [
        {"first_row": 0, "first_column": 1, "last_row": 0, "last_column": 2},
        {"first_row": 0, "first_column": 3, "last_row": 0, "last_column": 4},
    ],
    "top_root": {
        "name": "<TOP>",
        "start_idx": -1,
        "end_idx": -1,
        "children": [
            {
                "name": "2020",
                "start_idx": 1,
                "end_idx": 2,
                "children": [
                    {"name": "Q1", "start_idx": 1, "end_idx": 1, "children": []},
                    {"name": "Q2", "start_idx": 2, "end_idx": 2, "children": []},
                ],
            },
            {
                "name": "2021",
                "start_idx": 3,
                "end_idx": 4,
                "children": [
                    {"name": "Q1", "start_idx": 3, "end_idx": 3, "children": []},
                    {"name": "Q2", "start_idx": 4, "end_idx": 4, "children": []},
                ],
            },
        ],
    },
    "left_root": {
        "name": "<LEFT>",
        "start_idx": -1,
        "end_idx": -1,
        "children": [
            {"name": "Apple", "start_idx": 2, "end_idx": 2, "children": []},
            {"name": "Banana", "start_idx": 3, "end_idx": 3, "children": []},
        ],
    },
}


@pytest.fixture()
def sample_json_path(tmp_path: Path) -> Path:
    """Write sample HiTab JSON to a temp file and return its path."""
    p = tmp_path / "test_table_001.json"
    p.write_text(json.dumps(SAMPLE_HITAB_JSON), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Table schema tests
# ---------------------------------------------------------------------------


class TestHeaderNode:
    def test_leaf_detection(self) -> None:
        leaf = HeaderNode(name="Q1", span_start=1, span_end=1)
        parent = HeaderNode(name="2020", span_start=1, span_end=2, children=[leaf])
        assert leaf.is_leaf
        assert not parent.is_leaf

    def test_leaves(self) -> None:
        tree = HeaderNode(
            name="root",
            span_start=0,
            span_end=3,
            children=[
                HeaderNode(
                    name="A",
                    span_start=0,
                    span_end=1,
                    children=[
                        HeaderNode(name="A1", span_start=0, span_end=0),
                        HeaderNode(name="A2", span_start=1, span_end=1),
                    ],
                ),
                HeaderNode(name="B", span_start=2, span_end=3),
            ],
        )
        leaf_names = [n.name for n in tree.leaves()]
        assert leaf_names == ["A1", "A2", "B"]

    def test_ancestor_chain(self) -> None:
        tree = HeaderNode(
            name="<TOP>",
            span_start=-1,
            span_end=-1,
            children=[
                HeaderNode(
                    name="2020",
                    span_start=1,
                    span_end=2,
                    children=[
                        HeaderNode(name="Q1", span_start=1, span_end=1),
                        HeaderNode(name="Q2", span_start=2, span_end=2),
                    ],
                ),
            ],
        )
        chain = tree.ancestor_chain(2)
        assert chain == ["<TOP>", "2020", "Q2"]


class TestTable:
    def test_dimensions(self) -> None:
        cells = [
            [Cell(value="a"), Cell(value="b")],
            [Cell(value="c"), Cell(value="d")],
            [Cell(value="e"), Cell(value="f")],
        ]
        t = Table(cells=cells)
        assert t.n_rows == 3
        assert t.n_cols == 2

    def test_empty_table(self) -> None:
        t = Table(cells=[])
        assert t.n_rows == 0
        assert t.n_cols == 0


# ---------------------------------------------------------------------------
# HiTab loader tests
# ---------------------------------------------------------------------------


class TestLoadTable:
    def test_basic_load(self, sample_json_path: Path) -> None:
        table = load_table(sample_json_path)
        assert table.n_rows == 4
        assert table.n_cols == 5

    def test_metadata(self, sample_json_path: Path) -> None:
        table = load_table(sample_json_path)
        assert table.metadata["table_id"] == "test_table_001"
        assert table.metadata["top_header_rows_num"] == 2
        assert table.metadata["left_header_columns_num"] == 1

    def test_header_flag(self, sample_json_path: Path) -> None:
        table = load_table(sample_json_path)
        assert table.cells[0][1].is_header  # top header row
        assert table.cells[1][1].is_header  # top header row
        assert table.cells[2][0].is_header  # left header col
        assert not table.cells[2][1].is_header  # data cell

    def test_merged_cells(self, sample_json_path: Path) -> None:
        table = load_table(sample_json_path)
        assert len(table.merged_cells) == 2
        assert (0, 1, 0, 2) in table.merged_cells
        assert (0, 3, 0, 4) in table.merged_cells

    def test_span_values(self, sample_json_path: Path) -> None:
        table = load_table(sample_json_path)
        # "2020" spans columns 1-2
        assert table.cells[0][1].value == "2020"
        assert table.cells[0][1].col_span == 2
        assert table.cells[0][1].row_span == 1
        # covered cell has span 0
        assert table.cells[0][2].row_span == 0
        assert table.cells[0][2].col_span == 0

    def test_top_header_tree(self, sample_json_path: Path) -> None:
        table = load_table(sample_json_path)
        tree = table.top_header_tree
        assert tree.name == "<TOP>"
        assert len(tree.children) == 2
        assert tree.children[0].name == "2020"
        leaf_names = [n.name for n in tree.leaves()]
        assert leaf_names == ["Q1", "Q2", "Q1", "Q2"]

    def test_left_header_tree(self, sample_json_path: Path) -> None:
        table = load_table(sample_json_path)
        tree = table.left_header_tree
        assert tree.name == "<LEFT>"
        leaf_names = [n.name for n in tree.leaves()]
        assert leaf_names == ["Apple", "Banana"]

    def test_data_cell_values(self, sample_json_path: Path) -> None:
        table = load_table(sample_json_path)
        assert table.cells[2][1].value == "100"
        assert table.cells[3][4].value == "120"


class TestLoadTables:
    def test_load_all(self, tmp_path: Path) -> None:
        for i in range(3):
            data = {**SAMPLE_HITAB_JSON, "title": f"table_{i}"}
            (tmp_path / f"table_{i}.json").write_text(json.dumps(data), encoding="utf-8")
        tables = load_tables(tmp_path)
        assert len(tables) == 3

    def test_load_by_ids(self, tmp_path: Path) -> None:
        for i in range(3):
            data = {**SAMPLE_HITAB_JSON, "title": f"table_{i}"}
            (tmp_path / f"table_{i}.json").write_text(json.dumps(data), encoding="utf-8")
        tables = load_tables(tmp_path, table_ids=["table_0", "table_2"])
        assert len(tables) == 2
        ids = [t.metadata["table_id"] for t in tables]
        assert "table_0" in ids
        assert "table_2" in ids
