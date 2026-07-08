"""Tests for the S1 (flat) and S2 (header-path) serializers.

Uses a hand-built hierarchical fixture so the tests run without the HiTab
download, plus a FlatTable to cover the FinQA/WikiSQL degenerate case.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_agent.serialization import serialize, FlatTable  # noqa: E402
from rag_agent.serialization.base import Chunk  # noqa: E402


class FakeTable:
    """Minimal hierarchical TableView: 2 rows x 2 cols, 2-level column headers."""

    table_id = "t1"
    title = "Demo"

    _data = [[1234, 2345], [10, 20]]
    _cols = [["2023", "Q1"], ["2023", "Q2"]]
    _rows = [["Revenue"], ["Cost"]]

    @property
    def n_rows(self):
        return 2

    @property
    def n_cols(self):
        return 2

    def cell(self, r, c):
        return self._data[r][c]

    def col_path(self, c):
        return self._cols[c]

    def row_path(self, r):
        return self._rows[r]


def test_s2_header_path_prefix_matches_spec():
    chunks = serialize(FakeTable(), scheme="S2")
    assert len(chunks) == 2
    # The spec's canonical example: "Revenue > 2023 > Q1: 1,234"
    assert "Revenue > 2023 > Q1: 1234" in chunks[0].text
    assert "Revenue > 2023 > Q2: 2345" in chunks[0].text
    assert chunks[0].scheme == "S2"
    assert chunks[0].row_index == 0
    # header_paths carry full row+col provenance for every cell.
    assert chunks[0].header_paths[0] == ["Revenue", "2023", "Q1"]


def test_s1_is_flat_without_header_paths():
    chunks = serialize(FakeTable(), scheme="S1")
    assert len(chunks) == 2
    assert chunks[0].scheme == "S1"
    # Flat markdown row, no "A > B > C:" header-path prefixing.
    assert "| Revenue |" in chunks[0].text
    assert "2023 > Q1" not in chunks[0].text
    assert chunks[0].header_paths == []


def test_s2_cell_granularity_one_chunk_per_cell():
    chunks = serialize(FakeTable(), scheme="S2", granularity="cell")
    assert len(chunks) == 4  # 2x2
    assert all(c.kind == "cell" for c in chunks)
    ids = {c.chunk_id for c in chunks}
    assert "t1::S2::r0c0" in ids
    assert all(len(c.header_paths) == 1 for c in chunks)


def test_chunk_ids_unique_and_stable():
    chunks = serialize(FakeTable(), scheme="S2")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids == ["t1::S2::r0", "t1::S2::r1"]


def test_flat_table_degrades_gracefully():
    ft = FlatTable(
        table_id="fq1",
        columns=["year", "revenue"],
        rows=[["2023", "100"], ["2024", "120"]],
        row_headers=["", ""],
        title="Income",
    )
    s2 = serialize(ft, scheme="S2")
    assert len(s2) == 2
    # No row hierarchy -> "<column>: <value>".
    assert "year: 2023" in s2[0].text
    assert "revenue: 100" in s2[0].text
    s1 = serialize(ft, scheme="S1")
    assert "| 2023 | 100 |" in s1[0].text


def test_to_dict_roundtrip_keys():
    c = serialize(FakeTable(), scheme="S2")[0]
    d = c.to_dict()
    assert set(d) == {
        "table_id", "chunk_id", "text", "scheme", "kind",
        "row_index", "col_index", "header_paths", "metadata",
    }
    assert isinstance(c, Chunk)
