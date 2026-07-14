# SPDX-License-Identifier: MIT
"""Data-free unit tests for cell-level caption+header verbalization."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.bench.schema import BenchTable
from rag_agent.serialize.verbalize import verbalize_cell, verbalize_table, STYLES


@pytest.fixture
def hier_table():
    return BenchTable(
        table_id="t1",
        title="population by region",
        data=[[950, 3.1], [340, ""], [None, 1.2]],
        top_paths=[["census 2020", "count"], ["census 2020", "growth %"]],
        left_paths=[["korea", "seoul"], ["korea", "busan"], ["korea", "incheon"]],
    )


def test_short_uses_leaf_headers_no_caption(hier_table):
    s = verbalize_cell(hier_table, 0, 0, "short")
    assert s == "count for seoul is 950."
    assert "population" not in s


def test_medium_adds_caption(hier_table):
    s = verbalize_cell(hier_table, 0, 0, "medium")
    assert s == "In population by region, count for seoul is 950."


def test_long_uses_full_paths(hier_table):
    s = verbalize_cell(hier_table, 0, 1, "long")
    assert s == ("In population by region, census 2020, growth % "
                 "for korea, seoul is 3.1.")


def test_missing_title_and_row_path():
    t = BenchTable(table_id="t2", title="", data=[[7]],
                   top_paths=[["col a"]], left_paths=[[]])
    assert verbalize_cell(t, 0, 0, "medium") == "col a is 7."
    assert verbalize_cell(t, 0, 0, "long") == "col a is 7."


def test_verbalize_table_skips_empty_cells(hier_table):
    chunks = verbalize_table(hier_table, "short")
    # 6 cells, one None and one "" skipped -> 4
    assert len(chunks) == 4
    covered = {(c.rows[0], c.cols[0]) for c in chunks}
    assert (1, 1) not in covered and (2, 0) not in covered
    assert all(c.table_id == "t1" for c in chunks)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_unknown_style_raises(hier_table):
    with pytest.raises(ValueError):
        verbalize_cell(hier_table, 0, 0, "huge")
    assert set(STYLES) == {"short", "medium", "long"}
