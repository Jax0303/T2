# SPDX-License-Identifier: MIT
"""Data-free unit tests for the unified schema + S1/S2 serializers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.bench.schema import BenchTable, Chunk
from rag_agent.serialize import serialize_table, fulltable_chunk, S1, S2


def _toy() -> BenchTable:
    # 2x2 table: rows = assets>cash / assets>bonds, cols = 2022 / 2023
    return BenchTable(
        table_id="t1",
        title="Holdings",
        data=[[10, 11], [20, 21]],
        top_paths=[["year", "2022"], ["year", "2023"]],
        left_paths=[["assets", "cash"], ["assets", "bonds"]],
        source="toy",
    )


def test_dims_and_paths():
    t = _toy()
    assert t.n_rows == 2 and t.n_cols == 2
    assert t.full_path(0, 1) == ["assets", "cash", "year", "2023"]


def test_s1_is_flat_leaf_only():
    t = _toy()
    chunks = serialize_table(t, S1)
    assert len(chunks) == 2
    txt = chunks[0].text
    # S1 keeps leaf headers only — no ancestor path separators
    assert "2022: 10" in txt and "2023: 11" in txt
    assert "year >" not in txt and "assets >" not in txt


def test_s2_has_full_header_path():
    t = _toy()
    chunks = serialize_table(t, S2)
    assert "assets > cash > year > 2022: 10" in chunks[0].text
    assert "assets > bonds > year > 2023: 21" in chunks[1].text


def test_chunk_coverage_mapping():
    t = _toy()
    chunks = serialize_table(t, S2)
    # row chunk r covers every cell in that row, no other row
    assert chunks[0].covers(0, 0) and chunks[0].covers(0, 1)
    assert not chunks[0].covers(1, 0)


def test_fulltable_chunk_covers_everything():
    t = _toy()
    full = fulltable_chunk(t, S2)
    assert isinstance(full, Chunk)
    assert all(full.covers(r, c) for r in range(2) for c in range(2))
    assert full.chunk_id.endswith("#full")
