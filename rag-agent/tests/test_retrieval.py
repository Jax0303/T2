"""Tests for component 3: hybrid index + operand-targeted retrieval.

All deterministic via the NumPy HashingEncoder fallback (no torch / model
download). A guarded HiTab integration test runs only if the dataset is present.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from rag_agent.serialization import header_path as s2  # noqa: E402
from rag_agent.retrieve.encoders import HashingEncoder  # noqa: E402
from rag_agent.retrieve.hybrid_index import HybridIndex  # noqa: E402
from rag_agent.retrieve.operand_retrieval import (  # noqa: E402
    operand_recall_at_k,
    gold_operands_from_hitab,
)


class FakeTable:
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


def _enc():
    return HashingEncoder(dim=256)


def test_hybrid_index_finds_lexically_matching_cell():
    chunks = s2.serialize(FakeTable(), granularity="cell")
    idx = HybridIndex(chunks, encoder=_enc(), alpha=0.5)
    hits = idx.search("Revenue 2023 Q1", k=1)
    assert hits
    assert "Revenue > 2023 > Q1: 1234" in hits[0].chunk.text


def test_hybrid_alpha_endpoints_are_pure_backends():
    chunks = s2.serialize(FakeTable(), granularity="cell")
    bm_only = HybridIndex(chunks, encoder=_enc(), alpha=0.0)
    dn_only = HybridIndex(chunks, encoder=_enc(), alpha=1.0)
    # both should still rank the exact match first
    assert "Q1" in bm_only.search("Revenue 2023 Q1", k=1)[0].chunk.text
    assert "Q1" in dn_only.search("Revenue 2023 Q1", k=1)[0].chunk.text


def test_empty_index_returns_empty():
    idx = HybridIndex([], encoder=_enc())
    assert idx.search("anything", k=5) == []


def test_operand_recall_counts_covered_paths():
    chunks = s2.serialize(FakeTable(), granularity="cell")
    idx = HybridIndex(chunks, encoder=_enc(), alpha=0.5)
    hits = idx.search("Revenue 2023 Q1", k=4)
    gold = [["Revenue", "2023", "Q1"]]
    assert operand_recall_at_k(gold, hits, k=4) == 1.0
    # an operand that exists in no cell is never covered
    assert operand_recall_at_k([["Profit", "2099"]], hits, k=4) == 0.0


def test_operand_recall_empty_gold_is_zero():
    assert operand_recall_at_k([], [], k=5) == 0.0


def test_gold_operands_from_hitab_parses_entity_link():
    sample = {
        "linked_cells": {
            "entity_link": {
                "top": {"the fy 2017 budget": {"(0, 1)": "2017 actual"}},
                "left": {"pre-production": {"(18, 0)": "total"}},
                "top_left_corner": {},
            }
        }
    }
    gold = gold_operands_from_hitab(sample)
    assert gold == [["total", "2017 actual"]]


def test_gold_operands_single_axis():
    sample = {"linked_cells": {"entity_link": {"top": {"q": {"(0,1)": "revenue"}}, "left": {}}}}
    assert gold_operands_from_hitab(sample) == [["revenue"]]


def _hitab_available():
    from rag_agent.data.loader import _find_data_root
    try:
        _find_data_root("data/hitab")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _hitab_available(), reason="HiTab data not downloaded")
def test_operand_recall_monotonic_in_k_on_hitab():
    from rag_agent.data.loader import load_hitab
    from rag_agent.serialization import from_hitab_raw
    from rag_agent.retrieve.operand_retrieval import OperandTargetedRetriever

    samples = load_hitab(split="dev", max_samples=15)
    r = OperandTargetedRetriever(encoder=HashingEncoder(dim=512), alpha=0.5)
    means = {}
    for kk in (1, 5, 10):
        vals = []
        for s in samples:
            gold = gold_operands_from_hitab(s)
            if not gold:
                continue
            t = from_hitab_raw(s["table"])
            res = r.retrieve(s["question"], t, k=kk)
            vals.append(operand_recall_at_k(gold, res.retrieved, k=None))
        means[kk] = sum(vals) / len(vals) if vals else 0.0
    # recall should not decrease as k grows
    assert means[1] <= means[5] <= means[10]
