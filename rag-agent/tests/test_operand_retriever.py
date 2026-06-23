# SPDX-License-Identifier: MIT
"""Data-free unit tests for operand-targeted retrieval + operand_recall."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.bench.schema import BenchTable, GoldOperand
from rag_agent.serialize import serialize_table, S2
from rag_agent.retrieve.operand_retriever import (
    HybridRetriever, retrieve, operand_recall,
)


def _toy() -> BenchTable:
    # 3 rows so retrieval has something to discriminate
    return BenchTable(
        table_id="t1", title="Holdings",
        data=[[10, 11], [20, 21], [30, 31]],
        top_paths=[["year", "2022"], ["year", "2023"]],
        left_paths=[["assets", "cash"], ["assets", "bonds"], ["assets", "stocks"]],
        source="toy",
    )


def test_bm25_retriever_ranks_matching_row():
    t = _toy()
    r = HybridRetriever(serialize_table(t, S2), embedder=None)
    idx = r.search("bonds 2023", k=1)
    assert r.chunks[idx[0]].rows == [1]  # the bonds row


def test_operand_recall_perfect_when_row_retrieved():
    t = _toy()
    gold = [GoldOperand(row=1, col=1, header_path=["assets", "bonds", "year", "2023"], value=21)]
    res = retrieve("bonds in 2023", t, gold, mode="operand", k=3, matcher="fuzzy")
    assert operand_recall(res.retrieved, gold) == 1.0


def test_oracle_uses_gold_paths():
    t = _toy()
    gold = [GoldOperand(row=2, col=0, header_path=["assets", "stocks", "year", "2022"], value=30)]
    res = retrieve("irrelevant text", t, gold, mode="oracle", k=2, matcher="fuzzy")
    assert res.mode == "oracle"
    assert operand_recall(res.retrieved, gold) == 1.0


def test_recall_none_without_gold():
    assert operand_recall([], []) is None
