# SPDX-License-Identifier: MIT
"""The cascade retrieves the right TABLE first, then cells inside it.

Stage 1 (TableIndex) must rank the topically-matching table above distractors,
and cascade_retrieve must then surface the gold cell from that table. Runs on a
synthetic 3-table corpus with the HashingEncoder — no model, no dataset — so it
guards the wiring anywhere the suite runs.
"""
import pytest

from rag_agent.bench.schema import GoldOperand
from rag_agent.eval.operand_set import operand_set_completeness
from rag_agent.retrieve.cascade import TableIndex, cascade_retrieve
from rag_agent.retrieve.encoders import HashingEncoder
from rag_agent.retrieve.operand_retrieval import OperandTargetedRetriever
from rag_agent.stores.original_store import OriginalTable


def _tbl(tid, title, top, left, data):
    return OriginalTable(table_id=tid, title=title, data=data,
                         top_paths=top, left_paths=left)


@pytest.fixture
def corpus():
    sales = _tbl("sales", "Company Sales Revenue",
                 [["Region", "North"], ["Region", "South"]],
                 [["Year", "2020"], ["Year", "2021"]],
                 [[100, 200], [150, 250]])
    weather = _tbl("weather", "City Weather",
                   [["Metric", "Temperature"], ["Metric", "Rainfall"]],
                   [["Month", "January"], ["Month", "February"]],
                   [[5, 80], [7, 60]])
    pop = _tbl("population", "Country Population Census",
               [["Sex", "Male"], ["Sex", "Female"]],
               [["Country", "Korea"], ["Country", "Japan"]],
               [[26, 25], [63, 64]])
    return {t.table_id: t for t in (sales, weather, pop)}


def test_stage1_ranks_matching_table_first(corpus):
    idx = TableIndex(HashingEncoder(dim=1024)).build(corpus)
    ranked = idx.search("What were the North region sales revenue in 2021?", m=3)
    assert [tid for tid, _ in ranked][0] == "sales"
    # the match beats the best distractor by a clear margin
    assert ranked[0][1] > ranked[1][1]


@pytest.mark.parametrize("embed_resolver", [False, True])
def test_cascade_surfaces_gold_cell_from_gold_table(corpus, embed_resolver):
    enc = HashingEncoder(dim=1024)
    idx = TableIndex(enc).build(corpus)
    engine = OperandTargetedRetriever(encoder=enc, scheme="S3",
                                      embed_resolver=embed_resolver)
    res = cascade_retrieve("What were the North region sales revenue in 2021?",
                           idx, corpus, engine, m=1, k=5, gold_table_id="sales")
    assert res.gold_table_hit is True
    assert res.table_ids == ["sales"]
    cells = [(rc.chunk.row_index, rc.chunk.col_index) for rc in res.retrieved
             if rc.chunk.table_id == "sales" and rc.chunk.row_index is not None]
    assert (1, 0) in cells  # Year>2021 x Region>North
    gold = [GoldOperand(row=1, col=0, header_path=["Year", "2021", "Region", "North"])]
    assert operand_set_completeness(gold, cells) == 1


def test_missing_gold_table_yields_zero_completeness(corpus):
    """If stage 1 keeps only a wrong table, OSC is 0 — no spurious coverage."""
    enc = HashingEncoder(dim=1024)
    idx = TableIndex(enc).build(corpus)
    engine = OperandTargetedRetriever(encoder=enc, scheme="S3")
    # force the wrong table by asking with the gold set to a table we won't pick
    res = cascade_retrieve("average January rainfall in the city",
                           idx, corpus, engine, m=1, k=5, gold_table_id="sales")
    cells = [(rc.chunk.row_index, rc.chunk.col_index) for rc in res.retrieved
             if rc.chunk.table_id == "sales" and rc.chunk.row_index is not None]
    gold = [GoldOperand(row=1, col=0, header_path=["Year", "2021", "Region", "North"])]
    assert res.gold_table_hit is False
    assert operand_set_completeness(gold, cells) == 0
