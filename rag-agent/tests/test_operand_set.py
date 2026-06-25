# SPDX-License-Identifier: MIT
"""Unit tests for operand-set completeness metrics (no dataset needed)."""
from rag_agent.eval.operand_set import (
    aggregate_by,
    bin_depth,
    bin_scope,
    covered_gold_cells,
    evaluate_query,
    header_depth,
    operand_set_completeness,
    per_cell_recall,
    scope_size,
    summarize,
)


def test_osc_all_or_nothing():
    gold = [(0, 0), (1, 2)]
    assert operand_set_completeness(gold, [(0, 0), (1, 2), (5, 5)]) == 1
    assert operand_set_completeness(gold, [(0, 0)]) == 0  # missing one operand -> 0
    assert operand_set_completeness(gold, []) == 0


def test_osc_empty_gold_is_vacuous():
    assert operand_set_completeness([], [(0, 0)]) == 1


def test_per_cell_recall():
    gold = [(0, 0), (1, 2), (3, 4)]
    assert per_cell_recall(gold, [(0, 0), (1, 2)]) == 2 / 3
    assert per_cell_recall(gold, []) == 0.0
    assert per_cell_recall([], [(0, 0)]) == 1.0


def test_dedup_and_order_invariant():
    gold = [(1, 1), (1, 1), (2, 2)]  # duplicate gold cell
    assert scope_size(gold) == 2
    assert operand_set_completeness(gold, [(2, 2), (1, 1)]) == 1


def test_accepts_objects_with_row_col():
    class C:
        def __init__(self, r, c):
            self.row, self.col = r, c

    gold = [C(0, 0), C(1, 1)]
    assert operand_set_completeness(gold, [C(0, 0), C(1, 1)]) == 1
    assert per_cell_recall(gold, [(0, 0)]) == 0.5


def test_header_depth():
    assert header_depth([["a"], ["a", "b"]], [[], ["x"]]) == 2
    assert header_depth([["a"]], [[]]) == 1
    assert header_depth([], []) == 0


def test_bins():
    assert bin_scope(1) == "1"
    assert bin_scope(2) == "2"
    assert bin_scope(7) == "5-8"
    assert bin_scope(12) == "9+"
    assert bin_depth(1) == "flat(d<=1)"
    assert bin_depth(2) == "d2"
    assert bin_depth(3) == "d3+"


def test_evaluate_and_aggregate():
    recs = [
        evaluate_query([(0, 0)], [(0, 0)], header_depth_val=1, aggregation="sum"),
        evaluate_query([(0, 0), (1, 1)], [(0, 0)], header_depth_val=2, aggregation="sum"),
    ]
    assert recs[0]["osc"] == 1 and recs[0]["scope_size"] == 1
    assert recs[1]["osc"] == 0 and recs[1]["per_cell_recall"] == 0.5

    by = aggregate_by(recs, lambda r: r["aggregation"])
    assert by["sum"]["n"] == 2
    assert by["sum"]["osc"] == 0.5

    s = summarize(recs)
    assert s["overall"]["n"] == 2
    assert s["overall"]["osc"] == 0.5
    assert "by_scope" in s and "by_depth" in s and "by_aggregation" in s


def test_summarize_empty():
    s = summarize([])
    assert s["overall"]["n"] == 0


def test_covered_gold_cells_bridges_chunks():
    class Op:
        def __init__(self, r, c):
            self.row, self.col = r, c

    class Chunk:
        def __init__(self, rows, cols):
            self.rows, self.cols = rows, cols

        def covers(self, r, c):
            return r in self.rows and c in self.cols

    gold = [Op(0, 0), Op(1, 1)]
    chunks = [Chunk([0], [0])]  # covers (0,0) only
    covered = covered_gold_cells(gold, chunks)
    assert covered == [(0, 0)]
    assert operand_set_completeness(gold, covered) == 0  # (1,1) missing
    assert per_cell_recall(gold, covered) == 0.5
    chunks.append(Chunk([1], [1]))  # now covers (1,1) too
    assert operand_set_completeness(gold, covered_gold_cells(gold, chunks)) == 1
