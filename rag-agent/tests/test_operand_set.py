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


# --- rank-based OSC@k -------------------------------------------------------

def test_set_recall_at_k_all_or_nothing():
    from rag_agent.eval.operand_set import set_recall_at_k
    assert set_recall_at_k([1, 5, 10], 10) == 1
    assert set_recall_at_k([1, 5, 11], 10) == 0          # one miss kills the set
    assert set_recall_at_k([1, None], 50) == 0           # never retrieved
    assert set_recall_at_k({7: 3, 9: 50}, 50) == 1       # mapping (cell -> rank)
    assert set_recall_at_k([], 10) == 1                  # vacuous, W1 convention


def test_coverage_at_k_partial():
    from rag_agent.eval.operand_set import coverage_at_k
    assert abs(coverage_at_k([1, 5, 11], 10) - 2 / 3) < 1e-9
    assert coverage_at_k([None, None], 10) == 0.0
    assert coverage_at_k({1: 2, 4: 3}, 10) == 1.0


def test_osc_at_k_summary_counts_partial_queries():
    from rag_agent.eval.operand_set import osc_at_k_summary
    pop = [[1, 2], [1, 40], [None, None]]
    s = osc_at_k_summary(pop, ks=(10,))
    assert s["n_queries"] == 3
    assert s["set_recall@10"] == round(1 / 3, 4)
    assert s["coverage@10"] == round((1.0 + 0.5 + 0.0) / 3, 4)
    assert s["n_partial@10"] == 1                        # only the [1, 40] query


def test_paired_set_recall_flip_binomial():
    from rag_agent.eval.operand_set import paired_set_recall_flip
    a = [[1, 99], [1, 2], [None]]                        # covered@10: 0,1,0
    b = [[1, 2], [1, 2], [3]]                            # covered@10: 1,1,1
    r = paired_set_recall_flip(a, b, k=10)
    assert (r["gain"], r["loss"]) == (2, 0)
    assert abs(r["p_two_sided"] - 0.5) < 1e-9            # 2 flips, both gains
    r0 = paired_set_recall_flip([[1]], [[2]], k=10)
    assert r0["p_two_sided"] is None and r0["gain"] == r0["loss"] == 0
    try:
        paired_set_recall_flip(a, b[:2], k=10)
        assert False, "unaligned populations must raise"
    except ValueError:
        pass
