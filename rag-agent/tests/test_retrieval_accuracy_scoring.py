# SPDX-License-Identifier: MIT
"""The two rules Table 1 rests on: a fixed CELL budget, and pass/fail per query.

A row- or table-level index unit delivers many cells at once, so "top-20 units"
would hand those arms a far bigger context than the cell index gets and the
comparison would measure context size instead of ranking. The budget is
therefore counted in cells. And a query is scored PASS/FAIL, never partially:
`all` needs every gold cell, `any` needs one of the scope.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def fill(covers, order, budget):
    """The loop scripts/retrieval_accuracy.py uses, isolated."""
    got, n = set(), 0
    for p in order:
        if n >= budget:
            break
        got |= covers[p]
        n += len(covers[p])
    return got, n


def test_cell_budget_is_counted_in_cells_not_units():
    cell = [frozenset([("t", 0, j)]) for j in range(50)]
    got, n = fill(cell, range(50), 20)
    assert n == 20 and len(got) == 20

    # one row unit carrying 8 cells: three rows reach the budget, not twenty
    row = [frozenset(("t", i, j) for j in range(8)) for i in range(50)]
    got, n = fill(row, range(50), 20)
    assert n == 24 and len(got) == 24, (n, len(got))   # last unit may overshoot


def test_pass_fail_is_all_or_any_by_mode():
    got = {("t", 0, 0), ("t", 0, 1)}
    assert {("t", 0, 0)} <= got                          # all: subset holds
    assert not {("t", 0, 0), ("t", 9, 9)} <= got         # all: one missing -> fail
    assert {("t", 9, 9), ("t", 0, 1)} & got              # any: one is enough
    assert not {("t", 9, 9)} & got


def test_gold_shape_matches_the_scorer():
    """gold_target hands back exactly what the two rules above consume."""
    from rag_agent.bench import hitab_grid as hg
    from rag_agent.data.loader import load_samples
    s = load_samples(str(ROOT / "data/hitab"), "test")[0]
    tab = hg.load_table(s["table_id"], str(ROOT / "data/hitab"))
    cells, mode, why = hg.gold_target(s, tab)
    assert mode in ("all", "any")
    assert why is None or isinstance(why, str)
    assert all(isinstance(c, tuple) and len(c) == 3 for c in cells)


def test_type_accuracy_needs_every_gold_cell():
    """세 유형 표: gold 일부만 찾으면 FAIL, 제외는 분모 밖, macro 는 유형 정확도 평균."""
    from retrieval_accuracy import query_type, type_accuracy, type_row
    A, B, C, X, Y = (("t", 0, j) for j in range(5))

    def q(qid, gold, agg="none", mode="all", why=None):
        return {"query_id": qid, "gold": set(gold), "aggregation": agg,
                "mode": mode, "excluded": why}

    rows = [type_row(q("s1", [A]), {A, X}, 20),
            type_row(q("s2", [A]), {X}, 20),
            type_row(q("m1", [A, B, C]), {A, B, C, X}, 20),
            type_row(q("m2", [A, B, C]), {A, B, X, Y}, 20),        # 2/3 -> FAIL
            type_row(q("a1", [A, B], "diff"), {A, B, X}, 2),       # last unit overshoots
            type_row(q("e1", [A, B], why="gold_is_header_cell"), None, 20)]
    assert [r["query_type"] for r in rows[:5]] == ["single_cell"] * 2 + ["multi_cell"] * 2 + ["arithmetic"]
    assert query_type(q("h", [A, B], "argmax", mode="any")) == "header_answer"
    assert rows[3]["retrieval_success"] == 0 and rows[3]["evidence_recall"] == 0.6667
    assert (rows[3]["gold_cell_ids"], rows[3]["num_gold_cells"], rows[3]["num_gold_retrieved"],
            rows[3]["num_retrieved_cells"]) == (sorted([A, B, C]), 3, 2, 4)
    assert rows[4]["over_budget"] and rows[4]["retrieval_success"] == 1
    assert rows[5]["retrieval_success"] is None and rows[5]["unresolved_gold_reason"]

    ta = type_accuracy(rows)
    assert (ta["single_cell"]["n"], ta["single_cell"]["accuracy"]) == (2, 0.5)
    assert ta["multi_cell"]["mean_evidence_recall_DIAGNOSTIC"] == 0.8333
    assert (ta["arithmetic"]["n"], ta["arithmetic"]["accuracy"]) == (1, 1.0)
    o = ta["overall"]
    assert (o["n"], o["success"], o["accuracy"], o["macro_accuracy"]) == (5, 3, 0.6, 0.6667)
    assert (ta["n_excluded"], o["over_budget"]) == (1, 1)


if __name__ == "__main__":
    test_cell_budget_is_counted_in_cells_not_units()
    test_pass_fail_is_all_or_any_by_mode()
    test_gold_shape_matches_the_scorer()
    test_type_accuracy_needs_every_gold_cell()
    print("ok")
