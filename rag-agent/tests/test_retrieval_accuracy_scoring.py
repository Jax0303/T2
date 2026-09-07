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


if __name__ == "__main__":
    test_cell_budget_is_counted_in_cells_not_units()
    test_pass_fail_is_all_or_any_by_mode()
    test_gold_shape_matches_the_scorer()
    print("ok")
