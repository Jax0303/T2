# SPDX-License-Identifier: MIT
"""The mechanical error check behind PREREG-2026-09-14-header-v3.md 정정 3 (synthetic inputs only)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from analysis.header_confirmed_errors import compare, judge, own_total, words  # noqa: E402


class _Table:
    def __init__(self, grid, paths):
        self.grid, self.nhr, self.nhc, self._paths = grid, 1, 1, paths

    def row_path(self, i):
        return self._paths[i]


def test_only_rows_between_a_heading_and_its_own_total_are_judged():
    grid = [["", "2019"], ["Deposits:", ""], ["Demand", "1"], ["Total core deposits", "1"],
            ["Brokered", "2"], ["Total deposits", "3"], ["Debt", "4"]]
    paths = [["Deposits:"], ["Deposits:", "Demand"], ["Deposits:", "Total core deposits"], ["Brokered"],
             ["Deposits:", "Total deposits"], ["Debt"]]
    j = judge(_Table(grid, paths))
    assert j[4][0] == {"Deposits:"}
    assert not j[2][0] and j[2][2]
    assert not j[6][0] and not j[6][2]


def test_own_total_is_the_heading_word_for_word():
    assert own_total("Total deposits", words("Deposits:"))
    assert own_total("Sub-total Deposits", words("Deposits:"))
    assert not own_total("Total core deposits", words("Deposits:"))


def test_a_heading_lost_on_a_row_that_already_had_a_v2_error_is_rule_made():
    # counted per (row, heading): the row-level count put the second loss under "v2 부터 있음"
    base = {"t": {5: (frozenset({"A"}), frozenset(), True, ("x",))}}
    new = {"t": {5: (frozenset({"A", "B"}), frozenset(), True, ("y",))}}
    res, keys = compare(base, new)
    assert keys["E"]["v2_and_rule"] == [("t", 5, "A")]
    assert keys["E"]["rule_made"] == [("t", 5, "B")]
    assert res["E"]["rule_made"]["rows"] == res["E"]["v2_and_rule"]["rows"] == 1
