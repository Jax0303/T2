# SPDX-License-Identifier: MIT
"""Header rule v3.2 — v3.1 minus the two error types a mechanical check found v3.1 had made
and v2 had not (PREREG-2026-09-14-header-v3.md, 정정 3).

Synthetic tables only. Each test also pins what v3.1 does, so v3.1's results stay reproducible.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from rag_agent.reconstruct.header_grid import reconstruct_row_paths  # noqa: E402


def _table(*rows):
    return "<table>" + "".join(f"<tr>{r}</tr>" for r in rows) + "</table>"


# S4n2 ----------------------------------------------------------------------

def test_a_named_sub_total_does_not_close_a_section_whose_own_total_is_still_below():
    grid = [["", "2019"],
            ["Deposits:", ""],
            ["Demand", "1"],
            ["Total core deposits", "1"],
            ["Brokered", "2"],
            ["Total deposits", "3"],
            ["Debt", "4"]]
    assert reconstruct_row_paths(grid, 1, 1, close_on_total="named")[3] == ["Brokered"]      # v3.1
    v32 = reconstruct_row_paths(grid, 1, 1, close_on_total="own")
    assert v32[3] == ["Deposits:", "Brokered"]
    assert v32[4] == ["Deposits:", "Total deposits"]
    assert v32[5] == ["Debt"]                           # the own total still closes


def test_a_bare_total_does_not_close_it_either():
    grid = [["", "2019"],
            ["Financial Services Businesses:", ""],
            ["Gains", "1"],
            ["Total", "1"],
            ["Divested businesses", "2"],
            ["Total Financial Services Businesses", "3"],
            ["Closed Block", "4"]]
    assert reconstruct_row_paths(grid, 1, 1, close_on_total="named")[3] == ["Divested businesses"]
    v32 = reconstruct_row_paths(grid, 1, 1, close_on_total="own")
    assert v32[3] == ["Financial Services Businesses:", "Divested businesses"]
    assert v32[5] == ["Closed Block"]


def test_the_own_total_of_the_same_heading_further_down_does_not_count():
    # the heading comes back before any "Total segment A": that total belongs to the second section
    grid = [["", "2015"],
            ["Segment A:", ""],
            ["Services", "1"],
            ["Total segment A revenue", "1"],
            ["Other", "2"],
            ["Segment A:", ""],
            ["Services", "3"],
            ["Total segment A", "3"]]
    assert reconstruct_row_paths(grid, 1, 1, close_on_total="own")[3] == ["Other"]
    assert reconstruct_row_paths(grid, 1, 1, close_on_total="named")[3] == ["Other"]


# S5n2 ----------------------------------------------------------------------

def test_a_number_is_not_a_heading_that_tells_rows_apart():
    grid = [["", "2015"],
            ["Plans:", ""],
            ["Expense", "1"],
            ["", "544,823"],
            ["Other", "2"],
            ["Plans:", ""],
            ["Expense", "3"]]
    assert reconstruct_row_paths(grid, 1, 1, disambiguate="guarded")[5] == ["544,823", "Plans:", "Expense"]
    assert reconstruct_row_paths(grid, 1, 1, disambiguate="no_numbers")[5] == ["Plans:", "Expense"]
    grid[3][1] = "2014"                                 # a year is still a heading
    assert reconstruct_row_paths(grid, 1, 1, disambiguate="no_numbers")[5] == ["2014", "Plans:", "Expense"]


# end to end ---------------------------------------------------------------------

def test_build_tables_v3_2_is_v3_1_with_the_two_rules_replaced():
    from mh_arms import V31_RULES, V32_RULES, build_tables
    assert [r for r in V32_RULES if r not in V31_RULES] == ["S4n2", "S5n2"]
    assert len(V32_RULES) == len(V31_RULES)
    html = _table("<td></td><td>2019</td><td>2018</td>",
                  "<td>Deposits:</td><td></td><td></td>",
                  "<td>Demand</td><td>1</td><td>2</td>",
                  "<td>Total core deposits</td><td>1</td><td>2</td>",
                  "<td>Brokered</td><td>2</td><td>3</td>",
                  "<td>Total deposits</td><td>3</td><td>5</td>")
    docs = {"u": ([html], "{}", [])}
    t31 = build_tables(docs, "v3.1")[0]["u::0"].table
    t32 = build_tables(docs, "v3.2")[0]["u::0"].table
    assert (t31.nhr, t31.nhc) == (t32.nhr, t32.nhc) == (1, 1)
    assert t31.row_path(3) == ["Brokered"]
    assert t32.row_path(3) == ["Deposits:", "Brokered"]
