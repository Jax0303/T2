# SPDX-License-Identifier: MIT
"""Header rule v3.1 — v3 narrowed to the side effects v3 was measured to have
(PREREG-2026-09-14-header-v3.md, 정정 1).

Synthetic tables only: each test pins a side effect by its shape and checks that v3 still
does what it did (v3's results stay reproducible) while v3.1 does not.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from rag_agent.reconstruct.header_grid import (corner_scope, guess_n_header_rows,  # noqa: E402
                                               parse_html_table_layout, reconstruct_col_paths,
                                               reconstruct_row_paths)


def _table(*rows):
    return "<table>" + "".join(f"<tr>{r}</tr>" for r in rows) + "</table>"


# S1n -----------------------------------------------------------------------

def test_a_label_that_declares_a_span_still_stops_at_its_edge():
    html = _table('<td>year</td><td></td><td rowspan="2">Group A</td><td></td>',
                  "<td>2008</td><td>Alpha</td><td>Total</td>",
                  "<td>Revenue</td><td>1</td><td>2</td><td>3</td>")
    grid, cover = parse_html_table_layout(html)
    assert reconstruct_col_paths(grid, 2, 1, cover=cover, narrow=True) == [["Alpha"], ["Group A"], ["Total"]]


def test_a_one_by_one_label_keeps_the_v2_carry():
    # the author spanned "2013" but not "2014": a 1x1 cell declares no extent
    html = _table('<td></td><td>2014</td><td></td><td colspan="2">2013</td>',
                  "<td></td><td>$</td><td>%</td><td>$</td><td>%</td>",
                  "<td>Sales</td><td>1</td><td>2</td><td>3</td><td>4</td>")
    grid, cover = parse_html_table_layout(html)
    assert reconstruct_col_paths(grid, 2, 1, cover=cover)[1] == ["%"]                      # v3
    assert reconstruct_col_paths(grid, 2, 1, cover=cover, narrow=True) == \
        reconstruct_col_paths(grid, 2, 1)                                                  # v3.1 == v2


def test_covered_positions_are_not_filled():
    html = _table('<td></td><td colspan="2">2013</td><td colspan="2">2012</td>',
                  '<td></td><td colspan="4">(In millions)</td>',
                  "<td></td><td>A</td><td>B</td><td>A</td><td>B</td>",
                  "<td>Sales</td><td>1</td><td>2</td><td>3</td><td>4</td>")
    grid, cover = parse_html_table_layout(html)
    assert reconstruct_col_paths(grid, 3, 1, cover=cover)[2] == ["2012", "(In millions)", "A"]  # v3
    assert reconstruct_col_paths(grid, 3, 1, cover=cover, narrow=True) == reconstruct_col_paths(grid, 3, 1)


def test_a_cut_that_would_delete_the_columns_year_falls_back_to_v2():
    dated = _table('<td></td><td colspan="2">2014</td><td></td>',
                   "<td></td><td>$</td><td>%</td><td>Change</td>",
                   "<td>Sales</td><td>1</td><td>2</td><td>3</td>")
    grid, cover = parse_html_table_layout(dated)
    assert reconstruct_col_paths(grid, 2, 1, cover=cover)[2] == ["Change"]                # v3
    assert reconstruct_col_paths(grid, 2, 1, cover=cover, narrow=True)[2] == ["2014", "Change"]
    undated = dated.replace(">2014<", ">Years Ended<")
    grid, cover = parse_html_table_layout(undated)
    assert reconstruct_col_paths(grid, 2, 1, cover=cover, narrow=True)[2] == ["Change"]


def test_cutting_a_span_keeps_a_one_by_one_label_that_starts_after_it():
    # 정정 2: the cut used to clear every deeper level, so "Change" vanished from the % column
    html = _table('<td></td><td colspan="2">Years Ended</td><td></td><td></td>',
                  "<td></td><td>2015</td><td>2014</td><td>Change</td><td></td>",
                  "<td>Fees</td><td>1</td><td>2</td><td>3</td><td>25%</td>")
    grid, cover = parse_html_table_layout(html)
    assert reconstruct_col_paths(grid, 2, 1)[3] == ["Years Ended", "Change"]
    assert reconstruct_col_paths(grid, 2, 1, cover=cover, narrow=True)[2:] == [["Change"], ["Change"]]


def test_cutting_a_span_also_cuts_the_labels_that_started_inside_it():
    html = _table('<td></td><td colspan="2">Segment A</td><td></td>',
                  "<td></td><td>Rev</td><td>Inc</td><td></td>",
                  "<td>Fees</td><td>1</td><td>2</td><td>3</td>")
    grid, cover = parse_html_table_layout(html)
    assert reconstruct_col_paths(grid, 2, 1)[2] == ["Segment A", "Inc"]
    assert reconstruct_col_paths(grid, 2, 1, cover=cover, narrow=True)[2] == []


# S2n -----------------------------------------------------------------------

def test_only_a_lone_blank_label_row_loses_the_inherited_label():
    html = _table("<td></td><td>2013</td><td>2012</td>",
                  "<td>Drilling</td><td>17</td><td>15</td>",
                  "<td></td><td>45</td><td>41</td>",
                  "<td>Rating D</td><td>1</td><td>2</td>",
                  "<td></td><td>3</td><td>4</td>",
                  "<td></td><td>5</td><td>6</td>")
    grid, cover = parse_html_table_layout(html)
    v3 = reconstruct_row_paths(grid, 1, 1, cover=cover)
    assert v3 == [["Drilling"], [], ["Rating D"], [], []]
    v31 = reconstruct_row_paths(grid, 1, 1, cover=cover, isolated_blank_only=True)
    assert v31 == [["Drilling"], [], ["Rating D"], ["Rating D"], ["Rating D"]]


# S3n -----------------------------------------------------------------------

def test_signed_currency_rows_are_data_not_labels():
    grid = [["", "2014", "2013"],
            ["Financing costs", "$-0.11", "$ -0.10"],
            ["Other", "1", "2"]]
    assert guess_n_header_rows(grid, n_header_cols=1, rule="v3") == 2
    assert guess_n_header_rows(grid, n_header_cols=1, rule="v3.1") == 1


def test_v3_1_still_takes_real_label_rows():
    for row in (["Description", "Dec 31, 2011", "Dec 31, 2010"],
                ["Year ended", "1998", "1999(a)"],
                ["Contractual Obligations", "Total", "2012 – 2013"]):
        grid = [["", "Schedule", ""], row, ["Reserves", "$28,506", "$26,850"]]
        assert guess_n_header_rows(grid, n_header_cols=1, rule="v3.1") == 2, row


# S4n -----------------------------------------------------------------------

def test_a_total_closes_only_the_section_it_names():
    grid = [["", "2009"],
            ["ASSETS", ""],
            ["Investments", "5"],
            ["Total investments", "5"],
            ["Cash", "1"],
            ["TOTAL ASSETS", "6"],
            ["Debt", "2"]]
    v3 = reconstruct_row_paths(grid, 1, 1, close_on_total=True)
    assert v3[3] == ["Cash"]
    v31 = reconstruct_row_paths(grid, 1, 1, close_on_total="named")
    assert v31[3] == ["ASSETS", "Cash"]
    assert v31[5] == ["Debt"]


def test_bare_and_plural_totals_close():
    grid = [["", "2009"],
            ["Revenues:", ""],
            ["Product", "1"],
            ["Total revenue", "1"],
            ["Costs", "2"],
            ["Alternatives", ""],
            ["Core", "3"],
            ["Subtotal", "3"],
            ["Long-term", "4"]]
    v31 = reconstruct_row_paths(grid, 1, 1, close_on_total="named")
    assert v31[3] == ["Costs"] and v31[7] == ["Long-term"]


# S5n -----------------------------------------------------------------------

def test_a_closed_heading_is_not_put_back():
    grid = [["", "2015"],
            ["Fee revenue:", ""],
            ["Servicing", "1"],
            ["Total fee revenue", "5"],
            ["Other", "2"],
            ["Expenses:", ""],
            ["Servicing", "3"],
            ["Total expenses", "4"],
            ["Other", "6"]]
    v3 = reconstruct_row_paths(grid, 1, 1, close_on_total=True, disambiguate=True)
    assert v3[3] == ["Fee revenue:", "Other"]
    v31 = reconstruct_row_paths(grid, 1, 1, close_on_total="named", disambiguate="guarded")
    assert v31[3] == v31[7] == ["Other"]


def test_a_heading_with_another_year_is_not_added():
    # data between the headings, so v2 replaces them instead of nesting and the two paths collide
    grid = [["", "x"],
            ["Liabilities", ""],
            ["Loans 2017", "1"],
            ["As of 2016", ""],
            ["Deposits", "2"],
            ["Liabilities", ""],
            ["Loans 2017", "3"]]
    v2 = reconstruct_row_paths(grid, 1, 1)
    assert v2[1] == v2[5] == ["Liabilities", "Loans 2017"]
    v3 = reconstruct_row_paths(grid, 1, 1, disambiguate=True)
    assert v3[5] == ["As of 2016", "Liabilities", "Loans 2017"]
    v31 = reconstruct_row_paths(grid, 1, 1, disambiguate="guarded")
    assert v31[5] == ["Liabilities", "Loans 2017"]


# S6n -----------------------------------------------------------------------

def test_corner_scope_needs_undated_rows_and_a_short_corner():
    grid = [["As of December 31, 2008", "Amount"], ["Program 2012", "5"]]
    cols, rows = reconstruct_col_paths(grid, 1, 1), reconstruct_row_paths(grid, 1, 1)
    assert corner_scope(grid, 1, 1, cols) == "As of December 31, 2008"                   # v3
    assert corner_scope(grid, 1, 1, cols, row_paths=rows, max_len=60) == ""
    long = [["(In millions) 2007 Revenues and other income: consolidated segment totals", "Amount"],
            ["Program", "5"]]
    assert corner_scope(long, 1, 1, reconstruct_col_paths(long, 1, 1),
                        row_paths=reconstruct_row_paths(long, 1, 1), max_len=60) == ""


# end to end ---------------------------------------------------------------------

def test_build_tables_v3_1_keeps_the_case_fixes():
    from mh_arms import build_tables
    html = _table('<td>For the Year Ended December 31, 2008</td><td colspan="2">Segments</td>'
                  '<td rowspan="2">Other</td><td></td>',
                  "<td></td><td>Alpha</td><td>Beta</td><td>Total</td>",
                  "<td>Revenue</td><td>1</td><td>2</td><td>3</td><td>6</td>",
                  "<td>Costs</td><td>1</td><td>1</td><td>1</td><td>3</td>",
                  "<td></td><td>0</td><td>1</td><td>2</td><td>3</td>")
    docs = {"u": ([html], "{}", [])}
    t = build_tables(docs, "v3.1")[0]["u::0"].table
    scope = "For the Year Ended December 31, 2008"
    assert t.col_path(1) == [scope, "Segments", "Beta"]
    assert t.col_path(3) == [scope, "Total"]
    assert t.row_path(2) == []
