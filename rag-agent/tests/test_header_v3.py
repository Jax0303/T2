# SPDX-License-Identifier: MIT
"""Header rule v3 — restore the source table's own correspondence (PREREG-2026-09-14-header-v3.md).

Every table here is synthetic. Each test pins one conversion defect by its SHAPE, not by a
question, an answer or a dataset id, and checks that v2 still produces what it produced.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from rag_agent.reconstruct.header_grid import (corner_scope, guess_n_header_rows,  # noqa: E402
                                               parse_html_table, parse_html_table_layout,
                                               reconstruct_col_paths, reconstruct_row_paths)


def _table(*rows):
    return "<table>" + "".join(f"<tr>{r}</tr>" for r in rows) + "</table>"


def test_layout_parser_returns_the_same_grid():
    html = _table('<td></td><td colspan="2">2013</td><td rowspan="2">Other</td>',
                  "<td>x</td><td>a</td><td>b</td>",
                  "<td>r</td><td>1</td><td>2</td><td>3</td>")
    grid, cover = parse_html_table_layout(html)
    assert grid == parse_html_table(html)
    assert cover[0][2] == (0, 1)          # colspan-covered
    assert cover[1][3] == (0, 3)          # rowspan-covered
    assert cover[1][1] == (1, 1)          # its own <td>


# S1 -----------------------------------------------------------------------

def test_explicit_blank_in_a_marked_header_row_does_not_inherit_the_left_label():
    html = _table('<td>year</td><td></td><td rowspan="2">Group A</td><td></td>',
                  "<td>2008</td><td>Alpha</td><td>Total</td>",
                  "<td>Revenue</td><td>1</td><td>2</td><td>3</td>")
    grid, cover = parse_html_table_layout(html)
    assert reconstruct_col_paths(grid, 2, 1) == [["Alpha"], ["Group A"], ["Group A", "Total"]]
    assert reconstruct_col_paths(grid, 2, 1, cover=cover) == [["Alpha"], ["Group A"], ["Total"]]


def test_colspan_covers_exactly_its_columns():
    html = _table('<td></td><td colspan="2">2013</td><td>Change</td>',
                  "<td></td><td>Rev</td><td>Inc</td><td>%</td>",
                  "<td>Drilling</td><td>1</td><td>2</td><td>3</td>")
    grid, cover = parse_html_table_layout(html)
    assert reconstruct_col_paths(grid, 2, 1, cover=cover) == [["2013", "Rev"], ["2013", "Inc"],
                                                              ["Change", "%"]]


def test_an_empty_colspan_does_not_carry_the_label_before_it():
    html = _table('<td></td><td>2013</td><td colspan="2"></td>',
                  "<td></td><td>Rev</td><td>Inc</td><td>Cost</td>",
                  "<td>Drilling</td><td>1</td><td>2</td><td>3</td>")
    grid, cover = parse_html_table_layout(html)
    assert reconstruct_col_paths(grid, 2, 1, cover=cover) == [["2013", "Rev"], ["Inc"], ["Cost"]]


def test_rows_without_markup_and_padded_cells_keep_the_v2_carry():
    # no span markup anywhere in the header row: blanks may be visual spans, so v2 stands
    plain = _table("<td></td><td>Years ended</td><td></td><td></td>",
                   "<td></td><td>2013</td><td>2012</td><td>2011</td>",
                   "<td>Sales</td><td>1</td><td>2</td><td>3</td>")
    grid, cover = parse_html_table_layout(plain)
    assert reconstruct_col_paths(grid, 2, 1, cover=cover) == reconstruct_col_paths(grid, 2, 1)
    # a short row (cells missing, not empty) says nothing either, even in a marked row
    short = _table('<td></td><td colspan="2">Three Months</td>',
                   "<td>(in m)</td><td>Q4</td><td>Q3</td><td>FY13</td><td>FY12</td>",
                   "<td>Sales</td><td>1</td><td>2</td><td>3</td><td>4</td>")
    grid, cover = parse_html_table_layout(short)
    assert cover[0][3] is None and cover[0][4] is None
    assert reconstruct_col_paths(grid, 2, 1, cover=cover)[2:] == reconstruct_col_paths(grid, 2, 1)[2:]


# S2 -----------------------------------------------------------------------

def test_blank_row_label_does_not_inherit_the_previous_rows_label():
    # two value columns: a row with ONE filled cell is a section row by shape
    html = _table("<td></td><td>2013</td><td>2012</td>",
                  "<td>Drilling</td><td>17</td><td>15</td>",
                  "<td>Production</td><td>15</td><td>14</td>",
                  "<td></td><td>45</td><td>41</td>")
    grid, cover = parse_html_table_layout(html)
    assert reconstruct_row_paths(grid, 1, 1) == [["Drilling"], ["Production"], ["Production"]]
    assert reconstruct_row_paths(grid, 1, 1, cover=cover) == [["Drilling"], ["Production"], []]


def test_a_rowspan_label_still_covers_its_rows():
    html = _table("<td></td><td>2013</td><td>2012</td>",
                  '<td rowspan="2">Europe</td><td>1</td><td>1</td>',
                  "<td>2</td><td>2</td>",
                  "<td>Asia</td><td>3</td><td>3</td>")
    grid, cover = parse_html_table_layout(html)
    assert reconstruct_row_paths(grid, 1, 1, cover=cover) == [["Europe"], ["Europe"], ["Asia"]]


# S3 -----------------------------------------------------------------------

def test_a_text_label_row_under_a_blank_corner_is_header():
    grid = [["", "Deliveries", "", "Revenues", ""],
            ["Description", "Dec 31, 2011", "Dec 31, 2010", "Dec 31, 2011", "Dec 31, 2010"],
            ["Residential", "38,160", "37,963", "$704", "$733"]]
    assert guess_n_header_rows(grid, n_header_cols=1, rule="v2") == 1
    assert guess_n_header_rows(grid, n_header_cols=1, rule="v3") == 2


def test_footnoted_years_are_a_label_row():
    grid = [["", "Schedule", "", ""],
            ["Year ended", "1998", "1999(a)", "2000"],
            ["Reserves", "28,506", "26,850", "26,510"]]
    assert guess_n_header_rows(grid, n_header_cols=1, rule="v2") == 1
    assert guess_n_header_rows(grid, n_header_cols=1, rule="v3") == 2


def test_label_row_rule_does_not_swallow_data_rows():
    base = [["", "2014", "2013"]]
    for row in (["Revenue", "$1,000", "$900"],     # numbers
                ["Loans", "—", "—"],              # missing markers are data
                ["Note", "see below", ""],        # one filled cell is not a label row
                ["Rate", "4.5%", "text"]):        # any data-like cell ends the header
        grid = base + [row, ["Other", "1", "2"]]
        assert guess_n_header_rows(grid, n_header_cols=1, rule="v3") == 1, row


# S4 -----------------------------------------------------------------------

def test_a_total_row_closes_its_section():
    grid = [["", "2014"],
            ["Equity", "10"],
            ["Alternatives", ""],
            ["Core", "3"],
            ["Subtotal", "4"],
            ["Long-term", "20"],
            ["Total", "30"]]
    v2 = reconstruct_row_paths(grid, 1, 1)
    assert v2[4] == ["Alternatives", "Long-term"] and v2[5] == ["Alternatives", "Total"]
    v3 = reconstruct_row_paths(grid, 1, 1, close_on_total=True)
    assert v3[2] == ["Alternatives", "Core"]
    assert v3[3] == ["Alternatives", "Subtotal"]      # the total row itself stays in its section
    assert v3[4] == ["Long-term"] and v3[5] == ["Total"]


def test_closing_a_section_keeps_the_leading_scope():
    grid = [["", "2015"],
            ["Operations", ""],
            ["Revenues:", ""],
            ["Airplanes", "5"],
            ["Defense:", ""],
            ["Aircraft", "1"],
            ["Total Defense", "2"],
            ["Capital", "3"],
            ["Total revenues", "9"],
            ["Other", "1"]]
    v3 = reconstruct_row_paths(grid, 1, 1, close_on_total=True)
    assert v3[4] == ["Operations", "Defense:", "Aircraft"]
    assert v3[6] == ["Operations", "Capital"]
    assert v3[8] == ["Operations", "Other"]


# S5 -----------------------------------------------------------------------

def test_identical_paths_are_split_by_the_nearest_heading_above_them():
    grid = [["", "2015"],
            ["Revenues:", ""],
            ["Airplanes", "5"],
            ["Defense:", ""],
            ["Services", "9"],
            ["Backlog:", ""],
            ["Airplanes", "50"],
            ["Defense:", ""],
            ["Services", "16"]]
    v2 = reconstruct_row_paths(grid, 1, 1)
    assert v2[3] == v2[7] == ["Defense:", "Services"]
    v3 = reconstruct_row_paths(grid, 1, 1, disambiguate=True)
    assert v3[3] == ["Revenues:", "Defense:", "Services"]
    assert v3[7] == ["Backlog:", "Defense:", "Services"]
    # rows that were already distinct are untouched
    assert v3[1] == v2[1] and v3[5] == v2[5]


def test_headings_that_do_not_split_a_collision_are_not_added():
    grid = [["", "2015"],
            ["Assets:", ""],
            ["Other", "1"],
            ["Other", "2"]]
    assert reconstruct_row_paths(grid, 1, 1, disambiguate=True) == reconstruct_row_paths(grid, 1, 1)


# S6 -----------------------------------------------------------------------

def test_a_dated_corner_is_the_scope_of_undated_columns():
    grid = [["In millions of dollars at December 31, 2018", "Level 1", "Level 2"],
            ["Loans", "1", "2"]]
    assert corner_scope(grid, 1, 1, reconstruct_col_paths(grid, 1, 1)) == \
        "In millions of dollars at December 31, 2018"


def test_corner_scope_needs_a_year_and_undated_columns():
    undated = [["(Dollars in millions)", "Level 1"], ["Loans", "1"]]
    assert corner_scope(undated, 1, 1, reconstruct_col_paths(undated, 1, 1)) == ""
    dated_cols = [["December 31,", "2018", "2017"], ["Loans", "1", "2"]]
    assert corner_scope(dated_cols, 1, 1, reconstruct_col_paths(dated_cols, 1, 1)) == ""


# end to end: the MultiHiertt table object ------------------------------------

def test_build_tables_v3_applies_the_rules_and_v2_is_unchanged():
    from mh_arms import build_tables
    html = _table('<td>For the Year Ended December 31, 2008</td><td colspan="2">Segments</td>'
                  '<td rowspan="2">Other</td><td></td>',
                  "<td></td><td>Alpha</td><td>Beta</td><td>Total</td>",
                  "<td>Revenue</td><td>1</td><td>2</td><td>3</td><td>6</td>",
                  "<td>Costs</td><td>1</td><td>1</td><td>1</td><td>3</td>",
                  "<td></td><td>0</td><td>1</td><td>2</td><td>3</td>")
    docs = {"u": ([html], "{}", [])}
    v2 = build_tables(docs, "v2")[0]["u::0"].table
    v3 = build_tables(docs, "v3")[0]["u::0"].table
    assert (v2.nhr, v2.nhc) == (v3.nhr, v3.nhc) == (2, 1)
    assert v2.col_path(3) == ["Other", "Total"] and v2.row_path(2) == ["Costs"]
    scope = "For the Year Ended December 31, 2008"
    assert v3.col_path(1) == [scope, "Segments", "Beta"]
    assert v3.col_path(3) == [scope, "Total"]
    assert v3.row_path(2) == []
