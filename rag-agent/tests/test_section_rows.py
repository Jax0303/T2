# SPDX-License-Identifier: MIT
"""Section-row promotion on the row axis (rag_agent/reconstruct/header_grid.py)."""
from rag_agent.reconstruct import guess_n_header_rows, reconstruct_row_paths


def test_section_row_becomes_the_parent_level():
    # stub block is ONE column, so "north" has no column to live in; it is a
    # section row and every row under it is its child.
    grid = [
        ["region", "2023", "2024"],
        ["north", "", ""],
        ["seoul", "1", "2"],
        ["busan", "3", "4"],
        ["south", "", ""],
        ["daegu", "5", "6"],
    ]
    paths = reconstruct_row_paths(grid, n_header_rows=1, n_header_cols=1)
    assert paths[1] == ["north", "seoul"]
    assert paths[2] == ["north", "busan"]
    # a new section cuts the previous one off instead of bleeding past it
    assert paths[4] == ["south", "daegu"]
    # the heading is not its own child
    assert paths[0] == ["north"]


def test_section_label_need_not_sit_in_the_stub():
    grid = [
        ["sector", "immigrant", "third generation"],
        ["", "percent", ""],
        ["utilities", "0.5", "0.6"],
    ]
    paths = reconstruct_row_paths(grid, n_header_rows=1, n_header_cols=1)
    assert paths[1] == ["percent", "utilities"]


def test_a_leading_title_row_is_not_a_boundary():
    # Spreadsheet exports (RealHiTBench is PhpSpreadsheet HTML) open with a
    # one-cell title, which is a section row by shape. Treating it as the end
    # of the header block returned nhr=0 and wiped the header off 43 of 92
    # tables. HiTab grids have no title row, so no HiTab split catches this.
    grid = [
        ["Inland fisheries", "", "", ""],
        ["area", "2010", "2011", "2012"],
        ["Australia", "1376", "1084", "1074"],
    ]
    assert guess_n_header_rows(grid, n_header_cols=1) == 2


def test_a_section_row_under_a_real_header_still_ends_it():
    grid = [
        ["region", "2023", "2024"],
        ["north", "", ""],
        ["seoul", "1", "2"],
    ]
    assert guess_n_header_rows(grid, n_header_cols=1) == 1


def test_no_section_rows_is_unchanged():
    grid = [
        ["region", "2023"],
        ["seoul", "1"],
        ["busan", "3"],
    ]
    on = reconstruct_row_paths(grid, n_header_rows=1, n_header_cols=1)
    off = reconstruct_row_paths(grid, n_header_rows=1, n_header_cols=1,
                                use_section_rows=False)
    assert on == off == [["seoul"], ["busan"]]


def test_leading_section_run_is_a_persistent_scope():
    """The body's first heading run outlives the groups beneath it.

    "percent" is a unit that holds over the whole table; "age group" and
    "marital status" are the groups inside it. Treating a heading that follows
    data as a fresh stack drops the unit, so "marital status" reconstructs one
    segment short of the gold ['percent', 'marital status'] — 5.95pp of HiTab
    dev's row paths (.7607 -> .8202).
    """
    grid = [
        ["characteristic", "men", "women"],
        ["", "percent", ""],          # leading scope: holds for the whole table
        ["age group", "", ""],        # group heading inside that scope
        ["20 to 24", "10.3", "16.3"],
        ["25 to 29", "24.6", "27.0"],
        ["marital status", "", ""],   # follows data — replaces the group, NOT the scope
        ["married", "68.2", "77.4"],
    ]
    paths = reconstruct_row_paths(grid, n_header_rows=1, n_header_cols=1)
    assert paths[2] == ["percent", "age group", "20 to 24"], paths[2]
    assert paths[4] == ["percent", "marital status"], paths[4]
    assert paths[5] == ["percent", "marital status", "married"], paths[5]


def test_single_leading_heading_is_not_kept_as_scope():
    """One heading before the data is a group, not a scope — it must still be
    replaced. Keeping it would put "north" on every southern row."""
    grid = [
        ["region", "2023"],
        ["north", ""],
        ["seoul", "1"],
        ["south", ""],
        ["daegu", "5"],
    ]
    paths = reconstruct_row_paths(grid, n_header_rows=1, n_header_cols=1)
    assert paths[1] == ["north", "seoul"]
    assert paths[3] == ["south", "daegu"], paths[3]
    assert "north" not in paths[3]
