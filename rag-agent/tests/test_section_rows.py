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
