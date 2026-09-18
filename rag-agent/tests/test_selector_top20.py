# SPDX-License-Identifier: MIT
"""Pinning test for scripts/selector_top20_eval.py's selector-output parser --
the one piece of new branching logic in the top-20-then-select-1 pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from selector_top20_eval import majority_vote_by_cell, parse_selection, K


def test_parse_selection_plain_digit():
    assert parse_selection("7") == 7


def test_parse_selection_two_digit_in_range():
    assert parse_selection("13") == 13


def test_parse_selection_extracts_first_number_from_prose():
    assert parse_selection("Candidate 4 answers it.") == 4


def test_parse_selection_out_of_range_is_none():
    assert parse_selection(str(K + 1)) is None
    assert parse_selection("0") is None


def test_parse_selection_no_digit_is_none():
    assert parse_selection("none of them") is None


def test_majority_vote_by_cell_picks_the_plurality_cell():
    cand_cells = [("t1", 0, 0), ("t1", 0, 1), ("t1", 1, 0)]
    votes = [("t1", 0, 1), ("t1", 0, 1), ("t1", 1, 0)]
    assert majority_vote_by_cell(votes, cand_cells) == ("t1", 0, 1)


def test_majority_vote_by_cell_ties_break_by_hybrid_rank_order():
    cand_cells = [("t1", 0, 0), ("t1", 0, 1), ("t1", 1, 0)]
    votes = [("t1", 1, 0), ("t1", 0, 1)]  # 1-1 tie between rank-3 and rank-2 cells
    assert majority_vote_by_cell(votes, cand_cells) == ("t1", 0, 1)  # rank-2 beats rank-3
