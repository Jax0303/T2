# SPDX-License-Identifier: MIT
"""Pins the TreeThinker round-1 borrowing to what its prompt actually returns.

The prompt is verbatim from RealHiTBench (Findings of ACL 2025); what is ours is
the reply -> row-path mapping, which is what these pin. Reference:
cspzyy/RealHiTBench inference/answer_prompt_tree.py :: Generate_Tree.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.reconstruct.llm_tree import (GENERATE_TREE, parse_tuples,
                                            paths_from_tuples,
                                            reconstruct_row_paths_llm, to_latex)


class FakeLLM:
    """Records the call and replays a canned tuple list."""

    def __init__(self, reply):
        self.reply, self.seen = reply, None

    def complete(self, system, user, max_tokens=256):
        self.seen = (system, user)
        return self.reply


def test_parses_axis_level_span_and_text():
    nodes = parse_tuples("L=[(R0, 1, 4, Total), (C0, 1, 2, Count), (R1, 1, 2, Male)]")
    assert (0, 1, 4, "Total") in nodes
    assert (1, 1, 2, "Male") in nodes
    assert all(n[3] != "Count" for n in nodes)  # column tuples stay out of rows


def test_restated_list_wins():
    """Models revise the list mid-reply; the last statement is the revision."""
    nodes = parse_tuples("(R1, 3, 4, Femail)\n...on review...\n(R1, 3, 4, Female)")
    assert nodes == [(1, 3, 4, "Female")]


def test_reversed_span_is_tolerated():
    assert parse_tuples("(R0, 4, 1, Total)") == [(0, 1, 4, "Total")]


def test_path_is_the_covering_chain_shallowest_first():
    nodes = [(1, 1, 2, "Male"), (0, 1, 4, "Total"), (2, 2, 2, "Grade 2"),
             (2, 1, 1, "Grade 1"), (1, 3, 4, "Female")]
    assert paths_from_tuples(nodes, n_header_rows=1, n_rows=5) == [
        ["Total", "Male", "Grade 1"],
        ["Total", "Male", "Grade 2"],
        ["Total", "Female"],
        ["Total", "Female"],
    ]


def test_data_parent_row_is_not_duplicated_into_its_own_path():
    """The failure class this exists for: a parent row that also holds values."""
    nodes = [(0, 1, 3, "Total"), (1, 1, 1, "Total"), (1, 2, 2, "Male")]
    assert paths_from_tuples(nodes, 1, 3) == [["Total"], ["Total", "Male"]]


def test_rows_without_any_covering_node_are_empty_not_missing():
    """One path per data row, always — the caller indexes by row - n_header_rows."""
    assert paths_from_tuples([(0, 1, 1, "A")], 1, 4) == [["A"], [], []]


def test_latex_keeps_merge_blanks_and_escapes_ampersands():
    out = to_latex([["a", "b & c"], ["1", ""]]).splitlines()
    assert out[0] == "\\begin{tabular}{ll}"
    assert out[1] == "a & b \\& c \\\\"
    assert out[2] == "1 &  \\\\"


def test_drop_in_signature_matches_the_rule_based_reconstructor():
    llm = FakeLLM("L=[(R0, 2, 3, Total), (R1, 2, 2, Male), (R1, 3, 3, Female)]")
    grid = [["", "n"], ["hdr", ""], ["Male", "3"], ["Female", "4"]]
    paths = reconstruct_row_paths_llm(grid, 2, 1, llm)
    assert paths == [["Total", "Male"], ["Total", "Female"]]

    system, user = llm.seen
    assert system == GENERATE_TREE          # prompt reaches the model unedited
    assert "\\begin{tabular}" in user
    assert "Rows are numbered 0..3" in user  # positions are pinned to grid lines
