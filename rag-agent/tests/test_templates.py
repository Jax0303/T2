# SPDX-License-Identifier: MIT
"""The two S3 templates must not drift.

STRUCTURAL is pinned byte-for-byte because results already on disk were produced
under the retired ``length="long"`` preset and must stay reproducible from this
code. MT2NET is pinned to the one sentence Zhao et al. (2022) §4 publishes — if
that assertion ever fails, the "MT2Net reproduction" label is no longer true.
"""
import pytest

from rag_agent.serialization import caption
from rag_agent.serialization.templates import MT2NET, STRUCTURAL, render

PAPER_SENTENCE = ("For Innovation Systems of Segment, sales of product in 2018, "
                  "Year Ended December 31 is 2,894")


def test_mt2net_reproduces_the_published_example():
    got = render(MT2NET, "",
                 ["Segment", "Innovation Systems"],
                 ["Year Ended December 31", "sales of product in 2018"],
                 "2,894")
    assert got == PAPER_SENTENCE


@pytest.mark.parametrize("title,row,col,val,want", [
    ("Population by region", ["North America", "Canada"], ["2023", "Q1"], "1,234",
     "In the table 'Population by region', among North America > Canada, "
     "the value of 2023 > Q1 is 1,234."),
    # untitled: the leading capital is added, the label's own capitals survive.
    # This line read "Among total" until 2026-08-23, when str.capitalize() was
    # found to be case-folding every label in a corpus without titles.
    ("", ["Total"], [], "5", "Among Total, the value is 5."),
    ("T", [], ["%"], "", "In the table 'T', the value of % is ."),
    ("Tbl", ["a > b"], ["c"], None, "In the table 'Tbl', among a > b, the value of c."),
])
def test_structural_is_pinned(title, row, col, val, want):
    assert render(STRUCTURAL, title, row, col, val) == want


def test_retired_length_axis_fails_loudly():
    with pytest.raises(TypeError, match="length axis is retired"):
        caption.caption_sentence("t", ["r"], ["c"], value="1", length="long")


def test_untitled_keeps_label_capitals():
    """The case-fold regression this file exists to prevent coming back."""
    got = render(STRUCTURAL, "", ["Married mothers"], ["Total, all activities"], "122")
    assert "Married mothers" in got and "Total, all activities" in got
    assert got[0].isupper()
