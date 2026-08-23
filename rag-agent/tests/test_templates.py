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


def test_structural_compact_falls_back_to_s2_when_untitled():
    """No title -> no frame: the S2 string the cell arm indexes."""
    from rag_agent.serialization.templates import STRUCTURAL_COMPACT

    got = render(STRUCTURAL_COMPACT, "", ["Total"], ["Revenue", "2018"], 2894)
    assert got == "Total > Revenue > 2018: 2894"


def test_structural_compact_is_structural_when_titled():
    """A title earns the frame back, so titled corpora are untouched."""
    from rag_agent.serialization.templates import STRUCTURAL, STRUCTURAL_COMPACT

    args = ("Table 3", ["Total"], ["Revenue", "2018"], 2894)
    assert render(STRUCTURAL_COMPACT, *args) == render(STRUCTURAL, *args)


def test_structural_compact_untitled_header_scope_drops_the_colon():
    """value=None names a header scope; an S2 line with no value has no ': '."""
    from rag_agent.serialization.templates import STRUCTURAL_COMPACT

    assert render(STRUCTURAL_COMPACT, "", ["Total"], ["Revenue"]) == "Total > Revenue"


def test_structural_compact_collapses_whitespace_like_every_other_template():
    """fmt_value normalises label whitespace, so S3c is TOKEN-identical to the
    legacy raw S2 renderer, not byte-identical: 267/5320 AIT-QA cells and
    8652/148140 RealHiTBench cells differ by collapsed runs alone, and every one
    of them tokenises identically under bge-small. Pinned so the distinction is
    not rediscovered as a bug."""
    from rag_agent.serialization.templates import STRUCTURAL_COMPACT

    got = render(STRUCTURAL_COMPACT, "", [], ["Fuel Expense       (in millions)"], "$9,307")
    assert got == "Fuel Expense (in millions): $9,307"


def test_s2r_reverses_each_axis_and_keeps_every_token():
    """S2r is the capacity-controlled contrast: same tokens, leaf-first order.

    Every other template contrast in this repo changes sentence LENGTH, so a
    difference can always be blamed on how many cells fit the budget. This one
    cannot -- if the multiset ever stops matching, the experiment that prices
    word order apart from the sentence frame is no longer controlled.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from point3_reconstruction_cost import cell_text

    rp, cp, v = ["a", "b"], ["c", "d"], "1"
    s2 = cell_text(rp, cp, v, "S2")
    s2r = cell_text(rp[::-1], cp[::-1], v, "S2")
    assert s2 == "a > b > c > d: 1"
    assert s2r == "b > a > d > c: 1"
    # segment multiset, not whitespace split -- ": 1" rides on the last segment
    seg = lambda t: sorted(t.rsplit(": ", 1)[0].split(" > "))
    assert seg(s2) == seg(s2r)
