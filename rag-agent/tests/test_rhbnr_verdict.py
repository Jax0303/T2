# SPDX-License-Identifier: MIT
"""The verdict must read the prereg's bounds, not invent friendlier ones."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from rhbnr_verdict import BIG, BOUNDS, SMALL, verdict


def _rec(qid, tokens, cell, gold):
    return {"query_id": qid, "gold_table_tokens": tokens,
            "cell": {"answer_em": cell}, "goldtable": {"answer_em": gold}}


def test_crossover_passes_when_both_sides_clear_their_bound():
    recs = ([_rec(f"b{i}", 4000, 1, 0) for i in range(10)]      # big: cell wins
            + [_rec(f"s{i}", 300, 0, 1) for i in range(10)])    # small: table wins
    v = verdict(recs)["predictions"]
    assert v["P1_big_tables_cell_beats_goldtable"]["pass"]
    assert v["P2_small_tables_goldtable_beats_cell"]["pass"]
    assert v["P3_signs_oppose"]["pass"]


def test_no_crossover_fails_p3():
    recs = [_rec(f"b{i}", 4000, 1, 0) for i in range(10)] + \
           [_rec(f"s{i}", 300, 1, 0) for i in range(10)]        # cell wins both
    assert not verdict(recs)["predictions"]["P3_signs_oppose"]["pass"]


def test_bounds_are_the_prereg_numbers():
    assert BOUNDS["P1"] == 0.05 and BOUNDS["P2"] == 0.03
    assert BOUNDS["P4"] == (0.03, 0.08) and (BIG, SMALL) == (2048, 512)


def test_p7_is_reported_but_not_a_pass_field():
    recs = [_rec("a", 4000, 1, 0)]
    p7 = verdict(recs)["predictions"]["P7_vs_published_5.32"]
    assert "beats" in p7 and "pass" not in p7   # the claim does not rest on it
