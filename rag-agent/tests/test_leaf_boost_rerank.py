# SPDX-License-Identifier: MIT
"""Pinning test for scripts/leaf_boost_rerank.py's pure logic (leaf id dedup,
boost mask, McNemar) -- no encoder, no data files."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from leaf_boost_rerank import boost_mask, leaf_ids, mcnemar


def test_leaf_ids_dedups_across_cells():
    row_leaf = ["Total", "total", "Male", ""]
    col_leaf = ["2020", "2020", "", "Male"]
    row_ids, col_ids, uniq = leaf_ids(row_leaf, col_leaf)
    assert sorted(uniq) == ["2020", "male", "total"]
    assert row_ids[0] == row_ids[1]           # "Total"/"total" dedup case-insensitively
    assert row_ids[3] == -1                   # empty leaf -> no id


def test_boost_mask_counts_row_and_col_matches():
    row_leaf = ["Total", "Male", "Total"]
    col_leaf = ["2020", "2020", ""]
    row_ids, col_ids, uniq = leaf_ids(row_leaf, col_leaf)
    mask = boost_mask("Total in 2020", row_ids, col_ids, uniq)
    assert list(mask) == [2.0, 1.0, 1.0]      # row2 has no leaf column, so only row leaf counts


def test_boost_mask_no_match_is_zero():
    row_ids, col_ids, uniq = leaf_ids(["Female"], ["2019"])
    mask = boost_mask("how many people", row_ids, col_ids, uniq)
    assert list(mask) == [0.0]


def test_mcnemar_significant_when_all_discordant_favor_one_side():
    a = {i: True for i in range(20)}
    b = {i: (i >= 5) for i in range(20)}     # b loses 5 that a had, gains none
    out = mcnemar(a, b)
    assert out["baseline_only"] == 5
    assert out["boosted_only"] == 0
    assert out["p_value"] < 0.1


def test_mcnemar_no_discordant_pairs_is_p_one():
    a = {i: True for i in range(10)}
    out = mcnemar(a, dict(a))
    assert out["discordant"] == 0
    assert out["p_value"] == 1.0
