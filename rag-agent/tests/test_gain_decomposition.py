# SPDX-License-Identifier: MIT
"""Unit tests for the P5 pool-vs-ranking gain decomposition (no dataset)."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "osc_gain_decomposition",
    Path(__file__).resolve().parent.parent / "scripts" / "osc_gain_decomposition.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
covered, in_pool, decompose = _mod.covered, _mod.in_pool, _mod.decompose


def test_covered_requires_all_gold_within_k():
    assert covered({1: 3, 2: 10}, 10)
    assert not covered({1: 3, 2: 11}, 10)       # one operand ranked out
    assert not covered({1: 3, 2: None}, 10)     # one operand outside the pool


def test_in_pool_ignores_rank_only_membership():
    assert in_pool({1: 99, 2: 100})
    assert not in_pool({1: 1, 2: None})


def test_decompose_splits_pool_vs_rank_limited():
    # q0: flat misses a gold cell from its pool entirely -> pool-limited gain
    # q1: flat has all gold in pool but one ranked 40 -> rank-limited gain
    # q2: both cover -> no flip; q3: flat covers, S3 does not -> loss
    base = {0: {1: 2, 2: None}, 1: {1: 2, 2: 40}, 2: {1: 1, 2: 2}, 3: {1: 1, 2: 2}}
    treat = {0: {1: 1, 2: 2}, 1: {1: 1, 2: 2}, 2: {1: 3, 2: 4}, 3: {1: 1, 2: None}}
    d = decompose(base, treat, k=10)
    assert d["gain"] == 2 and d["loss"] == 1
    assert d["gain_pool_limited"] == 1 and d["gain_rank_limited"] == 1
    assert d["share_pool_limited"] == 0.5
    assert d["gain_cells_missing_from_base_pool"] == 1
    assert d["gain_cells_in_pool_ranked_below_k"] == 1


def test_oracle_is_pool_membership_capped_by_scope():
    # all gold in pool but scope 3 > k=2 -> even a perfect reranker fails
    base = {0: {1: 50, 2: 60, 3: 70}}
    treat = {0: {1: 1, 2: 2, 3: 3}}
    d = decompose(base, treat, k=2)
    assert d["oracle_base_set_recall@k"] == 0.0
    d10 = decompose(base, treat, k=10)
    assert d10["oracle_base_set_recall@k"] == 1.0
