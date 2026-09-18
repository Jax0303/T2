# SPDX-License-Identifier: MIT
"""Self-check for scripts/text_collision_ceiling.py's ceiling-math aggregation."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from text_collision_ceiling import ceiling_stats  # noqa: E402


def _row(cls, collision_leaf, collision_compact=False):
    return {"class": cls, "hybrid_antagonist_collision_leaf": collision_leaf,
           "hybrid_antagonist_collision_compact": collision_compact}


def test_ceiling_is_one_minus_collision_rate_over_the_whole_population():
    # 10 queries total, 4 misses, 1 of them a hard text collision.
    rows = [_row("shared_representation_failure", True),
           _row("shared_representation_failure", False),
           _row("combination_failure", False),
           _row("shared_representation_failure", False)]
    out = ceiling_stats(rows, n_pop=10)
    assert out["structural_leaf"]["hard_collision_misses"] == 1
    assert out["structural_leaf"]["fixable_gap_misses"] == 3
    assert out["structural_leaf"]["real_ceiling_R@1"] == 0.9  # 1 - 1/10


def test_by_class_only_counts_collisions_within_that_class():
    rows = [_row("shared_representation_failure", True),
           _row("combination_failure", True),
           _row("combination_failure", False)]
    out = ceiling_stats(rows, n_pop=3)
    assert out["by_class"]["shared_representation_failure"]["n"] == 1
    assert out["by_class"]["shared_representation_failure"]["collision_leaf"] == 1
    assert out["by_class"]["combination_failure"]["n"] == 2
    assert out["by_class"]["combination_failure"]["collision_leaf"] == 1


def test_no_misses_is_a_perfect_ceiling_not_a_division_error():
    out = ceiling_stats([], n_pop=5)
    assert out["structural_leaf"]["real_ceiling_R@1"] == 1.0
    assert out["structural_leaf"]["pct_of_misses"] is None
