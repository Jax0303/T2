# SPDX-License-Identifier: MIT
"""E-A pools must contain the gold set and nest across sizes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ea_pool_size_sweep import pool_for


def test_pools_hold_gold_and_nest():
    gold = [3, 17, 900]
    cache = {"max_size": 500}
    pools = {n: pool_for(0, gold, 5000, n, 42, cache) for n in (50, 100, 500)}
    for n, p in pools.items():
        assert len(p) == n, (n, len(p))
        assert len(set(p)) == n, "duplicate cells in pool"
        assert set(gold) <= set(p), "gold fell out of the pool"
    assert set(pools[50]) <= set(pools[100]) <= set(pools[500]), "pools do not nest"
