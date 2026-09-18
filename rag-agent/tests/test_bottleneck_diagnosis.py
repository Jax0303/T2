# SPDX-License-Identifier: MIT
"""Self-check for scripts/bottleneck_diagnosis.py's pure logic: rank bucketing,
top-1 error-class priority order, and the stratified sampler. No HiTab data, no
model — fakes stand in for HitabTable so classify_top1_error is exercised
without touching disk.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bottleneck_diagnosis import (classify_top1_error, condition_names,  # noqa: E402
                                  parse_condition, rank_bucket, stratified_sample)


class _FakeTable:
    def __init__(self, data, row_paths, col_paths):
        self.data = data
        self._rp, self._cp = row_paths, col_paths

    def row_path(self, i):
        return self._rp[i]

    def col_path(self, j):
        return self._cp[j]


class _FakeTab:
    def __init__(self, table):
        self.table = table


def test_rank_bucket_edges():
    assert [rank_bucket(r) for r in (1, 2, 3, 4, 5, 6, 10, 11, 20, 21, 500, None)] == [
        "1", "2", "3", "4-5", "4-5", "6-10", "6-10", "11-20", "11-20", ">20", ">20", ">20"]


def _table(rows=3, cols=3):
    data = [[f"v{i}{j}" for j in range(cols)] for i in range(rows)]
    row_paths = [[f"r{i}"] for i in range(rows)]
    col_paths = [[f"c{j}"] for j in range(cols)]
    return _FakeTable(data, row_paths, col_paths)


def test_classify_same_value_wins_over_everything_else():
    t = _table()
    t.data[0][0] = "42"
    t.data[2][2] = "42"                       # far away, same value
    tabs = {"t1": _FakeTab(t)}
    assert classify_top1_error(("t1", 2, 2), ("t1", 0, 0), tabs) == "same_value"


def test_classify_wrong_row_then_wrong_column():
    t = _table()
    tabs = {"t1": _FakeTab(t)}
    assert classify_top1_error(("t1", 1, 0), ("t1", 0, 0), tabs) == "wrong_row"
    assert classify_top1_error(("t1", 0, 1), ("t1", 0, 0), tabs) == "wrong_column"


def test_classify_same_leaf_header_before_nearby_and_other():
    t = _table(rows=4, cols=4)
    t._rp[3] = t._rp[0]                       # row 3 repeats row 0's leaf label
    tabs = {"t1": _FakeTab(t)}
    # different row AND column, but the predicted cell's row leaf == gold's
    assert classify_top1_error(("t1", 3, 2), ("t1", 0, 0), tabs) == "same_leaf_header"


def test_classify_nearby_cell_and_other_and_wrong_table():
    t = _table(rows=4, cols=4)
    tabs = {"t1": _FakeTab(t)}
    assert classify_top1_error(("t1", 1, 1), ("t1", 0, 0), tabs) == "nearby_cell"
    assert classify_top1_error(("t1", 3, 3), ("t1", 0, 0), tabs) == "other"
    tabs["t2"] = _FakeTab(_table())
    assert classify_top1_error(("t2", 0, 0), ("t1", 1, 1), tabs) == "wrong_table"


def test_stratified_sample_is_deterministic_and_disjoint(monkeypatch):
    import bottleneck_diagnosis as bd

    pop = {f"q{i}": {"gold_rank": (i % 25) + 1 if i % 25 < 24 else None}
           for i in range(200)}
    monkeypatch.setattr(bd, "load_primary_population", lambda: pop)
    a = stratified_sample(50, seed=1)
    b = stratified_sample(50, seed=1)
    c = stratified_sample(50, seed=2)
    assert a == b                             # same seed -> identical sample
    assert len(a) == 50 and len(set(a)) == 50
    assert a != c                             # different seed -> (almost certainly) different


def test_condition_names_roundtrip():
    names = condition_names()
    assert len(names) == 1 + 3 * 3 * 2
    assert parse_condition("gold_only") is None
    assert parse_condition("gold_9_same_value_last") == (9, "same_value", "last")
    assert parse_condition("gold_1_hard_negative_first") == (1, "hard_negative", "first")
