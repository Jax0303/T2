"""The greedy window fill that the MT2Net answer leg counts OSC from."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mt2net_retriever_baseline import fill_by_index

count = len          # one char = one token, so the arithmetic is readable


def test_skips_the_unfittable_and_keeps_going():
    texts = {0: "aaa", 1: "x" * 99, 2: "bb", 3: "c"}
    kept, used = fill_by_index([0, 1, 2, 3], texts, count, 6)
    # 1 is skipped, not a stopping point -- 2 and 3 still get in
    assert kept == [0, 2, 3] and used == 6


def test_rank_order_is_respected():
    texts = {0: "aa", 1: "aa", 2: "aa"}
    assert fill_by_index([2, 0, 1], texts, count, 4)[0] == [2, 0]


def test_osc_is_countable_from_what_was_kept():
    texts = {0: "aa", 1: "bb", 2: "cc"}
    kept, _ = fill_by_index([0, 1, 2], texts, count, 4)
    assert {0, 1} <= set(kept)          # gold pair fits
    assert not {0, 2} <= set(kept)      # this one does not


if __name__ == "__main__":
    test_skips_the_unfittable_and_keeps_going()
    test_rank_order_is_respected()
    test_osc_is_countable_from_what_was_kept()
    print("ok")
