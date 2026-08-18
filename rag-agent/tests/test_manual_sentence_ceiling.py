# SPDX-License-Identifier: MIT
"""The hand-written-sentence ceiling run reports paired significance itself."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from cell_retrieval_matrix import context_order
from manual_sentence_ceiling import mcnemar


class TestMcNemar(unittest.TestCase):
    def test_concordant_pairs_are_ignored(self):
        """Agreements carry no signal: 100 identical outcomes must not be a result."""
        self.assertEqual(mcnemar([1] * 100, [1] * 100)["exact_p"], 1.0)

    def test_discordant_counts_and_two_sided_p(self):
        a = [1, 1, 1, 0, 0]
        b = [0, 0, 0, 0, 1]
        # 3 only-a, 1 only-b -> two-sided binomial on 4 flips = 2 * (1+4)/16
        self.assertEqual(mcnemar(a, b), {"only_first": 3, "only_second": 1,
                                         "exact_p": 0.625})

    def test_symmetric_in_its_arguments(self):
        a, b = [1, 1, 0, 1, 0, 0], [0, 1, 1, 0, 0, 0]
        fwd, rev = mcnemar(a, b), mcnemar(b, a)
        self.assertEqual(fwd["exact_p"], rev["exact_p"])
        self.assertEqual((fwd["only_first"], fwd["only_second"]),
                         (rev["only_second"], rev["only_first"]))

    def test_p_never_exceeds_one(self):
        """min(n01,n10) doubling can overshoot 1 on an even split; it is clamped."""
        self.assertEqual(mcnemar([1, 0], [0, 1])["exact_p"], 1.0)


class TestContextOrder(unittest.TestCase):
    """--oracle-cell must make recall 1.0 without changing the context size."""

    def test_plain_is_the_retriever_head(self):
        self.assertEqual(context_order([4, 7, 1, 9], 1, 3, False), [4, 7, 1])

    def test_oracle_pins_gold_first_and_drops_the_tail(self):
        # gold 9 was rank 4 and outside k=3; it comes in, the last distractor goes
        self.assertEqual(context_order([4, 7, 1, 9], 9, 3, True), [9, 4, 7])

    def test_oracle_never_duplicates_a_gold_already_retrieved(self):
        got = context_order([4, 7, 1, 9], 7, 3, True)
        self.assertEqual(got, [7, 4, 1])
        self.assertEqual(len(set(got)), 3)


if __name__ == "__main__":
    unittest.main()
