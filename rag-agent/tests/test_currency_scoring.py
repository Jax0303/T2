# SPDX-License-Identifier: MIT
"""A currency mark must not decide an answer, and must not move HiTab."""
import unittest

from rag_agent.eval.metrics import hitab_exact_match, hitab_exact_match_text


class TestCurrency(unittest.TestCase):
    def test_dollar_gold_accepts_bare_number(self):
        """AIT-QA ships '$2.25'; a model answers '2.25'. That is the same answer."""
        self.assertTrue(hitab_exact_match_text("2.25", "$2.25"))
        self.assertTrue(hitab_exact_match_text("2783", "$2,783"))

    def test_dollar_prediction_accepts_bare_gold(self):
        """Stripped on both sides, so neither arm can gain from it."""
        self.assertTrue(hitab_exact_match_text("$1,480", "1,480"))

    def test_a_wrong_number_is_still_wrong(self):
        self.assertFalse(hitab_exact_match_text("2.26", "$2.25"))
        self.assertFalse(hitab_exact_match_text("$9,307", "$5,813"))

    def test_the_published_normalisations_still_hold(self):
        self.assertTrue(hitab_exact_match_text("24", "24%"))
        self.assertTrue(hitab_exact_match_text("1480", "1,480"))
        self.assertTrue(hitab_exact_match(["57.0", "31.2"], [57.0, 31.2]))
        self.assertFalse(hitab_exact_match_text("whole economy", "manufacturing"))

    def test_text_answers_are_untouched(self):
        self.assertTrue(hitab_exact_match_text("Whole economy", "whole economy"))


if __name__ == "__main__":
    unittest.main()


class TestMultiValueStringGold(unittest.TestCase):
    """RealHitBench stores multi-value answers as one comma-separated STRING."""

    def test_a_missing_space_is_not_a_wrong_answer(self):
        self.assertTrue(hitab_exact_match_text("128154,21538", "128154, 21538"))
        self.assertTrue(hitab_exact_match_text("24492, 24062", "24492, 24062"))

    def test_wrong_values_still_fail(self):
        self.assertFalse(hitab_exact_match_text("128154,21539", "128154, 21538"))
        self.assertFalse(hitab_exact_match_text("128154", "128154, 21538"))

    def test_order_still_matters(self):
        self.assertFalse(hitab_exact_match_text("21538, 128154", "128154, 21538"))

    def test_single_value_gold_is_untouched(self):
        self.assertTrue(hitab_exact_match_text("5813", "5813"))
        self.assertFalse(hitab_exact_match_text("5813", "9307"))
