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
