# SPDX-License-Identifier: MIT
"""Reader-output parsing: what the agent records as its answer."""
import unittest

from rag_agent.agent import _parse_final_answer


class TestParseFinalAnswer(unittest.TestCase):
    def test_marker_wins(self):
        raw = "Reasoning: 45 + 33 = 78.\nFinal answer: 78"
        self.assertEqual(_parse_final_answer(raw), "78")

    def test_marker_strips_trailing_prose(self):
        self.assertEqual(_parse_final_answer("Final answer: 2.27\nnote"), "2.27")

    def test_label_only_line_is_not_an_answer(self):
        """Small models emit 'Reasoning:' then run out of tokens. Scoring the
        label as the prediction is how a truncated run silently became a
        'wrong answer' instead of a non-answer."""
        self.assertEqual(_parse_final_answer("Reasoning:"), "")

    def test_no_marker_takes_the_conclusion_not_the_opener(self):
        """Without the marker we take the last content line as-is. We do NOT
        try to pull a number out of the prose: string answers ('living in a
        couple') are just as common, and a number-extraction heuristic would
        turn those into wrong answers rather than honest misses."""
        raw = "Reasoning:\nQuebec is 33, Alberta is 45.\nThe largest is 45"
        self.assertEqual(_parse_final_answer(raw), "The largest is 45")

    def test_empty_and_none(self):
        self.assertEqual(_parse_final_answer(""), "")
        self.assertEqual(_parse_final_answer(None), "")

    def test_answer_prefix_stripped(self):
        self.assertEqual(_parse_final_answer("The answer is 137.0"), "137.0")


if __name__ == "__main__":
    unittest.main()
