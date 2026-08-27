# SPDX-License-Identifier: MIT
"""The population that gives OSC up, and the arm that ignores the budget.

PREREG-2026-08-27-rhb-nr-large-tables. Both invariants here fail SILENTLY if
they break -- a wrong population or a fabricated OSC of 1.0 does not raise, it
just prints a different number.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from corpus_dump_vs_cell import (ARMS, RHB_POPS, osc_share,  # noqa: E402
                                 realhitbench_corpus)

HTML = """<table>
<tr><td>Region</td><td>2015</td><td>2016</td></tr>
<tr><td>Ontario</td><td>10</td><td>20</td></tr>
<tr><td>Quebec</td><td>30</td><td>40</td></tr>
<tr><td>Alberta</td><td>50</td><td>60</td></tr>
</table>"""

QA = [
    # answer "20" IS a cell -> survives the answer-match filter
    {"id": 1, "FileName": "t1", "Question": "Ontario in 2016?",
     "QuestionType": "Numerical Reasoning", "SubQType": "Calculation",
     "FinalAnswer": "20", "ProcessedAnswer": "20"},
    # answer 10+30+50 = 90 is written in NO cell -> the filter drops it
    {"id": 2, "FileName": "t1", "Question": "Total 2015?",
     "QuestionType": "Numerical Reasoning", "SubQType": "Calculation",
     "FinalAnswer": "90", "ProcessedAnswer": "90"},
    # an NR SubQType on a Fact Checking question: 4 of these exist in the real
    # file, which is why the filter is on QuestionType and not on SubQType
    {"id": 3, "FileName": "t1", "Question": "Is Quebec 30 in 2015?",
     "QuestionType": "Fact Checking", "SubQType": "Comparison",
     "FinalAnswer": "30", "ProcessedAnswer": "30"},
]


class TestRhbNrCorpus(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        (d / "html").mkdir()
        (d / "html" / "t1.html").write_text(HTML)
        (d / "QA_final.json").write_text(json.dumps({"queries": QA}))
        self.d = str(d)

    def tearDown(self):
        self.tmp.cleanup()

    def _ids(self, **kw):
        C = realhitbench_corpus(self.d, pin=False, **kw)
        return [q["query_id"] for q in C.queries]

    def test_default_keeps_only_answers_that_resolve_to_cells(self):
        self.assertEqual(self._ids(), ["1", "3"])

    def test_nr_population_keeps_the_computed_answer(self):
        """The 7.0% survival rate is the reason this population exists."""
        self.assertEqual(self._ids(**RHB_POPS["rhb_nr_all"]), ["1", "2"])

    def test_qtypes_filters_on_question_type_not_subqtype(self):
        """id 3 carries an NR SubQType but is Fact Checking -- it must not be in."""
        self.assertNotIn("3", self._ids(**RHB_POPS["rhb_nr_all"]))

    def test_the_computed_answer_query_carries_no_gold_cells(self):
        C = realhitbench_corpus(self.d, pin=False, **RHB_POPS["rhb_nr_all"])
        by_id = {q["query_id"]: q for q in C.queries}
        self.assertEqual(by_id["2"]["gold_cells"], set())
        self.assertTrue(by_id["1"]["gold_cells"])


class TestOscShare(unittest.TestCase):
    def test_no_gold_cells_is_undefined_not_perfect(self):
        """The bug this guards: len(hit) == len(gold) is True for two empty sets."""
        self.assertEqual(osc_share(set(), set()), (None, None))

    def test_complete_set_scores_one(self):
        self.assertEqual(osc_share({("t", 0, 0)}, {("t", 0, 0)}), (1, 1.0))

    def test_partial_set_scores_zero_but_reports_the_share(self):
        self.assertEqual(osc_share({("t", 0, 0)}, {("t", 0, 0), ("t", 0, 1)}),
                         (0, 0.5))


class TestGoldTableArm(unittest.TestCase):
    def test_goldtable_is_a_registered_arm(self):
        self.assertIn("goldtable", ARMS)


if __name__ == "__main__":
    unittest.main()
