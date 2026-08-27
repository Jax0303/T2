# SPDX-License-Identifier: MIT
"""The table-level discriminator of PREREG-2026-08-25-structural-discriminator.

S2h and S2hr differ on the one property the prereg exists to buy: S2hr is
CONSTANT across a table (so it can narrow retrieval to one table), S2h is not.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from corpus_dump_vs_cell import s2h_cell_text, s2h_prefixes  # noqa: E402


class C:
    """Two tables. t1's row axis has two roots, t2's has one."""
    cell_paths = [(["Ontario", "Toronto"], ["2016", "Male"], "122"),
                  (["Quebec", "Montreal"], ["2016", "Male"], "99"),
                  (["Ontario", "Toronto"], ["2016", "Male"], "7")]
    cell_owner = [("t1", 0, 0), ("t1", 1, 0), ("t2", 0, 0)]
    cell_text = ["Ontario > Toronto | 2016 > Male: 122",
                 "Quebec > Montreal | 2016 > Male: 99",
                 "Ontario > Toronto | 2016 > Male: 7"]


class TestPrefixes(unittest.TestCase):
    def test_s2h_excludes_labels_already_in_the_cell_path(self):
        got = s2h_prefixes(C, "S2h")
        self.assertNotIn("Ontario", got[0])          # in its own path
        self.assertIn("Quebec", got[0])              # the sibling root is not

    def test_s2h_is_not_constant_within_a_table(self):
        """Why it cannot narrow: the term differs cell to cell."""
        got = s2h_prefixes(C, "S2h")
        self.assertNotEqual(got[0], got[1])          # same table t1

    def test_s2hr_is_constant_within_a_table(self):
        got = s2h_prefixes(C, "S2hr")
        self.assertEqual(got[0], got[1])

    def test_s2hr_picks_the_rarest_root_across_the_corpus(self):
        """'Ontario' is a root in both tables, 'Quebec' in one -- t1 takes Quebec."""
        self.assertEqual(s2h_prefixes(C, "S2hr")[0], "Quebec")

    def test_text_falls_back_to_bare_s2_when_there_is_no_prefix(self):
        got = s2h_cell_text(C, "S2h")
        self.assertTrue(got[2].startswith("Ontario >"))   # t2 has one root, in path
        self.assertTrue(got[0].startswith("[Quebec] "))


if __name__ == "__main__":
    unittest.main()
