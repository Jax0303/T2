# SPDX-License-Identifier: MIT
"""Collision counting is per CELL, and cross-table means more than one table."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from cell_sentence_collision import collisions


class TestCollisions(unittest.TestCase):
    def test_all_distinct_is_zero(self):
        r = collisions(["a", "b", "c"], [("t1", 0, 0), ("t1", 0, 1), ("t2", 0, 0)])
        self.assertEqual((r["dup_cell_rate"], r["cross_table_cell_rate"]), (0.0, 0.0))
        self.assertEqual(r["distinct"], 3)

    def test_a_class_of_n_costs_n_cells_not_one(self):
        """A duplicate class of 3 puts 3 cells in trouble, not 1."""
        r = collisions(["x"] * 3 + ["y"], [("t1", 0, 0), ("t2", 0, 0), ("t3", 0, 0),
                                           ("t1", 0, 1)])
        self.assertEqual(r["dup_cell_rate"], 0.75)
        self.assertEqual(r["max_class"], 3)
        self.assertEqual(r["max_class_tables"], 3)

    def test_same_table_duplicate_is_not_cross_table(self):
        """A duplicate inside one table still points the reader at the right table."""
        r = collisions(["x", "x"], [("t1", 0, 0), ("t1", 1, 0)])
        self.assertEqual(r["dup_cell_rate"], 1.0)
        self.assertEqual(r["cross_table_cell_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
