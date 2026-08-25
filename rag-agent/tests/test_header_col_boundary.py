# SPDX-License-Identifier: MIT
"""The row-header region has to reach the column that holds the row's NAME.

Both shapes below are real RealHiTBench tables. In each the content scan stops
on a numeric column that sits INSIDE the row-header region, so the label every
query actually says lands in the data and every row of the table ends up with
the same address.
"""
import unittest

from rag_agent.reconstruct import guess_n_header_cols


class TestHeaderColBoundary(unittest.TestCase):
    def test_group_label_spans_unlabeled_stub_columns(self):
        """biology-table03: group label, numeric country CODE, country NAME, years."""
        grid = [["Product, country code and name", "", "", "2013", "2014"],
                ["Total", "3370", "Chile", "239,624", "279,026"],
                ["", "1220", "Canada", "148,547", "104,520"],
                ["", "4039", "Norway", "40,591", "57,896"]]
        self.assertEqual(guess_n_header_cols(grid, n_header_rows=1), 3)

    def test_leading_index_column_is_not_the_whole_stub(self):
        """economy-table137: BLS "Indent Level" (0,1,2,3), then the real label."""
        grid = [["Indent Level", "Expenditure category", "Relative importance", "Apr. 2022"],
                ["0", "All items", "100.000", "289.109"],
                ["1", "Food", "13.474", "298.711"],
                ["2", "Food at home", "08.663", "282.161"]]
        self.assertEqual(guess_n_header_cols(grid, n_header_rows=1), 2)

    def test_the_extension_needs_a_labelled_column_to_its_right(self):
        """Same table twice; only the last header cell differs.

        Without a label further right there is nothing saying the blanks are a
        stub the group label spans -- they are just an empty sheet. An earlier
        version of this rule extended anyway and swallowed RealHiTBench's blank
        header bands whole, which is worse than the miss it fixes.
        """
        with_label = [["Group", "", "", "2013"],
                      ["Total", "3370", "Chile", "239,624"],
                      ["", "1220", "Canada", "148,547"]]
        without = [["Group", "", "", ""],
                   ["Total", "3370", "Chile", "239,624"],
                   ["", "1220", "Canada", "148,547"]]
        self.assertEqual(guess_n_header_cols(with_label, n_header_rows=1), 3)
        self.assertLess(guess_n_header_cols(without, n_header_rows=1), 3)

    def test_a_plain_labelled_table_is_unchanged(self):
        grid = [["Item", "Country", "Budget", "Actual"],
                ["Pen", "USA", "15", "13"],
                ["", "UK", "29", "41"]]
        self.assertEqual(guess_n_header_cols(grid, n_header_rows=1), 2)


if __name__ == "__main__":
    unittest.main()
