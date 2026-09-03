# SPDX-License-Identifier: MIT
"""``hitab_{split}_lookup_multi`` is selected by ANSWER cells, not operands.

The first freeze of this population used ``len(gold_operands)>=2`` and 47 of its
81 queries had a one-cell answer -- ``_coords_of`` pools every ``quantity_link``
bucket, so a figure the question quotes becomes an operand. These four checks
pin the corrected definition so that mistake cannot come back silently.
"""
import json
import unittest
from pathlib import Path

from rag_agent.bench import population as pop_mod
from rag_agent.bench.hitab import load_queries, load_samples

DATA = "data/hitab"
SPLITS = ("dev", "test", "train")
EXPECTED_N = {"dev": 33, "test": 31, "train": 150}
ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}


def _frozen(split):
    got = pop_mod.read(f"hitab_{split}_lookup_multi")
    if got is None:
        raise unittest.SkipTest(f"hitab_{split}_lookup_multi is not frozen")
    return got


def _samples(split):
    return {s["id"]: s for s in load_samples(DATA, split)}


class TestLookupMultiPopulation(unittest.TestCase):
    def test_every_query_has_at_least_two_answer_cells(self):
        for split in SPLITS:
            ids, _ = _frozen(split)
            samples = _samples(split)
            for qid in ids:
                lc = (samples[qid].get("linked_cells") or {})
                ans = (lc.get("quantity_link") or {}).get("[ANSWER]") or {}
                self.assertGreaterEqual(
                    len(ans), 2, f"{split}/{qid} answer is {len(ans)} cell(s)")

    def test_no_arithmetic_queries(self):
        for split in SPLITS:
            ids, _ = _frozen(split)
            samples = _samples(split)
            for qid in ids:
                agg = samples[qid].get("aggregation")
                agg = agg[0] if isinstance(agg, list) and agg else agg
                self.assertEqual((agg or "none"), "none", f"{split}/{qid} agg={agg}")
                self.assertNotIn((agg or "none"), ARITH)

    def test_disjoint_from_single_cell_lookup_pool(self):
        for split in SPLITS:
            ids, _ = _frozen(split)
            single = pop_mod.read(f"hitab_{split}_lookup_all")
            if single is None:
                continue
            overlap = set(ids) & set(single[0])
            self.assertEqual(overlap, set(), f"{split} overlaps lookup_all: {overlap}")

    def test_header_declares_the_same_n_as_the_body(self):
        for split in SPLITS:
            ids, meta = _frozen(split)
            text = pop_mod.path(f"hitab_{split}_lookup_multi").read_text()
            declared = [l for l in text.splitlines() if l.startswith("# n=")]
            self.assertEqual(declared, [f"# n={len(ids)}"], split)
            self.assertEqual(meta.get("n"), len(ids), split)
            self.assertEqual(len(ids), EXPECTED_N[split], split)
            self.assertEqual(len(set(ids)), len(ids), f"{split} has duplicates")


if __name__ == "__main__":
    unittest.main()
