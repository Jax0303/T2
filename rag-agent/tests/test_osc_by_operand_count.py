"""The cell arm's OSC edge is worst where the operand set is largest.

Guards the finding, not just the code. Strict monotonicity in ``m`` does NOT
hold -- at 2048+ the m=1 dump arm saturates (OSC 1.000), so its delta collapses
to zero by ceiling rather than by weakness. What holds without exception, over
17 budget x retriever x corpus points, is that ``m>=3`` is the worst bucket.
"""
import json
import os

import pytest

from scripts.osc_by_operand_count import bucket, split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAMILIES = [
    (f"corpus_dump_vs_cell_dense_{b}", ) for b in (128, 256, 512, 1024, 2048, 4096)
] + [
    (f"corpus_dump_vs_cell_bm25_{b}", ) for b in (128, 256, 512, 1024, 2048, 4096)
] + [
    (f"corpus_dump_vs_cell_multihiertt_bm25_{b}", ) for b in (128, 256, 512, 1024, 2048)
]


def _path(stem):
    return os.path.join(ROOT, "results", f"{stem}_records.jsonl")


def test_bucket_edges():
    assert bucket(1) == "m=1"
    assert bucket(2) == "m=2"
    assert bucket(3) == bucket(12) == "m>=3"


@pytest.mark.parametrize("stem", [f[0] for f in FAMILIES])
def test_many_operand_bucket_is_always_the_worst(stem):
    got = split(_path(stem))
    assert got is not None, "records predate the m field"
    delta = {k: c - d for k, (_, d, c) in got.items()}
    assert delta["m>=3"] <= min(delta["m=1"], delta["m=2"]) + 1e-9, delta


def test_all_or_nothing_tax_is_the_cell_arm_s():
    """dump pays no tax by construction -- the table is in or out."""
    recs = [json.loads(l) for l in open(_path("corpus_dump_vs_cell_dense_2048"))]
    hard = [r for r in recs if r["m"] >= 3]
    tax = lambda a: (sum(r[a]["per_cell"] for r in hard)
                     - sum(r[a]["osc"] for r in hard)) / len(hard)
    assert tax("dump") == pytest.approx(0.0, abs=1e-9)
    assert tax("cell") > 0.15
