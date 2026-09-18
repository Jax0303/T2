# SPDX-License-Identifier: MIT
"""Self-check for scripts/bottleneck_root_cause.py: the margin bucketer, and an
end-to-end sanity check that predicted_structure's row_map/col_map-based
mapping (NOT tree_reconstruct_hitab_raw.py's value-matching align()) lands on
a real table without raising and produces plausible, non-gold-leaking paths.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bottleneck_root_cause import margin_bucket, predicted_structure  # noqa: E402
from bottleneck_diagnosis import split_corpus_table_ids  # noqa: E402
from rag_agent.bench import hitab_grid as hg  # noqa: E402


def test_margin_bucket_edges():
    assert margin_bucket(None) == "unknown"
    assert margin_bucket(-0.01) == "distractor_below_gold"
    assert margin_bucket(0.0) == "0.00-0.05"
    assert margin_bucket(0.049) == "0.00-0.05"
    assert margin_bucket(0.05) == "0.05-0.15"
    assert margin_bucket(0.149) == "0.05-0.15"
    assert margin_bucket(0.15) == "0.15-0.30"
    assert margin_bucket(0.29) == "0.15-0.30"
    assert margin_bucket(0.30) == ">=0.30"
    assert margin_bucket(1.0) == ">=0.30"


def test_predicted_structure_on_a_real_table_returns_plausible_paths():
    tid = split_corpus_table_ids()[0]
    row_path_fn, col_path_fn, meta = predicted_structure(tid, "data/hitab")
    assert row_path_fn is not None and col_path_fn is not None
    assert meta["nhr"] >= 1 and meta["nhc"] >= 1

    t = hg.load_table(tid, "data/hitab").table
    n_checked = 0
    for j in range(t.n_cols):
        cp = col_path_fn(j)
        if cp is None:
            continue
        assert isinstance(cp, list) and all(isinstance(s, str) for s in cp)
        n_checked += 1
    assert n_checked > 0                  # the mapping actually reaches data columns
