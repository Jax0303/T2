# SPDX-License-Identifier: MIT
"""The decomposition-matcher policy must equal the committed evidence.

``rag_agent.query.matcher_policy.best_matcher`` claims a best matcher per corpus.
The claim is only allowed to exist because ``results/operand_rag/<bench>/summary.json``
measured it. This test reads those files and fails if the policy drifts from the
argmax of the committed ``ceiling`` block — so the policy can never be edited to a
value the numbers do not support without a fresh run replacing the numbers too.

Pure stdlib: no torch/embedder, runs without the experiment venv.
"""
import json
from pathlib import Path

import pytest

from rag_agent.query.matcher_policy import BENCH_CEILING, best_matcher, is_measured

ROOT = Path(__file__).resolve().parents[1]
BENCHES = ("hitab", "finqa", "wikisql")


def _committed_ceiling(bench: str):
    p = ROOT / "results" / "operand_rag" / bench / "summary.json"
    if not p.exists():
        pytest.skip(f"committed ceiling missing: {p}")
    return json.loads(p.read_text())["ceiling"]


@pytest.mark.parametrize("bench", BENCHES)
def test_table_matches_committed_file(bench):
    """The hard-coded BENCH_CEILING mirrors results/ to 4 decimals."""
    committed = _committed_ceiling(bench)
    for matcher, val in BENCH_CEILING[bench].items():
        assert round(committed[matcher], 4) == round(val, 4), (
            f"{bench}/{matcher}: table has {val}, results file has {committed[matcher]}"
        )


@pytest.mark.parametrize("bench", BENCHES)
def test_best_matcher_is_argmax_of_committed(bench):
    """best_matcher() returns the top-scoring matcher in the committed file."""
    committed = _committed_ceiling(bench)
    expected = max(committed, key=committed.get)
    assert best_matcher(bench) == expected


def test_expected_winners():
    """The measured outcome, spelled out: hierarchical -> embedding, flat -> hybrid."""
    assert best_matcher("hitab") == "embedding"
    assert best_matcher("finqa") == "embedding"
    assert best_matcher("wikisql") == "hybrid"


def test_unmeasured_bench_uses_hierarchical_prior():
    """A corpus with no committed ceiling falls back to the embedding prior,
    and is flagged as not measured so no number is quoted for it by accident."""
    assert best_matcher("multihiertt") == "embedding"
    assert not is_measured("multihiertt")
    assert is_measured("hitab")
