# SPDX-License-Identifier: MIT
"""The free-text -> gold-shape bridge, and the scale rule it cannot paper over.

Two separate defects were costing correct answers in
``results/baseline_comparison_llm_records.jsonl``:

  * shape -- HiTab's eval is handed a LIST of predicted cell values, so a model
    that answered "57.0, 31.2, 11.8" to a multi-value gold failed on type before
    any number was compared. Fixed in ``hitab_exact_match_text``.
  * scale -- the solver prompt ordered a decimal fraction for any ratio the model
    computed, which is wrong whenever the source cells are already percentages.
    Fixed in the prompt, NOT in the scorer: the scorer must stay strict or its
    numbers stop being comparable to published HiTab accuracies.

The second class is pinned here as a scorer NON-behaviour, so nobody later
"fixes" a scale mismatch by loosening the metric.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from rag_agent.eval.metrics import hitab_exact_match, hitab_exact_match_text  # noqa: E402
from rag_agent.generate.answerer import _DIRECT_SYS  # noqa: E402


@pytest.mark.parametrize("pred,gold", [
    ("57.0, 31.2, 11.8", [57.0, 31.2, 11.8]),   # real record, was scored wrong
    ("13, 25", [13.0, 25.0]),                   # real record, was scored wrong
    ("53.5, 21.1, 25.5", [53.5, 21.1, 25.5]),
    ("13 and 25", [13.0, 25.0]),
    ("24.7 to 14.6", [24.7, 14.6]),
])
def test_multi_value_free_text_is_matched(pred, gold):
    assert hitab_exact_match_text(pred, gold)
    assert not hitab_exact_match(pred, gold), "bare scorer should still fail on shape"


@pytest.mark.parametrize("pred,gold", [
    ("264", [264.0]),
    ("73.4", [73.4]),
    ("living in a couple", ["living in a couple"]),
])
def test_single_value_behaviour_is_unchanged(pred, gold):
    assert hitab_exact_match_text(pred, gold) == hitab_exact_match(pred, gold) is True


@pytest.mark.parametrize("pred,gold", [
    ("0.018", [1.8]),        # percent cells divided by 100 -- a prompt bug
    ("0.194", [19.4]),
    ("18.2", [0.071998]),    # fraction multiplied by 100
    ("32.9", [33.8, 25.3]),  # genuinely wrong values, right shape
    ("57.0, 31.2", [57.0, 31.2, 11.8]),          # wrong count
    ("11.8, 31.2, 57.0", [57.0, 31.2, 11.8]),    # order is not free
])
def test_no_leniency_is_smuggled_in(pred, gold):
    assert not hitab_exact_match_text(pred, gold)


def test_prompt_forbids_rescaling_and_asks_for_the_gold_shape():
    assert "NEVER multiply or divide by 100" in _DIRECT_SYS
    assert "separated by commas" in _DIRECT_SYS
    # the retired wording ordered the exact rescaling the gold answers never do
    assert "report the raw decimal fraction" not in _DIRECT_SYS
