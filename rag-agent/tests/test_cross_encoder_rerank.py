# SPDX-License-Identifier: MIT
"""Pinning test for scripts/cross_encoder_rerank.py's McNemar helper -- no
model, no data files."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from cross_encoder_rerank import mcnemar


def test_mcnemar_keys_named_by_side():
    a = {1: True, 2: False, 3: True}
    b = {1: True, 2: True, 3: False}
    out = mcnemar(a, b, "baseline", "k20")
    assert out["baseline_only"] == 1     # query 3
    assert out["k20_only"] == 1          # query 2
    assert out["discordant"] == 2
    assert out["baseline_r1"] == round(2 / 3, 4)
    assert out["k20_r1"] == round(2 / 3, 4)


def test_mcnemar_identical_dicts_no_discordant():
    a = {1: True, 2: False}
    out = mcnemar(a, dict(a), "baseline", "k50")
    assert out["discordant"] == 0
    assert out["p_value"] == 1.0
