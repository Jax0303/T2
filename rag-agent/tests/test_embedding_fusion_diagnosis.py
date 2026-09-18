# SPDX-License-Identifier: MIT
"""Self-check for scripts/embedding_fusion_diagnosis.py's miss classifier."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from embedding_fusion_diagnosis import classify_miss  # noqa: E402


def test_either_channel_alone_correct_is_a_combination_failure():
    assert classify_miss(True, False) == "combination_failure"
    assert classify_miss(False, True) == "combination_failure"
    assert classify_miss(True, True) == "combination_failure"


def test_neither_channel_alone_correct_is_a_shared_representation_failure():
    assert classify_miss(False, False) == "shared_representation_failure"
