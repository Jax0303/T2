# SPDX-License-Identifier: MIT
"""Data-free unit tests for the answerer (no LLM): exec guard, extraction, eval."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.bench.schema import Chunk
from rag_agent.generate.answerer import (
    _safe_exec, _extract_code, _parse_number, format_context, evaluate_answer,
)


def test_safe_exec_arithmetic():
    assert _safe_exec("answer = (9.7 + 8.8 + 9) / (9.7 + 8.8 + 9 + 4.5)") == pytest.approx(0.86, abs=0.01)
    assert _safe_exec("answer = sum([1, 2, 3])") == 6.0


def test_safe_exec_blocks_import_and_attr():
    for bad in ["import os\nanswer=1", "answer = (1).__class__", "answer = open('x')"]:
        with pytest.raises(ValueError):
            _safe_exec(bad)


def test_extract_code_strips_fence_and_print():
    raw = "```python\nanswer = 5 + 5\nprint(answer)\n```"
    code = _extract_code(raw)
    assert "print" not in code
    assert _safe_exec(code) == 10.0


def test_parse_number_takes_last():
    assert _parse_number("the result is 1,234.5 dollars") == 1234.5
    assert _parse_number("no digits here") is None


def test_format_context_respects_budget():
    chunks = [Chunk("t", f"t#r{i}", "x" * 100, [i], [0]) for i in range(100)]
    ctx = format_context(chunks, max_context_tokens=50)  # 50*4=200 chars budget
    assert len(ctx) <= 300  # at most a couple chunks


def test_evaluate_answer_numeric_tolerance_and_string():
    assert evaluate_answer(154983.0, [154983]) is True
    assert evaluate_answer(100.3, [100]) is True       # within ±2% tolerance
    assert evaluate_answer(130.0, [100]) is False      # outside tolerance
    assert evaluate_answer("Guard-Forward", ["guard-forward"]) is True



# --- answer-shape parsing -------------------------------------------------

def test_bare_numbers_and_their_decoration():
    from rag_agent.generate.answerer import _as_bare_number
    for text, want in [("641", 641.0), ("641.0", 641.0), (" 641.0 ", 641.0),
                       ("1,426,352", 1426352.0), ("$1,234", 1234.0),
                       ("5.3%", 5.3), ("-2.5", -2.5), ("3.0.", 3.0)]:
        assert _as_bare_number(text) == want, text


def test_prose_and_labels_stay_strings():
    """A reply is only a number when the WHOLE reply is a number.

    The old rule dug the last number out of arbitrary prose, which turned gold
    string answers like "15 to 19" into 19.0 — a wrong answer where an honest
    string comparison would have been right. HiTab's own scorer does no prose
    extraction either.
    """
    from rag_agent.generate.answerer import _as_bare_number
    for text in ["15 to 19", "filipina women", "the answer is 641",
                 "women", "", "2015 to 2016"]:
        assert _as_bare_number(text) is None, text


def test_scale_rule_distinguishes_computed_from_printed():
    """HiTab stores a COMPUTED ratio as a decimal fraction (0.053), but a
    percentage already printed in a cell stays as printed (25.3). An unscoped
    "never multiply by 100" told the reader to divide looked-up cells by 100,
    which the official scorer counts as wrong."""
    from rag_agent.generate.answerer import _RATIO_RULE
    assert "COMPUTE" in _RATIO_RULE
    assert "as printed" in _RATIO_RULE
