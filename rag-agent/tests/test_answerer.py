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
