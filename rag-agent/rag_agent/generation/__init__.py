"""Answer generation + scoring (pipeline component 5).

Consumes the assembled context from component 4 and produces an answer via one
of two paths — ``direct`` (LLM reads and answers) or ``codegen`` (LLM writes
Python over the operand cells, executed deterministically). Scoring reuses the
paper-aligned metrics in :mod:`rag_agent.eval.metrics`.
"""
from __future__ import annotations

from ..eval.metrics import exact_match, numeric_match
from .codegen import CodeResult, extract_code, run_codegen
from .generator import Answerer, AnswerResult
from .mock import MockLLM
from .prompts import cells_from_chunks


def score_answer(pred, gold, rel_tol: float = 0.02) -> dict:
    """Score one prediction: exact match + tolerant numeric/substring match.

    ``exec_acc`` mirrors ``numeric_match`` here — the codegen path's executed
    value is scored with the same tolerant numeric comparison; the aggregate
    execution accuracy is the mean of ``nm`` over codegen answers.
    """
    em = exact_match(pred, gold)
    nm = numeric_match(pred, gold, rel_tol=rel_tol)
    return {"em": em, "nm": nm}


__all__ = [
    "Answerer",
    "AnswerResult",
    "MockLLM",
    "CodeResult",
    "run_codegen",
    "extract_code",
    "cells_from_chunks",
    "score_answer",
    "exact_match",
    "numeric_match",
]
