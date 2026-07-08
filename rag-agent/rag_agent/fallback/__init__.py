"""Coverage check + fallback (pipeline component 4).

Operand-targeted retrieval can silently miss operands — exactly the
retrieval-completeness gap the thesis measures. At inference time we cannot see
the gold operands, so we use two *self-assessed* signals to decide whether the
focused retrieval is trustworthy:

* **coverage rate** — of the operands HPIR decomposed, how many were actually
  grounded in a retrieved cell. Low coverage means the focused retrieval did not
  find what the query asked for.
* **HPIR confidence** — how strongly the query terms matched the decomposed
  operand header paths (see
  :func:`rag_agent.retrieve.operand_retrieval.decomposition_confidence`).

If either falls below its threshold the controller **falls back** to the whole
table (serialized rows, capped to a token budget) instead of the sparse operand
cells. Every decision is logged for the ablation that isolates the fallback's
value (full pipeline vs. no-fallback).

Note this is a design *consequence* of the decomposition ceiling, not a novel
retrieval trick: because HPIR cannot exceed ~0.685 operand recall, a safety net
above that ceiling is required for the pipeline to be reliable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..serialization import header_path as s2
from ..serialization.base import Chunk
from ..stores.original_store import OriginalTable
from ..retrieve.operand_retrieval import (
    OperandRetrievalResult,
    _covers,
    _norm_tokens,
)


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token), avoids a tokenizer dependency."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

@dataclass
class CoverageReport:
    n_operands: int
    n_covered: int
    covered: List[bool]            # per-operand, aligned with result.operands
    coverage_rate: float

    def to_dict(self) -> dict:
        return {
            "n_operands": self.n_operands,
            "n_covered": self.n_covered,
            "coverage_rate": round(self.coverage_rate, 4),
            "covered": self.covered,
        }


def operand_coverage(result: OperandRetrievalResult) -> CoverageReport:
    """Self-assessed coverage: which decomposed operands appear in retrieved cells.

    Unlike ``operand_recall_at_k`` (which needs gold operands), this compares the
    *decomposed* operands against the retrieved chunk header paths, so it is
    usable at inference time as a fallback trigger.
    """
    paths = result.covered_header_paths()
    covered: List[bool] = []
    for op in result.operands:
        gt = _norm_tokens(op.header_path)
        covered.append(any(_covers(gt, p) for p in paths))
    n = len(result.operands)
    n_cov = sum(covered)
    rate = (n_cov / n) if n else 0.0
    return CoverageReport(n_operands=n, n_covered=n_cov, covered=covered, coverage_rate=rate)


# ---------------------------------------------------------------------------
# Fallback decision
# ---------------------------------------------------------------------------

@dataclass
class FallbackDecision:
    triggered: bool
    reason: str                    # "ok" | "low_coverage" | "low_confidence" | "low_both" | "no_operands"
    coverage_rate: float
    confidence: float
    coverage_threshold: float
    confidence_threshold: float
    n_operands: int
    n_covered: int

    def to_dict(self) -> dict:
        return {
            "triggered": self.triggered,
            "reason": self.reason,
            "coverage_rate": round(self.coverage_rate, 4),
            "confidence": round(self.confidence, 4),
            "coverage_threshold": self.coverage_threshold,
            "confidence_threshold": self.confidence_threshold,
            "n_operands": self.n_operands,
            "n_covered": self.n_covered,
        }


def decide_fallback(
    result: OperandRetrievalResult,
    coverage: Optional[CoverageReport] = None,
    coverage_threshold: float = 0.7,
    confidence_threshold: float = 0.3,
) -> FallbackDecision:
    """Fallback iff coverage < tau_cov OR confidence < tau_conf."""
    cov = coverage or operand_coverage(result)
    low_cov = cov.coverage_rate < coverage_threshold
    low_conf = result.confidence < confidence_threshold

    if not result.operands:
        reason = "no_operands"
    elif low_cov and low_conf:
        reason = "low_both"
    elif low_cov:
        reason = "low_coverage"
    elif low_conf:
        reason = "low_confidence"
    else:
        reason = "ok"

    return FallbackDecision(
        triggered=(low_cov or low_conf or not result.operands),
        reason=reason,
        coverage_rate=cov.coverage_rate,
        confidence=result.confidence,
        coverage_threshold=coverage_threshold,
        confidence_threshold=confidence_threshold,
        n_operands=cov.n_operands,
        n_covered=cov.n_covered,
    )


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

@dataclass
class ContextBundle:
    chunks: List[Chunk]
    text: str
    used_fallback: bool
    n_chunks: int
    n_tokens: int
    truncated: bool
    decision: FallbackDecision

    def to_dict(self) -> dict:
        return {
            "used_fallback": self.used_fallback,
            "n_chunks": self.n_chunks,
            "n_tokens": self.n_tokens,
            "truncated": self.truncated,
            "decision": self.decision.to_dict(),
        }


def _pack(
    chunks: List[Chunk], max_tokens: int
) -> Tuple[List[str], List[Chunk], int, bool]:
    """Greedily pack chunk texts under a token budget."""
    texts: List[str] = []
    kept: List[Chunk] = []
    used = 0
    truncated = False
    for ch in chunks:
        t = estimate_tokens(ch.text)
        if used + t > max_tokens and kept:
            truncated = True
            break
        texts.append(ch.text)
        kept.append(ch)
        used += t
    return texts, kept, used, truncated


def build_context(
    result: OperandRetrievalResult,
    table: OriginalTable,
    decision: FallbackDecision,
    max_tokens: int = 4096,
    max_operand_cells: int = 12,
) -> ContextBundle:
    """Assemble the LLM context: operand cells normally, whole table on fallback.

    On fallback the table is serialized as S2 *rows* (header-path preserved) and
    packed under ``max_tokens`` so a large table never blows the budget.
    """
    if not decision.triggered:
        cells = [rc.chunk for rc in result.retrieved[:max_operand_cells]]
        texts, kept, used, truncated = _pack(cells, max_tokens)
        return ContextBundle(
            chunks=kept,
            text="\n".join(texts),
            used_fallback=False,
            n_chunks=len(kept),
            n_tokens=used,
            truncated=truncated,
            decision=decision,
        )

    # Fallback: whole-table S2 rows under budget.
    rows = s2.serialize(table, granularity="row")
    texts, kept, used, truncated = _pack(rows, max_tokens)
    return ContextBundle(
        chunks=kept,
        text="\n\n".join(texts),
        used_fallback=True,
        n_chunks=len(kept),
        n_tokens=used,
        truncated=truncated,
        decision=decision,
    )


def assemble_context(
    result: OperandRetrievalResult,
    table: OriginalTable,
    coverage_threshold: float = 0.7,
    confidence_threshold: float = 0.3,
    max_tokens: int = 4096,
    enable_fallback: bool = True,
) -> ContextBundle:
    """One-call coverage check + fallback decision + context assembly.

    ``enable_fallback=False`` forces the operand-cell context regardless of the
    signals — the no-fallback arm of the ablation.
    """
    cov = operand_coverage(result)
    decision = decide_fallback(result, cov, coverage_threshold, confidence_threshold)
    if not enable_fallback:
        decision.triggered = False
        decision.reason = "fallback_disabled"
    return build_context(result, table, decision, max_tokens=max_tokens)
