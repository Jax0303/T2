# SPDX-License-Identifier: MIT
"""Coverage check + fallback.

After operand-targeted retrieval we estimate, *without gold labels*, whether the
retrieved chunks actually cover the operands the decomposition asked for, and how
confident the decomposition was. If either is too low the pipeline falls back to
the full-table chunk so the generator still sees the answer cells — trading
tokens for safety. Every decision is logged so the fallback's value can be
ablated (full vs no-fallback).

  * ``coverage_rate`` — fraction of decomposed operands whose header-path match
    score clears ``score_floor``. A purely *self-supervised* proxy: with no gold
    labels, the only signal of a missed operand is that the decomposition could
    not confidently name its header path. (Measured caveat: this score is a weak
    miss predictor on HiTab — missed-operand queries average min-score 0.354 vs
    0.393 for fully-covered ones — so the trigger is deliberately conservative
    and the fallback's real value is established downstream by the full-vs-
    no-fallback answer-accuracy ablation, not by trigger precision.)
  * ``confidence``    — mean matcher score of the decomposed operands.

Fallback fires when ``coverage_rate < tau_cov`` OR ``confidence < tau_conf``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from ..bench.schema import BenchTable, Chunk
from ..serialize import serialize_table, fulltable_chunk, S2
from ..query.operand_decomposer import Operand


@dataclass
class CoverageReport:
    coverage_rate: float
    confidence: float
    n_operands: int
    n_covered: int
    fallback: bool
    reason: str
    tau_cov: float
    tau_conf: float

    def as_dict(self) -> dict:
        return {
            "coverage_rate": round(self.coverage_rate, 4),
            "confidence": round(self.confidence, 4),
            "n_operands": self.n_operands,
            "n_covered": self.n_covered,
            "fallback": self.fallback,
            "reason": self.reason,
        }


def assess(
    operands: Sequence[Operand],
    retrieved: Sequence[Chunk],
    table: BenchTable,
    tau_cov: float = 0.7,
    tau_conf: float = 0.0,
    score_floor: float = 0.3,
) -> CoverageReport:
    """Estimate coverage + confidence and decide whether to fall back.

    ``retrieved`` is accepted for interface symmetry but coverage is assessed
    from decomposition confidence only — checking the retrieved set against the
    operands we *searched for* is self-referential and near-always 1.0.
    """
    n = len(operands)
    covered = sum(1 for o in operands if o.score >= score_floor)
    coverage_rate = covered / n if n else 0.0
    confidence = (sum(o.score for o in operands) / n) if n else 0.0

    reasons = []
    if n == 0:
        reasons.append("no_operands")
    if coverage_rate < tau_cov:
        reasons.append(f"coverage<{tau_cov}")
    if confidence < tau_conf:
        reasons.append(f"confidence<{tau_conf}")
    fallback = bool(reasons)
    return CoverageReport(
        coverage_rate=coverage_rate,
        confidence=confidence,
        n_operands=n,
        n_covered=covered,
        fallback=fallback,
        reason=",".join(reasons) if reasons else "ok",
        tau_cov=tau_cov,
        tau_conf=tau_conf,
    )


def apply_fallback(
    retrieved: List[Chunk],
    table: BenchTable,
    report: CoverageReport,
    scheme: str = S2,
    max_full_rows: int = 60,
) -> List[Chunk]:
    """Return the context chunks, appending the full table when fallback fired.

    For large tables the full-table chunk would blow the token budget; above
    ``max_full_rows`` rows we instead append the table's individual row chunks
    (still the whole table, but the generator/truncation can window it).
    """
    if not report.fallback:
        return list(retrieved)
    out = list(retrieved)
    have = {c.chunk_id for c in out}
    if table.n_rows <= max_full_rows:
        full = fulltable_chunk(table, scheme)
        if full.chunk_id not in have:
            out.append(full)
    else:
        for ch in serialize_table(table, scheme):
            if ch.chunk_id not in have:
                out.append(ch)
    return out
