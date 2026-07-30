# SPDX-License-Identifier: MIT
"""완전성 판정 — the all-or-nothing completeness gate, as a RUNTIME control signal.

:mod:`rag_agent.eval.operand_set` measures Operand-Set Completeness *after the
fact*, against gold operands, to score a run. That is an evaluation metric and
cannot run at inference time — at inference there are no gold operands.

The invention disclosure requires the same all-or-nothing criterion to act
instead as a control signal *during* execution: when the retrieved evidence does
not contain every operand the requested computation needs, the executor asks for
a wider search budget or for header-scope enumeration, and the similarity search
and structural complement are re-run. This module is that gate.

What makes it gold-free: the operand set required by the computation is the one
the **query decomposition** produced (:class:`~rag_agent.query.operand_decomposer.Operand`),
not the gold answer's. So the judgment is "is every operand I decided I needed
actually grounded in a retrieved cell?" — answerable at inference time, and
all-or-nothing exactly as the metric is. A partial-credit reading would defeat
the purpose: one missing operand makes a sum wrong, so a gate that passes at
"most operands found" passes precisely the queries that go on to fail.

The escalation ladder mirrors the disclosure's two remedies, cheapest first:

1. **budget** — re-run the similarity search at a larger ``k``.
2. **structural** — additionally union in the aggregate rows the index already
   flagged (:mod:`rag_agent.retrieve.structure_attr`). This is what recovers an
   operand carrying no similarity signal, the case a bigger ``k`` alone cannot
   fix.
3. **enumerate** — fall back to deterministic header-scope enumeration
   (:func:`rag_agent.retrieve.header_enum.enumerate_scope`), which returns the
   scope's cells by tree structure instead of by rank.

Every attempt is recorded in :class:`GateTrace` so a run can report how often
the gate fired, and at which rung it was satisfied.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple

from .hybrid_index import RetrievedChunk
from .operand_retrieval import (OperandRetrievalResult, OperandTargetedRetriever,
                                _covers)

#: Default escalation ladder: (stage name, k multiplier, inject_structural).
DEFAULT_LADDER: Tuple[Tuple[str, int, bool], ...] = (
    ("budget", 4, False),
    ("structural", 4, True),
)


@dataclass
class CompletenessVerdict:
    """All-or-nothing judgment over the decomposed operand set."""

    complete: bool
    n_required: int
    n_grounded: int
    missing: List[str] = field(default_factory=list)  # ungrounded operand paths

    @property
    def coverage(self) -> float:
        """Partial view, for logging only — never the gate criterion."""
        return self.n_grounded / self.n_required if self.n_required else 1.0


def judge_completeness(result: OperandRetrievalResult) -> CompletenessVerdict:
    """1 iff EVERY decomposed operand is grounded in some retrieved index unit.

    An operand is grounded when a retrieved unit's header path contains all of
    the operand's path tokens — the same ``_covers`` predicate the repo's
    operand-recall metric uses, so gate and metric agree on what "found" means.

    A query that decomposed into no operands is vacuously complete: there is
    nothing for the gate to demand, and escalating would burn budget blindly.
    """
    paths = result.covered_header_paths()
    missing: List[str] = []
    for op in result.operands:
        if not any(_covers(op.key_tokens(), p) for p in paths):
            missing.append(" > ".join(op.header_path))
    n_req = len(result.operands)
    return CompletenessVerdict(
        complete=not missing,
        n_required=n_req,
        n_grounded=n_req - len(missing),
        missing=missing,
    )


@dataclass
class GateAttempt:
    stage: str          # "initial", then a ladder rung name
    k: int
    inject_structural: bool
    verdict: CompletenessVerdict


@dataclass
class GateTrace:
    """What the gate did, for the ablation that isolates its value."""

    attempts: List[GateAttempt] = field(default_factory=list)

    @property
    def fired(self) -> bool:
        """True if the initial retrieval was judged incomplete."""
        return bool(self.attempts) and not self.attempts[0].verdict.complete

    @property
    def resolved_at(self) -> Optional[str]:
        """Ladder rung that reached completeness, or None if never reached."""
        for a in self.attempts:
            if a.verdict.complete:
                return a.stage
        return None

    def to_dict(self) -> dict:
        return {
            "fired": self.fired,
            "resolved_at": self.resolved_at,
            "n_attempts": len(self.attempts),
            "attempts": [
                {"stage": a.stage, "k": a.k, "inject_structural": a.inject_structural,
                 "complete": a.verdict.complete, "n_required": a.verdict.n_required,
                 "n_grounded": a.verdict.n_grounded, "missing": a.verdict.missing}
                for a in self.attempts
            ],
        }


def retrieve_with_gate(
    retriever: OperandTargetedRetriever,
    query: str,
    table,
    k: int = 5,
    llm=None,
    ladder: Sequence[Tuple[str, int, bool]] = DEFAULT_LADDER,
    inject_structural: bool = False,
    enumerate_fallback: bool = True,
) -> Tuple[OperandRetrievalResult, GateTrace]:
    """Retrieve, judge completeness, and escalate while the judgment fails.

    Returns the last (widest) result together with the trace. The result is
    returned even when the ladder is exhausted without reaching completeness —
    the caller still has to answer, and ``trace.resolved_at is None`` marks the
    answer as one the gate could not vouch for.

    Note the gate can only ever be as good as the decomposition: an operand the
    decomposer never produced is not in ``n_required``, so the gate cannot know
    to look for it. It bounds retrieval failures, not decomposition failures.
    """
    result = retriever.retrieve(query, table, k=k, llm=llm,
                               inject_structural=inject_structural)
    verdict = judge_completeness(result)
    trace = GateTrace(attempts=[GateAttempt("initial", k, inject_structural, verdict)])
    if verdict.complete:
        return result, trace

    widest_k = k
    for stage, k_mult, inject in ladder:
        widest_k = max(k * k_mult, k + 1)
        # `or inject_structural`: a rung never revokes an already-enabled
        # complement — the ladder only ever widens.
        use_inject = inject or inject_structural
        result = retriever.retrieve(query, table, k=widest_k, llm=llm,
                                    inject_structural=use_inject)
        verdict = judge_completeness(result)
        trace.attempts.append(GateAttempt(stage, widest_k, use_inject, verdict))
        if verdict.complete:
            return result, trace

    if enumerate_fallback:
        result = _enumerate_rung(retriever, query, table, result)
        verdict = judge_completeness(result)
        trace.attempts.append(GateAttempt("enumerate", widest_k, True, verdict))
    return result, trace


def _enumerate_rung(retriever: OperandTargetedRetriever, query: str, table,
                    result: OperandRetrievalResult) -> OperandRetrievalResult:
    """헤더 범위 열거 — union in the resolved scope's cells by tree structure.

    Ranking has already failed by this point, so this rung stops ranking: it
    resolves the query to header paths that exist in the table and enumerates
    every numeric cell under them. Enumerated cells enter at score 0.0, like the
    structural complement — they were admitted by structure, not similarity.

    Cells with no corresponding index unit are skipped rather than synthesized,
    so the evidence set stays made of real index units.
    """
    from ..query.header_path_resolver import resolve_against_table
    from .header_enum import enumerate_scope

    if not hasattr(table, "cell_num"):
        return result  # enumeration needs numeric cell access
    intent = resolve_against_table(query, table)
    scope = enumerate_scope(table, intent.row_paths, intent.col_paths)

    index = retriever.index_table(table)
    by_cell = {(ch.row_index, ch.col_index): ch for ch in index.chunks
               if ch.row_index is not None and ch.col_index is not None}
    have = {rc.chunk.chunk_id for rc in result.retrieved}
    added = list(result.retrieved)
    n_new = 0
    for cell in scope.cells:
        ch = by_cell.get(cell)
        if ch is not None and ch.chunk_id not in have:
            added.append(RetrievedChunk(chunk=ch, score=0.0, bm25=0.0, dense=0.0))
            n_new += 1
    return replace(result, retrieved=added, n_injected=result.n_injected + n_new)
