"""Adaptive table-RAG agent.

Stage flow (skipped stages are recorded in the trace, not silently dropped):

  1. classify_query(q)   → QueryIntent
  2. plan_stages(intent) → ordered list of Stage to run
  3. if RETRIEVE       : vector search top-K candidates
  4. if VERIFY         : cross-check vs OriginalStore, rerank
  5. if SYMBOLIC       : cell extract via LLM → safe pandas-style eval
  6. if LLM_ANSWER     : LLM reads verified top-1 (fallback / non-arithmetic)

Returns a single AgentResult; trace is exhaustive enough that downstream
eval can re-derive per-stage metrics offline.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .extract.cell_extractor import extract_plan
from .extract.decomposition_extractor import extract_plan_decomposed
from .extract.symbolic_eval import evaluate_plan
from .llm.base import BaseLLM
from .retrieve.verifier import rerank, verify_against_original
from .router.policy import Plan, Stage, plan_stages
from .router.query_classifier import QueryIntent, QueryType, classify_query
from .stores.original_store import OriginalStore, OriginalTable
from .stores.vector_store import VectorStore


def _format_table_for_reader(t: OriginalTable, max_rows: int = 40) -> str:
    """Render the verified table for the LLM reader. Same convention as the
    extractor's format (so the model sees a stable schema across paths)."""
    cols_block = []
    for c in range(t.n_cols):
        p = t.col_path(c)
        cols_block.append(f"  col[{c}]: {' > '.join(p)}" if p else f"  col[{c}]: (blank)")
    rows_block = []
    for r in range(min(t.n_rows, max_rows)):
        p = t.row_path(r)
        rh = " > ".join(p) if p else f"row_{r}"
        vals = []
        for c in range(min(t.n_cols, 12)):
            v = t.cell(r, c)
            vals.append("" if v is None else str(v))
        rows_block.append(f"  row[{r}] ({rh}): " + " | ".join(vals))
    note = "" if t.n_rows <= max_rows else f"\n  (...{t.n_rows - max_rows} more rows truncated)"
    return (
        f"Title: {t.title}\n"
        f"Columns (header paths):\n" + "\n".join(cols_block) + "\n"
        f"Data:\n" + "\n".join(rows_block) + note
    )


_READER_SYSTEM = (
    "You are a precise table QA assistant. Answer ONLY from the table below. "
    "Think step by step using `Reasoning:` then give the final answer after "
    "`Final answer:` on its own line. If the final answer is a number, output "
    "just the number (no units, no commas, no '%'). For fractions/percentages "
    "match the form used in the table values. If multiple, comma-separate. "
    "If unanswerable, write `Final answer: N/A`."
)

_FINAL_RE = re.compile(r"final\s*answer\s*[:\-]\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL)


def _parse_final_answer(raw: str) -> str:
    m = _FINAL_RE.search(raw or "")
    if m:
        s = m.group(1).strip()
    else:
        s = next((ln.strip() for ln in (raw or "").splitlines() if ln.strip()), raw or "")
    s = re.sub(r"^(answer|the answer is|=|:)\s*", "", s, flags=re.IGNORECASE)
    return s.rstrip(".").strip()


@dataclass
class AgentResult:
    query: str
    intent: Dict
    plan: Dict
    vector_ranked: List[Dict] = field(default_factory=list)
    final_ranked: List[Dict] = field(default_factory=list)
    top_table_id: Optional[str] = None
    symbolic: Optional[Dict] = None
    reader: Optional[Dict] = None
    answer: Optional[str] = None
    source: Optional[str] = None       # "symbolic" | "reader" | "reasoning-only"
    elapsed_s: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "intent": self.intent,
            "plan": self.plan,
            "vector_ranked": self.vector_ranked,
            "final_ranked": self.final_ranked,
            "top_table_id": self.top_table_id,
            "symbolic": self.symbolic,
            "reader": self.reader,
            "answer": self.answer,
            "source": self.source,
            "elapsed_s": round(self.elapsed_s, 3),
        }


class RAGAgent:
    def __init__(
        self,
        original_store: OriginalStore,
        vector_store: VectorStore,
        llm: BaseLLM,
        symbolic_llm: Optional[BaseLLM] = None,
        top_k_vectors: int = 20,
        top_k_tables: int = 5,
        w_vector: float = 0.7,
        w_verify: float = 0.3,
        extractor: str = "original",
        use_verify: bool = True,
        use_symbolic: bool = True,
        oracle_retrieval: bool = False,
    ) -> None:
        self.original = original_store
        self.vector = vector_store
        self.llm = llm
        self.symbolic_llm = symbolic_llm or llm
        self.top_k_vectors = top_k_vectors
        self.top_k_tables = top_k_tables
        self.w_vector = w_vector
        self.w_verify = w_verify
        self.extractor = extractor
        self.use_verify = use_verify
        self.use_symbolic = use_symbolic
        self.oracle_retrieval = oracle_retrieval

    def run(self, query: str, gold_table_id: Optional[str] = None) -> AgentResult:
        t0 = time.time()
        intent = classify_query(query)
        plan = plan_stages(intent)

        # Override plan stages based on agent-level ablation flags
        if not self.use_verify:
            plan.stages = [s for s in plan.stages if s != Stage.VERIFY]
            plan.reason += " [verify OFF]"
        if not self.use_symbolic:
            plan.stages = [s for s in plan.stages if s != Stage.SYMBOLIC]
            plan.reason += " [symbolic OFF]"

        out = AgentResult(
            query=query,
            intent={"qtype": intent.qtype.value, "needs_table": intent.needs_table,
                    "needs_symbolic": intent.needs_symbolic, "signals": intent.keywords},
            plan={"stages": [s.value for s in plan.stages], "reason": plan.reason},
        )

        # --- oracle retrieval: skip vector search, use gold table directly ---
        if self.oracle_retrieval and gold_table_id:
            out.top_table_id = gold_table_id
            out.final_ranked = [{"table_id": gold_table_id, "vector_score": 1.0,
                                 "verify_confidence": 1.0, "final_score": 1.0}]
            hits = []
        elif Stage.RETRIEVE in plan.stages:
            hits = self.vector.search(query, self.top_k_vectors, self.top_k_tables)
            out.vector_ranked = [h.to_dict() for h in hits]
        else:
            hits = []

        if not self.oracle_retrieval:
            if Stage.VERIFY in plan.stages and hits:
                # For multi-cell arithmetic queries, generic header keywords like
                # "total" / "percentage" match too many tables, so the verifier
                # hurts more than it helps. Down-weight verify confidence and
                # lean on the vector score in those cases.
                if intent.qtype in (QueryType.MULTI_OP_FORMULA, QueryType.ARITHMETIC_AGG):
                    w_vec, w_ver = 0.9, 0.1
                else:
                    w_vec, w_ver = self.w_vector, self.w_verify
                ranked = rerank(query, hits, self.original,
                                w_vector=w_vec, w_verify=w_ver)
                out.final_ranked = ranked
                out.top_table_id = ranked[0]["table_id"] if ranked else None
            elif hits:
                out.final_ranked = [{"table_id": h.table_id, "vector_score": h.score,
                                     "verify_confidence": None,
                                     "final_score": h.score} for h in hits]
                out.top_table_id = hits[0].table_id

        top_table = self.original.get(out.top_table_id) if out.top_table_id else None

        # --- symbolic path ---
        if Stage.SYMBOLIC in plan.stages and top_table is not None:
            if self.extractor == "decomposition":
                plan_obj = extract_plan_decomposed(self.symbolic_llm, query, top_table)
            else:
                plan_obj = extract_plan(self.symbolic_llm, query, top_table)
            sym = evaluate_plan(plan_obj, top_table)
            # Gate: only adopt symbolic answer when (a) eval succeeded AND
            # (b) the expression is non-trivial (≥2 operators OR multi-cell).
            # Otherwise let the reader speak — this avoids the case where a
            # spurious x1-x2 displaces a correct name-answer from the reader.
            op_count = sum(ch in "+-*/" for ch in plan_obj.expression)
            non_trivial = op_count >= 1 and len(plan_obj.cells) >= 2
            multi_op = op_count >= 2
            adopt = sym.ok and (multi_op or (intent.qtype == QueryType.ARITHMETIC_AGG and non_trivial))
            out.symbolic = {
                "extracted_cells": [c.__dict__ for c in plan_obj.cells],
                "extracted_expression": plan_obj.expression,
                "raw_llm_output": plan_obj.raw_llm_output[:800],
                "parse_ok": plan_obj.parse_ok,
                "op_count": op_count, "adopted": adopt,
                **sym.to_dict(),
            }
            if adopt:
                out.answer = _format_symbolic_answer(sym.value)
                out.source = "symbolic"

        # --- LLM reader path (fallback or primary for non-arithmetic) ---
        if out.answer is None and Stage.LLM_ANSWER in plan.stages:
            if top_table is None and intent.qtype != QueryType.REASONING_ONLY:
                out.answer = ""
                out.source = "no_table"
            else:
                if top_table is None:
                    user = f"Question: {query}\n\nNo table is available. Answer briefly."
                else:
                    user = (f"Table:\n{_format_table_for_reader(top_table)}\n\n"
                            f"Question: {query}")
                raw = self.llm.complete(_READER_SYSTEM, user, max_tokens=384)
                ans = _parse_final_answer(raw)
                out.reader = {"raw_output": raw[:1200], "answer": ans}
                out.answer = ans
                out.source = out.source or "reader"

        out.elapsed_s = max(0.0, time.time() - t0)
        return out


def _format_symbolic_answer(v: float) -> str:
    """Render a numeric answer as a plain string (no thousands sep, trim 0s)."""
    if v == int(v):
        return str(int(v))
    # round to 4 decimals, strip trailing zeros
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return s
