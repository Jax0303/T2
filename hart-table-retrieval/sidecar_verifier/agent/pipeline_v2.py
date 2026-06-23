"""End-to-end pipeline v2: retrieve → verify → reconcile → route → read → trace.

Replaces the v1 ``VerifierAgent`` for the experimental track. The reader is
called via one of three routes depending on the query type:

  α (alpha)  full table markdown → LLM answer
  β (beta)   verifier-driven sub-table → LLM answer
  γ (gamma)  LLM generates pandas expression → sandbox eval

The router (rule-based; HiTab gold available for evaluation) decides which
route to take per query. On code-gen failure (syntax / runtime / timeout) we
fall back to β.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

from ..store.table_store import TableRecord, TableStore
from .reconciler import rerank
from .retriever import VectorRetriever
from .router import Route, route_query, gold_route_from_formula
from .sandbox import SandboxResult, run_sandboxed
from .subtable import SubtableResult, extract_subtable
from .tracer import TraceResult, trace
from .verifier import verify_hits


def _df_with_named_axes(rec: TableRecord):
    """Return a copy of rec.df whose columns and index match the labels we
    show the LLM in the schema prompt — so generated code can address cells
    by header-path strings."""
    df = rec.df.copy()
    col_names = [" / ".join(rec.col_header_path(c)) or f"col_{c}"
                 for c in range(df.shape[1])]
    row_names = [" / ".join(rec.row_header_path(r)) or f"row_{r}"
                 for r in range(df.shape[0])]
    df.columns = col_names
    df.index = row_names
    return df


class ReaderProto(Protocol):
    """Anything that satisfies this works as a v2 reader."""

    def answer_full(self, query: str, rec: TableRecord): ...
    def answer_subtable(self, query: str, sub: SubtableResult,
                        table_id: str = ..., title: str = ...): ...
    def code_for_query(self, query: str, rec: TableRecord): ...


@dataclass
class PipelineResult:
    query: str
    final_ranked: List[Dict]
    top_table_id: Optional[str]
    route: Route
    route_gold: Optional[Route]
    answer: str
    answer_source: str            # "alpha" | "beta" | "gamma_code" | "gamma_fallback_beta" | "abstain"
    code: Optional[str] = None
    sandbox_error: Optional[str] = None
    trace: Optional[TraceResult] = None
    raw_outputs: Dict = field(default_factory=dict)


class MockReader:
    """Stand-in reader for testing the pipeline without API access."""

    def answer_full(self, query, rec):
        from .answerer import AnswerResult
        return AnswerResult(answer="N/A", raw_output="(mock)", table_id=rec.table_id)

    def answer_subtable(self, query, sub, table_id="", title=""):
        from .answerer import AnswerResult
        return AnswerResult(answer="N/A", raw_output="(mock)", table_id=table_id)

    def code_for_query(self, query, rec):
        from .groq_reader import CodeResult
        return CodeResult(code="'N/A'", raw_output="(mock)", table_id=rec.table_id)


class VerifierAgentV2:
    def __init__(
        self,
        retriever: VectorRetriever,
        store: TableStore,
        reader: ReaderProto,
        w_verify: float = 0.2,
        top_k_vectors: int = 20,
        top_k_tables: int = 10,
    ) -> None:
        self.retriever = retriever
        self.store = store
        self.reader = reader
        self.w_verify = w_verify
        self.top_k_vectors = top_k_vectors
        self.top_k_tables = top_k_tables

    def run(self, query: str, *,
            gold_formulas=None, gold_answer=None,
            forced_top_table: Optional[str] = None) -> PipelineResult:
        # ---- 1. retrieve ----
        if forced_top_table is not None:
            # Oracle retrieval — pin the gold table at position 0
            ranked = [{"table_id": forced_top_table, "score": 1.0, "verification": {}}]
        else:
            vec_hits = self.retriever.retrieve(
                query, top_k_vectors=self.top_k_vectors, top_k_tables=self.top_k_tables,
            )
            verified = verify_hits(query, self.store, vec_hits)
            ranked = rerank(verified, w_vector=1.0 - self.w_verify, w_verify=self.w_verify)

        top_id = ranked[0]["table_id"] if ranked else None
        top_rec = self.store.get(top_id) if top_id else None

        # ---- 2. route ----
        route = route_query(query)
        route_gold: Optional[Route] = None
        if gold_formulas is not None:
            route_gold = gold_route_from_formula(gold_formulas, gold_answer)

        # ---- 3. read ----
        answer = "N/A"
        answer_source = "abstain"
        code: Optional[str] = None
        sandbox_error: Optional[str] = None
        raw_outputs: Dict = {}

        if top_rec is None:
            return PipelineResult(
                query=query, final_ranked=ranked, top_table_id=top_id,
                route=route, route_gold=route_gold,
                answer=answer, answer_source=answer_source,
                code=code, sandbox_error=sandbox_error, trace=None, raw_outputs=raw_outputs,
            )

        if route == "gamma":
            code_res = self.reader.code_for_query(query, top_rec)
            code = code_res.code
            raw_outputs["code_raw"] = code_res.raw_output
            # Use a renamed view so the schema we showed the LLM matches the
            # actual column / index labels at exec time.
            df_named = _df_with_named_axes(top_rec)
            sandbox = run_sandboxed(code, df_named, timeout_sec=5)
            if sandbox.ok:
                answer = str(sandbox.value) if sandbox.value is not None else "N/A"
                answer_source = "gamma_code"
            else:
                sandbox_error = f"{sandbox.error_type}: {sandbox.error}"
                # Fallback to beta (sub-table → LLM)
                sub = extract_subtable(query, top_rec)
                ans = self.reader.answer_subtable(
                    query, sub, table_id=top_id, title=top_rec.title,
                )
                answer = ans.answer
                answer_source = "gamma_fallback_beta"
                raw_outputs["beta_raw"] = ans.raw_output
        elif route == "beta":
            sub = extract_subtable(query, top_rec)
            ans = self.reader.answer_subtable(
                query, sub, table_id=top_id, title=top_rec.title,
            )
            answer = ans.answer
            answer_source = "beta"
            raw_outputs["beta_raw"] = ans.raw_output
        else:  # alpha
            ans = self.reader.answer_full(query, top_rec)
            answer = ans.answer
            answer_source = "alpha"
            raw_outputs["alpha_raw"] = ans.raw_output

        # ---- 4. trace ----
        trace_res = None
        try:
            trace_res = trace(answer, top_rec)
        except Exception:
            trace_res = None

        return PipelineResult(
            query=query, final_ranked=ranked, top_table_id=top_id,
            route=route, route_gold=route_gold,
            answer=answer, answer_source=answer_source,
            code=code, sandbox_error=sandbox_error,
            trace=trace_res, raw_outputs=raw_outputs,
        )
