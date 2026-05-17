"""End-to-end agent: retrieve → query-aware verify → reconcile → (optional) answer trace."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..store.table_store import TableStore
from .answerer import AnswerResult, LocalLLMAnswerer
from .reconciler import disagreement, filter_only, filter_then_rerank, rerank
from .retriever import VectorRetriever
from .tracer import TraceResult, trace
from .verifier import verify_hits


@dataclass
class AgentResult:
    query: str
    vector_ranked: List[Dict]
    verified_hits: List[Dict]            # verified, vector order
    final_ranked: List[Dict]             # after reconciler
    disagreement_signal: Dict
    answer: Optional[AnswerResult] = None
    trace: Optional[TraceResult] = None


class VerifierAgent:
    """
    mode:
      - "rerank"        : fuse vector + verification (default)
      - "filter"        : drop low-confidence candidates, keep vector order
      - "filter+rerank" : drop then fuse
    """

    def __init__(
        self,
        retriever: VectorRetriever,
        store: TableStore,
        mode: str = "rerank",
        w_vector: float = 0.8,
        w_verify: float = 0.2,
        filter_threshold: float = 0.2,
        answerer: Optional[LocalLLMAnswerer] = None,
    ) -> None:
        self.retriever = retriever
        self.store = store
        self.mode = mode
        self.w_vector = w_vector
        self.w_verify = w_verify
        self.filter_threshold = filter_threshold
        self.answerer = answerer

    def _reconcile(self, hits: List[Dict]) -> List[Dict]:
        if self.mode == "filter":
            return filter_only(hits, threshold=self.filter_threshold)
        if self.mode == "rerank":
            return rerank(hits, w_vector=self.w_vector, w_verify=self.w_verify)
        if self.mode == "filter+rerank":
            return filter_then_rerank(
                hits, threshold=self.filter_threshold,
                w_vector=self.w_vector, w_verify=self.w_verify,
            )
        raise ValueError(f"Unknown mode: {self.mode}")

    def run(
        self,
        query: str,
        top_k_vectors: int = 20,
        top_k_tables: int = 10,
        candidate_answer: Optional[str] = None,
    ) -> AgentResult:
        vector_hits = self.retriever.retrieve(
            query, top_k_vectors=top_k_vectors, top_k_tables=top_k_tables
        )
        verified = verify_hits(query, self.store, vector_hits)
        final = self._reconcile(verified)
        signal = disagreement(vector_hits, final)

        answer_result: Optional[AnswerResult] = None
        trace_result: Optional[TraceResult] = None

        top_rec = self.store.get(final[0]["table_id"]) if final else None
        if top_rec is not None:
            # If no candidate provided and LLM is available, generate one.
            if candidate_answer is None and self.answerer is not None:
                answer_result = self.answerer.answer(query, top_rec)
                candidate_answer = answer_result.answer
            if candidate_answer:
                trace_result = trace(candidate_answer, top_rec)

        return AgentResult(
            query=query,
            vector_ranked=vector_hits,
            verified_hits=verified,
            final_ranked=final,
            disagreement_signal=signal,
            answer=answer_result,
            trace=trace_result,
        )
