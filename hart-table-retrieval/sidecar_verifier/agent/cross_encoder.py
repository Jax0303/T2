"""Cross-encoder reranker — standard strong rerank baseline.

Sits between vector retrieval and the verifier. Takes the top-K candidates from
the dense retriever and rescoring them with a (query, document) cross-encoder
(default ``BAAI/bge-reranker-base``). CPU-friendly.
"""
from __future__ import annotations

import os
from typing import Dict, List

from sentence_transformers import CrossEncoder


_DEFAULT_MODEL = "BAAI/bge-reranker-base"


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        max_pairs: int = 50,
        device: str | None = None,
    ) -> None:
        # Keep HF cache on D drive if env not already set elsewhere.
        os.environ.setdefault("HF_HOME", "/mnt/d/hart_data/hf_cache")
        os.environ.setdefault("HF_HUB_CACHE", "/mnt/d/hart_data/hf_cache")
        self.model_name = model_name
        self.max_pairs = max_pairs
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, hits: List[Dict]) -> List[Dict]:
        """Rescore ``hits`` (List[{table_id, score, chunk_text, ...}]) with the
        cross-encoder. Annotates ``ce_score`` and re-sorts. Original ``score`` is
        preserved for downstream consumers (e.g. verifier ensemble).
        """
        if not hits:
            return hits
        truncated = hits[: self.max_pairs]
        pairs = [(query, str(h.get("chunk_text") or "")) for h in truncated]
        scores = self.model.predict(pairs, show_progress_bar=False)
        for h, s in zip(truncated, scores):
            h["ce_score"] = float(s)
        # Anything beyond max_pairs gets sent to the back, preserving relative order.
        for h in hits[self.max_pairs:]:
            h["ce_score"] = float("-inf")
        return sorted(hits, key=lambda h: -h.get("ce_score", float("-inf")))
