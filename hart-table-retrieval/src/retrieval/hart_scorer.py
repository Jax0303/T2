import numpy as np
from collections import defaultdict
from typing import List, Tuple


class HARTScorer:
    def __init__(self, embedder, alpha: float = 0.5):
        self.embedder = embedder
        self.alpha = alpha
        self._header_cache: dict[str, np.ndarray] = {}

    def _get_header_embedding(self, header_text: str) -> np.ndarray:
        if header_text not in self._header_cache:
            emb = self.embedder.embed([header_text])
            self._header_cache[header_text] = emb[0]
        return self._header_cache[header_text]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def compute_structural_alignment(
        self, query_embedding: np.ndarray, path: List[str], depth: int
    ) -> float:
        if not path or depth == 0:
            return 0.0

        d = depth
        total = 0.0
        for k_idx, header in enumerate(path):
            k = k_idx + 1
            h_emb = self._get_header_embedding(header)
            cos_sim = self._cosine_similarity(query_embedding, h_emb)
            weight = k / d
            total += cos_sim * weight
        return total / d

    def score_table(
        self, query_embedding: np.ndarray, sub_doc_results: List[dict]
    ) -> float:
        score = 0.0
        for doc in sub_doc_results:
            content_sim = doc["content_similarity"]
            path = doc.get("path", [])
            depth = doc.get("depth", 0)

            if path:
                alignment = self.compute_structural_alignment(
                    query_embedding, path, depth
                )
                combined = content_sim * (self.alpha + (1 - self.alpha) * alignment)
            else:
                combined = content_sim
            score += combined
        return score

    def rank_tables(
        self, query: str, candidate_results: List[dict], top_k: int = 10
    ) -> List[Tuple[str, float]]:
        query_embedding = self.embedder.embed([query])[0]

        grouped = defaultdict(list)
        for r in candidate_results:
            grouped[r["table_id"]].append(r)

        table_scores = []
        for table_id, docs in grouped.items():
            score = self.score_table(query_embedding, docs)
            table_scores.append((table_id, score))

        table_scores.sort(key=lambda x: x[1], reverse=True)
        return table_scores[:top_k]
