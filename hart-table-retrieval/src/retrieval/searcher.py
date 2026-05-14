import json
import logging
from collections import defaultdict
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class TableSearcher:
    def __init__(self, collection, embedder, hart_scorer=None):
        self.collection = collection
        self.embedder = embedder
        self.hart_scorer = hart_scorer

    def search(
        self,
        query: str,
        top_k_vectors: int = 50,
        top_k_tables: int = 10,
    ) -> List[dict]:
        """
        Search for tables matching the query.
        Returns list of {"table_id": str, "score": float} sorted by score desc.
        """
        query_embedding = self.embedder.embed([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k_vectors, self.collection.count()),
            include=["metadatas", "distances", "documents"],
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        # Convert distances to similarities (ChromaDB cosine: distance = 1 - sim)
        candidate_results = []
        for doc_id, meta, dist in zip(ids, metadatas, distances):
            similarity = 1.0 - dist  # cosine distance to similarity
            path = json.loads(meta.get("path", "[]"))
            candidate_results.append({
                "doc_id": doc_id,
                "table_id": meta.get("table_id", "unknown"),
                "content_similarity": similarity,
                "path": path,
                "depth": meta.get("depth", 0),
            })

        if self.hart_scorer:
            ranked = self.hart_scorer.rank_tables(
                query, candidate_results, top_k=top_k_tables
            )
            return [{"table_id": tid, "score": score} for tid, score in ranked]
        else:
            # Group by table_id, take max similarity
            grouped = defaultdict(float)
            for r in candidate_results:
                tid = r["table_id"]
                grouped[tid] = max(grouped[tid], r["content_similarity"])

            table_scores = sorted(grouped.items(), key=lambda x: x[1], reverse=True)
            return [
                {"table_id": tid, "score": score}
                for tid, score in table_scores[:top_k_tables]
            ]

    def search_batch(
        self,
        queries: List[str],
        top_k_vectors: int = 50,
        top_k_tables: int = 10,
    ) -> List[List[dict]]:
        """Search for multiple queries."""
        return [
            self.search(q, top_k_vectors, top_k_tables)
            for q in queries
        ]
