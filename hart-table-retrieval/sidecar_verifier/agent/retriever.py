"""Thin wrapper around the existing ChromaDB collection used by HART."""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional

import chromadb

from src.retrieval.embedder import EmbedderFactory
from src.retrieval.indexer import _model_short_name, _sanitize_collection_name


class VectorRetriever:
    def __init__(
        self,
        chroma_dir: str,
        model_name: str = "BAAI/bge-large-en-v1.5",
        serializer: str = "plain_markdown",
        device: Optional[str] = None,
    ) -> None:
        self.client = chromadb.PersistentClient(path=chroma_dir)
        model_config = {"name": model_name, "type": "sentence-transformer", "device": device}
        self.embedder = EmbedderFactory.create(model_config)
        col_name = _sanitize_collection_name(
            f"{serializer}_{_model_short_name(model_name)}"
        )
        self.collection = self.client.get_collection(col_name)
        self.serializer = serializer

    def retrieve(self, query: str, top_k_vectors: int = 20, top_k_tables: int = 5) -> List[Dict]:
        q_emb = self.embedder.embed([query])[0]
        if hasattr(q_emb, "tolist"):
            q_emb = q_emb.tolist()
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k_vectors,
            include=["metadatas", "documents", "distances"],
        )

        # Aggregate hits by table_id; keep best (smallest distance) per table.
        per_table: "OrderedDict[str, Dict]" = OrderedDict()
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        for vec_id, doc, meta, dist in zip(ids, docs, metas, dists):
            tid = meta.get("table_id") or meta.get("uid") or vec_id.split("__")[0]
            score = 1.0 - float(dist)
            if tid not in per_table or score > per_table[tid]["score"]:
                per_table[tid] = {
                    "table_id": tid,
                    "score": score,
                    "vector_id": vec_id,
                    "chunk_text": doc,
                    "meta": meta,
                }
        ranked = sorted(per_table.values(), key=lambda x: -x["score"])
        return ranked[:top_k_tables]
