"""Vector store: thin wrapper around ChromaDB + a SentenceTransformer embedder.

Reuses the existing prebuilt chroma index if present so we don't re-embed.
The serializer the chroma collection was built with is inferred from
``collection_name`` and is independent of the OriginalStore.
"""
from __future__ import annotations

import logging
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional

import chromadb

logger = logging.getLogger(__name__)


def _short_model(name: str) -> str:
    return name.split("/")[-1].lower().replace("-", "_").replace(".", "_")


def _sanitize_collection_name(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    return name[:63] if len(name) > 63 else name


@dataclass
class VectorHit:
    table_id: str
    score: float            # cosine sim ≈ 1 - distance
    vector_id: str
    chunk_text: str
    meta: dict

    def to_dict(self) -> dict:
        return {
            "table_id": self.table_id,
            "score": round(float(self.score), 4),
            "vector_id": self.vector_id,
            "chunk_text": self.chunk_text[:400],
        }


class VectorStore:
    """ChromaDB-backed dense retriever. GPU embedding when CUDA is available."""

    def __init__(
        self,
        chroma_dir: str,
        embedder_model: str = "BAAI/bge-large-en-v1.5",
        serializer: str = "plain_markdown",
        device: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer
        import torch

        self.chroma_dir = chroma_dir
        self.embedder_model = embedder_model
        self.serializer = serializer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(embedder_model, device=self.device, trust_remote_code=True)
        self._batch = 128 if self.device == "cuda" else 32
        logger.info("VectorStore embedder=%s device=%s", embedder_model, self.device)

        self.client = chromadb.PersistentClient(path=chroma_dir)
        col_name = collection_name or _sanitize_collection_name(
            f"{serializer}_{_short_model(embedder_model)}"
        )
        self.collection = self.client.get_collection(col_name)
        logger.info("VectorStore collection=%s size=%d", col_name, self.collection.count())

    # ---- inference-time API ----

    def embed_query(self, text: str):
        v = self.model.encode([text], batch_size=self._batch, convert_to_numpy=True, show_progress_bar=False)[0]
        return v.tolist()

    def search(self, query: str, top_k_vectors: int = 20, top_k_tables: int = 5) -> List[VectorHit]:
        q_emb = self.embed_query(query)
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k_vectors,
            include=["metadatas", "documents", "distances"],
        )
        per_table: "OrderedDict[str, VectorHit]" = OrderedDict()
        for vec_id, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            tid = meta.get("table_id") or meta.get("uid") or vec_id.split("__")[0]
            score = 1.0 - float(dist)
            if tid not in per_table or score > per_table[tid].score:
                per_table[tid] = VectorHit(
                    table_id=tid, score=score, vector_id=vec_id, chunk_text=doc, meta=meta or {}
                )
        ranked = sorted(per_table.values(), key=lambda h: -h.score)
        return ranked[:top_k_tables]

    def __len__(self) -> int:
        return self.collection.count()
