import json
import logging
import re
from typing import List, Tuple

from tqdm import tqdm

logger = logging.getLogger(__name__)


def _sanitize_collection_name(name: str) -> str:
    """ChromaDB collection names: 3-63 chars, alphanumeric/underscores/hyphens."""
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_-")
    if len(name) < 3:
        name = name + "_col"
    return name[:63]


def _model_short_name(model_name: str) -> str:
    """Shorten model name for collection naming."""
    parts = model_name.split("/")
    short = parts[-1] if len(parts) > 1 else parts[0]
    return re.sub(r"[^a-zA-Z0-9]", "_", short)


class TableIndexer:
    def __init__(self, chroma_client, embedder, serializer_name: str, model_name: str):
        self.client = chroma_client
        self.embedder = embedder
        self.serializer_name = serializer_name
        self.model_name = model_name

        col_name = _sanitize_collection_name(
            f"{serializer_name}_{_model_short_name(model_name)}"
        )
        self.collection = self.client.get_or_create_collection(
            name=col_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Using collection: %s", col_name)

    def index_documents(
        self,
        documents: List[Tuple[str, dict]],
        batch_size: int = 100,
    ) -> int:
        """
        Index serialized documents into ChromaDB.
        documents: list of (text, metadata) tuples from a serializer.
        Returns: number of documents indexed.
        """
        total = len(documents)
        if total == 0:
            return 0

        for start in tqdm(range(0, total, batch_size), desc="Indexing"):
            batch = documents[start : start + batch_size]
            texts = [t for t, _ in batch]
            metas = [m for _, m in batch]

            embeddings = self.embedder.embed(texts)

            ids = []
            chroma_metas = []
            for i, meta in enumerate(metas):
                table_id = meta.get("table_id", "unknown")
                path = meta.get("path", [])
                depth = meta.get("depth", 0)

                if path:
                    doc_id = f"{table_id}_{start + i}"
                else:
                    doc_id = table_id

                ids.append(doc_id)
                chroma_metas.append({
                    "table_id": table_id,
                    "path": json.dumps(path),
                    "depth": depth,
                    "serializer": self.serializer_name,
                    "model": self.model_name,
                })

            self.collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=chroma_metas,
            )

        logger.info("Indexed %d documents into collection", total)
        return total

    def get_stats(self) -> dict:
        count = self.collection.count()
        return {
            "collection_name": self.collection.name,
            "total_vectors": count,
            "serializer": self.serializer_name,
            "model": self.model_name,
        }
