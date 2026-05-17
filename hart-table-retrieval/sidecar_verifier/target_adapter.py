"""TARGET-benchmark adapter for our VerifierAgent.

Implements two retrievers:
  - VectorBaselineRetriever  : dense vector retrieval only (baseline)
  - VerifierRerankRetriever  : same vector retrieval + our query-aware verifier rerank

Both share the same embedding index, so the comparison is apples-to-apples — the only
difference is whether the rerank step runs.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import chromadb
import pandas as pd

from target_benchmark.retrievers import AbsCustomEmbeddingRetriever

from src.retrieval.embedder import EmbedderFactory
from sidecar_verifier.agent.reconciler import rerank
from sidecar_verifier.agent.verifier import verify_hits
from sidecar_verifier.store.table_store import TableStore


_DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"


def _table_to_markdown(df: pd.DataFrame, title: str = "", max_rows: int = 100) -> str:
    """Light serialization: title + markdown table (truncated)."""
    n = min(df.shape[0], max_rows)
    head = df.iloc[:n]
    try:
        body = head.to_markdown(index=False)
    except Exception:
        body = head.to_string(index=False)
    return (f"# {title}\n\n" if title else "") + body


def _df_to_hitab_like(df: pd.DataFrame, table_id: str) -> dict:
    """Wrap a DataFrame in the dict shape our TableStore.build_table_record consumes."""
    cols = [str(c) for c in df.columns]
    # Top-root with single level: each column header is a leaf.
    top_children = {
        c: {"name": c, "value": c, "type": "data", "line_idx": 0, "children_dict": {}}
        for c in cols
    }
    top_root = {"name": "ROOT", "value": "ROOT", "type": "root", "line_idx": -1,
                "children_dict": top_children}
    # Left-root flat: each row labelled by row index.
    left_children = {
        str(i): {"name": str(i), "value": str(i), "type": "data", "line_idx": 0, "children_dict": {}}
        for i in range(df.shape[0])
    }
    left_root = {"name": "ROOT", "value": "ROOT", "type": "root", "line_idx": -1,
                 "children_dict": left_children}
    data = [[{"value": v} for v in row] for row in df.itertuples(index=False, name=None)]
    return {"table_id": table_id, "title": "", "top_root": top_root, "left_root": left_root, "data": data}


class _BaseChromaRetriever(AbsCustomEmbeddingRetriever):
    """Shared embedding + indexing logic. Subclasses override `retrieve`."""

    def __init__(
        self,
        chroma_dir: str = "/home/user/T2/hart-table-retrieval/data/target_chroma",
        model_name: str = _DEFAULT_MODEL,
        top_k_vectors: int = 50,
    ) -> None:
        super().__init__(expected_corpus_format="dataframe")
        self.chroma_dir = chroma_dir
        self.model_name = model_name
        self.top_k_vectors = top_k_vectors

        Path(chroma_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.embedder = EmbedderFactory.create({"name": model_name, "type": "sentence-transformer"})
        self._stores: Dict[str, TableStore] = {}
        self._collections: Dict[str, "chromadb.Collection"] = {}
        # Maps composite "db_id||table_id" -> (db_id, table_id) for retrieve()
        self._id_to_pair: Dict[str, Dict[str, Tuple[str, str]]] = {}

    # ---- TARGET API ----
    def embed_corpus(self, dataset_name: str, corpus: Iterable[Dict]) -> None:
        col_name = f"target_{dataset_name}".replace("-", "_")
        try:
            self._collections[dataset_name] = self.client.get_collection(col_name)
            print(f"  [{dataset_name}] reusing existing collection ({self._collections[dataset_name].count()} vecs)")
        except Exception:
            self._collections[dataset_name] = self.client.create_collection(
                col_name, metadata={"hnsw:space": "cosine"}
            )

        store = TableStore()
        id_map: Dict[str, Tuple[str, str]] = {}

        col = self._collections[dataset_name]
        existing = col.count()

        texts: List[str] = []
        ids: List[str] = []
        metas: List[Dict] = []

        for batch in corpus:
            # batch is dict-of-lists (column-batched). Iterate row-wise.
            db_ids = batch.get("database_id", [])
            table_ids = batch.get("table_id", [])
            tables = batch.get("table", [])
            for db_id, table_id, table in zip(db_ids, table_ids, tables):
                if table is None:
                    continue
                if not isinstance(table, pd.DataFrame):
                    try:
                        table = pd.DataFrame(table)
                    except Exception:
                        continue

                composite = f"{db_id}||{table_id}"
                id_map[composite] = (str(db_id), str(table_id))
                store.add(_df_to_hitab_like(table, composite))

                text = _table_to_markdown(table)
                texts.append(text)
                ids.append(composite)
                metas.append({"table_id": composite, "db_id": str(db_id)})

        self._stores[dataset_name] = store
        self._id_to_pair[dataset_name] = id_map

        if existing == 0 and texts:
            B = 64
            for i in range(0, len(texts), B):
                chunk_texts = texts[i:i + B]
                emb = self.embedder.embed(chunk_texts)
                col.upsert(
                    ids=ids[i:i + B],
                    documents=chunk_texts,
                    embeddings=emb.tolist(),
                    metadatas=metas[i:i + B],
                )
            print(f"  [{dataset_name}] indexed {len(texts)} tables; store={len(store)}")

    # ---- helpers ----
    def _vector_search(self, query: str, dataset_name: str) -> List[Dict]:
        col = self._collections[dataset_name]
        q_emb = self.embedder.embed([query])[0].tolist()
        res = col.query(query_embeddings=[q_emb], n_results=self.top_k_vectors,
                        include=["metadatas", "documents", "distances"])
        per: "OrderedDict[str, Dict]" = OrderedDict()
        for vec_id, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                            res["metadatas"][0], res["distances"][0]):
            tid = meta.get("table_id") or vec_id
            score = 1.0 - float(dist)
            if tid not in per or score > per[tid]["score"]:
                per[tid] = {"table_id": tid, "score": score, "chunk_text": doc, "meta": meta}
        return sorted(per.values(), key=lambda x: -x["score"])


class VectorBaselineRetriever(_BaseChromaRetriever):
    def retrieve(self, query: str, dataset_name: str, top_k: int, **kwargs) -> List[Tuple]:
        hits = self._vector_search(query, dataset_name)[:top_k]
        id_map = self._id_to_pair[dataset_name]
        return [id_map[h["table_id"]] for h in hits if h["table_id"] in id_map]


class VerifierRerankRetriever(_BaseChromaRetriever):
    def __init__(self, *args, w_vector: float = 0.8, w_verify: float = 0.2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.w_vector = w_vector
        self.w_verify = w_verify

    def retrieve(self, query: str, dataset_name: str, top_k: int, **kwargs) -> List[Tuple]:
        hits = self._vector_search(query, dataset_name)
        verified = verify_hits(query, self._stores[dataset_name], hits)
        ranked = rerank(verified, w_vector=self.w_vector, w_verify=self.w_verify)[:top_k]
        id_map = self._id_to_pair[dataset_name]
        return [id_map[h["table_id"]] for h in ranked if h["table_id"] in id_map]
