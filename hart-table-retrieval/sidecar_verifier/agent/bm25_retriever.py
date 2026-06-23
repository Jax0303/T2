"""BM25 sparse retriever — drop-in replacement for VectorRetriever.

Uses ``rank_bm25.BM25Okapi`` over the same serialized table texts as the dense
index, so the only thing being compared is the retrieval signal (sparse vs
dense). Implements ``.retrieve(query, top_k_vectors, top_k_tables) -> List[Dict]``
with the same return schema, so it plugs into ``VerifierAgent`` unchanged.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

from rank_bm25 import BM25Okapi


_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """BM25 over a pre-serialized table corpus.

    Parameters
    ----------
    corpus : iterable of (table_id, doc_text)
        One document per unique table — same serialization used by the dense
        retriever (plain_markdown / json_kv / header_path).
    """

    def __init__(self, corpus: Iterable[Tuple[str, str]]) -> None:
        self.table_ids: List[str] = []
        self.docs: List[str] = []
        for tid, doc in corpus:
            self.table_ids.append(tid)
            self.docs.append(doc)
        if not self.docs:
            raise ValueError("BM25Retriever: empty corpus")
        self.tokenized = [_tokenize(d) for d in self.docs]
        self.bm25 = BM25Okapi(self.tokenized)

    def retrieve(
        self,
        query: str,
        top_k_vectors: int = 20,  # ignored — BM25 is 1 doc per table
        top_k_tables: int = 5,
    ) -> List[Dict]:
        q_tokens = _tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        # argpartition for top-k then sort
        k = min(top_k_tables, len(scores))
        if k == 0:
            return []
        # indices sorted by descending score
        import numpy as np
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        out: List[Dict] = []
        for i in idx:
            out.append({
                "table_id": self.table_ids[i],
                "score": float(scores[i]),
                "vector_id": self.table_ids[i],
                "chunk_text": self.docs[i],
                "meta": {"table_id": self.table_ids[i]},
            })
        return out


def build_bm25_from_samples(samples, serializer) -> BM25Retriever:
    """Build a BM25Retriever using ``serializer`` over unique tables in ``samples``.

    ``serializer`` is an instance from ``src.serializers``. We call
    ``serializer.serialize(table_data, header_tree=None)`` and concatenate the
    returned chunks per table.
    """
    seen = set()
    corpus: List[Tuple[str, str]] = []
    for s in samples:
        # Only use samples that actually have an attached table — skip stubs
        # where loader.load_table() couldn't find the file.
        if "table" not in s:
            continue
        table = s["table"]
        tid = table.get("table_id") or table.get("uid")
        if not tid or tid in seen:
            continue
        seen.add(tid)
        try:
            chunks = serializer.serialize(table)
        except TypeError:
            chunks = serializer.serialize(table, header_tree=None)
        doc = "\n".join(text for text, _meta in chunks).strip()
        if not doc:
            continue
        corpus.append((tid, doc))
    return BM25Retriever(corpus)
