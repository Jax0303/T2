# SPDX-License-Identifier: MIT
"""Stage 1 of the cascade: retrieve the TABLE(s) a query needs, before cells.

The operand-targeted retriever searches cells *inside one table* — every result
in this repo handed it the gold table from the dataset annotation, so table
selection was an oracle, not a measured step. This module adds the missing first
stage: rank whole tables against the query, keep the top-M, and let cell
retrieval run only inside them.

Each table is one retrieval unit: its S3 cell sentences joined into a single
document, embedded with the SAME encoder that embeds the cells
(embedding-consistency rule, ``encoders.py``). ``cascade_retrieve`` picks the
top-M tables and runs a per-table cell retriever inside them, merging cells by
score. ``gold_table_id`` is optional and used only to report whether the gold
table survived stage 1 — it never steers retrieval.

The pipeline this completes:  query -> [TableIndex: top-M tables]
                                     -> [OperandTargetedRetriever: cells] -> reader
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..serialization import caption as s3
from ..stores.original_store import OriginalTable
from .encoders import Encoder


def table_document(table: OriginalTable, template: str = "structural") -> str:
    """One retrieval document per table: all its S3 cell sentences, joined.

    Same serializer as the cell index, so stage 1 ranks tables in the same
    text space the cells are later ranked in.
    """
    chunks = s3.serialize(table, granularity="cell", template=template)
    return "\n".join(c.text for c in chunks)


class TableIndex:
    """Whole-table dense index over a corpus: query -> top-M table ids."""

    def __init__(self, encoder: Encoder, template: str = "structural") -> None:
        self.encoder = encoder
        self.template = template
        self.table_ids: List[str] = []
        self._vecs: Optional[np.ndarray] = None  # (n_tables, dim), L2-normalized

    def build(self, tables: Dict[str, OriginalTable]) -> "TableIndex":
        self.table_ids = list(tables.keys())
        docs = [table_document(tables[t], self.template) for t in self.table_ids]
        # encoders return L2-normalized rows, so a dot product is cosine.
        self._vecs = np.asarray(self.encoder.encode(docs), dtype=np.float32)
        return self

    def search(self, query: str, m: int = 1) -> List[Tuple[str, float]]:
        """Top-``m`` (table_id, score), best first. Empty if the index is empty."""
        if self._vecs is None or not self.table_ids:
            return []
        q = np.asarray(self.encoder.encode([query]), dtype=np.float32)[0]
        scores = self._vecs @ q
        order = np.argsort(-scores)[: max(1, m)]
        return [(self.table_ids[int(i)], float(scores[int(i)])) for i in order]


@dataclass
class CascadeResult:
    table_ids: List[str]                 # tables chosen in stage 1, ranked
    gold_table_hit: bool                 # was the gold table among them (if known)
    retrieved: list                      # merged cell RetrievedChunk, best first
    per_table: dict = field(default_factory=dict)  # tid -> OperandRetrievalResult


def cascade_retrieve(
    query: str,
    table_index: TableIndex,
    tables: Dict[str, OriginalTable],
    cell_retriever,
    m: int = 1,
    k: int = 5,
    gold_table_id: Optional[str] = None,
) -> CascadeResult:
    """Stage 1 (tables) then stage 2 (cells inside them), merged by score.

    ``cell_retriever`` is an :class:`OperandTargetedRetriever`; it caches its
    per-table index internally, so calling it across the top-M tables is cheap.
    Cells are merged on ``chunk_id`` (unique corpus-wide, table_id embedded), so
    two tables never collide.
    """
    chosen = [tid for tid, _ in table_index.search(query, m)]
    hit = gold_table_id in chosen if gold_table_id is not None else False
    merged: Dict[str, object] = {}
    per_table = {}
    for tid in chosen:
        if tid not in tables:
            continue
        r = cell_retriever.retrieve(query, tables[tid], k=k)
        per_table[tid] = r
        for rc in r.retrieved:
            cur = merged.get(rc.chunk.chunk_id)
            if cur is None or rc.score > cur.score:
                merged[rc.chunk.chunk_id] = rc
    retrieved = sorted(merged.values(), key=lambda h: -h.score)[:k]
    return CascadeResult(table_ids=chosen, gold_table_hit=hit,
                         retrieved=retrieved, per_table=per_table)
