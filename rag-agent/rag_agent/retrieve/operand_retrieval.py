"""Operand-targeted retrieval (pipeline component 3).

Instead of issuing one query for the whole question, HPIR decomposes the
question into the individual *operands* it needs — each a hierarchical header
path identifying a cell — and the retriever fetches each operand separately
from a hybrid index over the table's S2 cell chunks. The union of those hits is
the evidence passed downstream.

The headline metric is ``operand_recall@k``: of the operands the gold answer
actually depends on, how many were surfaced in the top-k retrieved cells. The
thesis argument is that this recall — not whole-table retrieval — is the
binding constraint, and that it is capped by the HPIR decomposition ceiling.

Gold-operand extraction from HiTab uses ``linked_cells.entity_link`` (the real
row/column header leaves the answer binds to). Cell-coordinate reconciliation
across HiTab's merged-header grid is deliberately avoided here; the header-path
level is both well-defined and the level the thesis cares about.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..query.header_path_resolver import extract_target_terms, resolve_intent
from ..serialization import header_path as s2
from ..serialization.base import Chunk
from ..stores.original_store import OriginalTable
from .encoders import Encoder
from .hybrid_index import HybridIndex, RetrievedChunk

_NORM_RE = re.compile(r"[^a-z0-9]+")


def _norm_tokens(parts: Sequence[str]) -> List[str]:
    toks: List[str] = []
    for p in parts:
        toks += [t for t in _NORM_RE.split(str(p).lower()) if t]
    return toks


# ---------------------------------------------------------------------------
# Operand decomposition
# ---------------------------------------------------------------------------

@dataclass
class Operand:
    """One header-path target the question needs, plus its retrieval query."""

    header_path: List[str]            # row_path + col_path (most specific)
    row_path: List[str] = field(default_factory=list)
    col_path: List[str] = field(default_factory=list)
    value_type: str = "number"
    query_text: str = ""

    def key_tokens(self) -> List[str]:
        return _norm_tokens(self.header_path)


def decompose_operands(
    query: str,
    table: OriginalTable,
    llm=None,
    max_rows: int = 2,
    max_cols: int = 2,
) -> List[Operand]:
    """Decompose ``query`` into operand header-path targets via HPIR.

    Pairs the top row header paths with the top column header paths into
    (row, col) operands; if one axis is empty, falls back to single-axis
    operands so retrieval still has something to target.
    """
    intent = resolve_intent(query, table, llm=llm, top_n_cols=max_cols, top_n_rows=max_rows)
    rows = intent.row_paths[:max_rows] or [[]]
    cols = intent.col_paths[:max_cols] or [[]]
    value_type = "number" if intent.needs_symbolic else "value"

    operands: List[Operand] = []
    seen: set = set()
    for rp in rows:
        for cp in cols:
            path = list(rp) + list(cp)
            if not path:
                continue
            key = " > ".join(path)
            if key in seen:
                continue
            seen.add(key)
            operands.append(
                Operand(
                    header_path=path,
                    row_path=list(rp),
                    col_path=list(cp),
                    value_type=value_type,
                    query_text=" ".join(path),
                )
            )
    return operands


def decomposition_confidence(
    query: str, table: OriginalTable, operands: Sequence[Operand]
) -> float:
    """How well the decomposed operands are grounded in the query, in [0, 1].

    This is the runtime HPIR-confidence signal the fallback controller reads. It
    measures how strongly the query's header-candidate terms match each operand's
    header path (via the store's own fuzzy scorer), averaged over operands. Zero
    operands means zero confidence (the query could not be decomposed at all).
    """
    if not operands:
        return 0.0
    terms = extract_target_terms(query)
    if not terms:
        return 0.0
    query_str = " ".join(terms)
    scores: List[float] = []
    for op in operands:
        s_col = table._fuzzy_score(query_str, op.col_path) if op.col_path else 0.0
        s_row = table._fuzzy_score(query_str, op.row_path) if op.row_path else 0.0
        scores.append(max(s_col, s_row))
    return float(sum(scores) / len(scores))


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

@dataclass
class OperandHit:
    operand: Operand
    chunks: List[RetrievedChunk]


@dataclass
class OperandRetrievalResult:
    query: str
    table_id: str
    operands: List[Operand]
    per_operand: List[OperandHit]
    retrieved: List[RetrievedChunk]   # deduped union, best-score first
    confidence: float = 0.0           # HPIR decomposition confidence in [0, 1]

    def covered_header_paths(self) -> List[List[str]]:
        paths: List[List[str]] = []
        for rc in self.retrieved:
            paths.extend(rc.chunk.header_paths)
        return paths


class OperandTargetedRetriever:
    """Indexes S2 cell chunks per table and retrieves operand by operand."""

    def __init__(self, encoder: Optional[Encoder] = None, alpha: float = 0.5) -> None:
        self.encoder = encoder
        self.alpha = alpha
        self._index_cache: Dict[str, HybridIndex] = {}

    def index_table(self, table: OriginalTable) -> HybridIndex:
        if table.table_id not in self._index_cache:
            chunks = s2.serialize(table, granularity="cell")
            self._index_cache[table.table_id] = HybridIndex(
                chunks, encoder=self.encoder, alpha=self.alpha
            )
            # Reuse the index's encoder for later tables so embeddings stay consistent.
            self.encoder = self._index_cache[table.table_id].encoder
        return self._index_cache[table.table_id]

    def retrieve(
        self,
        query: str,
        table: OriginalTable,
        k: int = 5,
        llm=None,
    ) -> OperandRetrievalResult:
        index = self.index_table(table)
        operands = decompose_operands(query, table, llm=llm)

        per_operand: List[OperandHit] = []
        best: Dict[str, RetrievedChunk] = {}
        for op in operands:
            hits = index.search(op.query_text, k=k)
            per_operand.append(OperandHit(operand=op, chunks=hits))
            for h in hits:
                cur = best.get(h.chunk.chunk_id)
                if cur is None or h.score > cur.score:
                    best[h.chunk.chunk_id] = h

        retrieved = sorted(best.values(), key=lambda h: -h.score)
        return OperandRetrievalResult(
            query=query,
            table_id=table.table_id,
            operands=operands,
            per_operand=per_operand,
            retrieved=retrieved,
            confidence=decomposition_confidence(query, table, operands),
        )


# ---------------------------------------------------------------------------
# Gold operands + recall metric
# ---------------------------------------------------------------------------

def _covers(gold_tokens: List[str], chunk_path: Sequence[str]) -> bool:
    """A chunk path covers a gold operand if it contains all gold tokens."""
    if not gold_tokens:
        return False
    chunk_tokens = set(_norm_tokens(chunk_path))
    return all(t in chunk_tokens for t in gold_tokens)


def operand_recall_at_k(
    gold_operands: Sequence[Sequence[str]],
    retrieved_chunks: Sequence[RetrievedChunk],
    k: Optional[int] = None,
) -> float:
    """Fraction of gold operand header-paths covered by the top-k retrieved cells.

    ``gold_operands`` is a list of header-path token sequences (e.g.
    ``[["total", "2017 actual"], ...]``). ``retrieved_chunks`` are ranked; if
    ``k`` is given only the first ``k`` are considered.
    """
    golds = [list(g) for g in gold_operands if g]
    if not golds:
        return 0.0
    chunks = list(retrieved_chunks)
    if k is not None:
        chunks = chunks[:k]
    paths = [p for rc in chunks for p in rc.chunk.header_paths]

    covered = 0
    for g in golds:
        gt = _norm_tokens(g)
        if any(_covers(gt, p) for p in paths):
            covered += 1
    return covered / len(golds)


def gold_operands_from_hitab(sample: dict) -> List[List[str]]:
    """Extract gold operand header paths from a HiTab sample's ``linked_cells``.

    Uses ``entity_link.top`` (column header leaves) and ``entity_link.left``
    (row header leaves). Each gold operand is the (row-leaf, col-leaf) pair the
    answer binds to; when only one axis is linked, that single leaf is the
    operand. Returns a list of header-path token sequences.
    """
    linked = sample.get("linked_cells") or {}
    entity = linked.get("entity_link") or {}

    def _leaves(axis: str) -> List[str]:
        out: List[str] = []
        for _phrase, coord_map in (entity.get(axis) or {}).items():
            for _coord, leaf in (coord_map or {}).items():
                if leaf:
                    out.append(str(leaf))
        return out

    col_leaves = _leaves("top")
    row_leaves = _leaves("left")

    operands: List[List[str]] = []
    if row_leaves and col_leaves:
        for r in row_leaves:
            for c in col_leaves:
                operands.append([r, c])
    else:
        for leaf in row_leaves + col_leaves:
            operands.append([leaf])

    # dedup, preserve order
    seen: set = set()
    uniq: List[List[str]] = []
    for op in operands:
        key = " > ".join(op)
        if key not in seen:
            seen.add(key)
            uniq.append(op)
    return uniq
