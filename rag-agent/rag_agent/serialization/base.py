"""Serialization primitives shared by the S1 (flat) and S2 (header-path) schemes.

The pipeline turns a parsed table into a list of :class:`Chunk` objects. Each
chunk is the unit that gets embedded / indexed downstream, so the two schemes
differ only in *how* a row is rendered to text — the chunk envelope (ids,
row index, header-path metadata) is identical. Keeping that envelope shared is
what makes the S1-vs-S2 ablation in the thesis a clean, single-variable
comparison.

A ``TableView`` is the minimal read interface the serializers need. The
hierarchical HiTab table (:class:`rag_agent.stores.original_store.OriginalTable`)
already satisfies it; :class:`rag_agent.serialization.flat_table.FlatTable`
provides the same interface for flat FinQA / WikiSQL tables.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, runtime_checkable


@runtime_checkable
class TableView(Protocol):
    """Read-only view a serializer needs over any table.

    ``col_path(c)`` / ``row_path(r)`` return the *header path* for a column /
    row as an ordered list of header segments (outermost first). For a flat
    table ``col_path`` is a single-element list and ``row_path`` is empty.
    """

    table_id: str
    title: str

    @property
    def n_rows(self) -> int: ...

    @property
    def n_cols(self) -> int: ...

    def cell(self, row: int, col: int): ...

    def col_path(self, col: int) -> List[str]: ...

    def row_path(self, row: int) -> List[str]: ...


@dataclass
class Chunk:
    """One retrieval unit produced by a serializer.

    Attributes
    ----------
    table_id, chunk_id:
        ``chunk_id`` is unique within a corpus, of the form
        ``"{table_id}::{scheme}::r{row}"`` (or ``::r{row}c{col}`` for cell
        granularity).
    text:
        The string that gets embedded / BM25-indexed.
    scheme:
        ``"S1"`` (flat) or ``"S2"`` (header-path).
    kind:
        ``"row"`` (default), ``"cell"``, or ``"table"``.
    row_index / col_index:
        Provenance back into the source table (``None`` for ``table`` kind).
    header_paths:
        Full header path(s) attached to this chunk — for a row chunk, one
        entry per cell: ``row_path + col_path``. Used by the operand-targeted
        retriever and by ``coverage`` checks, and logged for error analysis.
    """

    table_id: str
    chunk_id: str
    text: str
    scheme: str
    kind: str = "row"
    row_index: Optional[int] = None
    col_index: Optional[int] = None
    header_paths: List[List[str]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "table_id": self.table_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "scheme": self.scheme,
            "kind": self.kind,
            "row_index": self.row_index,
            "col_index": self.col_index,
            "header_paths": self.header_paths,
            "metadata": self.metadata,
        }


def fmt_value(v: object) -> str:
    """Render a cell value to a single clean token-friendly string."""
    if v is None:
        return ""
    s = str(v).strip()
    # Collapse internal whitespace / newlines so a cell never breaks the
    # one-cell-per-line layout that S2 relies on.
    return " ".join(s.split())


def leaf(path: List[str]) -> str:
    """Last (most specific) segment of a header path, or '' if empty."""
    return path[-1] if path else ""


def join_path(path: List[str], sep: str = " > ") -> str:
    """Render a header path, dropping empty segments."""
    return sep.join(p for p in (fmt_value(s) for s in path) if p)
