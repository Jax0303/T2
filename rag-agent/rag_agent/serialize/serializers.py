# SPDX-License-Identifier: MIT
"""S1 (flat) and S2 (structure-preserving) row-level serializers.

Both consume a :class:`~rag_agent.bench.schema.BenchTable` and emit a list of
:class:`~rag_agent.bench.schema.Chunk` — one per data row, annotated with the
cells it covers. Keeping serialization benchmark-agnostic (it reads only the
unified header-path representation) is what lets the same retrieval/eval code run
on HiTab, FinQA and WikiSQL.
"""
from __future__ import annotations

from typing import List

from ..bench.schema import BenchTable, Chunk

S1 = "s1_flat"
S2 = "s2_headerpath"
SCHEMES = (S1, S2)


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _leaf(path: List[str]) -> str:
    return path[-1] if path else ""


def _row_text_s1(table: BenchTable, r: int) -> str:
    """Flat markdown row: leaf column headers only, no hierarchy/row path."""
    row_leaf = _leaf(table.row_path(r))
    cells = []
    for c in range(table.n_cols):
        col_leaf = _leaf(table.col_path(c))
        v = _fmt(table.cell(r, c))
        label = f"{col_leaf}: {v}" if col_leaf else v
        cells.append(label)
    head = f"{table.title} | " if table.title else ""
    prefix = f"{row_leaf} | " if row_leaf else ""
    return head + prefix + " | ".join(cells)


def _row_text_s2(table: BenchTable, r: int) -> str:
    """Structure-preserving row: every cell prefixed with its full header path."""
    row_path = [p for p in table.row_path(r) if p]
    cells = []
    for c in range(table.n_cols):
        full = row_path + [p for p in table.col_path(c) if p]
        v = _fmt(table.cell(r, c))
        label = (" > ".join(full) + f": {v}") if full else v
        cells.append(label)
    head = f"{table.title} | " if table.title else ""
    return head + " | ".join(cells)


def serialize_table(table: BenchTable, scheme: str = S2) -> List[Chunk]:
    """Serialize ``table`` into row-level chunks under the given scheme."""
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")
    render = _row_text_s1 if scheme == S1 else _row_text_s2
    all_cols = list(range(table.n_cols))
    chunks: List[Chunk] = []
    for r in range(table.n_rows):
        chunks.append(Chunk(
            table_id=table.table_id,
            chunk_id=f"{table.table_id}#r{r}",
            text=render(table, r),
            rows=[r],
            cols=all_cols,
        ))
    return chunks


def fulltable_chunk(table: BenchTable, scheme: str = S2) -> Chunk:
    """The whole table as a single chunk — used as the fallback context."""
    rows_text = [(_row_text_s1 if scheme == S1 else _row_text_s2)(table, r)
                 for r in range(table.n_rows)]
    text = (f"{table.title}\n" if table.title else "") + "\n".join(rows_text)
    return Chunk(
        table_id=table.table_id,
        chunk_id=f"{table.table_id}#full",
        text=text,
        rows=list(range(table.n_rows)),
        cols=list(range(table.n_cols)),
    )
