"""S2 — Structure-preserving (header-path) serialization.

Every cell is rendered with its full hierarchical header path as a prefix::

    Revenue > 2023 > Q1: 1,234

where the path is ``row_path(r) + col_path(c)``. This keeps the table's
relational structure inside the chunk text instead of discarding it the way a
flat markdown row does, which is the mechanism the thesis claims reduces
"structural information loss" at indexing time.

Two granularities are supported:

* ``"row"`` (default) — one chunk per row, holding every cell of that row,
  matching the spec's "chunk = 1 row + header-path prefix".
* ``"cell"`` — one chunk per cell, used by the operand-targeted retriever
  which needs to fetch individual operands rather than whole rows.
"""
from __future__ import annotations

from typing import List

from .base import Chunk, TableView, fmt_value, join_path


SCHEME = "S2"


def _cell_path(table: TableView, row: int, col: int) -> List[str]:
    """Full header path for a cell: row header path then column header path."""
    return list(table.row_path(row)) + list(table.col_path(col))


def _cell_line(table: TableView, row: int, col: int) -> str:
    path = _cell_path(table, row, col)
    value = fmt_value(table.cell(row, col))
    prefix = join_path(path)
    return f"{prefix}: {value}" if prefix else value


def serialize(
    table: TableView,
    granularity: str = "row",
    include_title: bool = True,
) -> List[Chunk]:
    """Serialize ``table`` with a header-path prefix on every cell."""
    if granularity not in ("row", "cell"):
        raise ValueError(f"granularity must be 'row' or 'cell', got {granularity!r}")

    title = fmt_value(table.title)
    title_prefix = [title] if (include_title and title) else []
    chunks: List[Chunk] = []

    if granularity == "row":
        for r in range(table.n_rows):
            lines = [_cell_line(table, r, c) for c in range(table.n_cols)]
            text = "\n".join(title_prefix + lines)
            chunks.append(
                Chunk(
                    table_id=table.table_id,
                    chunk_id=f"{table.table_id}::{SCHEME}::r{r}",
                    text=text,
                    scheme=SCHEME,
                    kind="row",
                    row_index=r,
                    header_paths=[_cell_path(table, r, c) for c in range(table.n_cols)],
                )
            )
        return chunks

    # cell granularity
    for r in range(table.n_rows):
        for c in range(table.n_cols):
            text = "\n".join(title_prefix + [_cell_line(table, r, c)])
            chunks.append(
                Chunk(
                    table_id=table.table_id,
                    chunk_id=f"{table.table_id}::{SCHEME}::r{r}c{c}",
                    text=text,
                    scheme=SCHEME,
                    kind="cell",
                    row_index=r,
                    col_index=c,
                    header_paths=[_cell_path(table, r, c)],
                )
            )
    return chunks
