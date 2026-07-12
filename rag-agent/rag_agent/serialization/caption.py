"""S3 — Natural-language caption serialization.

Where S2 encodes header hierarchy as a ``>``-joined path prefix
(``Revenue > 2023 > Q1: 1,234``), S3 renders the same cell as a sentence a
retriever's embedder was actually pretrained on, e.g.::

    Among the regional population distribution, Seoul is 950.

The hypothesis this scheme tests: sentence-shaped text should embed closer to
a natural-language question than a ``>``-delimited path does, at the cost of
being longer (more tokens per cell, cheaper per-chunk information density).
Three ``length`` presets trade off how much header/title context each
sentence spells out — this is the "experiment with sentence length" axis:

* ``"short"``  — ``"{col}: {value}"`` prefixed by the row header if present.
* ``"medium"`` — ``"For {row path}, {col path} is {value}."``
* ``"long"``   — ``"In the table '{title}', among {row path}, the value of
  {col path} is {value}."``

Two granularities, matching S1/S2:

* ``"row"``   — one chunk per row (default).
* ``"cell"``  — one chunk per cell.
* ``"table"`` — the whole table as a single chunk (title stated once, not
  repeated per sentence) — the "1 table = 1 chunk" baseline.
"""
from __future__ import annotations

from typing import List

from .base import Chunk, TableView, fmt_value, join_path


SCHEME = "S3"
LENGTHS = ("short", "medium", "long")


def _cell_sentence(table: TableView, row: int, col: int, length: str, include_title: bool) -> str:
    row_path = join_path(table.row_path(row))
    col_path = join_path(table.col_path(col))
    value = fmt_value(table.cell(row, col))
    title = fmt_value(table.title) if include_title else ""

    if length == "short":
        if row_path and col_path:
            return f"{row_path} {col_path}: {value}."
        label = col_path or row_path
        return f"{label}: {value}." if label else f"{value}."

    if length == "medium":
        if row_path and col_path:
            return f"For {row_path}, {col_path} is {value}."
        if col_path:
            return f"{col_path} is {value}."
        if row_path:
            return f"{row_path} is {value}."
        return f"The value is {value}."

    if length == "long":
        clause = f"among {row_path}, " if row_path else ""
        what = f"the value of {col_path}" if col_path else "the value"
        if title:
            return f"In the table '{title}', {clause}{what} is {value}."
        return f"{clause}{what} is {value}.".capitalize()

    raise ValueError(f"length must be one of {LENGTHS}, got {length!r}")


def serialize(
    table: TableView,
    length: str = "medium",
    granularity: str = "row",
    include_title: bool = True,
) -> List[Chunk]:
    """Serialize ``table`` into natural-language caption sentences."""
    if length not in LENGTHS:
        raise ValueError(f"length must be one of {LENGTHS}, got {length!r}")
    if granularity not in ("row", "cell", "table"):
        raise ValueError(f"granularity must be 'row', 'cell' or 'table', got {granularity!r}")

    # "long" already states the title inside every sentence; for row/table
    # chunks that would repeat it on every line, so state it once up front
    # instead and drop it from the per-cell sentence.
    per_cell_title = include_title and granularity == "cell"
    title = fmt_value(table.title)
    title_line = [title] if (include_title and title and granularity != "cell") else []

    def cell_line(r: int, c: int) -> str:
        return _cell_sentence(table, r, c, length, per_cell_title)

    if granularity == "cell":
        chunks: List[Chunk] = []
        for r in range(table.n_rows):
            for c in range(table.n_cols):
                text = cell_line(r, c)
                chunks.append(
                    Chunk(
                        table_id=table.table_id,
                        chunk_id=f"{table.table_id}::{SCHEME}::{length}::r{r}c{c}",
                        text=text,
                        scheme=SCHEME,
                        kind="cell",
                        row_index=r,
                        col_index=c,
                        header_paths=[list(table.row_path(r)) + list(table.col_path(c))],
                        metadata={"length": length},
                    )
                )
        return chunks

    if granularity == "row":
        chunks = []
        for r in range(table.n_rows):
            lines = [cell_line(r, c) for c in range(table.n_cols)]
            text = "\n".join(title_line + lines)
            chunks.append(
                Chunk(
                    table_id=table.table_id,
                    chunk_id=f"{table.table_id}::{SCHEME}::{length}::r{r}",
                    text=text,
                    scheme=SCHEME,
                    kind="row",
                    row_index=r,
                    header_paths=[list(table.row_path(r)) + list(table.col_path(c)) for c in range(table.n_cols)],
                    metadata={"length": length},
                )
            )
        return chunks

    # granularity == "table": every cell's sentence in one chunk.
    lines = [cell_line(r, c) for r in range(table.n_rows) for c in range(table.n_cols)]
    text = "\n".join(title_line + lines)
    return [
        Chunk(
            table_id=table.table_id,
            chunk_id=f"{table.table_id}::{SCHEME}::{length}::table",
            text=text,
            scheme=SCHEME,
            kind="table",
            header_paths=[
                list(table.row_path(r)) + list(table.col_path(c))
                for r in range(table.n_rows) for c in range(table.n_cols)
            ],
            metadata={"length": length},
        )
    ]
