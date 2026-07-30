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

from typing import List, Sequence

from .base import Chunk, TableView, fmt_value, join_path


SCHEME = "S3"
LENGTHS = ("short", "medium", "long")


def caption_sentence(
    title,
    row_path: Sequence[str],
    col_path: Sequence[str],
    value=None,
    length: str = "medium",
) -> str:
    """Render ONE index-unit sentence from a title + the two header paths.

    This is the single source of truth for the S3 sentence template — the
    "table caption + column header path + row header path + cell value into a
    sentence template" index unit. :func:`_cell_sentence` is a thin adapter over
    it, so anything that needs the same rendering (bench selectors, ablations)
    can call this directly instead of hand-rolling a lookalike that silently
    drifts from what the index actually contains.

    Paths are joined with :func:`~rag_agent.serialization.base.join_path`, which
    formats each segment and drops empty ones — so a path carrying a blank
    segment does not produce a dangling ``"North America > "``.

    ``value=None`` renders the same sentence with the ``"is {value}"`` predicate
    omitted, naming a header *scope* rather than asserting a cell's contents.
    That form addresses a candidate header node, not an index unit, so use it
    only where the thing being ranked is a node (e.g. the column/row selection
    benches) — the deployed index always renders with a value.
    """
    if length not in LENGTHS:
        raise ValueError(f"length must be one of {LENGTHS}, got {length!r}")
    row_s = join_path(row_path)
    col_s = join_path(col_path)
    title_s = fmt_value(title) if title else ""
    has_val = value is not None
    val_s = fmt_value(value) if has_val else ""

    if length == "short":
        tail = f": {val_s}." if has_val else "."
        if row_s and col_s:
            return f"{row_s} {col_s}{tail}"
        label = col_s or row_s
        return f"{label}{tail}" if label else (f"{val_s}." if has_val else ".")

    if length == "medium":
        pred = f" is {val_s}" if has_val else ""
        if row_s and col_s:
            return f"For {row_s}, {col_s}{pred}."
        if col_s:
            return f"{col_s}{pred}."
        if row_s:
            return f"{row_s}{pred}."
        return f"The value{pred}." if has_val else "The value."

    # length == "long"
    clause = f"among {row_s}, " if row_s else ""
    what = f"the value of {col_s}" if col_s else "the value"
    pred = f" is {val_s}" if has_val else ""
    if title_s:
        return f"In the table '{title_s}', {clause}{what}{pred}."
    return f"{clause}{what}{pred}.".capitalize()


def _cell_sentence(table: TableView, row: int, col: int, length: str, include_title: bool) -> str:
    # fmt_value first: an empty cell is a value that happens to be blank ("is ."),
    # NOT an absent value — passing None through would drop the predicate and
    # silently change every blank cell's index unit.
    return caption_sentence(
        table.title if include_title else "",
        table.row_path(row),
        table.col_path(col),
        value=fmt_value(table.cell(row, col)),
        length=length,
    )


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
