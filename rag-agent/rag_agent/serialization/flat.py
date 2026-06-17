"""S1 — Flat markdown serialization (the structure-agnostic baseline).

Each data row becomes one chunk rendered as a small markdown table: the title,
the column-header row, and the single data row. The row's own header (the leaf
of its ``row_path``) is prepended as a first column so the row stays
identifiable, but **no hierarchical header path is attached to the cells** —
that is the whole point of S1. It is the control condition against which S2's
header-path prefixing is measured.
"""
from __future__ import annotations

from typing import List

from .base import Chunk, TableView, fmt_value, leaf


SCHEME = "S1"


def _column_headers(table: TableView) -> List[str]:
    return [leaf(table.col_path(c)) or f"col_{c}" for c in range(table.n_cols)]


def _md_row(cells: List[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def serialize(table: TableView, include_title: bool = True) -> List[Chunk]:
    """Serialize ``table`` into one flat-markdown chunk per data row."""
    headers = _column_headers(table)
    has_row_header = any(table.row_path(r) for r in range(table.n_rows))

    head_cells = (["row"] if has_row_header else []) + headers
    header_line = _md_row(head_cells)
    sep_line = _md_row(["---"] * len(head_cells))
    title = fmt_value(table.title)

    chunks: List[Chunk] = []
    for r in range(table.n_rows):
        data_cells = [fmt_value(table.cell(r, c)) for c in range(table.n_cols)]
        if has_row_header:
            data_cells = [leaf(table.row_path(r))] + data_cells
        row_line = _md_row(data_cells)

        parts: List[str] = []
        if include_title and title:
            parts.append(title)
        parts += [header_line, sep_line, row_line]
        text = "\n".join(parts)

        chunks.append(
            Chunk(
                table_id=table.table_id,
                chunk_id=f"{table.table_id}::{SCHEME}::r{r}",
                text=text,
                scheme=SCHEME,
                kind="row",
                row_index=r,
            )
        )
    return chunks
