# SPDX-License-Identifier: MIT
"""Cell-level caption+header verbalization ("서울은 950입니다" style).

Turns each nonempty data cell into one natural-language sentence built from the
table caption (title) and the cell's hierarchical header paths, e.g.

    In table 1 federal r&d spending, 2017 actual for department of defense is 49197.

Three length styles form the sentence-length ablation:

* ``short``  — leaf headers only, no caption:      "{col_leaf} for {row_leaf} is {v}."
* ``medium`` — caption + leaf headers:             "In {title}, {col_leaf} for {row_leaf} is {v}."
* ``long``   — caption + full hierarchical paths:  "In {title}, {col_path} for {row_path} is {v}."

Each sentence becomes a :class:`~rag_agent.bench.schema.Chunk` covering exactly
one cell (rows=[r], cols=[c]), so retrieval can be scored both at table level
(max over a table's sentences) and at cell level (against gold operands).
"""
from __future__ import annotations

from typing import List

from ..bench.schema import BenchTable, Chunk

STYLES = ("short", "medium", "long")


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _clean_path(path: List[str]) -> List[str]:
    return [str(p).strip() for p in path if p and str(p).strip()]


def _subject(col_part: str, row_part: str) -> str:
    """Compose "X for Y" from whichever of column/row descriptions exist."""
    if col_part and row_part:
        return f"{col_part} for {row_part}"
    return col_part or row_part or "the value"


def verbalize_cell(table: BenchTable, r: int, c: int, style: str) -> str:
    v = _fmt(table.cell(r, c))
    row_path = _clean_path(table.row_path(r))
    col_path = _clean_path(table.col_path(c))
    title = (table.title or "").strip()

    if style == "short":
        subj = _subject(col_path[-1] if col_path else "",
                        row_path[-1] if row_path else "")
        return f"{subj} is {v}."
    if style == "medium":
        subj = _subject(col_path[-1] if col_path else "",
                        row_path[-1] if row_path else "")
        head = f"In {title}, " if title else ""
        return f"{head}{subj} is {v}."
    if style == "long":
        subj = _subject(", ".join(col_path), ", ".join(row_path))
        head = f"In {title}, " if title else ""
        return f"{head}{subj} is {v}."
    raise ValueError(f"unknown style {style!r}; expected one of {STYLES}")


def verbalize_table(table: BenchTable, style: str) -> List[Chunk]:
    """One sentence Chunk per nonempty data cell."""
    chunks: List[Chunk] = []
    for r in range(table.n_rows):
        for c in range(table.n_cols):
            raw = table.cell(r, c)
            if raw is None or str(raw).strip() == "":
                continue
            chunks.append(Chunk(
                table_id=table.table_id,
                chunk_id=f"{table.table_id}#c{r}_{c}",
                text=verbalize_cell(table, r, c, style),
                rows=[r],
                cols=[c],
            ))
    return chunks
