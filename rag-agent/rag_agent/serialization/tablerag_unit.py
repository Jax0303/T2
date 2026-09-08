# SPDX-License-Identifier: MIT
"""TableRAG's cell index unit, ported as a serialization scheme.

Reproduces ``build_cell_corpus`` from TableRAG (Chen et al., NeurIPS 2024,
arXiv:2410.04739, ``google-research/table_rag/agent/retriever.py``) so the
baseline is something this repo executes rather than paraphrases. It exists to
answer one question and no other: **does putting the hierarchical header path
inside the index unit beat the strongest published cell-level alternative?**

What TableRAG indexes, verbatim from its source:

* a **categorical** cell becomes ``{"column_name": "...", "cell_value": "..."}``
* a **numeric column** becomes ONE summary doc,
  ``{"column_name": "...", "dtype": "...", "min": ..., "max": ...}`` — the
  individual numbers are never encoded
* the categorical docs are **deduplicated** by (column, value) and kept by
  frequency up to ``max_encode_cell``

Three consequences matter for the comparison, and the write-up has to state all
three or the baseline is misrepresented:

1. **No row identity.** ``{"column_name": "2016", "cell_value": "2207"}`` does
   not say which row it came from. TableRAG does not need it to: it is a ReAct
   agent that uses retrieved cells as *hints for writing pandas code* against
   the whole table, not as the evidence set handed to a reader.
2. **Numbers are not retrievable individually**, so an operand-set metric over
   numeric cells is not a task this corpus was built for.
3. It assumes a flat frame with one header row.

So this is NOT "we beat TableRAG". It is an ablation of the index unit with the
retrieval machinery, solver, budget and scorer held fixed. Label it that way.
``column_name_mode`` exists for the same reason: ``"leaf"`` is what a plain
``pd.read_html`` of a hierarchical table yields, and ``"path"`` hands TableRAG
the joined header path — the stronger variant. Report both; winning only
against the weaker one is not a result.
"""
from __future__ import annotations

from collections import Counter
from typing import List, Optional

from .base import Chunk, TableView, fmt_value, join_path

SCHEME = "TRAG"

#: TableRAG's default cell-encoding budget B (``--max_encode_cell``).
DEFAULT_MAX_ENCODE_CELL = 10000


def _col_name(table: TableView, c: int, mode: str) -> str:
    path = [fmt_value(s) for s in table.col_path(c) if fmt_value(s)]
    if not path:
        return f"col{c}"
    return path[-1] if mode == "leaf" else join_path(path)


def _is_numeric_column(table: TableView, c: int) -> bool:
    """Mirror pandas' inference on a clean read: a column is numeric only if
    every non-empty cell in it parses as a number, otherwise it is ``object``.

    Matching pandas matters because TableRAG branches on ``col.dtype``, and that
    branch is what decides whether the column's cells are encoded at all.
    """
    seen = False
    for r in range(table.n_rows):
        v = fmt_value(table.cell(r, c))
        if not v:
            continue
        seen = True
        try:
            float(v.replace(",", ""))
        except ValueError:
            return False
    return seen


def schema_doc(table: TableView, c: int, mode: str = "leaf") -> str:
    """The one summary doc a numeric column contributes."""
    name = _col_name(table, c, mode)
    nums = [float(fmt_value(table.cell(r, c)).replace(",", ""))
            for r in range(table.n_rows)
            if fmt_value(table.cell(r, c))]
    lo = min(nums) if nums else 0
    hi = max(nums) if nums else 0
    return f'{{"column_name": "{name}", "dtype": "float64", "min": {lo}, "max": {hi}}}'


def cell_doc(table: TableView, r: int, c: int, mode: str = "leaf") -> str:
    """The doc a single categorical cell contributes."""
    return (f'{{"column_name": "{_col_name(table, c, mode)}", '
            f'"cell_value": "{fmt_value(table.cell(r, c))}"}}')


def serialize(
    table: TableView,
    max_encode_cell: int = DEFAULT_MAX_ENCODE_CELL,
    column_name_mode: str = "leaf",
    **kwargs,
) -> List[Chunk]:
    """TableRAG's retrieval corpus for one table.

    Unlike every other serializer here the output is NOT one chunk per cell:
    numeric columns collapse to a single summary and categorical cells are
    deduplicated. ``row_index`` is therefore ``None`` on a deduplicated cell
    chunk — there is no single row it belongs to, which is the property the
    comparison is about.
    """
    if kwargs:
        raise TypeError(f"unexpected keyword arguments: {sorted(kwargs)}")
    if column_name_mode not in ("leaf", "path"):
        raise ValueError("column_name_mode must be 'leaf' or 'path', got "
                         f"{column_name_mode!r}")

    chunks: List[Chunk] = []

    def add(text: str, kind: str, c: int) -> None:
        chunks.append(Chunk(
            table_id=table.table_id,
            chunk_id=f"{table.table_id}::{SCHEME}::{len(chunks)}",
            text=text,
            scheme=SCHEME,
            kind="cell",
            # None on both kinds, and that is the point: a numeric column's
            # summary belongs to no row, and a deduplicated categorical doc
            # stands for every row that shares the value.
            row_index=None,
            col_index=c,
            header_paths=[list(table.col_path(c))],
            metadata={"tablerag_kind": kind,
                      "column_name_mode": column_name_mode},
        ))

    numeric = [c for c in range(table.n_cols) if _is_numeric_column(table, c)]
    for c in numeric:
        add(schema_doc(table, c, column_name_mode), "schema", c)

    # Categorical cells, deduplicated and ranked by frequency exactly as
    # TableRAG does — the budget is spent on the most common values.
    counts: Counter = Counter()
    origin = {}
    for c in range(table.n_cols):
        if c in numeric:
            continue
        for r in range(table.n_rows):
            if not fmt_value(table.cell(r, c)):
                continue
            doc = cell_doc(table, r, c, column_name_mode)
            counts[doc] += 1
            origin.setdefault(doc, c)

    budget = max(max_encode_cell - len(chunks), 0)
    for doc, _ in counts.most_common(budget):
        add(doc, "cell", origin[doc])

    return chunks
