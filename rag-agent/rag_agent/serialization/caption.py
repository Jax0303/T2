"""S3 — cell -> sentence index units.

Each cell becomes one sentence carrying its hierarchical row and column header
paths, so the retriever scores text an embedder was pretrained on rather than a
delimited path fragment.

Which sentence is rendered is a **template** choice, not a length choice — see
:mod:`rag_agent.serialization.templates`. The ``short``/``medium``/``long``
preset axis is retired: sentence length is no longer an experimental variable.

* ``templates.STRUCTURAL`` — this work's index unit. Byte-identical to the old
  ``length="long"``, so results on disk produced under ``"long"`` remain
  reproducible from this code.
* ``templates.MT2NET`` — provisional reproduction of Zhao et al. (2022) §4.

Granularity is unchanged: ``"row"`` (default), ``"cell"``, ``"table"``.
"""
from __future__ import annotations

from typing import List, Sequence

from .base import Chunk, TableView, fmt_value
from .templates import MT2NET, STRUCTURAL, TEMPLATES, render


SCHEME = "S3"


def _reject_length(kwargs) -> None:
    if "length" in kwargs:
        raise TypeError(
            "the short/medium/long length axis is retired; pass "
            "template='structural' (was length='long') or template='mt2net' "
            "(the provisional MT2Net reproduction). See "
            "rag_agent/serialization/templates.py"
        )


def caption_sentence(
    title,
    row_path: Sequence[str],
    col_path: Sequence[str],
    value=None,
    template: str = STRUCTURAL,
    **kwargs,
) -> str:
    """Render ONE index-unit sentence from a title + the two header paths.

    Single source of truth for the S3 sentence, so anything needing the same
    rendering (bench selectors, ablations) calls this instead of hand-rolling a
    lookalike that silently drifts from what the index actually contains.

    ``value=None`` omits the ``"is {value}"`` predicate, naming a header *scope*
    rather than a cell's contents — use only where the ranked thing is a header
    node. The deployed index always renders with a value.
    """
    _reject_length(kwargs)
    if kwargs:
        raise TypeError(f"unexpected keyword arguments: {sorted(kwargs)}")
    return render(template, title, row_path, col_path, value)


def _cell_sentence(table: TableView, row: int, col: int, template: str,
                   include_title: bool) -> str:
    # fmt_value first: an empty cell is a value that happens to be blank ("is ."),
    # NOT an absent value — passing None through would drop the predicate and
    # silently change every blank cell's index unit.
    return caption_sentence(
        table.title if include_title else "",
        table.row_path(row),
        table.col_path(col),
        value=fmt_value(table.cell(row, col)),
        template=template,
    )


def serialize(
    table: TableView,
    template: str = STRUCTURAL,
    granularity: str = "row",
    include_title: bool = True,
    **kwargs,
) -> List[Chunk]:
    """Serialize ``table`` into cell-sentence chunks under ``template``."""
    _reject_length(kwargs)
    if kwargs:
        raise TypeError(f"unexpected keyword arguments: {sorted(kwargs)}")
    if template not in TEMPLATES:
        raise ValueError(f"template must be one of {TEMPLATES}, got {template!r}")
    if granularity not in ("row", "cell", "table"):
        raise ValueError(f"granularity must be 'row', 'cell' or 'table', got {granularity!r}")

    # STRUCTURAL states the title inside every sentence; for row/table chunks
    # that would repeat it on every line, so state it once up front instead.
    per_cell_title = include_title and granularity == "cell"
    title = fmt_value(table.title)
    title_line = [title] if (include_title and title and granularity != "cell") else []

    def cell_line(r: int, c: int) -> str:
        return _cell_sentence(table, r, c, template, per_cell_title)

    if granularity == "cell":
        chunks: List[Chunk] = []
        for r in range(table.n_rows):
            for c in range(table.n_cols):
                text = cell_line(r, c)
                chunks.append(
                    Chunk(
                        table_id=table.table_id,
                        chunk_id=f"{table.table_id}::{SCHEME}::{template}::r{r}c{c}",
                        text=text,
                        scheme=SCHEME,
                        kind="cell",
                        row_index=r,
                        col_index=c,
                        header_paths=[list(table.row_path(r)) + list(table.col_path(c))],
                        metadata={"template": template},
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
                    chunk_id=f"{table.table_id}::{SCHEME}::{template}::r{r}",
                    text=text,
                    scheme=SCHEME,
                    kind="row",
                    row_index=r,
                    header_paths=[list(table.row_path(r)) + list(table.col_path(c)) for c in range(table.n_cols)],
                    metadata={"template": template},
                )
            )
        return chunks

    # granularity == "table": every cell's sentence in one chunk.
    lines = [cell_line(r, c) for r in range(table.n_rows) for c in range(table.n_cols)]
    text = "\n".join(title_line + lines)
    return [
        Chunk(
            table_id=table.table_id,
            chunk_id=f"{table.table_id}::{SCHEME}::{template}::table",
            text=text,
            scheme=SCHEME,
            kind="table",
            header_paths=[
                list(table.row_path(r)) + list(table.col_path(c))
                for r in range(table.n_rows) for c in range(table.n_cols)
            ],
            metadata={"template": template},
        )
    ]
