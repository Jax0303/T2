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

Granularity is unchanged: ``"row"`` (default), ``"cell"``, ``"table"``.
"""
from __future__ import annotations

from typing import List, Sequence

from .base import Chunk, TableView, fmt_value
from .templates import STRUCTURAL, TEMPLATES, render


SCHEME = "S3"


def _reject_length(kwargs) -> None:
    if "length" in kwargs:
        raise TypeError(
            "the short/medium/long length axis is retired; pass "
            "template='structural' (was length='long'). See "
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


# --- title collision: the frame's title slot, when the title does not identify
# the table. STRUCTURAL_COMPACT already drops the frame for an EMPTY title
# ("with no title it asserts nothing while still spending budget"). A title 61
# HiTab dev tables share ('career statistics') asserts nothing about WHICH table
# either, and is worse than empty: it makes those 61 tables' cells read alike.
# Measured on hitab_dev_lookup_all (results/tableconf/VERDICT.md): 179 of 424
# tables share a title, and their queries lose the whole table 45.3% of the time
# against 9.8% for uniquely titled ones (Fisher p=2.1e-13).
def with_page_title(title: str, entry) -> str:
    """Prefix ToTTo's PAGE title to HiTab's SECTION title.

    HiTab's ``title`` is ToTTo's *section* title ('career statistics'); the page
    title names the entity the question is about ('Hristo Yanev'). The deployed
    index carries the prefix, so anything that rebuilds a cell sentence outside
    the index — the reader's gold/oracle conditions, ablations — has to apply
    the same rule here or it hands the reader a title the corpus never held.
    """
    pg = (entry or {}).get("page_title", "").strip()
    if not pg:
        return title
    return f"{pg}: {title}" if title else pg


TITLE_MODES = ("raw", "drop", "sig", "page")
SIG_TOKENS = 3


def effective_titles(tids, title, cell_owner, cell_paths, mode="raw",
                     k=SIG_TOKENS, page_titles=None):
    """``{table_id: the string to hand caption_sentence as the title}``.

    ``raw``  the title as given -- byte-identical to not calling this at all.
    ``drop`` a non-unique title becomes ``""``, taking the empty-title path.
    ``sig``  a non-unique title gains the table's most table-specific header
             tokens, so every cell of the table carries the table's identity.
             35.3% of the cells in shared-title tables hold no token of their
             own that is rare across tables, so ``drop`` leaves those cells with
             nothing pointing at their table; ``sig`` is what covers them.

    Token specificity is document frequency over TABLES: the rarest tokens of
    this table's header paths win, ties broken alphabetically so the signature
    is deterministic.
    """
    if mode not in TITLE_MODES:
        raise ValueError(f"mode must be one of {TITLE_MODES}, got {mode!r}")
    out = {t: title.get(t, "") for t in tids}
    if mode == "raw":
        return out

    if mode == "page":
        # HiTab's `title` is ToTTo's SECTION title ('career statistics'); the
        # PAGE title -- the entity the question names ('Ian Miller (footballer,
        # born 1955)') -- was dropped when HiTab took the tables. This mode puts
        # it back. Unlike drop/sig it adds information the corpus did not hold,
        # which is why those two failed (results/titlemode/VERDICT.md).
        # Recovery and its verification: results/tableconf/TOTTO_RECOVERY.md.
        if not page_titles:
            raise ValueError("mode 'page' needs page_titles")
        for t in tids:
            out[t] = with_page_title(out[t], page_titles.get(t))
        return out

    shared = {}
    for t in tids:
        shared.setdefault(out[t].strip().lower(), []).append(t)
    dup = {t for ts in shared.values() if len(ts) > 1 for t in ts}
    if mode == "drop":
        return {t: ("" if t in dup else out[t]) for t in tids}

    df, toks = {}, {}
    for n, (t, _i, _j) in enumerate(cell_owner):
        rp, cp, _v = cell_paths[n]
        for w in " ".join([*rp, *cp]).lower().split():
            toks.setdefault(t, set()).add(w)
    for t, ws in toks.items():
        for w in ws:
            df[w] = df.get(w, 0) + 1
    for t in dup:
        best = sorted(toks.get(t, ()), key=lambda w: (df[w], w))[:k]
        out[t] = f"{out[t]}: {', '.join(best)}" if best else out[t]
    return out
