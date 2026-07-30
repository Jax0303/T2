# SPDX-License-Identifier: MIT
"""구조 속성 부여부 — index-time structural attribution of aggregate rows.

The invention disclosure specifies that the aggregate-row determination is made
**once per table during index construction** and stored as a *structural
attribute* on each index unit; the complementary-retrieval component then merely
*looks the attribute up* at query time. Two claims rest on that placement:

* the injected set is decided by structure (header paths + cell values), never
  by the similarity signal it is meant to compensate for, and
* no per-query cost is incurred by the determination itself.

Before this module the two detection rules lived in
:mod:`rag_agent.retrieve.header_enum` as free functions that every caller
invoked inside its own query loop, so the second property held only by accident
of how a given script happened to be written. Here the determination is bound to
the table's index, computed on first use and reused for every later query.

The two rules are unchanged and still owned by ``header_enum``:

* **Rule 1 (lexical)** — :func:`~rag_agent.retrieve.header_enum.is_total_row`:
  the row's own leaf label matches a total/overall/all pattern, or its header
  path is empty (an unparsed left header, dominantly a table-level aggregate).
* **Rule 2 (structural)** —
  :func:`~rag_agent.retrieve.header_enum.is_total_row_structural`: the row's
  values equal the sum of its tree children or of its siblings within a relative
  tolerance, across a majority of the columns where both sides carry data. No
  label text is consulted, so it is language-independent and catches the
  unnamed aggregate rows Rule 1 cannot.

The stored attribute is their union, matching
:func:`~rag_agent.retrieve.header_enum.total_like_rows_hybrid`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..serialization.base import Chunk
from .header_enum import (is_total_row, is_total_row_structural,
                          total_like_rows_hybrid)

Cell = Tuple[int, int]

#: Chunk.metadata key holding the structural attribute (True on an index unit
#: whose row was determined to be an aggregate row).
AGG_ROW_KEY = "is_aggregate_row"

#: Chunk.metadata key recording which rule fired: "lexical", "structural",
#: "both", or absent. Kept for the ablation that separates the two rules.
AGG_RULE_KEY = "aggregate_rule"


def _rule_label(table, r: int) -> Optional[str]:
    lex = is_total_row(table, r)
    struct = is_total_row_structural(table, r)
    if lex and struct:
        return "both"
    if lex:
        return "lexical"
    if struct:
        return "structural"
    return None


@dataclass
class StructuralAttributes:
    """The structural attributes of ONE table, determined once at index time.

    ``aggregate_rows`` is the union of the two rules; ``rule_by_row`` records
    which rule fired so the lexical/structural contributions stay separable.
    """

    table_id: str
    aggregate_rows: Set[int] = field(default_factory=set)
    rule_by_row: Dict[int, str] = field(default_factory=dict)

    @classmethod
    def compute(cls, table) -> "StructuralAttributes":
        """Run both rules over ``table`` — the once-per-table cost.

        Tables whose view cannot supply numeric cells (``cell_num``) degrade to
        Rule 1 alone rather than failing: the lexical rule needs only header
        paths, which every :class:`~rag_agent.serialization.base.TableView` has.
        """
        if not hasattr(table, "cell_num"):
            rows = {r for r in range(table.n_rows) if is_total_row(table, r)}
            return cls(table_id=table.table_id, aggregate_rows=rows,
                       rule_by_row={r: "lexical" for r in rows})
        rows = total_like_rows_hybrid(table)
        return cls(
            table_id=table.table_id,
            aggregate_rows=set(rows),
            rule_by_row={r: lbl for r in rows
                         if (lbl := _rule_label(table, r)) is not None},
        )

    def stamp(self, chunks: Sequence[Chunk]) -> Sequence[Chunk]:
        """Write the attribute onto each index unit, in place.

        Row- and cell-granularity chunks carry a ``row_index``; table chunks do
        not and are left unstamped (they span every row, so the attribute is
        not well defined for them).
        """
        for ch in chunks:
            if ch.row_index is None:
                continue
            if ch.row_index in self.aggregate_rows:
                ch.metadata[AGG_ROW_KEY] = True
                rule = self.rule_by_row.get(ch.row_index)
                if rule:
                    ch.metadata[AGG_RULE_KEY] = rule
            else:
                ch.metadata[AGG_ROW_KEY] = False
        return chunks


def aggregate_chunks(chunks: Iterable[Chunk]) -> List[Chunk]:
    """Query-time lookup: the index units already flagged as aggregate rows.

    A dict/attribute read over the indexed chunks — the determination itself
    happened at index time in :meth:`StructuralAttributes.compute`.
    """
    return [ch for ch in chunks if ch.metadata.get(AGG_ROW_KEY)]


def aggregate_cells(chunks: Iterable[Chunk]) -> Set[Cell]:
    """``(row, col)`` of every flagged cell-granularity index unit."""
    return {(ch.row_index, ch.col_index) for ch in aggregate_chunks(chunks)
            if ch.row_index is not None and ch.col_index is not None}
