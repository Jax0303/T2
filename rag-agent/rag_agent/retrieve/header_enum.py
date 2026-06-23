# SPDX-License-Identifier: MIT
"""Deterministic header-tree enumeration of an aggregation scope (E2 treatment).

Given a query's resolved header-path predicates (from
``rag_agent.query.header_path_resolver.resolve_against_table``), enumerate every
leaf cell under the matched scope nodes — rather than ranking cells by similarity.
The retrieved operand set is the numeric cells in the product of the matched row
leaves and matched column leaves.

This is the structural prescription behind H2: a header-tree node *is* an
aggregation scope, so its operand set is recovered by enumeration, not top-k
similarity. Whether the resolved predicate is correct is the decomposition
ceiling (reported separately via row/col axis coverage).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Set, Tuple

Cell = Tuple[int, int]


@dataclass
class ScopeEnumeration:
    cells: Set[Cell]          # numeric operand cells in the enumerated scope
    rows: Set[int]            # matched row leaves
    cols: Set[int]            # matched col leaves
    row_fallback: bool        # True if no row predicate matched -> whole axis
    col_fallback: bool


def _match_axis(table, paths: Sequence[Sequence[str]], axis: str) -> Set[int]:
    finder = table.find_rows_by_header if axis == "row" else table.find_cols_by_header
    out: Set[int] = set()
    for p in paths:
        if not p:
            continue
        out.update(finder(" > ".join(p)))
    return out


def enumerate_scope(table, row_paths: Sequence[Sequence[str]],
                    col_paths: Sequence[Sequence[str]],
                    row_fallback_all: bool = True,
                    col_fallback_all: bool = True) -> ScopeEnumeration:
    """Enumerate the numeric operand cells under the resolved header scope.

    An unmatched axis is treated as *the whole axis is the scope* (fallback),
    which matches the common case where a query constrains one axis (e.g. a row
    entity) and aggregates across the other (e.g. all year columns). The fallback
    flags are returned so callers can report the effective retrieval-set size.
    """
    rows = _match_axis(table, row_paths, "row")
    cols = _match_axis(table, col_paths, "col")
    row_fb = col_fb = False
    if not rows and row_fallback_all:
        rows = set(range(table.n_rows)); row_fb = True
    if not cols and col_fallback_all:
        cols = set(range(table.n_cols)); col_fb = True
    cells = {(r, c) for r in rows for c in cols if table.cell_num(r, c) is not None}
    return ScopeEnumeration(cells=cells, rows=rows, cols=cols,
                            row_fallback=row_fb, col_fallback=col_fb)
