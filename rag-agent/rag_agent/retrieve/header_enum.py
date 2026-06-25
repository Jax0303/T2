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

import re
from dataclasses import dataclass
from typing import List, Sequence, Set, Tuple

Cell = Tuple[int, int]

# A "total-like" header segment: a table/section aggregate row (total, overall, or
# a top-level "all <X>" rollup). The row-axis diagnosis
# (`scripts/diag_row_failures.py`) found that 68% of hybrid-resolver row-axis
# failures miss exactly such a row — ratio/share ("what % of total") queries whose
# operand set is a sub-scope PLUS a total row that lives at a different header
# level (often an empty/unparsed path), which a header-text resolver cannot bind.
_TOTALISH = re.compile(r"\b(total|overall)\b|^all\b", re.I)
# Ratio/share question cues -> the operand set very likely needs a total denominator.
_RATIO_CUES = re.compile(
    r"\b(percent|percentage|share|proportion|fraction|ratio|times|per\b|out of)\b", re.I)


def is_total_row(table, r: int) -> bool:
    """True if row ``r``'s left-header path denotes a table/section total.

    An empty path (unparsed left header) is treated as total-like: in HiTab these
    are overwhelmingly the table-level aggregate rows the resolver cannot name.
    """
    path = table.row_path(r)
    if not path:
        return True
    return any(_TOTALISH.search(str(seg)) for seg in path)


def total_like_rows(table) -> Set[int]:
    """All numeric-bearing total-like rows (the candidate denominators)."""
    return {r for r in range(table.n_rows)
            if is_total_row(table, r)
            and any(table.cell_num(r, c) is not None for c in range(table.n_cols))}


def is_ratio_query(question: str) -> bool:
    """Heuristic: does the question ask for a share/ratio (needs a total)?"""
    return bool(_RATIO_CUES.search(question or ""))


def expand_sibling_groups(table, matched_rows: Set[int]) -> Set[int]:
    """Expand each matched row to its full sibling group (same immediate parent).

    For every matched row, add all rows sharing its immediate-parent path prefix
    (``row_path[:-1]``). Turns a partially-matched child set into the whole group
    under the parent node — the structural prescription for parent_expandable /
    sibling_subset failures. No-op for depth-1 (parent-less) rows.
    """
    parents = set()
    for r in matched_rows:
        p = table.row_path(r)
        if len(p) >= 2:
            parents.add(tuple(p[:-1]))
    if not parents:
        return set(matched_rows)
    out = set(matched_rows)
    for r in range(table.n_rows):
        p = table.row_path(r)
        if len(p) >= 2 and tuple(p[:-1]) in parents:
            out.add(r)
    return out


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
                    col_fallback_all: bool = True,
                    add_total_rows: bool = False,
                    expand_siblings: bool = False) -> ScopeEnumeration:
    """Enumerate the numeric operand cells under the resolved header scope.

    An unmatched axis is treated as *the whole axis is the scope* (fallback),
    which matches the common case where a query constrains one axis (e.g. a row
    entity) and aggregates across the other (e.g. all year columns). The fallback
    flags are returned so callers can report the effective retrieval-set size.

    Diagnosis-driven row augmentations (applied only when the row axis actually
    matched a scope, i.e. not the whole-axis fallback):
      * ``add_total_rows``  — union in table/section total rows (the share/ratio
        denominator the resolver can't name); see :func:`total_like_rows`.
      * ``expand_siblings`` — expand each matched row to its full sibling group;
        see :func:`expand_sibling_groups`.
    """
    rows = _match_axis(table, row_paths, "row")
    cols = _match_axis(table, col_paths, "col")
    row_fb = col_fb = False
    if not rows and row_fallback_all:
        rows = set(range(table.n_rows)); row_fb = True
    if not cols and col_fallback_all:
        cols = set(range(table.n_cols)); col_fb = True
    if not row_fb:  # augment only a genuinely-matched (sub-axis) scope
        if expand_siblings:
            rows = expand_sibling_groups(table, rows)
        if add_total_rows:
            rows = rows | total_like_rows(table)
    cells = {(r, c) for r in rows for c in cols if table.cell_num(r, c) is not None}
    return ScopeEnumeration(cells=cells, rows=rows, cols=cols,
                            row_fallback=row_fb, col_fallback=col_fb)
