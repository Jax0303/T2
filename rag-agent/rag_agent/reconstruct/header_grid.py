"""Reconstruct hierarchical header PATHS from a raw ("flattened") 2D grid —
the "build the tree from the raw table, before anything is embedded" step.

HiTab hands the header tree to us pre-parsed (``top_root``/``left_root`` in
the raw JSON — see :mod:`rag_agent.stores.original_store`). Most real-world
raw tables do NOT: a merged header cell shows its value only at the first
covered row/column, leaving the rest blank (this is what you get from an
HTML table with ``rowspan``/``colspan`` once you throw the span markup away
and keep only the rendered grid, or from copy-pasting a spreadsheet with
merged cells into a CSV). This module reconstructs the per-column / per-row
header path a human reader would infer, by forward-filling each blank from
its nearest non-blank ancestor — with no access to a pre-built tree.

Two very different raw sources feed this same algorithm:

* :mod:`scripts.tree_reconstruct_hitab` — synthetically flattens HiTab's own
  gold tree into a blank-after-first grid, then checks reconstruction
  against the REAL gold paths (clean, exact ground truth).
* :mod:`scripts.tree_reconstruct_multihiertt` — parses real scraped HTML
  tables (``rowspan``/``colspan``, including malformed markup) from
  MultiHiertt, which never had a gold tree to begin with.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Dict, List

Grid = List[List[str]]


# ---------------------------------------------------------------------------
# HTML -> raw grid (occupancy-grid resolution of rowspan/colspan)
# ---------------------------------------------------------------------------

class _TableHTMLParser(HTMLParser):
    """Collects <tr>/<td>/<th> cells with their rowspan/colspan, tolerant of
    the malformed ``<td rowspan="2"SomeText>`` tags MultiHiertt's raw HTML
    occasionally contains (a stray missing ``>`` after the last attribute)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: List[List[dict]] = []
        self._cur_row: List[dict] | None = None
        self._cur_cell: dict | None = None

    def handle_starttag(self, tag, attrs):
        attrs_d = dict(attrs)
        if tag == "tr":
            self._cur_row = []
        elif tag in ("td", "th"):
            self._cur_cell = {
                "text": "",
                "rowspan": int(attrs_d.get("rowspan") or 1),
                "colspan": int(attrs_d.get("colspan") or 1),
            }

    def handle_data(self, data):
        if self._cur_cell is not None:
            self._cur_cell["text"] += data

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cur_cell is not None:
            self._cur_cell["text"] = self._cur_cell["text"].strip().rstrip(">").strip()
            if self._cur_row is not None:
                self._cur_row.append(self._cur_cell)
            self._cur_cell = None
        elif tag == "tr" and self._cur_row is not None:
            self.rows.append(self._cur_row)
            self._cur_row = None


# Fixes `<td colspan="2" rowspan="2"SomeText></td>` (missing '>' right after
# the last span attribute) into `...rowspan="2">SomeText></td>` so the text
# is recovered as cell content instead of being swallowed as bogus attributes.
_FIX_MISSING_GT = re.compile(r'((?:col|row)span="\d+")(?=[^\s>])')


def parse_html_table(html: str) -> Grid:
    """Parse an HTML table into a *blank-after-first* 2D grid: a spanning
    cell's text is placed only at its (row, col) origin; every other grid
    position it covers is left as ``""`` — simulating a raw dump that lost
    its rowspan/colspan markup, the realistic case this module targets."""
    fixed = _FIX_MISSING_GT.sub(r"\1>", html)
    p = _TableHTMLParser()
    p.feed(fixed)
    raw_rows = p.rows
    if not raw_rows:
        return []

    active_rowspans: Dict[int, int] = {}
    grid: Grid = []
    for row_cells in raw_rows:
        row_out: List[str] = []
        col = 0
        ci = 0
        while ci < len(row_cells) or (active_rowspans and col <= max(active_rowspans)):
            if col in active_rowspans:
                row_out.append("")
                active_rowspans[col] -= 1
                if active_rowspans[col] <= 0:
                    del active_rowspans[col]
                col += 1
                continue
            if ci >= len(row_cells):
                col += 1
                continue
            cell = row_cells[ci]
            ci += 1
            row_out.append(cell["text"])
            span_c = max(cell["colspan"], 1)
            span_r = max(cell["rowspan"], 1)
            for _ in range(1, span_c):
                row_out.append("")
            if span_r > 1:
                for cc in range(col, col + span_c):
                    active_rowspans[cc] = max(active_rowspans.get(cc, 0), span_r - 1)
            col += span_c
        grid.append(row_out)

    width = max(len(r) for r in grid)
    for r in grid:
        r.extend([""] * (width - len(r)))
    return grid


# ---------------------------------------------------------------------------
# blank-after-first grid -> header paths (the actual reconstruction)
# ---------------------------------------------------------------------------

def _hierarchical_carry(levels: List[List[str]]) -> List[List[str]]:
    """Shared span-tracking fill for both axes.

    ``levels[d][i]`` is the header cell at depth ``d`` for data line ``i``
    (columns for the top axis, rows for the left axis). A blank cell means
    "the merged span begun at the last non-blank cell at this depth is still
    covering me" — but a span belongs to ONE tree node, so it can never
    outlive its parent: whenever a shallower depth starts a new label, every
    deeper carry is cut off instead of bleeding into the new parent's region
    (the dominant error mode of naive per-depth forward fill).
    """
    n_depths = len(levels)
    n_lines = len(levels[0]) if levels else 0
    carry = [""] * n_depths
    out: List[List[str]] = []
    for i in range(n_lines):
        for d in range(n_depths):
            cell = levels[d][i].strip() if i < len(levels[d]) else ""
            if cell:
                carry[d] = cell
                for e in range(d + 1, n_depths):
                    carry[e] = ""
        out.append([seg for seg in carry if seg])
    return out


def reconstruct_col_paths(grid: Grid, n_header_rows: int, n_header_cols: int = 1) -> List[List[str]]:
    """One header path per DATA column (columns >= ``n_header_cols``),
    restricted to the data-column region so a row-header label sitting in
    column 0 never bleeds into the column paths."""
    n_cols = len(grid[0]) if grid else 0
    if n_header_rows <= 0:
        return [[] for _ in range(max(n_cols - n_header_cols, 0))]
    levels = [grid[r][n_header_cols:] for r in range(n_header_rows)]
    return _hierarchical_carry(levels)


def reconstruct_row_paths(grid: Grid, n_header_rows: int, n_header_cols: int = 1) -> List[List[str]]:
    """One header path per DATA row (rows >= ``n_header_rows``), mirroring
    :func:`reconstruct_col_paths`."""
    n_rows = len(grid)
    if n_header_cols <= 0:
        return [[] for _ in range(max(n_rows - n_header_rows, 0))]
    levels = [[grid[r][c] for r in range(n_header_rows, n_rows)]
              for c in range(n_header_cols)]
    return _hierarchical_carry(levels)


# ---------------------------------------------------------------------------
# header/data row boundary heuristic
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"^[\$\(\-]?[\d,]+\.?\d*%?\)?$")
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")


def looks_numeric(s: str) -> bool:
    """True for cell text that looks like actual tabulated data (currency,
    percentages, comma-grouped magnitudes) — deliberately excludes bare
    4-digit years, which read as numeric but are almost always column-header
    labels ("2004", "2003", ...), not data."""
    s = s.strip()
    if not s or _YEAR_RE.match(s):
        return False
    return bool(_NUM_RE.match(s))


def _left_region_blank(grid: Grid, r: int, n_header_cols: int) -> bool:
    return not any(c.strip() for c in grid[r][:n_header_cols])


# A stub-column entry that is FULLY parenthesized is a units annotation
# ("(in millions)", "(Dollars in thousands)"), not a row label: financial
# tables put these inside the header block, so the corner scan must not read
# them as the first data row. Genuine labels that merely start with a paren
# ("(Loss) income ...") are not fully wrapped and stay untouched.
_PAREN_NOTE_RE = re.compile(r"^\(.*\)$", re.DOTALL)


def _left_region_units_note(grid: Grid, r: int, n_header_cols: int) -> bool:
    text = " ".join(c.strip() for c in grid[r][:n_header_cols] if c.strip())
    return bool(text) and bool(_PAREN_NOTE_RE.match(text))


def _numeric_data_row(grid: Grid, r: int, n_header_cols: int) -> bool:
    cells = grid[r][n_header_cols:]
    n_nonblank = sum(1 for c in cells if c.strip())
    n_num = sum(1 for c in cells if looks_numeric(c))
    return bool(n_nonblank) and n_num / n_nonblank >= 0.5


def guess_n_header_rows(grid: Grid, n_header_cols: int = 1, max_header_rows: int = 8) -> int:
    """Guess how many top rows are column headers, two signals in priority order.

    **Blank-corner signal (primary, when applicable).** In a table with row
    labels, column-header rows leave the row-label region blank (the top-left
    corner block), and the first data row is the first row that puts text
    there. When the corner IS blank at row 0 we trust this transition
    outright: it also covers the rows the numeric signal is blind to —
    section-header rows (label on the left, all data cells blank), string-only
    data rows, and numeric-looking sub-header rows ("1", "2") that would
    otherwise end the header block early. Guarded by the row-0 check because
    a corner that *carries text* ("Item", ...) says nothing about where the
    header ends, so we fall through.

    **Numeric-ratio signal (fallback).** First row where >=50% of the
    non-row-label cells look numeric is the first data row.
    """
    if not grid:
        return 0
    limit = min(len(grid), max_header_rows)

    if n_header_cols > 0 and grid[0][:n_header_cols] and _left_region_blank(grid, 0, n_header_cols):
        for r in range(1, limit + 1):
            if r >= len(grid):
                break
            if not _left_region_blank(grid, r, n_header_cols) and \
                    not _left_region_units_note(grid, r, n_header_cols):
                return r
        # No row label ever appears (e.g. the table has no real row headers):
        # the corner signal is void — fall through to the numeric scan.

    for r in range(limit):
        if _numeric_data_row(grid, r, n_header_cols):
            return r
    return limit
