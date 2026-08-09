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


def section_label(row: List[str]) -> str:
    """The label of a SECTION row, or "" if this is not one.

    A section row carries exactly one text and no data — a group heading whose
    scope is every row beneath it until the next one. It is the row axis's only
    way to express a level the stub block has no column for, and unlike the
    column axis it leaves no merged span behind: the source marked it by
    indentation, which the published grid has already stripped.

    The label is not always in the stub. HiTab writes these as ``('', 'percent',
    '', '')`` as often as ``('other', '', '', '')``, so keying on the stub
    column alone finds well under half of them.
    """
    filled = [str(x).strip() for x in row if str(x).strip()]
    return filled[0] if len(filled) == 1 else ""


def reconstruct_row_paths(grid: Grid, n_header_rows: int, n_header_cols: int = 1,
                          use_section_rows: bool = True) -> List[List[str]]:
    """One header path per DATA row (rows >= ``n_header_rows``), mirroring
    :func:`reconstruct_col_paths`.

    ``use_section_rows`` prepends the section-row level above the stub columns,
    so a hierarchy deeper than the stub block still has somewhere to live.
    Pass False for the stub-only reconstruction this replaced.
    """
    n_rows = len(grid)
    if n_header_cols <= 0:
        return [[] for _ in range(max(n_rows - n_header_rows, 0))]
    body = range(n_header_rows, n_rows)
    levels = [[grid[r][c] for r in body] for c in range(n_header_cols)]

    # Only when the stub is a single column. With two or more the hierarchy
    # already has columns to live in, and a lone filled row is far more often a
    # spacer or unit annotation than a level -- promoting it there costs more
    # than it recovers (HiTab dev expressible bucket .9971 -> .9881).
    if use_section_rows and n_header_cols == 1:
        # Consecutive section rows declare NESTED levels: nothing separates a
        # heading from its sub-heading but adjacency, so a run of length L is a
        # stack L deep. A section row that follows data opens a fresh stack --
        # whether it is a sibling of the last heading or its child is not
        # recoverable from a grid that dropped the indentation.
        stacks, stack, run = [], [], 0
        for i, r in enumerate(body):
            lab = section_label(grid[r])
            if lab:
                stack = (stack if run else [])[:] + [lab]
                run += 1
                # the heading is not also its own child: blank its stub so the
                # label cannot appear twice in the same path
                for d in range(n_header_cols):
                    levels[d][i] = ""
            else:
                run = 0
            stacks.append(stack)
        depth = max((len(s) for s in stacks), default=0)
        if depth:
            levels = [[s[d] if d < len(s) else "" for s in stacks]
                      for d in range(depth)] + levels

    return _hierarchical_carry(levels)


def parse_html_table_with_merges(html: str):
    """Like :func:`parse_html_table`, but ALSO returns the ``merged_regions``
    (``rowspan``/``colspan`` geometry) instead of discarding it after resolving
    the occupancy grid. Returns ``(grid, merged_regions)`` where ``grid`` is the
    same blank-after-first grid and each spanning cell contributes one region
    ``{first_row,last_row,first_column,last_column}`` — so the markup can be
    CONSUMED via :func:`reconstruct_paths_with_merges` rather than re-inferred
    from blanks (the realistic case when a scrape KEEPS the span attributes)."""
    fixed = _FIX_MISSING_GT.sub(r"\1>", html)
    p = _TableHTMLParser()
    p.feed(fixed)
    raw_rows = p.rows
    if not raw_rows:
        return [], []

    active_rowspans: Dict[int, int] = {}
    grid: Grid = []
    merges: List[dict] = []
    for row_cells in raw_rows:
        row_out: List[str] = []
        r = len(grid)
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
            if span_c > 1 or span_r > 1:
                merges.append({"first_row": r, "last_row": r + span_r - 1,
                               "first_column": col, "last_column": col + span_c - 1})
            for _ in range(1, span_c):
                row_out.append("")
            if span_r > 1:
                for cc in range(col, col + span_c):
                    active_rowspans[cc] = max(active_rowspans.get(cc, 0), span_r - 1)
            col += span_c
        grid.append(row_out)

    width = max((len(rr) for rr in grid), default=0)
    for rr in grid:
        rr.extend([""] * (width - len(rr)))
    return grid, merges


# ---------------------------------------------------------------------------
# markup-aware reconstruction: consume merged_regions instead of guessing spans
# ---------------------------------------------------------------------------

def _fill_merges(grid: Grid, merged_regions: List[dict]) -> Grid:
    """Return a rectangular copy of ``grid`` with every merged region's origin
    text written into *all* cells it covers.

    The texts-only path (:func:`reconstruct_col_paths`) has to INFER spans from
    blank-after-first cells; here the span geometry is given by the markup
    (``merged_regions``: ``{first_row,last_row,first_column,last_column}``),
    so the fill is exact rather than heuristic.
    """
    width = max((len(r) for r in grid), default=0)
    filled = [list(r) + [""] * (width - len(r)) for r in grid]
    n_rows = len(filled)
    for m in merged_regions or []:
        r0, c0 = m.get("first_row", -1), m.get("first_column", -1)
        r1, c1 = m.get("last_row", r0), m.get("last_column", c0)
        if not (0 <= r0 < n_rows and 0 <= c0 < width):
            continue
        val = filled[r0][c0]
        for r in range(r0, min(r1, n_rows - 1) + 1):
            for c in range(c0, min(c1, width - 1) + 1):
                filled[r][c] = val
    return filled


def reconstruct_paths_with_merges(
    grid: Grid,
    merged_regions: List[dict],
    n_header_rows: int,
    n_header_cols: int = 1,
):
    """Markup-aware counterpart to :func:`reconstruct_col_paths` /
    :func:`reconstruct_row_paths`, returning ``(col_paths, row_paths)``.

    Two things the texts-only path cannot do, both driven by ``merged_regions``:

    1. **Exact span fill.** Each header cell's coverage is taken from the markup,
       not inferred from blanks — no over/under-carry.
    2. **Full-width header-band qualifier lift.** A header-band row whose entire
       data-column region is one merged value (e.g. table 1008's ``percent``
       spanning cols 1-8 at row 2) is not a column-distinguishing header — HiTab
       folds it into the LEFT (row) axis as a common ancestor. This row is dropped
       from the column paths and prepended to every row path. This is exactly the
       parent the texts-only reconstructor loses (the cell sits in the header
       band, outside the stub-column region ``reconstruct_row_paths`` scans).
    """
    filled = _fill_merges(grid, merged_regions)
    n_rows = len(filled)
    n_cols = len(filled[0]) if filled else 0
    nhr = max(0, min(n_header_rows, n_rows))
    nhc = max(0, min(n_header_cols, n_cols))

    # header-band rows that are a single full-width value -> row-axis qualifiers
    qualifiers: List[str] = []
    qualifier_rows = set()
    for r in range(nhr):
        data_vals = {filled[r][c].strip() for c in range(nhc, n_cols)}
        data_vals.discard("")
        if len(data_vals) == 1:
            qualifiers.append(next(iter(data_vals)))
            qualifier_rows.add(r)
    header_rows = [r for r in range(nhr) if r not in qualifier_rows]

    def _dedup(seq: List[str]) -> List[str]:
        out: List[str] = []
        for v in seq:
            v = v.strip()
            if v and (not out or out[-1] != v):
                out.append(v)
        return out

    col_paths = [_dedup([filled[r][c] for r in header_rows])
                 for c in range(nhc, n_cols)]
    row_paths = [_dedup(qualifiers + [filled[r][c] for c in range(nhc)])
                 for r in range(nhr, n_rows)]
    return col_paths, row_paths


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


def _section_boundary(grid: Grid, r: int) -> bool:
    """A section row ends the header block.

    Both signals below walk straight past one: its stub is blank, so the corner
    scan sees no row label, and it holds no numbers, so the numeric scan sees no
    data. That is the dominant boundary error -- 93 of 121 dev misses overshoot
    by exactly one, and five of the first six examples are a section row sitting
    immediately under the header. It belongs to the row-header structure of the
    DATA region (gold puts the boundary above it), so stopping here is the same
    fact `section_label` already encodes, not a patch. A parenthesised units
    note is excluded: those really do live inside the header block.
    """
    lab = section_label(grid[r])
    return bool(lab) and not _PAREN_NOTE_RE.match(lab.strip())


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
            if _section_boundary(grid, r):
                return r
            if not _left_region_blank(grid, r, n_header_cols) and \
                    not _left_region_units_note(grid, r, n_header_cols):
                return r
        # No row label ever appears (e.g. the table has no real row headers):
        # the corner signal is void — fall through to the numeric scan.

    # A section row can only END a header block, never BE one: on a spreadsheet
    # export the first row is a one-cell TITLE, which is a section row by shape
    # and made this loop return a header block of zero on 43 of 92 RealHiTBench
    # tables. HiTab grids carry no title row, so no HiTab split could show it.
    # So the rule waits until a row that actually looks like a header (two or
    # more filled cells) has been seen.
    seen_header_row = False
    for r in range(limit):
        if seen_header_row and _section_boundary(grid, r):
            return r
        if _numeric_data_row(grid, r, n_header_cols):
            return r
        seen_header_row = seen_header_row or sum(
            1 for x in grid[r] if str(x).strip()) >= 2

    # Neither signal fired: a text-valued table (Wikipedia discographies, cast
    # lists) where no numeric rule can ever find the first data row. Returning
    # `limit` here declared the top EIGHT rows header on 9 dev tables whose real
    # answer was 2 -- the worst available guess. Header rows carry the blanks
    # their merges leave behind, so the first row with no blank at all is the
    # first data row.
    for r in range(1, limit):
        if all(str(x).strip() for x in grid[r]):
            return r
    return 1


# Statistical agencies write "no data" as a marker, not a blank. A data column
# full of these is still a data column; counting the markers as text is what
# made the column guess overshoot by one on 374 of 2,043 HiTab tables.
_MISSING_MARKERS = {"..", "...", ".", "-", "--", "–", "—", "n/a", "na", "x",
                    "f", "e", "r", "*", "†", "‡", "nil", "none", "not available"}


def _data_like(s: str) -> bool:
    return looks_numeric(s) or s.strip().lower() in _MISSING_MARKERS


def _numeric_data_col(grid: Grid, c: int, n_header_rows: int) -> bool:
    cells = [grid[r][c] for r in range(n_header_rows, len(grid))
             if c < len(grid[r])]
    n_nonblank = sum(1 for x in cells if x.strip())
    n_num = sum(1 for x in cells if _data_like(x))
    return bool(n_nonblank) and n_num / n_nonblank >= 0.5


def guess_n_header_cols(grid: Grid, n_header_rows: int = 1,
                        max_header_cols: int = 4) -> int:
    """Guess how many left columns are row headers — the column-axis mirror of
    :func:`guess_n_header_rows`, so callers stop passing a hardcoded
    ``n_header_cols=1``.

    **Blank-tail signal (primary).** A stub column's label sits at the *top* of
    the header block and spans down, so the block's LAST header row is blank
    over the stub columns and carries the leaf column labels everywhere else::

        club     season   league                 <- header row 0
        (blank)  (blank)  division  apps  goals  <- last header row -> 2 stubs

    Measured on 2,043 HiTab tables against gold: 91.3% exact, versus 76.0% for
    the content-type scan below. The gap is entirely text-valued tables (sports
    and election tables whose data cells are team names and dates), where no
    numeric-vs-text rule can find the first data column at all.

    **Content-type scan (fallback, when the header block is one row deep).** The
    first column whose cells below the header block are >=50% data-like.

    Returns at least 1: treating column 0 as data would leave the row axis with
    no labels, strictly worse than the one-column default.

    Deliberately NOT solved here: a hierarchy encoded as parent rows *inside* one
    stub column (HiTab does this in 41% of tables) is invisible to any column
    count — see docs/RECONSTRUCTION_VALIDITY.md.
    """
    if not grid or not grid[0]:
        return 1
    limit = min(len(grid[0]), max_header_cols)

    tail = 0
    if n_header_rows >= 2 and n_header_rows - 1 < len(grid):
        last = grid[n_header_rows - 1]
        for c in range(min(len(last), limit)):
            if last[c].strip():
                break
            tail += 1

    scan = None
    for c in range(limit):
        if _numeric_data_col(grid, c, n_header_rows):
            scan = max(c, 1)
            break

    # The blank tail overcounts a data column whose label is merged DOWNWARD:
    # "mean" sits at header row 1 and the last header row is blank under it, so
    # the tail reads it as a second stub. The content scan sees straight through
    # that -- its cells are numbers. So where the scan actually found a data
    # column, it caps the tail; where it found none (the text-valued sports and
    # election tables the tail exists for) the tail stands alone.
    if tail >= 1:
        return min(tail, scan) if scan is not None else tail
    if scan is not None:
        return scan
    return max(limit, 1)
