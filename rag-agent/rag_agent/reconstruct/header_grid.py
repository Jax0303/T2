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


# v3 (PREREG-2026-09-14-header-v3.md): a level the source itself leaves empty. The carry
# stops here instead of treating the blank as the tail of a merge it never belonged to.
_HARD = "\x00"


def parse_html_table_layout(html: str):
    """:func:`parse_html_table`'s grid plus where every position came from (header rule v3).

    ``cover[r][c]`` is the position of the ``<td>`` that produced ``(r, c)``: the cell itself,
    the span's origin for a position a colspan/rowspan covers, or None for padding added because
    the row had fewer cells than the widest row. The grid is the one :func:`parse_html_table`
    returns, so the rules that read ``cover`` change paths, never cell coordinates.
    """
    fixed = _FIX_MISSING_GT.sub(r"\1>", html)
    p = _TableHTMLParser()
    p.feed(fixed)
    if not p.rows:
        return [], []
    active: Dict[int, int] = {}
    origin: Dict[int, tuple] = {}
    grid: Grid = []
    cover: List[list] = []
    for row_cells in p.rows:
        r = len(grid)
        row_out: List[str] = []
        row_cov: list = []
        col = ci = 0
        while ci < len(row_cells) or (active and col <= max(active)):
            if col in active:
                row_out.append("")
                row_cov.append(origin[col])
                active[col] -= 1
                if active[col] <= 0:
                    del active[col]
                col += 1
                continue
            if ci >= len(row_cells):
                col += 1
                continue
            cell = row_cells[ci]
            ci += 1
            here = (r, len(row_out))
            row_out.append(cell["text"])
            row_cov.append(here)
            span_c, span_r = max(cell["colspan"], 1), max(cell["rowspan"], 1)
            for _ in range(1, span_c):
                row_out.append("")
                row_cov.append(here)
            if span_r > 1:
                for cc in range(col, col + span_c):
                    active[cc] = max(active.get(cc, 0), span_r - 1)
                    origin[cc] = here
            col += span_c
        grid.append(row_out)
        cover.append(row_cov)
    width = max(len(x) for x in grid)
    for x, cv in zip(grid, cover):
        cv.extend([None] * (width - len(x)))
        x.extend([""] * (width - len(x)))
    return grid, cover


def _marked_rows(cover) -> set:
    """Rows whose markup declares a span: a span's origin row or a row a span covers."""
    rows = set()
    for r, line in enumerate(cover):
        for c, o in enumerate(line):
            if o is not None and o != (r, c):
                rows.update((r, o[0]))
    return rows


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
            cell = levels[d][i] if i < len(levels[d]) else ""
            if cell == _HARD:
                # v3: the source says this position is empty, not merged
                for e in range(d, n_depths):
                    carry[e] = ""
                continue
            cell = cell.strip()
            if cell:
                carry[d] = cell
                for e in range(d + 1, n_depths):
                    carry[e] = ""
        out.append([seg for seg in carry if seg])
    return out


def _years(path) -> set:
    return {y for seg in path for y in _YEAR_IN.findall(seg)}


def _narrow_col_paths(grid: Grid, n_header_rows: int, n_header_cols: int, cover) -> List[List[str]]:
    """v3.1 S1n (PREREG-2026-09-14-header-v3.md 정정 1): S1 kept to the labels that declare an extent.

    A carry stops only where the nearest label to its left declared a span (colspan/rowspan) that
    does not cover this position. A 1x1 label declares nothing, so its visual span is still inferred,
    and covered positions are not filled -- v2 already handled both, and S1's fill added nothing but
    units notes. A cut that would take a four-digit year off a column hands that column back to v2.
    """
    v2 = reconstruct_col_paths(grid, n_header_rows, n_header_cols)
    span_cols: Dict[tuple, set] = {}
    extent: Dict[tuple, int] = {}      # positions a cell covers: >1 means it declared a colspan or rowspan
    for line in cover:
        for c, p in enumerate(line):
            if p is not None:
                span_cols.setdefault(p, set()).add(c)
                extent[p] = extent.get(p, 0) + 1
    # Replay the v2 carry keeping each segment's source cell, then drop only what the markup
    # contradicts: a declared label past its span, and the labels that started inside that span.
    # A deeper 1x1 label that began after the span ended keeps its inferred carry
    # (정정 2: the first v3.1 run cleared it too and emptied 1,673 column paths).
    src = [None] * n_header_rows
    out = []
    for c in range(n_header_cols, len(grid[0])):
        for d in range(n_header_rows):
            if grid[d][c].strip():
                src[d] = (d, c)
                for e in range(d + 1, n_header_rows):
                    src[e] = None
        keep, cut_cols = [], set()
        for d, s in enumerate(src):
            if s is None:
                continue
            o = cover[d][c]
            if s[1] in cut_cols or (s != (d, c) and o is not None and o != s and extent[s] > 1):
                cut_cols |= span_cols[s] if extent[s] > 1 else set()
                continue
            keep.append(grid[s[0]][s[1]].strip())
        out.append(keep)
    return [p if not (_years(q) - _years(p)) else q for p, q in zip(out, v2)]


def reconstruct_col_paths(grid: Grid, n_header_rows: int, n_header_cols: int = 1,
                          cover=None, narrow: bool = False) -> List[List[str]]:
    """One header path per DATA column (columns >= ``n_header_cols``),
    restricted to the data-column region so a row-header label sitting in
    column 0 never bleeds into the column paths.

    ``cover`` (from :func:`parse_html_table_layout`) turns on v3 rule S1. In a header row that
    declares any span, the markup decides: a covered position takes its origin's text, and a
    blank ``<td>`` of its own is empty rather than the tail of the label to its left. Rows with
    no span markup, and padded positions, keep the inferred carry.
    """
    n_cols = len(grid[0]) if grid else 0
    if n_header_rows <= 0:
        return [[] for _ in range(max(n_cols - n_header_cols, 0))]
    if cover is None:
        levels = [grid[r][n_header_cols:] for r in range(n_header_rows)]
        return _hierarchical_carry(levels)
    if narrow:
        return _narrow_col_paths(grid, n_header_rows, n_header_cols, cover)
    marked = _marked_rows(cover)
    levels = []
    for r in range(n_header_rows):
        line = []
        for c in range(n_header_cols, n_cols):
            text, o = grid[r][c].strip(), cover[r][c]
            if text or o is None:
                line.append(text)
            elif o != (r, c):
                line.append(grid[o[0]][o[1]].strip() or _HARD)
            else:
                line.append(_HARD if r in marked else "")
        levels.append(line)
    # a rowspan hands the same label to the rows it covers; say it once
    return [[s for k, s in enumerate(p) if k == 0 or s != p[k - 1]]
            for p in _hierarchical_carry(levels)]


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


def band_qualifiers(grid: Grid, n_header_rows: int, n_header_cols: int) -> List[str]:
    """Single-value rows INSIDE the column header band, outermost first.

    A unit or scope annotation is written as its own header-band row holding one
    value and nothing else — ``('', 'percent', '', '')`` above a block of
    percentages. Structurally it is a section row that happens to sit above the
    data instead of inside it, and HiTab's gold puts it on the ROW path: the row
    for "marital status" is ``['percent', 'marital status']``.

    MEASURED AND REJECTED — kept as the record of a hypothesis that failed, so it
    is not proposed again. Promoting these onto every row path scores **0 gains
    and 110 losses** on HiTab dev (row EM .7607 -> .7454). What the rule actually
    picks up is the STUB COLUMN'S OWN NAME — "age group at symptom onset",
    "canadian community health survey cycle" — which names the label column and
    belongs to no row's path. The genuine qualifiers ("percent") were already
    being recovered by the body section-row pass, so there was nothing left to
    win. Wired off in :func:`reconstruct_row_paths`; do not turn it on without
    a discriminator that separates a scope annotation from a column name.

    The last header row is excluded whatever it looks like: that row is the
    column leaves, and a one-column table would otherwise donate its only column
    header to every row path.
    """
    out = []
    for r in range(max(0, n_header_rows - 1)):
        if r >= len(grid):
            break
        lab = section_label(grid[r])
        # Only a value in the DATA region counts. A lone label in the stub is the
        # stub column's own name ("Agency"), which belongs to no row's path.
        if lab and lab not in [str(x).strip() for x in grid[r][:n_header_cols]]:
            out.append(lab)
    return out


_TOTAL_RE = re.compile(r"^\s*(?:sub-?\s*)?totals?\b", re.I)


def reconstruct_row_paths(grid: Grid, n_header_rows: int, n_header_cols: int = 1,
                          use_section_rows: bool = True,
                          use_band_qualifiers: bool = True,
                          cover=None, close_on_total=False,
                          disambiguate=False,
                          isolated_blank_only: bool = False) -> List[List[str]]:
    """One header path per DATA row (rows >= ``n_header_rows``), mirroring
    :func:`reconstruct_col_paths`.

    ``use_section_rows`` prepends the section-row level above the stub columns,
    so a hierarchy deeper than the stub block still has somewhere to live.
    Pass False for the stub-only reconstruction this replaced.

    ``use_band_qualifiers`` additionally prepends :func:`band_qualifiers` — the
    single-value rows in the column header band — as the outermost level.

    v3 (PREREG-2026-09-14-header-v3.md), all off by default:

    * ``cover`` — S2: a data row whose last stub cell is its own blank ``<td>`` (no rowspan
      over it) has no label of its own; it does not inherit the row above's.
    * ``close_on_total`` — S4: rows after a ``Total``/``Subtotal`` row leave that row's
      innermost section. The leading scope is never closed.
    * ``disambiguate`` — S5: data rows left with identical paths get the nearest section
      headings above them that are not on the path, only as many as tell them apart.

    v3.1: ``close_on_total="named"`` (S4n), ``disambiguate="guarded"`` (S5n). v3.2:
    ``close_on_total="own"`` (S4n2) also keeps a section open while its own ``Total <heading>`` row is
    still below; ``disambiguate="no_numbers"`` (S5n2) also refuses a number as a heading.
    """
    n_rows = len(grid)
    if n_header_cols <= 0:
        return [[] for _ in range(max(n_rows - n_header_rows, 0))]
    body = range(n_header_rows, n_rows)
    levels = [[grid[r][c] for r in body] for c in range(n_header_cols)]

    def has_data(r):
        return any(str(x).strip() for x in grid[r][n_header_cols:])

    if cover is not None:
        leaf = n_header_cols - 1
        blank = [not str(grid[r][leaf]).strip() and cover[r][leaf] == (r, leaf) and has_data(r)
                 for r in body]
        for i, b in enumerate(blank):
            # v3.1 S2n: a run of blank-label rows is more often one visual group than a list of totals
            lone = not (i and blank[i - 1]) and not (i + 1 < len(blank) and blank[i + 1])
            if b and (lone or not isolated_blank_only):
                levels[leaf][i] = _HARD

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
        # The FIRST section run of the body is the table's own scope, not a
        # sibling of what follows: a unit or population row ("percent",
        # "current $millions") sits above the header block and stays in force
        # over every group beneath it. Resetting to [] on the next heading drops
        # it, which is why "marital status" reconstructs as ['marital status']
        # where the gold reads ['percent', 'marital status']. `keep` is how much
        # of the stack a later heading inherits instead of clearing.
        keep = 0
        seen_data = False
        # v3 bookkeeping: which body row each heading came from, every heading seen so far,
        # and whether S4 closed the stack down to nothing. With S4/S5 off no path reads it.
        sidxs, sidx, trails, trail, hard0, emptied = [], [], [], [], [], False
        closeds, closed = [], frozenset()   # v3.1: headings a total row has closed so far
        for i, r in enumerate(body):
            lab = section_label(grid[r])
            if lab:
                if run == 0 and seen_data and keep:
                    stack, sidx = stack[:keep] + [lab], sidx[:keep] + [i]
                else:
                    stack = (stack if run else [])[:] + [lab]
                    sidx = (sidx if run else [])[:] + [i]
                trail, emptied = trail + [(i, lab)], False
                run += 1
                # the heading is not also its own child: blank its stub so the
                # label cannot appear twice in the same path
                for d in range(n_header_cols):
                    levels[d][i] = ""
            else:
                # The leading run's outer levels become the persistent scope the
                # moment data proves the run has ended. Only the run's LAST
                # heading is a group label that later headings replace.
                if run and not seen_data:
                    keep = max(0, len(stack) - 1)
                if any(str(x).strip() for x in grid[r][n_header_cols:]):
                    seen_data = True
                run = 0
            stacks.append(stack)
            sidxs.append(sidx)
            trails.append(trail)
            hard0.append(emptied)
            closeds.append(closed)
            label = str(grid[r][n_header_cols - 1])
            if (close_on_total and not lab and len(stack) > keep and has_data(r)
                    and _TOTAL_RE.match(label)
                    and (close_on_total not in ("named", "own") or _total_names(label, stack[-1]))
                    # v3.2 S4n2: while the section's own total is still below, any other total is a sub-total
                    and (close_on_total != "own" or _own_total(label, stack[-1])
                         or not _own_total_below(grid, r, n_header_cols, stack[-1]))):
                closed = closed | {sidx[-1]}
                stack, sidx = stack[:-1], sidx[:-1]
                emptied = not stack
        stub = levels
        levels = _stack_levels(stacks, hard0) + stub
        if disambiguate:
            guarded = disambiguate in ("guarded", "no_numbers")
            stacks = _split_identical_paths(stacks, sidxs, trails, levels,
                                            [bool(section_label(grid[r])) or not has_data(r)
                                             for r in body],
                                            closed=closeds if guarded else None, other_years=guarded,
                                            skip_numbers=disambiguate == "no_numbers")
            levels = _stack_levels(stacks, hard0) + stub

    # NOT applied: see band_qualifiers' docstring for why it stays off.
    if use_band_qualifiers and False:  # pragma: no cover
        quals = band_qualifiers(grid, n_header_rows, n_header_cols)
        n_body = len(list(body))
        levels = [[q] * n_body for q in quals] + levels

    return _hierarchical_carry(levels)


def _stack_levels(stacks, hard0) -> List[List[str]]:
    """Section stacks as levels for :func:`_hierarchical_carry`, outermost first.

    A row that S4 closed down to no section gets a hard blank at the top depth, so the carry
    cannot hand back the heading it just left.
    """
    depth = max((len(s) for s in stacks), default=0)
    return [[s[d] if d < len(s) else (_HARD if d == 0 and h else "")
             for s, h in zip(stacks, hard0)] for d in range(depth)]


_STOP = {"and", "of", "the", "in", "for", "to", "a", "an"}


def _total_names(label: str, heading: str) -> bool:
    """v3.1 S4n: a bare Total/Subtotal, or a total that carries every word of the heading.

    Case, a plural s, stop words and footnote marks ("-1", "(a)") are ignored, so "Total revenue"
    closes "Revenues:" and "TOTAL ASSETS" closes "ASSETS", but "Total investments" does not.
    """
    def norm(s):
        return {w[:-1] if len(w) > 3 and w.endswith("s") else w
                for w in re.findall(r"[a-z0-9]+", s.lower())
                if w not in _STOP and len(w) > 1 and not w.isdigit()}
    rest = norm(_TOTAL_RE.sub("", label, count=1))
    return not rest or norm(heading) <= rest


def _own_total(label: str, heading: str) -> bool:
    """v3.2 S4n2: ``Total <heading>`` word for word — case and punctuation ignored, nothing added."""
    words = re.findall(r"[a-z0-9]+", heading.lower())
    return (bool(words) and bool(_TOTAL_RE.match(label))
            and re.findall(r"[a-z0-9]+", _TOTAL_RE.sub("", label, count=1).lower()) == words)


def _own_total_below(grid: Grid, r: int, n_header_cols: int, heading: str) -> bool:
    """v3.2 S4n2: a data row below ``r`` is the section's own total, before the heading comes back."""
    for y in range(r + 1, len(grid)):
        if section_label(grid[y]) == heading:
            return False
        if (_own_total(str(grid[y][n_header_cols - 1]), heading)
                and any(str(x).strip() for x in grid[y][n_header_cols:])):
            return True
    return False


def _split_identical_paths(stacks, sidxs, trails, levels, skip, closed=None, other_years=False,
                           skip_numbers=False):
    """S5: tell apart data rows whose paths are identical, with headings the source put above them.

    A row's candidates are the section headings above it that are not on its path, nearest first.
    A tied group takes the shortest prefix of candidates at which its members differ; members
    still tied continue from there. A group no prefix splits is left as it was.

    v3.1 S5n guards: ``closed[i]`` holds headings a total row closed before row ``i`` -- they no
    longer scope it -- and ``other_years`` refuses a heading dated differently from the path.
    v3.2 S5n2: ``skip_numbers`` refuses a heading that is a number ("544,823"); a year is not one.
    """
    paths = _hierarchical_carry(levels)
    groups: Dict[tuple, List[int]] = {}
    for i, p in enumerate(paths):
        if not skip[i]:
            groups.setdefault(tuple(p), []).append(i)
    extra: Dict[int, list] = {}

    def split(group, cands, done):
        for k in range(done + 1, max(len(cands[i]) for i in group) + 1):
            keys = {i: tuple(lab for _, lab in cands[i][:k]) for i in group}
            if len(set(keys.values())) > 1:
                ties: Dict[tuple, List[int]] = {}
                for i in group:
                    extra[i] = cands[i][:k]
                    ties.setdefault(keys[i], []).append(i)
                for tie in ties.values():
                    if len(tie) > 1:
                        split(tie, cands, k)
                return

    for group in groups.values():
        if len(group) < 2:
            continue
        cands = {}
        for i in group:
            seen, cands[i], dated = set(paths[i]), [], _years(paths[i])
            for ix, lab in reversed(trails[i]):
                if closed is not None and ix in closed[i]:
                    continue
                if other_years and dated and _years([lab]) - dated:
                    continue
                if skip_numbers and _number_like(lab):
                    continue
                if lab not in seen:
                    seen.add(lab)
                    cands[i].append((ix, lab))
        split(group, cands, 0)
    out = list(stacks)
    for i, add in extra.items():
        out[i] = [lab for _, lab in sorted(set(zip(sidxs[i], stacks[i])) | set(add))]
    return out


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

    # Row paths reuse reconstruct_row_paths' section-row stack. Reading the stub
    # columns alone -- which is all this function used to do -- cannot express a
    # level that has no stub column to live in, and on HiTab dev that is most of
    # the row axis: stub-only scores .5446 against .8202 for the texts-only path.
    # The grid handed over is filled in the STUB columns (exact merge spans, the
    # thing this function is for) and original in the data columns, because
    # section_label needs a section row's data region to still read as empty --
    # _fill_merges would have painted a full-width heading across every column.
    hybrid = [[filled[r][c] if c < nhc else
               (grid[r][c] if r < len(grid) and c < len(grid[r]) else "")
               for c in range(n_cols)]
              for r in range(n_rows)]
    row_paths = [_dedup(qualifiers + p)
                 for p in reconstruct_row_paths(hybrid, nhr, nhc)]
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


# v2 (PREREG-2026-09-13-header-units-note.md): the same annotation without the
# parentheses -- "$ in millions", "in millions of dollars", "Dollars in
# thousands" -- sat on the year row of 4.4% of MultiHiertt tables and ended the
# header block there, deleting the years from every cell path below it.
_UNITS_NOTE_RE = re.compile(r"\bin\s+(thousands|millions|billions)\b", re.I)


def _left_region_units_note(grid: Grid, r: int, n_header_cols: int,
                            rule: str = "v1") -> bool:
    text = " ".join(c.strip() for c in grid[r][:n_header_cols] if c.strip())
    if not text:
        return False
    if _PAREN_NOTE_RE.match(text):
        return True
    return rule in ("v2", "v3") and len(text) <= 60 and bool(_UNITS_NOTE_RE.search(text))


def _year_label_row(grid: Grid, r: int, n_header_cols: int) -> bool:
    """v2: every filled non-stub cell is a bare year -> a column-label row.

    `looks_numeric` already refuses to call a bare year data; this applies the
    same fact to the corner scan, whatever the stub cell on that row says.
    """
    cells = [c.strip() for c in grid[r][n_header_cols:] if c.strip()]
    return bool(cells) and all(_YEAR_RE.match(c) for c in cells)


def _number_like(s: str) -> bool:
    """v3.1 S3n: a number once currency, sign, parentheses and percent come off ("$-0.11",
    "$ -355", "(1.2)%"). A lone four-digit year is not, nor a year range or a footnoted year."""
    t = re.sub(r"[\s$,]", "", s).strip("()%").lstrip("-–—+").strip("()%")
    return bool(re.fullmatch(r"\d*\.?\d+", t)) and not re.fullmatch(r"(?:19|20)\d{2}", t)


def _label_row(grid: Grid, r: int, n_header_cols: int, numbers: bool = False) -> bool:
    """v3 S3: two or more filled non-stub cells, and none of them looks like data.

    The v2 year rule widened to the label rows it missed: footnoted years ("1999(a)"),
    dates ("December 31, 2011") and text leaves ("Variation"). `_data_like` counts
    missing markers as data, so a row of dashes still ends the header block.
    ``numbers`` (v3.1 S3n) also counts the signed currency `looks_numeric` misses as data --
    v3 moved 127 such rows into the header, two of them holding a query's gold cells.
    """
    cells = [c.strip() for c in grid[r][n_header_cols:] if c.strip()]
    return len(cells) >= 2 and not any(_data_like(c) or (numbers and _number_like(c)) for c in cells)


_YEAR_IN = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")


def corner_scope(grid: Grid, n_header_rows: int, n_header_cols: int, col_paths,
                 row_paths=None, max_len: int = 0) -> str:
    """v3 S6: the corner block's text, when it dates a table whose columns carry no year.

    The corner (header rows x stub columns) heads no data line, so no path ever read it, and
    a table's only date often sits there ("In millions of dollars at December 31, 2018").
    Only a corner with a four-digit year counts, and only when no column path has one: a
    corner that names the stub column ("(Dollars in millions)") stays out of every path.
    """
    # v3.1 S6n: a year on any row path means the rows date the table, and a corner past
    # ``max_len`` is a header block that swallowed body labels -- neither is one scope for all.
    if any(_YEAR_IN.search(seg) for paths in (col_paths, row_paths or []) for path in paths for seg in path):
        return ""
    text = " ".join(" ".join(str(grid[r][c]).split())
                    for r in range(min(n_header_rows, len(grid)))
                    for c in range(n_header_cols) if str(grid[r][c]).strip())
    if max_len and len(text) > max_len:
        return ""
    return text if _YEAR_IN.search(text) else ""


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


def guess_n_header_rows(grid: Grid, n_header_cols: int = 1, max_header_rows: int = 8,
                        rule: str = "v1") -> int:
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
    # v3.1 is v3 with S3n: the label-row test also reads signed currency as data
    numbers, rule = rule == "v3.1", ("v3" if rule == "v3.1" else rule)

    if n_header_cols > 0 and grid[0][:n_header_cols] and _left_region_blank(grid, 0, n_header_cols):
        for r in range(1, limit + 1):
            if r >= len(grid):
                break
            if _section_boundary(grid, r):
                return r
            if not _left_region_blank(grid, r, n_header_cols) and \
                    not _left_region_units_note(grid, r, n_header_cols, rule) and \
                    not (rule in ("v2", "v3") and _year_label_row(grid, r, n_header_cols)) and \
                    not (rule == "v3" and _label_row(grid, r, n_header_cols, numbers)):
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


def _col_unlabeled(grid: Grid, c: int, n_header_rows: int) -> bool:
    """Is column ``c`` blank across every header row?

    With ``parse_html_table_with_merges`` resolving colspans before this runs, a
    real data column carries a label somewhere in the header band ("2013"). A
    column that is blank all the way down the band is one the stub's group label
    spans over -- part of the row-header region, not the data.
    """
    cells = [grid[r][c] for r in range(min(n_header_rows, len(grid)))
             if c < len(grid[r])]
    return bool(cells) and not any(str(x).strip() for x in cells)


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

    Measured against gold with ``scripts/diag_boundary.py`` on 2026-08-26:
    92.4% exact on train (2,043 tables), 91.8% on dev (424), 91.8% on test (414),
    versus 76.0% for the content-type scan below. (The 91.3% this line used to
    quote was the TEST split before that day's column-boundary fix, not train.) The gap is entirely text-valued tables (sports
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
        nhc = min(tail, scan) if scan is not None else tail
    elif scan is not None:
        nhc = scan
    else:
        nhc = max(limit, 1)

    # Both signals above stop at the FIRST data-like column, and a numeric column
    # inside the stub region ends them one or two columns early. RealHiTBench's
    # `biology-table03` is the shape: a group label spanning col 0, a numeric
    # country CODE in col 1, the country NAME in col 2, years from col 3. The
    # scan stops at the code, so the name -- the label every query actually
    # says -- lands in the data and every row of the table gets the same address.
    # An unlabeled column cannot be data, so the region extends across it --
    # but ONLY while a labeled column still lies to the right. Without that
    # guard a spreadsheet export whose header band is blank (RealHiTBench ships
    # these: `education-table03` has two empty header rows) reads as all stub
    # and swallows the table, which is worse than the miss it fixes.
    # A row-header region has to contain at least one column of LABELS. When
    # every column the signals accepted is itself data-like, the scan stopped on
    # an index or code column: BLS tables lead with "Indent Level" (0, 1, 2, 3)
    # and put the row's name in the NEXT column, so the address ends up carrying
    # a line number where the query says "Cereals and bakery products".
    if nhc and all(_numeric_data_col(grid, c, n_header_rows) for c in range(nhc)):
        for c in range(nhc, limit):
            if not _numeric_data_col(grid, c, n_header_rows):
                nhc = c + 1
                break

    width = max((len(r) for r in grid[:max(1, n_header_rows)]), default=0)
    labeled = [c for c in range(width) if not _col_unlabeled(grid, c, n_header_rows)]
    last_labeled = max(labeled) if labeled else -1
    while nhc < limit and nhc < last_labeled and _col_unlabeled(grid, nhc, n_header_rows):
        nhc += 1
    return nhc
