# SPDX-License-Identifier: MIT
"""OTSL (Optimised Table Structure Language) serializer.

OTSL tokens (from Lysak et al., 2023 / IBM Docling):
  C  <text>  — regular cell with content
  L          — left-looking span (colspan continuation)
  U          — up-looking span (rowspan continuation)
  X          — cross span (both row and col continuation)
  NL         — new line (end of row)

This implementation encodes the cell grid into OTSL tokens and decodes
them back.  Header trees are NOT preserved (only the cell grid + spans).

We use tab (``\\t``) as a token separator so that cell values containing
reserved words (C, L, U, X, NL) are not ambiguous.
"""

from __future__ import annotations

from src.io.table_schema import Cell, Table
from src.serializers.base import SerializerBase

# Token constants
_C = "C"
_L = "L"
_U = "U"
_X = "X"
_NL = "NL"

_SEP = "\t"


class OtslSerializer(SerializerBase):
    """OTSL token-based table serializer."""

    # ------------------------------------------------------------------
    # serialize
    # ------------------------------------------------------------------
    def serialize(self, table: Table) -> str:
        """Encode *table* as an OTSL token sequence.

        Strategy: build an auxiliary *origin* grid that maps every (r, c)
        to the (origin_r, origin_c) of the cell that spans over it, then
        assign tokens accordingly.

        Token format per cell: ``C<tab>value`` or bare ``L`` / ``U`` / ``X``.
        Rows end with ``NL``.  Tokens are separated by ``<tab>``.
        """
        if not table.cells:
            return ""

        n_rows = table.n_rows
        n_cols = table.n_cols

        # Build origin map: (r, c) -> (origin_r, origin_c)
        origin: list[list[tuple[int, int]]] = [
            [(r, c) for c in range(n_cols)] for r in range(n_rows)
        ]
        for row_idx, row in enumerate(table.cells):
            for col_idx, cell in enumerate(row):
                if cell.row_span > 1 or cell.col_span > 1:
                    rs = cell.row_span
                    cs = cell.col_span
                    for dr in range(rs):
                        for dc in range(cs):
                            rr, cc = row_idx + dr, col_idx + dc
                            if rr < n_rows and cc < n_cols:
                                origin[rr][cc] = (row_idx, col_idx)

        tokens: list[str] = []
        for r in range(n_rows):
            for c in range(n_cols):
                or_, oc = origin[r][c]
                if or_ == r and oc == c:
                    val = table.cells[r][c].value
                    tokens.append(f"{_C}{_SEP}{val}")
                elif or_ == r and oc < c:
                    tokens.append(_L)
                elif or_ < r and oc == c:
                    tokens.append(_U)
                else:
                    tokens.append(_X)
            tokens.append(_NL)

        return _SEP.join(tokens)

    # ------------------------------------------------------------------
    # parse
    # ------------------------------------------------------------------
    def parse(self, text: str) -> Table:
        """Decode an OTSL token sequence back into a Table."""
        if not text.strip():
            return Table(cells=[])

        raw_tokens = _tokenize(text)

        rows_tokens: list[list[str]] = []
        current: list[str] = []
        for tok in raw_tokens:
            if tok == _NL:
                if current:
                    rows_tokens.append(current)
                current = []
            else:
                current.append(tok)
        if current:
            rows_tokens.append(current)

        if not rows_tokens:
            return Table(cells=[])

        n_rows = len(rows_tokens)
        n_cols = max(len(r) for r in rows_tokens)

        # Pad rows.
        for row in rows_tokens:
            while len(row) < n_cols:
                row.append(f"{_C}\t")

        # First pass: place origin cells.
        grid: list[list[Cell | None]] = [[None] * n_cols for _ in range(n_rows)]
        for r in range(n_rows):
            for c in range(n_cols):
                tok = rows_tokens[r][c]
                if tok.startswith(f"{_C}\t") or tok == _C:
                    val = tok.split("\t", 1)[1] if "\t" in tok else ""
                    grid[r][c] = Cell(value=val, row_span=1, col_span=1)

        # Second pass: expand spans.
        merged: list[tuple[int, int, int, int]] = []
        for r in range(n_rows):
            for c in range(n_cols):
                cell = grid[r][c]
                if cell is not None and cell.row_span >= 1:
                    raw_tok = rows_tokens[r][c]
                    if not (raw_tok.startswith(f"{_C}\t") or raw_tok == _C):
                        continue
                    # Compute how far L and U/X extend.
                    cs = 1
                    while c + cs < n_cols and rows_tokens[r][c + cs] == _L:
                        cs += 1
                    rs = 1
                    while r + rs < n_rows and rows_tokens[r + rs][c] in (_U, _X):
                        rs += 1

                    cell.row_span = rs
                    cell.col_span = cs

                    if rs > 1 or cs > 1:
                        merged.append((r, c, r + rs - 1, c + cs - 1))

                    # Fill covered positions.
                    for dr in range(rs):
                        for dc in range(cs):
                            if dr == 0 and dc == 0:
                                continue
                            grid[r + dr][c + dc] = Cell(
                                value="", row_span=0, col_span=0,
                            )

        # Replace remaining Nones.
        cells: list[list[Cell]] = []
        for row in grid:
            cells.append([c if c is not None else Cell(value="") for c in row])

        return Table(cells=cells, merged_cells=merged)


def _tokenize(text: str) -> list[str]:
    """Split tab-separated OTSL text into a token list.

    ``C\\tvalue`` is one token.  ``L``, ``U``, ``X``, ``NL`` are bare
    single tokens.
    """
    tokens: list[str] = []
    parts = text.split(_SEP)
    i = 0
    while i < len(parts):
        p = parts[i].strip()
        if p == _C:
            # Next part is the cell value.
            if i + 1 < len(parts):
                tokens.append(f"{_C}\t{parts[i + 1]}")
                i += 2
            else:
                tokens.append(f"{_C}\t")
                i += 1
        elif p in (_L, _U, _X, _NL):
            tokens.append(p)
            i += 1
        elif p == "":
            i += 1
        else:
            # Value fragment that follows C — shouldn't happen with tab sep.
            tokens.append(f"{_C}\t{p}")
            i += 1
    return tokens
