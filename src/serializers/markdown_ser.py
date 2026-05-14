# SPDX-License-Identifier: MIT
"""Markdown serializer — intentionally lossy.

Markdown pipe tables cannot represent:
* merged cells (rowspan / colspan)
* hierarchical headers (multi-row headers are flattened)
Only the first row is treated as the header row on round-trip.
"""

from __future__ import annotations

import re

from src.io.table_schema import Cell, Table
from src.serializers.base import SerializerBase


class MarkdownSerializer(SerializerBase):
    """Lossy Markdown pipe-table serializer."""

    # ------------------------------------------------------------------
    # serialize
    # ------------------------------------------------------------------
    def serialize(self, table: Table) -> str:
        """Flatten *table* into a Markdown pipe table.

        * Covered cells (span 0) are replaced with empty strings.
        * All rows are emitted; only the first row gets a separator line
          beneath it (Markdown convention).
        """
        if not table.cells:
            return ""

        n_cols = table.n_cols
        lines: list[str] = []
        for r_idx, row in enumerate(table.cells):
            cols: list[str] = []
            for cell in row:
                val = cell.value if (cell.row_span != 0 or cell.col_span != 0) else ""
                cols.append(val)
            # Pad / trim to n_cols.
            cols = (cols + [""] * n_cols)[:n_cols]
            lines.append("| " + " | ".join(cols) + " |")
            if r_idx == 0:
                lines.append("| " + " | ".join(["---"] * n_cols) + " |")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # parse
    # ------------------------------------------------------------------
    def parse(self, text: str) -> Table:
        """Parse a Markdown pipe table back into a Table.

        The separator line (``| --- | --- |``) is skipped.
        The first data row is tagged as header.
        No merged-cell or tree information is recovered.
        """
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        rows: list[list[str]] = []
        for line in lines:
            # Skip separator.
            if re.fullmatch(r"\|[\s\-:|]+\|", line):
                continue
            # Split on pipes, strip outer empties.
            parts = [p.strip() for p in line.split("|")]
            if parts and parts[0] == "":
                parts = parts[1:]
            if parts and parts[-1] == "":
                parts = parts[:-1]
            rows.append(parts)

        if not rows:
            return Table(cells=[])

        cells: list[list[Cell]] = []
        for r_idx, row in enumerate(rows):
            cell_row = [
                Cell(value=v, is_header=(r_idx == 0)) for v in row
            ]
            cells.append(cell_row)

        return Table(cells=cells)
