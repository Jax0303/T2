# SPDX-License-Identifier: MIT
"""CSV serializer — lossy (no span / tree information)."""

from __future__ import annotations

import csv
import io

from src.io.table_schema import Cell, Table
from src.serializers.base import SerializerBase


class CsvSerializer(SerializerBase):
    """Standard CSV serializer.  Merged cells write value only at origin."""

    # ------------------------------------------------------------------
    # serialize
    # ------------------------------------------------------------------
    def serialize(self, table: Table) -> str:
        """Write *table* as RFC-4180 CSV text."""
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        for row in table.cells:
            writer.writerow([
                cell.value if (cell.row_span != 0 or cell.col_span != 0) else ""
                for cell in row
            ])
        return buf.getvalue()

    # ------------------------------------------------------------------
    # parse
    # ------------------------------------------------------------------
    def parse(self, text: str) -> Table:
        """Parse CSV text into a flat Table (no header/merge info)."""
        reader = csv.reader(io.StringIO(text))
        cells: list[list[Cell]] = []
        for row in reader:
            cells.append([Cell(value=v) for v in row])
        return Table(cells=cells)
