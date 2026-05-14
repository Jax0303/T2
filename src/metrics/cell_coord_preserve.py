# SPDX-License-Identifier: MIT
"""Cell Coordinate Preservation Rate metric.

Measures the fraction of (row, col) → value mappings that survive a
serialisation round-trip.  Only non-covered cells are counted.
"""

from __future__ import annotations

from src.io.table_schema import Table


def cell_coord_preservation(table_orig: Table, table_recovered: Table) -> float:
    """Fraction of cells whose (row, col) → value mapping is preserved.

    Args:
        table_orig: Ground-truth table.
        table_recovered: Table after serialisation round-trip.

    Returns:
        Preservation rate in [0, 1].
    """
    total = 0
    match = 0

    for r in range(table_orig.n_rows):
        for c in range(table_orig.n_cols):
            orig_cell = table_orig.cells[r][c]
            # Skip covered cells (part of a merge, not origin).
            if orig_cell.row_span == 0 and orig_cell.col_span == 0:
                continue
            total += 1

            # Check if recovered table has the same position.
            if r < table_recovered.n_rows and c < table_recovered.n_cols:
                rec_cell = table_recovered.cells[r][c]
                if rec_cell.value == orig_cell.value:
                    match += 1

    return match / total if total > 0 else 1.0
