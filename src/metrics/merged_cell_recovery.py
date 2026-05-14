# SPDX-License-Identifier: MIT
"""Merged Cell Recovery Rate metric.

Measures the fraction of ground-truth merged regions that are exactly
recovered after a serialisation round-trip.
"""

from __future__ import annotations

from src.io.table_schema import Table


def merged_cell_recovery(table_orig: Table, table_recovered: Table) -> float:
    """Fraction of original merged-cell regions recovered exactly.

    A merge (r1, c1, r2, c2) is considered *recovered* if the same tuple
    appears in the recovered table's ``merged_cells`` list **or** the
    origin cell at (r1, c1) has the correct ``row_span`` and ``col_span``.

    Args:
        table_orig: Ground-truth table.
        table_recovered: Table after serialisation round-trip.

    Returns:
        Recovery rate in [0, 1].  Returns 1.0 if there are no merges.
    """
    if not table_orig.merged_cells:
        return 1.0

    rec_set = set(table_recovered.merged_cells)

    match = 0
    for merge in table_orig.merged_cells:
        r1, c1, r2, c2 = merge
        expected_rs = r2 - r1 + 1
        expected_cs = c2 - c1 + 1

        if merge in rec_set:
            match += 1
        elif (
            r1 < table_recovered.n_rows
            and c1 < table_recovered.n_cols
            and table_recovered.cells[r1][c1].row_span == expected_rs
            and table_recovered.cells[r1][c1].col_span == expected_cs
        ):
            match += 1

    return match / len(table_orig.merged_cells)
