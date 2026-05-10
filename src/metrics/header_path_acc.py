# SPDX-License-Identifier: MIT
"""Header Path Accuracy metric.

For each leaf data cell, compare the ground-truth ancestor header chain
(from root to leaf in the top/left header tree) with the chain recovered
after a serialisation round-trip.

Because lossy serialisers may not reconstruct header trees, we fall back
to cell-value heuristics when trees are absent.
"""

from __future__ import annotations

from src.io.table_schema import Table


def _collect_leaf_header_paths(
    table: Table,
) -> dict[tuple[int, int], tuple[list[str], list[str]]]:
    """For each non-header cell, collect (top_chain, left_chain).

    Returns:
        Mapping of (row, col) → (top_ancestor_names, left_ancestor_names).
    """
    top_tree = table.top_header_tree
    left_tree = table.left_header_tree

    n_top = table.metadata.get("top_header_rows_num", 0)
    n_left = table.metadata.get("left_header_columns_num", 0)

    paths: dict[tuple[int, int], tuple[list[str], list[str]]] = {}
    for r in range(table.n_rows):
        for c in range(table.n_cols):
            cell = table.cells[r][c]
            if cell.is_header or (cell.row_span == 0 and cell.col_span == 0):
                continue

            # Top header chain for column c.
            top_chain = top_tree.ancestor_chain(c)
            # Remove virtual root name.
            if top_chain and top_chain[0] in ("<TOP>", "<ROOT>"):
                top_chain = top_chain[1:]

            # Left header chain for row r.
            left_chain = left_tree.ancestor_chain(r)
            if left_chain and left_chain[0] in ("<LEFT>", "<ROOT>"):
                left_chain = left_chain[1:]

            paths[(r, c)] = (top_chain, left_chain)

    return paths


def header_path_accuracy(table_orig: Table, table_recovered: Table) -> float:
    """Compute header-path accuracy between original and recovered tables.

    For every data cell present in both tables, compare the concatenated
    header ancestor chain.  The score is the fraction of cells whose
    chains match exactly.

    Args:
        table_orig: Ground-truth table.
        table_recovered: Table after serialisation round-trip.

    Returns:
        Accuracy in [0, 1].
    """
    orig_paths = _collect_leaf_header_paths(table_orig)
    rec_paths = _collect_leaf_header_paths(table_recovered)

    if not orig_paths:
        return 1.0  # no data cells to compare

    match = 0
    total = 0
    for (r, c), (orig_top, orig_left) in orig_paths.items():
        rec_top, rec_left = rec_paths.get((r, c), ([], []))
        total += 1
        if orig_top == rec_top and orig_left == rec_left:
            match += 1

    return match / total if total > 0 else 1.0
