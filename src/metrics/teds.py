# SPDX-License-Identifier: MIT
"""Tree-Edit Distance Similarity (TEDS) metric.

Follows the PubTabNet evaluation protocol.  We convert each Table into
a labelled tree mirroring its HTML structure, compute the tree-edit
distance using the *zss* library, and normalise to [0, 1].

For large tables the tree-edit distance computation is O(n²) or worse.
A ``max_nodes`` guard skips the expensive computation and returns NaN.
"""

from __future__ import annotations

import math

import zss  # type: ignore[import-untyped]

from src.io.table_schema import Table

# Tables whose tree exceeds this node count are skipped (NaN).
_MAX_TREE_NODES = 200


# ---------------------------------------------------------------------------
# Table → zss Node conversion
# ---------------------------------------------------------------------------

def _table_to_zss(table: Table) -> zss.Node:
    """Convert a Table to a zss labelled tree.

    The tree mirrors an HTML ``<table>`` element:
        table → tr → td/th (with attrs) → text value
    """
    root = zss.Node("table")
    for row in table.cells:
        tr = zss.Node("tr")
        for cell in row:
            if cell.row_span == 0 and cell.col_span == 0:
                continue  # skip covered cells
            attrs: list[str] = []
            tag = "th" if cell.is_header else "td"
            if cell.row_span > 1:
                attrs.append(f"rowspan={cell.row_span}")
            if cell.col_span > 1:
                attrs.append(f"colspan={cell.col_span}")
            label = f"{tag}:{':'.join(attrs)}" if attrs else tag
            td_node = zss.Node(label)
            if cell.value:
                td_node.addkid(zss.Node(cell.value))
            tr.addkid(td_node)
        root.addkid(tr)
    return root


def _tree_size(node: zss.Node) -> int:
    """Count total nodes in a zss tree."""
    return 1 + sum(_tree_size(c) for c in node.children)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def teds(
    table_orig: Table,
    table_recovered: Table,
    max_nodes: int = _MAX_TREE_NODES,
) -> float:
    """Compute TEDS between two tables.

    Args:
        table_orig: Ground-truth table.
        table_recovered: Table recovered after serialisation round-trip.
        max_nodes: Skip computation and return NaN if either tree
            exceeds this size.  Set to 0 to disable the guard.

    Returns:
        Similarity score in [0, 1].  1.0 means identical structure.
        ``float('nan')`` if the table is too large.
    """
    tree1 = _table_to_zss(table_orig)
    tree2 = _table_to_zss(table_recovered)

    size1 = _tree_size(tree1)
    size2 = _tree_size(tree2)

    if max_nodes > 0 and max(size1, size2) > max_nodes:
        return float("nan")

    max_size = max(size1, size2)
    if max_size == 0:
        return 1.0

    dist: int = zss.simple_distance(tree1, tree2)
    return 1.0 - dist / max_size
