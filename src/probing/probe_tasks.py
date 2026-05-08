# SPDX-License-Identifier: MIT
"""Probe task dataset builders.

Three binary/multi-class classification tasks that test whether
structural table information is encoded in embedder hidden states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.io.table_schema import Table


@dataclass
class ProbeDataset:
    """Container for probe task data."""

    X: np.ndarray  # (n_samples, hidden_dim)
    y: np.ndarray  # (n_samples,) integer labels
    label_names: dict[int, str]
    task_name: str


# ---------------------------------------------------------------------------
# Task 1: Parent header prediction
# ---------------------------------------------------------------------------

def build_parent_header_task(
    tables: list[Table],
    hidden_states: dict[str, np.ndarray],
    layer: int,
    n_classes: int = 8,
    serializer_name: str = "html",
    seed: int = 42,
) -> ProbeDataset:
    """Predict the hash-bucket of a cell's parent header text.

    For each data cell, we look up its top-header ancestor chain and hash
    the first-level header name into ``n_classes`` buckets.

    Args:
        tables: List of Table objects.
        hidden_states: Mapping of cell-text → hidden states array
            (from HiddenStateExtractor).
        layer: Which layer index to use.
        n_classes: Number of hash buckets for header names.
        serializer_name: Used only for logging/metadata.
        seed: Random seed.

    Returns:
        A ProbeDataset.
    """
    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for table in tables:
        top_tree = table.top_header_tree
        for r in range(table.n_rows):
            for c in range(table.n_cols):
                cell = table.cells[r][c]
                if cell.is_header or (cell.row_span == 0 and cell.col_span == 0):
                    continue
                chain = top_tree.ancestor_chain(c)
                if len(chain) < 2:
                    continue
                header_name = chain[1]  # first non-root ancestor
                label = hash(header_name) % n_classes

                key = cell.value
                if key in hidden_states:
                    vec = hidden_states[key][layer]
                    X_list.append(vec)
                    y_list.append(label)

    if not X_list:
        return ProbeDataset(
            X=np.empty((0, 0)),
            y=np.empty(0, dtype=int),
            label_names={i: f"bucket_{i}" for i in range(n_classes)},
            task_name="parent_header",
        )

    return ProbeDataset(
        X=np.stack(X_list),
        y=np.array(y_list, dtype=int),
        label_names={i: f"bucket_{i}" for i in range(n_classes)},
        task_name="parent_header",
    )


# ---------------------------------------------------------------------------
# Task 2: Cell coordinate bucket prediction
# ---------------------------------------------------------------------------

def build_cell_coord_task(
    tables: list[Table],
    hidden_states: dict[str, np.ndarray],
    layer: int,
    n_row_buckets: int = 4,
    n_col_buckets: int = 4,
    seed: int = 42,
) -> ProbeDataset:
    """Predict the (row_bucket, col_bucket) of a data cell.

    The label is ``row_bucket * n_col_buckets + col_bucket``.

    Args:
        tables: List of Table objects.
        hidden_states: Mapping of cell-text → hidden states array.
        layer: Layer index.
        n_row_buckets: Number of row buckets.
        n_col_buckets: Number of column buckets.
        seed: Random seed.

    Returns:
        A ProbeDataset.
    """
    n_classes = n_row_buckets * n_col_buckets
    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for table in tables:
        max_r = max(table.n_rows, 1)
        max_c = max(table.n_cols, 1)
        for r in range(table.n_rows):
            for c in range(table.n_cols):
                cell = table.cells[r][c]
                if cell.is_header or (cell.row_span == 0 and cell.col_span == 0):
                    continue
                rb = min(int(r / max_r * n_row_buckets), n_row_buckets - 1)
                cb = min(int(c / max_c * n_col_buckets), n_col_buckets - 1)
                label = rb * n_col_buckets + cb

                key = cell.value
                if key in hidden_states:
                    X_list.append(hidden_states[key][layer])
                    y_list.append(label)

    if not X_list:
        return ProbeDataset(
            X=np.empty((0, 0)),
            y=np.empty(0, dtype=int),
            label_names={i: f"coord_{i}" for i in range(n_classes)},
            task_name="cell_coord",
        )

    return ProbeDataset(
        X=np.stack(X_list),
        y=np.array(y_list, dtype=int),
        label_names={i: f"coord_{i}" for i in range(n_classes)},
        task_name="cell_coord",
    )


# ---------------------------------------------------------------------------
# Task 3: Same-row classification (pair task)
# ---------------------------------------------------------------------------

def build_same_row_task(
    tables: list[Table],
    hidden_states: dict[str, np.ndarray],
    layer: int,
    max_pairs_per_table: int = 50,
    seed: int = 42,
) -> ProbeDataset:
    """Binary classification: are two cells from the same row?

    For each table, sample pairs of data cells.  Positive pairs share a
    row; negative pairs come from different rows.  The input to the probe
    is the concatenation (or difference) of the two cell embeddings —
    here we use ``|v1 - v2|`` (element-wise absolute difference) so that
    the probe input dimension equals hidden_dim.

    Args:
        tables: List of Table objects.
        hidden_states: Mapping of cell-text → hidden states array.
        layer: Layer index.
        max_pairs_per_table: Maximum pairs to sample per table.
        seed: Random seed.

    Returns:
        A ProbeDataset with binary labels (0 = different row, 1 = same row).
    """
    rng = np.random.default_rng(seed)
    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for table in tables:
        # Collect data cells with their row index and hidden state.
        cell_info: list[tuple[int, np.ndarray]] = []
        for r in range(table.n_rows):
            for c in range(table.n_cols):
                cell = table.cells[r][c]
                if cell.is_header or (cell.row_span == 0 and cell.col_span == 0):
                    continue
                key = cell.value
                if key in hidden_states:
                    cell_info.append((r, hidden_states[key][layer]))

        if len(cell_info) < 2:
            continue

        n_pairs = min(max_pairs_per_table, len(cell_info) * (len(cell_info) - 1) // 2)
        for _ in range(n_pairs):
            i, j = rng.choice(len(cell_info), size=2, replace=False)
            r_i, v_i = cell_info[i]
            r_j, v_j = cell_info[j]
            diff = np.abs(v_i - v_j)
            label = 1 if r_i == r_j else 0
            X_list.append(diff)
            y_list.append(label)

    if not X_list:
        return ProbeDataset(
            X=np.empty((0, 0)),
            y=np.empty(0, dtype=int),
            label_names={0: "different_row", 1: "same_row"},
            task_name="same_row",
        )

    return ProbeDataset(
        X=np.stack(X_list),
        y=np.array(y_list, dtype=int),
        label_names={0: "different_row", 1: "same_row"},
        task_name="same_row",
    )
