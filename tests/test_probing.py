# SPDX-License-Identifier: MIT
"""Tests for probing infrastructure using toy hidden states."""

from __future__ import annotations

import numpy as np
import pytest

from src.io.table_schema import Cell, HeaderNode, Table
from src.probing.probe_classifier import train_probe
from src.probing.probe_tasks import (
    ProbeDataset,
    build_cell_coord_task,
    build_parent_header_task,
    build_same_row_task,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

HIDDEN_DIM = 32
N_LAYERS = 13  # embedding + 12 layers


def _toy_table() -> Table:
    """4×4 table: 1 header row, 1 header col, 2×3 data region."""
    return Table(
        cells=[
            [
                Cell("", is_header=True),
                Cell("A", is_header=True),
                Cell("B", is_header=True),
                Cell("C", is_header=True),
            ],
            [Cell("R1", is_header=True), Cell("10"), Cell("20"), Cell("30")],
            [Cell("R2", is_header=True), Cell("40"), Cell("50"), Cell("60")],
            [Cell("R3", is_header=True), Cell("70"), Cell("80"), Cell("90")],
        ],
        top_header_tree=HeaderNode(
            name="<TOP>", span_start=-1, span_end=-1,
            children=[
                HeaderNode(name="A", span_start=1, span_end=1),
                HeaderNode(name="B", span_start=2, span_end=2),
                HeaderNode(name="C", span_start=3, span_end=3),
            ],
        ),
        left_header_tree=HeaderNode(
            name="<LEFT>", span_start=-1, span_end=-1,
            children=[
                HeaderNode(name="R1", span_start=1, span_end=1),
                HeaderNode(name="R2", span_start=2, span_end=2),
                HeaderNode(name="R3", span_start=3, span_end=3),
            ],
        ),
        metadata={"top_header_rows_num": 1, "left_header_columns_num": 1},
    )


def _toy_hidden_states(seed: int = 42) -> dict[str, np.ndarray]:
    """Create deterministic toy hidden states for each unique cell value.

    Returns:
        Mapping of cell-value → array of shape (N_LAYERS, HIDDEN_DIM).
    """
    rng = np.random.default_rng(seed)
    values = [
        "", "A", "B", "C",
        "R1", "R2", "R3",
        "10", "20", "30",
        "40", "50", "60",
        "70", "80", "90",
    ]
    return {v: rng.standard_normal((N_LAYERS, HIDDEN_DIM)).astype(np.float32) for v in values}


# ---------------------------------------------------------------------------
# Task builders
# ---------------------------------------------------------------------------

class TestBuildParentHeaderTask:
    def test_produces_samples(self) -> None:
        ds = build_parent_header_task(
            [_toy_table()], _toy_hidden_states(), layer=6,
        )
        assert ds.task_name == "parent_header"
        assert ds.X.shape[0] > 0
        assert ds.X.shape[1] == HIDDEN_DIM
        assert len(ds.y) == ds.X.shape[0]

    def test_empty_on_no_tree(self) -> None:
        t = _toy_table()
        t.top_header_tree = HeaderNode(name="<ROOT>", span_start=-1, span_end=-1)
        ds = build_parent_header_task([t], _toy_hidden_states(), layer=6)
        assert ds.X.shape[0] == 0


class TestBuildCellCoordTask:
    def test_produces_samples(self) -> None:
        ds = build_cell_coord_task(
            [_toy_table()], _toy_hidden_states(), layer=6,
        )
        assert ds.task_name == "cell_coord"
        assert ds.X.shape[0] > 0

    def test_label_range(self) -> None:
        ds = build_cell_coord_task(
            [_toy_table()], _toy_hidden_states(), layer=6,
            n_row_buckets=2, n_col_buckets=2,
        )
        assert ds.y.max() < 4  # 2×2 = 4 classes


class TestBuildSameRowTask:
    def test_produces_pairs(self) -> None:
        ds = build_same_row_task(
            [_toy_table()], _toy_hidden_states(), layer=6,
        )
        assert ds.task_name == "same_row"
        assert ds.X.shape[0] > 0
        assert set(np.unique(ds.y).tolist()).issubset({0, 1})

    def test_balanced_ish(self) -> None:
        ds = build_same_row_task(
            [_toy_table()], _toy_hidden_states(), layer=6,
            max_pairs_per_table=200,
        )
        ratio = ds.y.mean()
        # With 3 rows × 3 cols, ~1/3 pairs are same-row.
        assert 0.05 < ratio < 0.95


# ---------------------------------------------------------------------------
# Probe classifier
# ---------------------------------------------------------------------------

class TestTrainProbe:
    def _make_separable_dataset(self, n: int = 200) -> ProbeDataset:
        """Create a linearly separable toy dataset."""
        rng = np.random.default_rng(42)
        X0 = rng.standard_normal((n // 2, HIDDEN_DIM)).astype(np.float32) - 2.0
        X1 = rng.standard_normal((n // 2, HIDDEN_DIM)).astype(np.float32) + 2.0
        X = np.concatenate([X0, X1])
        y = np.array([0] * (n // 2) + [1] * (n // 2))
        return ProbeDataset(X=X, y=y, label_names={0: "neg", 1: "pos"}, task_name="toy")

    def test_linear_probe_learns(self) -> None:
        ds = self._make_separable_dataset()
        result = train_probe(ds, layer=0, probe_type="linear")
        assert result.accuracy > 0.8
        assert result.selectivity > 0.0

    def test_mlp_probe_learns(self) -> None:
        ds = self._make_separable_dataset()
        result = train_probe(ds, layer=0, probe_type="mlp")
        assert result.accuracy > 0.8

    def test_too_few_samples(self) -> None:
        ds = ProbeDataset(
            X=np.zeros((5, HIDDEN_DIM)),
            y=np.array([0, 1, 0, 1, 0]),
            label_names={0: "a", 1: "b"},
            task_name="tiny",
        )
        result = train_probe(ds, layer=0, probe_type="linear")
        assert result.accuracy == 0.0
        assert result.n_train == 0

    def test_selectivity_control(self) -> None:
        """Control accuracy should be lower than real accuracy."""
        ds = self._make_separable_dataset()
        result = train_probe(ds, layer=0, probe_type="linear")
        assert result.control_accuracy < result.accuracy
