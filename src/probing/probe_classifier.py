# SPDX-License-Identifier: MIT
"""Linear and MLP probe classifiers with selectivity control.

Implements the Hewitt & Liang (2019) selectivity metric: the difference
in accuracy between the real-label probe and a control probe trained on
randomly permuted labels.  High selectivity means the probe captures
genuine linguistic structure, not just memorisation capacity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from src.probing.probe_tasks import ProbeDataset


@dataclass
class ProbeResult:
    """Result of a single probe training run."""

    task_name: str
    probe_type: str  # "linear" or "mlp"
    layer: int
    accuracy: float
    control_accuracy: float
    selectivity: float  # accuracy - control_accuracy
    n_train: int
    n_test: int


def _make_probe(
    probe_type: Literal["linear", "mlp"],
    n_classes: int,
    seed: int = 42,
) -> LogisticRegression | MLPClassifier:
    """Instantiate a probe classifier."""
    if probe_type == "linear":
        return LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=seed,
        )
    return MLPClassifier(
        hidden_layer_sizes=(256,),
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=seed,
    )


def train_probe(
    dataset: ProbeDataset,
    layer: int,
    probe_type: Literal["linear", "mlp"] = "linear",
    test_size: float = 0.2,
    seed: int = 42,
) -> ProbeResult:
    """Train a probe and its selectivity control.

    Args:
        dataset: Probe task dataset (X, y).
        layer: Layer index (for metadata only).
        probe_type: ``"linear"`` or ``"mlp"``.
        test_size: Fraction held out for evaluation.
        seed: Random seed.

    Returns:
        A ProbeResult with accuracy, control accuracy, and selectivity.
    """
    if dataset.X.shape[0] < 10:
        return ProbeResult(
            task_name=dataset.task_name,
            probe_type=probe_type,
            layer=layer,
            accuracy=0.0,
            control_accuracy=0.0,
            selectivity=0.0,
            n_train=0,
            n_test=0,
        )

    n_classes = len(set(dataset.y.tolist()))

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X, dataset.y, test_size=test_size, random_state=seed, stratify=dataset.y,
    )

    # ---- Real probe ----
    probe = _make_probe(probe_type, n_classes, seed)
    probe.fit(X_train, y_train)
    accuracy = float(probe.score(X_test, y_test))

    # ---- Control probe (random labels) ----
    rng = np.random.default_rng(seed)
    y_train_rand = rng.permutation(y_train)
    control = _make_probe(probe_type, n_classes, seed)
    control.fit(X_train, y_train_rand)
    control_accuracy = float(control.score(X_test, y_test))

    return ProbeResult(
        task_name=dataset.task_name,
        probe_type=probe_type,
        layer=layer,
        accuracy=accuracy,
        control_accuracy=control_accuracy,
        selectivity=accuracy - control_accuracy,
        n_train=len(y_train),
        n_test=len(y_test),
    )
