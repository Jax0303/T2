# SPDX-License-Identifier: MIT
"""Deterministic seeding for reproducibility."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Fix all relevant random seeds.

    Args:
        seed: Integer seed (default 42).
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002

    # Optional: seed torch if available.
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
