# SPDX-License-Identifier: MIT
"""Run provenance: one call that seeds every RNG and describes the environment.

``RESEARCH_STRUCTURE.md`` §6 (재현성 부채) lists seed / device / torch version as
unrecorded in result files. A script calls :func:`run_env` once, right after
parsing args, and splices the returned dict into its result JSON under ``"env"``.
"""
from __future__ import annotations

import random
from datetime import datetime, timezone

import numpy as np


def run_env(seed: int, embed_model: str) -> dict:
    """Seed ``random`` / ``numpy`` / ``torch`` and return the result-JSON ``env`` block.

    ``torch`` is optional here for the same reason it is optional in
    :mod:`rag_agent.retrieve.encoders` (CPU containers, CI): when it is absent
    ``torch`` is ``None`` and the device is ``"cpu"``.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch_version = None
    device = "cpu"
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch_version = torch.__version__
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)

    return {
        "seed": seed,
        "device": device,
        "torch": torch_version,
        "embed_model": embed_model,
        "run_started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
