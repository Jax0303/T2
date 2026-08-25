# SPDX-License-Identifier: MIT
"""Run provenance: one call that seeds every RNG and describes the environment.

``RESEARCH_STRUCTURE.md`` §6 (재현성 부채) lists seed / device / torch version as
unrecorded in result files. A script calls :func:`run_env` once, right after
parsing args, and splices the returned dict into its result JSON under ``"env"``.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

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

    from .retrieve.hybrid_index import resolve_dense_backend

    return {
        "seed": seed,
        "device": device,
        "torch": torch_version,
        "embed_model": embed_model,
        # which vector backend the dense stage will use; "auto" degrades to
        # numpy when faiss is absent, and this is where that shows up
        "vector_backend": resolve_dense_backend(),
        "run_started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def guard_resume(records_path, env: dict, reader: str | None = None,
                 population: str | None = None, force: bool = False, **extra) -> None:
    """Refuse to append to a records file another run configuration produced.

    ``--resume`` appends to whatever is already on disk. That is right when the
    run is the same one continuing and wrong when anything that decides the
    numbers has changed underneath -- a different reader above all (the repo
    moved from hosted API models to a local one, so every pre-existing records
    file was written by a different model than the default writes now).

    Stores a sidecar next to the records file on first write and compares on
    every later one. ``force`` overrides, for the case where the mismatch is
    understood and wanted.

    ``**extra`` is for whatever else decides the numbers, the cell scheme above
    all: ``embed_model`` names the encoder but not what it was asked to embed,
    so two runs over different cell sentences wrote the same sidecar and one
    resumed off the other's records.
    """
    keys = {"reader": reader, "seed": env.get("seed"),
            "embed_model": env.get("embed_model"), "population": population, **extra}
    side = Path(str(records_path) + ".run.json")
    if not side.exists():
        side.parent.mkdir(parents=True, exist_ok=True)
        side.write_text(json.dumps(keys, ensure_ascii=False, indent=2) + "\n")
        return
    old = json.loads(side.read_text())
    diff = {k: (old.get(k), v) for k, v in keys.items()
            if v is not None and old.get(k) is not None and old.get(k) != v}
    if diff and not force:
        lines = "\n".join(f"  {k}: on disk {o!r} != now {n!r}" for k, (o, n) in diff.items())
        raise RuntimeError(
            f"{records_path} was written under a different configuration:\n{lines}\n"
            f"Mixing them puts two configurations in one file. Use a new --out, "
            f"or pass --force-resume if the mix is intended.")
    if not diff:
        merged = {**old, **{k: v for k, v in keys.items() if v is not None}}
        if merged != old:
            side.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n")
