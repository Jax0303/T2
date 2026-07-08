# SPDX-License-Identifier: MIT
"""Single entry point for loading any benchmark into the unified schema."""
from __future__ import annotations

from typing import Optional

from . import hitab, finqa, wikisql

BENCHMARKS = ("hitab", "finqa", "wikisql")
_DEFAULT_SPLIT = {"hitab": "dev", "finqa": "validation", "wikisql": "validation"}


def load(
    name: str,
    split: Optional[str] = None,
    max_samples: Optional[int] = None,
    data_dir: str = "data/hitab",
    cache_dir: Optional[str] = "data/hf_cache",
) -> tuple:
    """Return ``(queries, tables)`` for benchmark ``name``.

    ``queries`` is ``List[BenchQuery]`` (with gold operands resolved), ``tables``
    is ``Dict[table_id, BenchTable]``.
    """
    name = name.lower()
    split = split or _DEFAULT_SPLIT.get(name, "validation")
    if name == "hitab":
        return hitab.load_queries(data_dir, split, max_samples)
    if name == "finqa":
        return finqa.load_queries(split, max_samples, cache_dir)
    if name == "wikisql":
        return wikisql.load_queries(split, max_samples, cache_dir)
    raise ValueError(f"unknown benchmark {name!r}; expected one of {BENCHMARKS}")
