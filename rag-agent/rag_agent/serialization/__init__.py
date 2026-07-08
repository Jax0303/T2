"""Table serialization schemes for the operand-targeted Table-RAG pipeline.

Two schemes, sharing the :class:`Chunk` envelope so they can be ablated
single-variable:

* **S1** (:mod:`rag_agent.serialization.flat`) — flat markdown rows, the
  structure-agnostic baseline.
* **S2** (:mod:`rag_agent.serialization.header_path`) — every cell prefixed
  with its hierarchical header path (``Revenue > 2023 > Q1: 1,234``).

Use :func:`serialize` to pick a scheme by name::

    from rag_agent.serialization import serialize
    chunks = serialize(table, scheme="S2")
"""
from __future__ import annotations

from typing import List

from . import flat, header_path
from .base import Chunk, TableView, fmt_value, join_path, leaf
from .flat_table import FlatTable, from_finqa, from_wikisql

SCHEMES = ("S1", "S2")


def serialize(table: TableView, scheme: str = "S2", **kwargs) -> List[Chunk]:
    """Serialize ``table`` under ``scheme`` (``"S1"`` or ``"S2"``).

    Extra keyword arguments are forwarded to the underlying serializer
    (e.g. ``granularity="cell"`` for S2, ``include_title=False`` for both).
    """
    if scheme == "S1":
        return flat.serialize(table, **kwargs)
    if scheme == "S2":
        return header_path.serialize(table, **kwargs)
    raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")


def from_hitab_raw(raw: dict):
    """Convenience: parse a raw HiTab table dict into an OriginalTable view.

    Imported lazily so the serialization package stays usable (e.g. on flat
    FinQA/WikiSQL tables) without pulling in pandas via the HiTab store.
    """
    from rag_agent.stores.original_store import build_original_table

    return build_original_table(raw)


__all__ = [
    "Chunk",
    "TableView",
    "FlatTable",
    "SCHEMES",
    "serialize",
    "from_hitab_raw",
    "from_finqa",
    "from_wikisql",
    "fmt_value",
    "join_path",
    "leaf",
    "flat",
    "header_path",
]
