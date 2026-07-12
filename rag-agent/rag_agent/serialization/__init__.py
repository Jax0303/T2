"""Table serialization schemes for the operand-targeted Table-RAG pipeline.

Three schemes, sharing the :class:`Chunk` envelope so they can be ablated
single-variable:

* **S1** (:mod:`rag_agent.serialization.flat`) — flat markdown rows, the
  structure-agnostic baseline.
* **S2** (:mod:`rag_agent.serialization.header_path`) — every cell prefixed
  with its hierarchical header path (``Revenue > 2023 > Q1: 1,234``).
* **S3** (:mod:`rag_agent.serialization.caption`) — every cell rendered as a
  natural-language sentence (``Among Revenue 2023 Q1, the value is 1,234.``),
  with a ``length`` preset (``"short"``/``"medium"``/``"long"``).

Use :func:`serialize` to pick a scheme by name::

    from rag_agent.serialization import serialize
    chunks = serialize(table, scheme="S2")
"""
from __future__ import annotations

from typing import List

from . import caption, flat, header_path
from .base import Chunk, TableView, fmt_value, join_path, leaf
from .flat_table import FlatTable, from_finqa, from_wikisql

SCHEMES = ("S1", "S2", "S3")


def serialize(table: TableView, scheme: str = "S2", **kwargs) -> List[Chunk]:
    """Serialize ``table`` under ``scheme`` (``"S1"``, ``"S2"`` or ``"S3"``).

    Extra keyword arguments are forwarded to the underlying serializer
    (e.g. ``granularity="cell"`` for S2/S3, ``include_title=False`` for all
    three, ``length="short"|"medium"|"long"`` for S3).
    """
    if scheme == "S1":
        return flat.serialize(table, **kwargs)
    if scheme == "S2":
        return header_path.serialize(table, **kwargs)
    if scheme == "S3":
        return caption.serialize(table, **kwargs)
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
    "caption",
]
