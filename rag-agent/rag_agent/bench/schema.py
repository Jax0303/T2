# SPDX-License-Identifier: MIT
"""Unified table/query/operand schema shared by all benchmarks.

A benchmark loader's only job is to fill these dataclasses; nothing downstream
knows which benchmark a table came from. Header structure is represented
uniformly as *per-row* and *per-column* header paths:

  * ``top_paths[c]``  — the column header path of column ``c`` (e.g.
    ``["current $millions", "2014"]``). For a flat table this is ``[col_name]``.
  * ``left_paths[r]`` — the row header path of row ``r`` (e.g.
    ``["assets", "cash"]``). For a flat table this is ``[]`` (or a stub).

A :class:`GoldOperand` is a single data cell the gold answer depends on, tagged
with its full header path ``left_paths[r] + top_paths[c]`` — this is the unit
``operand_recall`` and ``header_path_match_accuracy`` are measured against.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class GoldOperand:
    """One answer-bearing data cell, identified by coordinate and header path."""

    row: int
    col: int
    header_path: List[str]          # left_paths[row] + top_paths[col]
    value: Optional[float] = None   # numeric value if parseable
    value_type: str = "number"      # "number" | "string"

    def path_str(self) -> str:
        return " > ".join(self.header_path)


@dataclass
class BenchTable:
    """A table in the unified representation."""

    table_id: str
    title: str
    data: List[List[Any]]                       # data[row][col] raw cell values
    top_paths: List[List[str]] = field(default_factory=list)   # len == n_cols
    left_paths: List[List[str]] = field(default_factory=list)  # len == n_rows
    source: str = ""                            # benchmark name

    @property
    def n_rows(self) -> int:
        return len(self.data)

    @property
    def n_cols(self) -> int:
        return len(self.data[0]) if self.data else 0

    def col_path(self, col: int) -> List[str]:
        return self.top_paths[col] if 0 <= col < len(self.top_paths) else []

    def row_path(self, row: int) -> List[str]:
        return self.left_paths[row] if 0 <= row < len(self.left_paths) else []

    def cell(self, row: int, col: int) -> Any:
        if 0 <= row < self.n_rows and 0 <= col < self.n_cols:
            return self.data[row][col]
        return None

    def full_path(self, row: int, col: int) -> List[str]:
        """Header path identifying cell (row, col): left path then col path."""
        return [p for p in self.row_path(row) if p] + [p for p in self.col_path(col) if p]


@dataclass
class BenchQuery:
    """A natural-language question with its gold table and gold operands."""

    query_id: str
    question: str
    gold_table_id: str
    answer: List[Any]
    gold_operands: List[GoldOperand] = field(default_factory=list)
    aggregation: Optional[str] = None
    split: str = ""
    source: str = ""


@dataclass
class Chunk:
    """A serialized retrieval unit and the data cells it covers.

    ``rows``/``cols`` record which data cells the chunk's text contains, so a
    retrieved chunk can be checked against gold operands: an operand ``(r, c)``
    is *covered* by a chunk iff ``r in chunk.rows and c in chunk.cols``.
    """

    table_id: str
    chunk_id: str
    text: str
    rows: List[int] = field(default_factory=list)
    cols: List[int] = field(default_factory=list)

    def covers(self, row: int, col: int) -> bool:
        return row in self.rows and col in self.cols
