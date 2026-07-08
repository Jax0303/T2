"""A :class:`~rag_agent.serialization.base.TableView` over flat tables.

FinQA and WikiSQL tables have no header hierarchy: a single header row and a
2-D body. ``FlatTable`` exposes them through the same interface the serializers
use for hierarchical HiTab tables, so S1 and S2 run unchanged. For a flat table
``col_path`` is the single column header and ``row_path`` is empty, which makes
S2 degrade gracefully to ``"<column>: <value>"`` per cell.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class FlatTable:
    table_id: str
    columns: List[str]
    rows: List[List[object]]
    title: str = ""
    row_headers: List[str] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        if self.columns:
            return len(self.columns)
        return len(self.rows[0]) if self.rows else 0

    def cell(self, row: int, col: int):
        if 0 <= row < self.n_rows and 0 <= col < len(self.rows[row]):
            return self.rows[row][col]
        return None

    def col_path(self, col: int) -> List[str]:
        if 0 <= col < len(self.columns) and str(self.columns[col]).strip():
            return [str(self.columns[col])]
        return []

    def row_path(self, row: int) -> List[str]:
        if 0 <= row < len(self.row_headers) and str(self.row_headers[row]).strip():
            return [str(self.row_headers[row])]
        return []


def from_finqa(record: dict) -> FlatTable:
    """Build a FlatTable from a FinQA record.

    FinQA stores the table under ``"table"`` as a list of rows where the first
    row is the header and the first column is a row label.
    """
    table = record.get("table") or record.get("table_ori") or []
    table = [list(map(str, r)) for r in table]
    if not table:
        return FlatTable(table_id=str(record.get("id", "finqa")), columns=[], rows=[])
    header, body = table[0], table[1:]
    row_headers = [r[0] if r else "" for r in body]
    return FlatTable(
        table_id=str(record.get("id", "finqa")),
        columns=header,
        rows=body,
        row_headers=row_headers,
    )


def from_wikisql(record: dict) -> FlatTable:
    """Build a FlatTable from a HuggingFace ``wikisql`` record's table field."""
    table = record.get("table") or {}
    return FlatTable(
        table_id=str(table.get("id") or record.get("id", "wikisql")),
        columns=[str(h) for h in table.get("header", [])],
        rows=[list(r) for r in table.get("rows", [])],
        title=str(table.get("caption", "") or ""),
    )
