"""Pandas-backed table store. Preserves header paths alongside the flat 2D cells."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class TableRecord:
    table_id: str
    title: str
    df: pd.DataFrame
    top_header_paths: List[List[str]]
    left_header_paths: List[List[str]]
    raw: dict = field(repr=False)

    def cell(self, row: int, col: int):
        return self.df.iat[row, col]

    def col_header_path(self, col: int) -> List[str]:
        return self.top_header_paths[col] if col < len(self.top_header_paths) else []

    def row_header_path(self, row: int) -> List[str]:
        return self.left_header_paths[row] if row < len(self.left_header_paths) else []

    def find_value(self, value) -> List[Tuple[int, int]]:
        hits: List[Tuple[int, int]] = []
        for r in range(self.df.shape[0]):
            for c in range(self.df.shape[1]):
                v = self.df.iat[r, c]
                if v is None:
                    continue
                if isinstance(value, (int, float)) and isinstance(v, (int, float)):
                    if float(v) == float(value):
                        hits.append((r, c))
                elif str(v).strip().lower() == str(value).strip().lower():
                    hits.append((r, c))
        return hits


def _flatten_header(root: dict) -> List[List[str]]:
    paths: List[List[str]] = []

    def walk(node: dict, prefix: List[str]):
        children = node.get("children_dict") or node.get("children") or []
        name = node.get("name") or node.get("value") or ""
        path = prefix + [str(name)] if name else prefix
        if not children:
            paths.append([p for p in path if p])
            return
        if isinstance(children, dict):
            iter_children = children.values()
        else:
            iter_children = children
        for child in iter_children:
            if isinstance(child, dict):
                walk(child, path)

    walk(root, [])
    # Drop the synthetic root if it produced an empty leading path
    return [p for p in paths if p]


def build_table_record(table: dict) -> TableRecord:
    table_id = table.get("table_id") or table.get("uid") or "unknown"
    title = table.get("title", "")

    top_paths = _flatten_header(table.get("top_root") or {})
    left_paths = _flatten_header(table.get("left_root") or {})

    rows = []
    for row in table.get("data") or []:
        rows.append([cell.get("value") if isinstance(cell, dict) else cell for cell in row])

    df = pd.DataFrame(rows)
    # Use last segment of each top header path as flat column label (kept for display)
    if top_paths and len(top_paths) == df.shape[1]:
        df.columns = [p[-1] if p else f"col_{i}" for i, p in enumerate(top_paths)]
    return TableRecord(
        table_id=table_id,
        title=title,
        df=df,
        top_header_paths=top_paths,
        left_header_paths=left_paths,
        raw=table,
    )


class TableStore:
    """In-memory store of TableRecord keyed by table_id."""

    def __init__(self) -> None:
        self._tables: Dict[str, TableRecord] = {}

    def add(self, table: dict) -> TableRecord:
        rec = build_table_record(table)
        self._tables[rec.table_id] = rec
        return rec

    def get(self, table_id: str) -> Optional[TableRecord]:
        return self._tables.get(table_id)

    def __len__(self) -> int:
        return len(self._tables)

    def ids(self) -> List[str]:
        return list(self._tables.keys())
