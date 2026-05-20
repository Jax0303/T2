"""Original-data store.

Keeps HiTab tables in their parsed 2D form (data matrix + header trees).
Exposes the lookups the symbolic-compute path needs:

  - get(table_id) -> OriginalTable
  - OriginalTable.value_at(row_header_path, col_header_path) -> float | str | None
  - OriginalTable.find_rows_by_header(token) / find_cols_by_header(token)
  - OriginalTable.excel_ref_to_rc("B21") -> (row_idx, col_idx) | None

This store is intentionally NOT shared with the vector index; the agent
verifies vector hits against this store rather than reading the serialized
text it already saw during retrieval.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd


_NUM_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+\.?\d*")


def _parse_paths(root: dict) -> Tuple[List[List[str]], Dict[int, List[str]]]:
    """Walk a HiTab top_root / left_root tree.

    Returns (leaf_paths_in_order, line_idx -> path).
    The synthetic root sentinel value (`<TOP>` / `<LEFT>` / `<ROOT>`) is dropped.
    """
    leaf_paths: List[List[str]] = []
    by_line_idx: Dict[int, List[str]] = {}

    def walk(node: dict, prefix: List[str]):
        name = str(node.get("value") or node.get("name") or "").strip()
        path = prefix + [name] if name and name not in ("<TOP>", "<LEFT>", "<ROOT>") else prefix
        children = node.get("children_dict") or node.get("children") or []
        if isinstance(children, dict):
            children = list(children.values())
        if not children:
            leaf_paths.append(path)
            li = node.get("line_idx")
            if li is not None:
                by_line_idx[int(li)] = path
            return
        for ch in children:
            if isinstance(ch, dict):
                walk(ch, path)

    if root:
        walk(root, [])
    return leaf_paths, by_line_idx


def _to_float(v) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "")
    if not s:
        return None
    m = _NUM_RE.fullmatch(s) or _NUM_RE.match(s)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return None
    return None


@dataclass
class OriginalTable:
    """Parsed 2D HiTab table with header trees preserved."""

    table_id: str
    title: str
    data: List[List[object]]        # raw values, rows × cols
    top_paths: List[List[str]]      # column header paths, one per col
    left_paths: List[List[str]]     # row header paths, one per row
    top_paths_by_col: Dict[int, List[str]] = field(default_factory=dict)
    left_paths_by_row: Dict[int, List[str]] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return len(self.data)

    @property
    def n_cols(self) -> int:
        return len(self.data[0]) if self.data else 0

    def cell(self, row: int, col: int):
        if 0 <= row < self.n_rows and 0 <= col < self.n_cols:
            return self.data[row][col]
        return None

    def cell_num(self, row: int, col: int) -> Optional[float]:
        return _to_float(self.cell(row, col))

    def col_path(self, col: int) -> List[str]:
        return self.top_paths_by_col.get(col) or (
            self.top_paths[col] if col < len(self.top_paths) else []
        )

    def row_path(self, row: int) -> List[str]:
        return self.left_paths_by_row.get(row) or (
            self.left_paths[row] if row < len(self.left_paths) else []
        )

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.data:
            rows.append([_to_float(v) if _to_float(v) is not None else v for v in r])
        df = pd.DataFrame(rows)
        # use last-segment headers for display; multi-level kept in self.top_paths
        df.columns = [
            (self.top_paths[c][-1] if c < len(self.top_paths) and self.top_paths[c] else f"col_{c}")
            for c in range(df.shape[1])
        ]
        return df

    # ---- header-path lookup (the inference-time symbolic path uses this) ----

    def find_cols_by_header(self, token: str) -> List[int]:
        """Return col indices whose top path contains ``token`` (case-insensitive substring)."""
        token_l = token.lower().strip()
        hits = []
        for c in range(self.n_cols):
            path = " :: ".join(self.col_path(c)).lower()
            if token_l in path:
                hits.append(c)
        return hits

    def find_rows_by_header(self, token: str) -> List[int]:
        token_l = token.lower().strip()
        hits = []
        for r in range(self.n_rows):
            path = " :: ".join(self.row_path(r)).lower()
            if token_l in path:
                hits.append(r)
        return hits

    def resolve(self, row_header: str, col_header: str) -> Optional[Tuple[int, int, object]]:
        """Resolve a (row_header, col_header) pair to a single cell.

        Both args are matched as case-insensitive substrings against the FULL
        header paths. The most specific match (longest path) wins when ties exist;
        ties broken by appearance order.
        """
        col_cands = self.find_cols_by_header(col_header) if col_header else list(range(self.n_cols))
        row_cands = self.find_rows_by_header(row_header) if row_header else list(range(self.n_rows))
        if not col_cands or not row_cands:
            return None

        # Prefer the most-specific match: longest path string that still contains the token.
        def specificity(p: List[str]) -> int:
            return sum(len(s) for s in p)

        col_cands.sort(key=lambda c: -specificity(self.col_path(c)))
        row_cands.sort(key=lambda r: -specificity(self.row_path(r)))
        r, c = row_cands[0], col_cands[0]
        return (r, c, self.cell(r, c))

    # ---- Excel-style ref support (kept for ad-hoc debugging / gold-supervised eval) ----

    def excel_ref_to_rc(self, ref: str, header_row_offset: int = 0) -> Optional[Tuple[int, int]]:
        """Translate ``B21`` to (row_idx, col_idx).

        HiTab's `reference_cells_map` gives the gold offset; without it, we
        assume Excel row 1 corresponds to the FIRST DATA row by default — the
        caller should pass `header_row_offset` if they know the header depth.
        """
        m = re.fullmatch(r"([A-Za-z]+)(\d+)", ref.strip())
        if not m:
            return None
        col_letters, row_str = m.group(1).upper(), m.group(2)
        col = 0
        for ch in col_letters:
            col = col * 26 + (ord(ch) - ord("A") + 1)
        col -= 1
        row = int(row_str) - 1 - header_row_offset
        if 0 <= row < self.n_rows and 0 <= col < self.n_cols:
            return (row, col)
        return None


def build_original_table(raw: dict) -> OriginalTable:
    table_id = raw.get("table_id") or raw.get("uid") or "unknown"
    title = raw.get("title", "") or ""

    top_paths, top_by_li = _parse_paths(raw.get("top_root") or {})
    left_paths, left_by_li = _parse_paths(raw.get("left_root") or {})

    rows: List[List[object]] = []
    for r in raw.get("data") or []:
        rows.append([cell.get("value") if isinstance(cell, dict) else cell for cell in r])

    n_cols = len(rows[0]) if rows else 0
    n_rows = len(rows)

    # Prefer per-line_idx mapping when present, else align by leaf order.
    top_paths_by_col: Dict[int, List[str]] = {}
    if top_by_li:
        for li, p in top_by_li.items():
            if 0 <= li < n_cols:
                top_paths_by_col[li] = p
    else:
        for i, p in enumerate(top_paths[:n_cols]):
            top_paths_by_col[i] = p

    left_paths_by_row: Dict[int, List[str]] = {}
    if left_by_li:
        for li, p in left_by_li.items():
            if 0 <= li < n_rows:
                left_paths_by_row[li] = p
    else:
        for i, p in enumerate(left_paths[:n_rows]):
            left_paths_by_row[i] = p

    return OriginalTable(
        table_id=table_id,
        title=title,
        data=rows,
        top_paths=[top_paths_by_col.get(c, []) for c in range(n_cols)],
        left_paths=[left_paths_by_row.get(r, []) for r in range(n_rows)],
        top_paths_by_col=top_paths_by_col,
        left_paths_by_row=left_paths_by_row,
    )


class OriginalStore:
    """In-memory store of OriginalTable keyed by table_id."""

    def __init__(self) -> None:
        self._tables: Dict[str, OriginalTable] = {}

    def add(self, raw: dict) -> OriginalTable:
        t = build_original_table(raw)
        self._tables[t.table_id] = t
        return t

    def get(self, table_id: str) -> Optional[OriginalTable]:
        return self._tables.get(table_id)

    def __len__(self) -> int:
        return len(self._tables)

    def ids(self) -> List[str]:
        return list(self._tables.keys())
