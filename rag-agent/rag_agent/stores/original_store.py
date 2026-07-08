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
from difflib import SequenceMatcher
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
        # An internal node can itself own a data row/col (its line_idx), distinct
        # from its children's rows/cols — record it regardless of leaf-ness, or
        # that row/col's header path silently comes back empty.
        li = node.get("line_idx")
        if li is not None:
            by_line_idx[int(li)] = path
        if not children:
            leaf_paths.append(path)
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

    def _match_path(self, query: str, path: list) -> bool:
        """Robust path match.

        ``query`` can be a full path with any separator (' > ', ' :: ', '/'),
        a single leaf, or a substring. Match if EVERY token in the query
        (after splitting on common separators) appears as a **word-bounded**
        substring of the joined actual path. Word boundaries prevent
        "east asia" from matching the row "southeast asia".
        """
        if not path:
            return False
        actual = " :: ".join(str(s) for s in path).lower()
        q = query.lower().strip()
        tokens = [t.strip() for t in re.split(r"\s*(?:::|>|/|\|)\s*", q) if t.strip()]
        if not tokens:
            return False
        for t in tokens:
            # Word-boundary search; falls back to plain substring if the token
            # contains regex-unfriendly characters that re.escape doesn't help.
            pat = r"(?<![A-Za-z0-9])" + re.escape(t) + r"(?![A-Za-z0-9])"
            if not re.search(pat, actual):
                return False
        return True

    def find_cols_by_header(self, token: str) -> List[int]:
        hits = []
        for c in range(self.n_cols):
            if self._match_path(token, self.col_path(c)):
                hits.append(c)
        return hits

    def find_rows_by_header(self, token: str) -> List[int]:
        hits = []
        for r in range(self.n_rows):
            if self._match_path(token, self.row_path(r)):
                hits.append(r)
        return hits

    # ---- fuzzy fallbacks (used only when exact word-boundary match fails) ----

    @staticmethod
    def _norm_tokens(s: str) -> List[str]:
        return [t for t in re.split(r"[^A-Za-z0-9]+", str(s).lower()) if t]

    def _fuzzy_score(self, query: str, path: List[str]) -> float:
        """Score how well ``query`` matches ``path``.

        Combines (a) token overlap and (b) string similarity of the joined
        path. Returns 0.0 when there is no shared content token, which keeps
        unrelated headers from accidentally winning.
        """
        if not path:
            return 0.0
        q_toks = set(self._norm_tokens(query))
        if not q_toks:
            return 0.0
        joined = " ".join(self._norm_tokens(" ".join(path)))
        p_toks = set(joined.split())
        if not p_toks:
            return 0.0
        overlap = q_toks & p_toks
        if not overlap:
            return 0.0
        # Token-overlap component: recall of query tokens that show up in path.
        recall = len(overlap) / len(q_toks)
        # Similarity component: catches close-but-not-exact spellings.
        ratio = SequenceMatcher(None, " ".join(sorted(q_toks)), joined).ratio()
        return 0.7 * recall + 0.3 * ratio

    def _fuzzy_find_rows(self, query: str, threshold: float = 0.4) -> List[int]:
        scored = [(self._fuzzy_score(query, self.row_path(r)), r) for r in range(self.n_rows)]
        scored = [(s, r) for s, r in scored if s >= threshold]
        scored.sort(key=lambda sr: -sr[0])
        return [r for _, r in scored]

    def _fuzzy_find_cols(self, query: str, threshold: float = 0.4) -> List[int]:
        scored = [(self._fuzzy_score(query, self.col_path(c)), c) for c in range(self.n_cols)]
        scored = [(s, c) for s, c in scored if s >= threshold]
        scored.sort(key=lambda sc: -sc[0])
        return [c for _, c in scored]

    def resolve(self, row_header: str, col_header: str) -> Optional[Tuple[int, int, object]]:
        """Resolve a (row_header, col_header) pair to a single cell.

        Exact (word-bounded) match is tried first; if either axis returns no
        candidates, a token/similarity-based fuzzy fallback is attempted.
        For exact matches we tie-break by path specificity (longest wins);
        for fuzzy results we preserve fuzzy-score order so that the
        best-scoring header is not overridden by an unrelated longer one.
        """
        col_cands = self.find_cols_by_header(col_header) if col_header else list(range(self.n_cols))
        row_cands = self.find_rows_by_header(row_header) if row_header else list(range(self.n_rows))
        col_fuzzy = False
        row_fuzzy = False

        if col_header and not col_cands:
            col_cands = self._fuzzy_find_cols(col_header)
            col_fuzzy = True
        if row_header and not row_cands:
            row_cands = self._fuzzy_find_rows(row_header)
            row_fuzzy = True

        if not col_cands or not row_cands:
            return None

        def specificity(p: List[str]) -> int:
            return sum(len(s) for s in p)

        if not col_fuzzy:
            col_cands.sort(key=lambda c: -specificity(self.col_path(c)))
        if not row_fuzzy:
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
