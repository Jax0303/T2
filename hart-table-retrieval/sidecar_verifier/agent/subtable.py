"""Query-aware sub-table extractor (route β).

Reuses the verifier's keyword/numeric signals to pinpoint which rows and
columns in a candidate table are relevant to the query. Returns a smaller
DataFrame that can be serialized for the LLM, avoiding the "lost-in-the-middle"
effect of dumping the full table.

Strategy
--------
- Column candidates: any column whose ``top_header_path`` contains at least one
  query keyword (case-insensitive substring on header tokens).
- Row candidates: any row whose ``left_header_path`` contains a query keyword,
  OR any row that contains a cell numerically matching a query number.
- If a side has zero matches, fall back to "all rows" / "all columns" for that
  side — better than emitting an empty sub-table.
- Always include at least the top-1 matched row's neighbours (1 row above and
  below) so the LLM sees local context. Same idea for columns when matched
  count is 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

import pandas as pd

from .verifier import _keywords, _parse_numbers
from ..store.table_store import TableRecord


@dataclass
class SubtableResult:
    df: pd.DataFrame
    rows: List[int]
    cols: List[int]
    matched_keywords: List[str]
    matched_numbers: List[float]
    fallback_rows: bool
    fallback_cols: bool

    @property
    def shape(self):
        return self.df.shape


def _header_text(rec: TableRecord, axis: str, idx: int) -> str:
    if axis == "top":
        path = rec.top_header_paths[idx] if idx < len(rec.top_header_paths) else []
    else:
        path = rec.left_header_paths[idx] if idx < len(rec.left_header_paths) else []
    return " ".join(str(p) for p in path).lower()


def _match_columns(query_kws: Set[str], rec: TableRecord) -> List[int]:
    if not query_kws:
        return []
    cols = []
    for c in range(rec.df.shape[1]):
        text = _header_text(rec, "top", c)
        if any(kw in text for kw in query_kws):
            cols.append(c)
    return cols


def _match_rows_by_header(query_kws: Set[str], rec: TableRecord) -> List[int]:
    if not query_kws:
        return []
    rows = []
    for r in range(rec.df.shape[0]):
        text = _header_text(rec, "left", r)
        if any(kw in text for kw in query_kws):
            rows.append(r)
    return rows


def _match_rows_by_number(query_nums: List[float], rec: TableRecord,
                          cols: List[int]) -> List[int]:
    if not query_nums:
        return []
    targets = set(round(n, 3) for n in query_nums)
    use_cols = cols if cols else range(rec.df.shape[1])
    rows = []
    for r in range(rec.df.shape[0]):
        for c in use_cols:
            v = rec.df.iat[r, c]
            try:
                vn = float(str(v).replace(",", "")) if v is not None else None
            except (ValueError, TypeError):
                vn = None
            if vn is not None and round(vn, 3) in targets:
                rows.append(r)
                break
    return rows


def _expand_neighbours(indices: List[int], max_len: int) -> List[int]:
    """If exactly 1 hit, also include the row above and below for context."""
    if len(indices) != 1:
        return indices
    i = indices[0]
    out = [j for j in (i - 1, i, i + 1) if 0 <= j < max_len]
    return out


def extract_subtable(
    query: str,
    rec: TableRecord,
    *,
    max_rows: int = 12,
    max_cols: int = 10,
) -> SubtableResult:
    q_kws = set(_keywords(query))
    q_nums = _parse_numbers(query)

    cols = _match_columns(q_kws, rec)
    rows_h = _match_rows_by_header(q_kws, rec)
    rows_n = _match_rows_by_number(q_nums, rec, cols)
    rows = sorted(set(rows_h) | set(rows_n))

    fallback_rows = not rows
    fallback_cols = not cols
    if fallback_rows:
        rows = list(range(min(rec.df.shape[0], max_rows)))
    if fallback_cols:
        cols = list(range(min(rec.df.shape[1], max_cols)))

    rows = _expand_neighbours(rows, rec.df.shape[0])
    cols = _expand_neighbours(cols, rec.df.shape[1])
    rows = rows[:max_rows]
    cols = cols[:max_cols]

    # Build sub-DataFrame, preserving header path labels.
    col_names = [" / ".join(rec.col_header_path(c)) or f"col_{c}" for c in cols]
    row_labels = [" / ".join(rec.row_header_path(r)) or f"row_{r}" for r in rows]
    sub = rec.df.iloc[rows, cols].copy()
    sub.columns = col_names
    sub.index = row_labels

    return SubtableResult(
        df=sub,
        rows=rows,
        cols=cols,
        matched_keywords=sorted(q_kws & _set_table_tokens(rec)),
        matched_numbers=q_nums,
        fallback_rows=fallback_rows,
        fallback_cols=fallback_cols,
    )


def _set_table_tokens(rec: TableRecord) -> Set[str]:
    tokens: Set[str] = set()
    tokens.update(_keywords(rec.title))
    for path in rec.top_header_paths + rec.left_header_paths:
        for h in path:
            tokens.update(_keywords(str(h)))
    return tokens


def render_subtable_for_llm(res: SubtableResult, title: str = "") -> str:
    """Render the sub-table as markdown for inclusion in an LLM prompt."""
    head = f"# {title}\n\n" if title else ""
    try:
        body = res.df.to_markdown(index=True)
    except Exception:
        body = res.df.to_string(index=True)
    note = []
    if res.fallback_rows:
        note.append("(no row-level match; showing all rows)")
    if res.fallback_cols:
        note.append("(no column-level match; showing all columns)")
    suffix = ("\n\n" + " ".join(note)) if note else ""
    return head + body + suffix
