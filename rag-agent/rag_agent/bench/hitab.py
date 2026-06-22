# SPDX-License-Identifier: MIT
"""HiTab → unified schema adapter.

Tables and questions are read from the local HiTab dump (reusing
``rag_agent.data.loader`` and ``build_original_table``). Gold operands are the
answer-bearing data cells listed in ``linked_cells.quantity_link``.

Coordinate caveat (measured): HiTab's ``linked_cells`` coordinates are in the
*original full grid* (header rows included) and do **not** index the data-only
matrix — on 200 dev tables the raw coords landed in range 235/384 times but the
value matched only 1/235. The exact cell *values*, however, are recoverable:
value-matching against the data matrix locates the operand 382/384 (99.5%),
uniquely 277/384 (72%). We therefore resolve each operand by value, breaking ties
with the ``entity_link`` row/column header names the question references.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from ..data.loader import load_samples, load_table
from ..stores.original_store import build_original_table, _to_float, OriginalTable
from .schema import BenchTable, BenchQuery, GoldOperand

SOURCE = "hitab"
_COORD_RE = re.compile(r"\(\s*\d+\s*,\s*\d+\s*\)")


def _bench_table_from_original(ot: OriginalTable) -> BenchTable:
    return BenchTable(
        table_id=ot.table_id,
        title=ot.title,
        data=ot.data,
        top_paths=[ot.col_path(c) for c in range(ot.n_cols)],
        left_paths=[ot.row_path(r) for r in range(ot.n_rows)],
        source=SOURCE,
    )


def _value_index(ot: OriginalTable) -> Dict[float, List[tuple]]:
    idx: Dict[float, List[tuple]] = {}
    for r in range(ot.n_rows):
        for c in range(ot.n_cols):
            fv = _to_float(ot.cell(r, c))
            if fv is not None:
                idx.setdefault(round(fv, 4), []).append((r, c))
    return idx


def _entity_headers(entity_link: dict) -> tuple:
    """Collect the row- and column-header leaf strings the question references."""
    rows, cols = set(), set()
    for axis, bucket in (("left", rows), ("top", cols)):
        for _phrase, cellmap in (entity_link.get(axis) or {}).items():
            for _coord, header in (cellmap or {}).items():
                if header:
                    bucket.add(str(header).strip().lower())
    return rows, cols


def _pick(candidates: List[tuple], ot: OriginalTable,
          row_hdrs: set, col_hdrs: set) -> tuple:
    """Tie-break value-matched candidates by header overlap with entity_link."""
    if len(candidates) == 1:
        return candidates[0]

    def score(rc) -> int:
        r, c = rc
        rl = (ot.row_path(r)[-1] if ot.row_path(r) else "").lower()
        cl = (ot.col_path(c)[-1] if ot.col_path(c) else "").lower()
        s = 0
        if rl and any(rl in h or h in rl for h in row_hdrs):
            s += 1
        if cl and any(cl in h or h in cl for h in col_hdrs):
            s += 1
        return s

    return max(candidates, key=score)


def resolve_gold_operands(ot: OriginalTable, linked_cells: dict) -> List[GoldOperand]:
    """Resolve ``quantity_link`` cells to data-space operands by value-matching."""
    ql = (linked_cells or {}).get("quantity_link") or {}
    row_hdrs, col_hdrs = _entity_headers((linked_cells or {}).get("entity_link") or {})
    vidx = _value_index(ot)
    ops: List[GoldOperand] = []
    seen = set()
    for _coord, val in (
        (coord, v)
        for bucket in ql.values() if isinstance(bucket, dict)
        for coord, v in bucket.items()
    ):
        fv = _to_float(val)
        if fv is None:
            continue
        cands = vidx.get(round(fv, 4))
        if not cands:
            continue
        r, c = _pick(cands, ot, row_hdrs, col_hdrs)
        if (r, c) in seen:
            continue
        seen.add((r, c))
        ops.append(GoldOperand(
            row=r, col=c,
            header_path=ot.full_path(r, c) if hasattr(ot, "full_path")
            else [p for p in ot.row_path(r) if p] + [p for p in ot.col_path(c) if p],
            value=fv, value_type="number",
        ))
    return ops


def load_tables(data_dir: str = "data/hitab", table_ids: Optional[set] = None) -> Dict[str, BenchTable]:
    """Load the requested tables (or all referenced by ``table_ids``) as BenchTables."""
    out: Dict[str, BenchTable] = {}
    if table_ids is None:
        return out
    for tid in table_ids:
        raw = load_table(tid, data_dir)
        if raw is None:
            continue
        out[tid] = _bench_table_from_original(build_original_table(raw))
    return out


def load_queries(
    data_dir: str = "data/hitab",
    split: str = "dev",
    max_samples: Optional[int] = None,
) -> tuple:
    """Return ``(queries, tables)`` for the split, with gold operands resolved."""
    samples = load_samples(data_dir, split, max_samples)
    tables: Dict[str, BenchTable] = {}
    ot_cache: Dict[str, OriginalTable] = {}
    queries: List[BenchQuery] = []
    for s in samples:
        tid = s.get("table_id")
        if tid not in ot_cache:
            raw = load_table(tid, data_dir)
            if raw is None:
                continue
            ot = build_original_table(raw)
            ot_cache[tid] = ot
            tables[tid] = _bench_table_from_original(ot)
        ot = ot_cache[tid]
        agg = s.get("aggregation")
        queries.append(BenchQuery(
            query_id=s.get("id"),
            question=s.get("question", ""),
            gold_table_id=tid,
            answer=s.get("answer", []),
            gold_operands=resolve_gold_operands(ot, s.get("linked_cells") or {}),
            aggregation=agg[0] if isinstance(agg, list) and agg else agg,
            split=split,
            source=SOURCE,
        ))
    return queries, tables
