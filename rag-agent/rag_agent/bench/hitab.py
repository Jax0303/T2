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


def _parse_coord(coord: str) -> Optional[tuple]:
    """``"(17, 1)"`` -> ``(17, 1)`` (full-grid row, col)."""
    nums = re.findall(r"\d+", str(coord))
    return (int(nums[0]), int(nums[1])) if len(nums) >= 2 else None


def _coord_offset(ot: OriginalTable, coords: List[tuple],
                  max_off: int = 6) -> Optional[tuple]:
    """Find the single (H, W) mapping every full-grid coord to a data cell whose
    value matches. ``linked_cells`` index the *full merged grid* (header block
    included); the data matrix drops it, so data[r][c] = full[r+H][c+W] where
    (H, W) is the header block's (rows, cols). The joint value constraint over
    all of a query's operands pins (H, W) down (measured: a consistent offset
    exists for 98.9% of dev tables)."""
    if not coords:
        return None
    for H in range(max_off):
        for W in range(max_off):
            ok = True
            for i, j, fv in coords:
                r, c = i - H, j - W
                if not (0 <= r < ot.n_rows and 0 <= c < ot.n_cols
                        and _to_float(ot.data[r][c]) == fv):
                    ok = False
                    break
            if ok:
                return H, W
    return None


def _operand_at(ot: OriginalTable, r: int, c: int, fv: float) -> GoldOperand:
    return GoldOperand(
        row=r, col=c,
        header_path=ot.full_path(r, c) if hasattr(ot, "full_path")
        else [p for p in ot.row_path(r) if p] + [p for p in ot.col_path(c) if p],
        value=fv, value_type="number",
    )


def resolve_gold_operands(ot: OriginalTable, linked_cells: dict) -> List[GoldOperand]:
    """Resolve ``quantity_link`` cells to data-space operands.

    Primary path: map the annotation's full-grid coordinates to the data matrix
    by the table's header-block offset (the *true* annotated cell). Fall back to
    value-matching only when no consistent offset is found (~1% of tables)."""
    ql = (linked_cells or {}).get("quantity_link") or {}
    coords: List[tuple] = []
    for bucket in ql.values():
        if not isinstance(bucket, dict):
            continue
        for coord, val in bucket.items():
            fv = _to_float(val)
            ij = _parse_coord(coord)
            if fv is not None and ij is not None:
                coords.append((ij[0], ij[1], fv))

    off = _coord_offset(ot, coords)
    ops: List[GoldOperand] = []
    seen = set()
    if off is not None:
        H, W = off
        for i, j, fv in coords:
            r, c = i - H, j - W
            if (r, c) in seen:
                continue
            seen.add((r, c))
            ops.append(_operand_at(ot, r, c, fv))
        return ops

    # Fallback: value-matching with entity_link header tie-break.
    row_hdrs, col_hdrs = _entity_headers((linked_cells or {}).get("entity_link") or {})
    vidx = _value_index(ot)
    for _i, _j, fv in coords:
        cands = vidx.get(round(fv, 4))
        if not cands:
            continue
        r, c = _pick(cands, ot, row_hdrs, col_hdrs)
        if (r, c) in seen:
            continue
        seen.add((r, c))
        ops.append(_operand_at(ot, r, c, fv))
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
