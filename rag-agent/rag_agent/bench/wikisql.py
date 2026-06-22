# SPDX-License-Identifier: MIT
"""WikiSQL → unified schema adapter (the flat-table control benchmark).

Loaded from the Hub's parquet branch (the legacy loading script is unsupported
by modern ``datasets``). WikiSQL tables are single-level/flat, so a column's
header path is just ``[header]`` and rows carry no header path — this is exactly
the "simple flat table" control the spec asks for.

The parquet export omits the materialised answer, so we execute the gold SQL
(``sel`` / ``agg`` / ``conds``) against the table. Gold operands are the cells the
query reads: the selected column in each matching row, plus the condition columns
in those rows.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .schema import BenchTable, BenchQuery, GoldOperand

SOURCE = "wikisql"
_AGG = ["", "max", "min", "count", "sum", "avg"]
_OP = ["=", ">", "<"]
_PARQUET_REV = "refs/convert/parquet"


def _to_num(v) -> Optional[float]:
    try:
        return float(str(v).replace(",", "").replace("$", "").strip())
    except (ValueError, AttributeError):
        return None


def _match_rows(rows: List[list], conds: dict) -> List[int]:
    cols = conds.get("column_index", [])
    ops = conds.get("operator_index", [])
    vals = conds.get("condition", [])
    out = []
    for r, row in enumerate(rows):
        ok = True
        for ci, oi, cv in zip(cols, ops, vals):
            if ci >= len(row):
                ok = False
                break
            cell = str(row[ci]).strip().lower()
            cond = str(cv).strip().lower()
            op = _OP[oi] if oi < len(_OP) else "="
            if op == "=":
                ok = cell == cond
            else:
                a, b = _to_num(row[ci]), _to_num(cv)
                ok = a is not None and b is not None and ((a > b) if op == ">" else (a < b))
            if not ok:
                break
        if ok:
            out.append(r)
    return out


def _execute(table: dict, sql: dict) -> Tuple[list, List[int]]:
    """Return (answer_list, matching_row_indices)."""
    rows = table.get("rows", [])
    sel = sql.get("sel", 0)
    agg = sql.get("agg", 0)
    matched = _match_rows(rows, sql.get("conds", {}))
    picked = [rows[r][sel] for r in matched if sel < len(rows[r])]
    if agg == 0:
        return picked, matched
    nums = [n for n in (_to_num(v) for v in picked) if n is not None]
    if agg == 3:                       # count
        return [len(picked)], matched
    if not nums:
        return picked, matched
    res = {1: max, 2: min, 4: sum}.get(agg)
    if res is not None:
        return [res(nums)], matched
    if agg == 5:                       # avg
        return [sum(nums) / len(nums)], matched
    return picked, matched


def _bench_table(table: dict) -> BenchTable:
    header = table.get("header", [])
    rows = table.get("rows", [])
    return BenchTable(
        table_id=table.get("id", ""),
        title=table.get("caption") or table.get("page_title") or "",
        data=rows,
        top_paths=[[h] for h in header],   # flat: one header per column
        left_paths=[[] for _ in rows],     # no row headers
        source=SOURCE,
    )


def _operands(table: dict, sql: dict, matched: List[int]) -> List[GoldOperand]:
    header = table.get("header", [])
    rows = table.get("rows", [])
    sel = sql.get("sel", 0)
    cond_cols = sql.get("conds", {}).get("column_index", [])
    targets = [sel] + list(cond_cols)
    ops: List[GoldOperand] = []
    seen = set()
    for r in matched:
        for c in targets:
            if c >= len(rows[r]) or (r, c) in seen:
                continue
            seen.add((r, c))
            val = _to_num(rows[r][c])
            ops.append(GoldOperand(
                row=r, col=c,
                header_path=[header[c]] if c < len(header) else [],
                value=val,
                value_type="number" if val is not None else "string",
            ))
    return ops


def load_queries(
    split: str = "validation",
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> tuple:
    """Return ``(queries, tables)`` from the WikiSQL parquet export."""
    from datasets import load_dataset

    sl = f"{split}[:{max_samples}]" if max_samples else split
    ds = load_dataset("Salesforce/wikisql", revision=_PARQUET_REV, split=sl,
                      cache_dir=cache_dir)
    tables: Dict[str, BenchTable] = {}
    queries: List[BenchQuery] = []
    for i, ex in enumerate(ds):
        table, sql = ex["table"], ex["sql"]
        tid = table.get("id") or f"wikisql_{split}_{i}"
        table["id"] = tid
        if tid not in tables:
            tables[tid] = _bench_table(table)
        answer, matched = _execute(table, sql)
        queries.append(BenchQuery(
            query_id=f"{split}_{i}",
            question=ex["question"],
            gold_table_id=tid,
            answer=answer,
            gold_operands=_operands(table, sql, matched),
            aggregation=_AGG[sql.get("agg", 0)] or "none",
            split=split,
            source=SOURCE,
        ))
    return queries, tables
