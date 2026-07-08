# SPDX-License-Identifier: MIT
"""FinQA → unified schema adapter (financial numeric reasoning).

Loaded from ``dreamerdeo/finqa`` (parquet branch). A FinQA ``table`` is a flat
grid where ``table[0]`` is the header row and each data row's first cell is a row
label — so a column's header path is ``[header]`` and a row's path is
``[row_label]``.

Gold operands are recovered from ``gold_evidence``, whose clauses have the form
``the {row} of {col} is {value}``. Each parsed (row, col) is matched back to a
data cell; clauses that don't resolve are dropped (logged by the caller).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from .schema import BenchTable, BenchQuery, GoldOperand

SOURCE = "finqa"
_PARQUET_REV = "refs/convert/parquet"
# "the american express of total volume ( billions ) is 647 ;"
_EVID_RE = re.compile(r"the\s+(.+?)\s+of\s+(.+?)\s+is\s+(.+?)\s*(?:;|$)")


def _to_num(v) -> Optional[float]:
    s = str(v).replace(",", "").replace("$", "").replace("%", "").strip()
    m = re.search(r"-?\d+\.?\d*", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _bench_table(table: List[list], tid: str) -> BenchTable:
    header = table[0] if table else []
    data = table[1:] if len(table) > 1 else []
    return BenchTable(
        table_id=tid,
        title="",
        data=data,
        top_paths=[[str(h)] for h in header],
        left_paths=[[str(row[0])] if row else [] for row in data],
        source=SOURCE,
    )


def _resolve_operands(table: List[list], bt: BenchTable, gold_evidence) -> List[GoldOperand]:
    if not table or len(table) < 2:
        return []
    header = [_norm(h) for h in table[0]]
    data = table[1:]
    row_labels = [_norm(row[0]) if row else "" for row in data]
    text = " ".join(gold_evidence) if isinstance(gold_evidence, list) else str(gold_evidence or "")
    ops: List[GoldOperand] = []
    seen = set()
    for rname, cname, val in _EVID_RE.findall(text):
        rn, cn = _norm(rname), _norm(cname)
        r = next((i for i, lbl in enumerate(row_labels) if lbl and (lbl == rn or lbl in rn or rn in lbl)), None)
        c = next((j for j, h in enumerate(header) if h and (h == cn or h in cn or cn in h)), None)
        if r is None or c is None or (r, c) in seen:
            continue
        seen.add((r, c))
        fv = _to_num(val)
        ops.append(GoldOperand(
            row=r, col=c,
            header_path=bt.full_path(r, c),   # row_label > col_header (matches candidate_paths)
            value=fv,
            value_type="number" if fv is not None else "string",
        ))
    return ops


def load_queries(
    split: str = "validation",
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> tuple:
    """Return ``(queries, tables)`` from the FinQA parquet export."""
    from datasets import load_dataset

    sl = f"{split}[:{max_samples}]" if max_samples else split
    ds = load_dataset("dreamerdeo/finqa", revision=_PARQUET_REV, split=sl,
                      cache_dir=cache_dir)
    tables: Dict[str, BenchTable] = {}
    queries: List[BenchQuery] = []
    for i, ex in enumerate(ds):
        tid = ex.get("id") or f"finqa_{split}_{i}"
        table = ex.get("table") or []
        if tid not in tables:
            tables[tid] = _bench_table(table, tid)
        bt = tables[tid]
        ans = _to_num(ex.get("answer"))
        queries.append(BenchQuery(
            query_id=tid,
            question=ex.get("question", ""),
            gold_table_id=tid,
            answer=[ans] if ans is not None else [ex.get("answer")],
            gold_operands=_resolve_operands(table, bt, ex.get("gold_evidence")),
            aggregation="finqa_program",
            split=split,
            source=SOURCE,
        ))
    return queries, tables
