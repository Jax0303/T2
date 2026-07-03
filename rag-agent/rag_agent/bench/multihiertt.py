# SPDX-License-Identifier: MIT
"""MultiHiertt → unified schema adapter (2nd hierarchical-table OSC dataset).

MultiHiertt (Zhao et al., ACL 2022) documents hold several **hierarchical**
finance tables (HTML with colspan group headers + section rows) plus text.
``qa.table_evidence`` lists the supporting cells as ``"{table}-{row}-{col}"``
coordinates — a direct gold **operand set**, which HiTab aside no other
hierarchical benchmark provides. That makes OSC computable on a second dataset.

Scope taken here (the subset our per-table pipeline addresses):
  * ``question_type == "arithmetic"`` (programs over numeric operands),
  * no text evidence (operands come from tables only),
  * all evidence cells inside **one** table (multi-table retrieval is an
    orthogonal axis we do not model),
  * >=2 distinct evidence cells (aggregation scope, mirrors HiTab m>=2),
  * every evidence cell **value-validated** against ``table_description``
    (the dataset's own cell rendering) — queries with any mismatch are dropped,
    so gold fidelity is audited, not assumed.

Parsing: leading rows containing ``colspan`` cells are column group headers;
the first span-free row closes the header block. A data row whose non-label
cells are all empty is a **section header**; following rows get
``[section, label]`` as their left path (2-level row hierarchy).
"""
from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .schema import BenchTable, BenchQuery, GoldOperand

SOURCE = "multihiertt"

_OP_MAP = {"add": "sum", "subtract": "diff", "divide": "div",
           "multiply": "mult", "exp": "exp", "greater": "comp"}
_DESC_VAL_RE = re.compile(r"\bis\s+(.+?)\s*\.?\s*$")


def _to_num(v) -> Optional[float]:
    s = str(v).replace(",", "").replace("$", "").replace("%", "").strip()
    s = s.replace("(", "-").replace(")", "")  # accounting negatives
    m = re.search(r"-?\d+\.?\d*", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


class _TableHTMLParser(HTMLParser):
    """<table> HTML → grid with colspan/rowspan expanded; tracks span rows."""

    def __init__(self):
        super().__init__()
        self.rows: List[List[str]] = []
        self.row_had_colspan: List[bool] = []
        self._cur: Optional[List[str]] = None
        self._cell: Optional[List[str]] = None
        self._colspan = 1
        self._rowspan = 1
        self._had_colspan = False
        # pending rowspans: col index -> [remaining_rows, value]
        self._pending: Dict[int, List] = {}

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "tr":
            self._cur = []
            self._had_colspan = False
        elif tag in ("td", "th") and self._cur is not None:
            # fill any pending rowspan cells that land at the current position
            while len(self._cur) in self._pending:
                c = len(self._cur)
                self._cur.append(self._pending[c][1])
                self._pending[c][0] -= 1
                if self._pending[c][0] <= 0:
                    del self._pending[c]
            self._cell = []
            self._colspan = int(a.get("colspan", 1) or 1)
            self._rowspan = int(a.get("rowspan", 1) or 1)
            if self._colspan > 1:
                self._had_colspan = True

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None and self._cur is not None:
            val = re.sub(r"\s+", " ", "".join(self._cell)).strip()
            start = len(self._cur)
            for i in range(self._colspan):
                self._cur.append(val)
                if self._rowspan > 1:
                    self._pending[start + i] = [self._rowspan - 1, val]
            self._cell = None
        elif tag == "tr" and self._cur is not None:
            while len(self._cur) in self._pending:
                c = len(self._cur)
                self._cur.append(self._pending[c][1])
                self._pending[c][0] -= 1
                if self._pending[c][0] <= 0:
                    del self._pending[c]
            self.rows.append(self._cur)
            self.row_had_colspan.append(self._had_colspan)
            self._cur = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def parse_table_html(html: str) -> Tuple[List[List[str]], int]:
    """Return ``(grid, n_header_rows)`` for one MultiHiertt HTML table."""
    p = _TableHTMLParser()
    p.feed(html)
    rows = p.rows
    if not rows:
        return [], 0
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    # header block: row 0, plus group-header rows -> closed by first span-free row
    n_header = 1
    if p.row_had_colspan and p.row_had_colspan[0]:
        i = 1
        while i < len(rows) and p.row_had_colspan[i]:
            i += 1
        n_header = min(i + 1, len(rows))  # include the leaf-header row after groups
    return rows, n_header


def _bench_table(grid: List[List[str]], n_header: int, tid: str) -> BenchTable:
    header_rows = grid[:n_header]
    data = grid[n_header:]
    n_cols = len(grid[0]) if grid else 0

    top_paths: List[List[str]] = []
    for c in range(n_cols):
        segs: List[str] = []
        for hr in header_rows:
            v = hr[c]
            if v and (not segs or segs[-1] != v):
                segs.append(v)
        top_paths.append(segs)

    left_paths: List[List[str]] = []
    section = ""
    for row in data:
        label = row[0]
        is_section = bool(label) and all(not v for v in row[1:])
        if is_section:
            section = label
            left_paths.append([label])
        else:
            left_paths.append([section, label] if section and label
                              else ([label] if label else []))
    # normalize value cells ("$1,350", "(451)") to floats so the shared
    # OriginalTable/_to_float numeric parsing (tuned for HiTab's raw numbers)
    # sees them; col 0 stays raw (row labels)
    norm = []
    for row in data:
        nr = list(row)
        for c in range(1, len(nr)):
            v = _to_num(nr[c])
            if v is not None:
                nr[c] = v
        norm.append(nr)
    return BenchTable(table_id=tid, title="", data=norm,
                      top_paths=top_paths, left_paths=left_paths, source=SOURCE)


def _desc_value(desc: str) -> Optional[float]:
    m = _DESC_VAL_RE.search(desc or "")
    return _to_num(m.group(1)) if m else None


def _values_match(a: Optional[float], b: Optional[float]) -> bool:
    if a is None or b is None:
        return False
    if a == b:
        return True
    return abs(a - b) <= 1e-3 * max(abs(a), abs(b))


def load_queries(data_dir: str = "data/multihiertt", split: str = "dev",
                 max_samples: Optional[int] = None) -> tuple:
    """Return ``(queries, tables, stats)`` for the single-table arithmetic subset.

    ``stats`` reports the population funnel and the evidence-cell validation
    rate so gold fidelity is part of the artifact.
    """
    samples = json.loads(Path(data_dir, f"{split}.json").read_text())
    if max_samples:
        samples = samples[:max_samples]

    tables: Dict[str, BenchTable] = {}
    queries: List[BenchQuery] = []
    stats = dict(total=len(samples), arithmetic=0, table_only=0, single_table=0,
                 m2=0, validated=0, cells_checked=0, cells_matched=0)

    for s in samples:
        qa = s["qa"]
        if qa.get("question_type") != "arithmetic":
            continue
        stats["arithmetic"] += 1
        if qa.get("text_evidence"):
            continue
        stats["table_only"] += 1
        evid = sorted(set(qa.get("table_evidence") or []))
        tids = {e.split("-")[0] for e in evid}
        if len(tids) != 1:
            continue
        stats["single_table"] += 1
        if len(evid) < 2:
            continue
        stats["m2"] += 1

        ti = int(next(iter(tids)))
        tid = f"{s['uid']}__t{ti}"
        if tid not in tables:
            grid, n_header = parse_table_html(s["tables"][ti])
            if not grid:
                continue
            tables[tid] = _bench_table(grid, n_header, tid)
        bt = tables[tid]

        # resolve + validate gold operands against table_description
        ops: List[GoldOperand] = []
        ok = True
        grid_header_offset = None
        for e in evid:
            _, r, c = (int(x) for x in e.split("-"))
            # evidence rows count header rows; recover offset from bt
            if grid_header_offset is None:
                grid_header_offset = _header_offset(s["tables"][ti])
            dr = r - grid_header_offset
            cell_v = _to_num(bt.cell(dr, c)) if dr >= 0 else None
            desc_v = _desc_value(s["table_description"].get(e, ""))
            stats["cells_checked"] += 1
            if not _values_match(cell_v, desc_v):
                ok = False
                continue
            stats["cells_matched"] += 1
            ops.append(GoldOperand(row=dr, col=c,
                                   header_path=bt.full_path(dr, c), value=cell_v))
        if not ok or len({(o.row, o.col) for o in ops}) < 2:
            continue
        stats["validated"] += 1

        prog_ops = re.findall(r"(\w+)\(", qa.get("program") or "")
        agg = _OP_MAP.get(prog_ops[0], prog_ops[0]) if prog_ops else "arith"
        queries.append(BenchQuery(
            query_id=s["uid"], question=qa.get("question", ""),
            gold_table_id=tid, answer=[qa.get("answer")],
            gold_operands=ops, aggregation=agg, split=split, source=SOURCE))

    used = {q.gold_table_id for q in queries}
    tables = {k: v for k, v in tables.items() if k in used}
    return queries, tables, stats


_HEADER_CACHE: Dict[int, int] = {}


def _header_offset(html: str) -> int:
    key = hash(html)
    if key not in _HEADER_CACHE:
        _HEADER_CACHE[key] = parse_table_html(html)[1]
    return _HEADER_CACHE[key]
