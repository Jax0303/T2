#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""W1 — extract the operand gold set for OSC evaluation.

For each HiTab query, resolve its gold operand cells (reusing
``rag_agent.bench.hitab.load_queries``, which already value-matches
``linked_cells.quantity_link`` into data space) and tag every query with the two
stratification variables the spec needs:

  * scope size  m = number of distinct gold operand cells
  * header depth d = max header-path length over the table's top/left trees

Emits one JSON record per query to ``results/operand_gold.jsonl`` and prints an
integrity report (orphan / missing-operand counts) so downstream OSC numbers are
interpreted against a known-clean gold set.

Run:
    python3 scripts/build_operand_gold.py --split dev
    python3 scripts/build_operand_gold.py --split dev --max 200   # quick check
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from rag_agent.bench.hitab import load_queries
from rag_agent.eval.operand_set import bin_depth, bin_scope, header_depth


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max", type=int, default=None, help="cap #queries (debug)")
    ap.add_argument("--out", default="results/operand_gold.jsonl")
    args = ap.parse_args()

    queries, tables = load_queries(args.data_dir, args.split, args.max)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_total = len(queries)
    n_empty = 0          # queries with zero resolved operands (orphans)
    n_no_table = 0       # gold table missing from the loaded set
    depth_hist: Counter = Counter()
    scope_hist: Counter = Counter()
    agg_hist: Counter = Counter()

    written = 0
    with open(args.out, "w") as fh:
        for q in queries:
            tbl = tables.get(q.gold_table_id)
            if tbl is None:
                n_no_table += 1
                continue
            d = header_depth(tbl.top_paths, tbl.left_paths)
            operands = [
                {"row": op.row, "col": op.col,
                 "header_path": op.header_path, "value": op.value}
                for op in q.gold_operands
            ]
            m = len({(op["row"], op["col"]) for op in operands})
            if m == 0:
                n_empty += 1
            depth_hist[bin_depth(d)] += 1
            scope_hist[bin_scope(m)] += 1
            agg_hist[q.aggregation or "none"] += 1
            fh.write(json.dumps({
                "query_id": q.query_id,
                "table_id": q.gold_table_id,
                "question": q.question,
                "answer": q.answer,
                "aggregation": q.aggregation or "none",
                "scope_size": m,
                "header_depth": d,
                "gold_operands": operands,
            }) + "\n")
            written += 1

    report = {
        "split": args.split,
        "n_queries_loaded": n_total,
        "n_written": written,
        "n_gold_table_missing": n_no_table,
        "n_empty_operand_set": n_empty,
        "empty_operand_rate": round(n_empty / written, 4) if written else None,
        "by_depth": dict(sorted(depth_hist.items())),
        "by_scope": dict(sorted(scope_hist.items())),
        "by_aggregation": dict(agg_hist.most_common()),
        "out": args.out,
    }
    rep_path = os.path.splitext(args.out)[0] + "_report.json"
    with open(rep_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwrote {written} records -> {args.out}\nreport -> {rep_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
