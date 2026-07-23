#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""G2 gate (PLAN_reconstruction_accuracy.md) — do reconstruction errors overlap gold operands?

Go/no-go for the "improve header-tree reconstruction" lever. Reconstruction is
only worth fixing if its per-table errors actually land on cells that are gold
operands for some query; otherwise closing col 96.7% / row 88.2% cannot move
set-EM / answer accuracy.

Method (identical error model to tree_reconstruct_hitab.py): render each table's
KNOWN gold header paths into a synthetic blank-after-first grid, run the real
reconstructor, and mark a column/row index as *mis-reconstructed* iff its rebuilt
path != gold path (token-normalised). A gold operand cell (r,c) is *affected* iff
its column c or its row r is mis-reconstructed — i.e. its S2/S3 serialization
would carry a wrong header path.

Reports, for all-with-gold and the m>=2 arithmetic-scope slice:
  * operand-cell level: share of gold operand cells on a mis-reconstructed path
  * QUERY level: share of queries with >=1 affected operand  (set-EM ceiling —
    one wrong-path operand can zero an all-or-nothing query)
  * PLAN's literal framing: share of all mis-reconstructed data cells that are
    gold operands (precision of the error set onto gold)

Run: python3 scripts/g2_reconstruct_gold_overlap.py --data-dir /mnt/d/hart_data/hitab/HiTab
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_agent.bench.hitab import load_queries
from rag_agent.reconstruct import reconstruct_col_paths, reconstruct_row_paths


def flatten_to_grid(top_paths, left_paths):
    """Render known gold header paths into a synthetic blank-after-first grid.
    (copied from tree_reconstruct_hitab.py so the error set matches that report)"""
    n_header_rows = max((len(p) for p in top_paths), default=0)
    n_header_cols = max((len(p) for p in left_paths), default=0)
    n_cols, n_rows = len(top_paths), len(left_paths)
    width = n_header_cols + n_cols
    height = n_header_rows + n_rows
    grid = [["" for _ in range(width)] for _ in range(height)]
    for c, path in enumerate(top_paths):
        for d in range(n_header_rows):
            label = path[d] if d < len(path) else ""
            prev = top_paths[c - 1][d] if c > 0 and d < len(top_paths[c - 1]) else None
            if label and label != prev:
                grid[d][n_header_cols + c] = label
    for r, path in enumerate(left_paths):
        for d in range(n_header_cols):
            label = path[d] if d < len(path) else ""
            prev = left_paths[r - 1][d] if r > 0 and d < len(left_paths[r - 1]) else None
            if label and label != prev:
                grid[n_header_rows + r][d] = label
    for r in range(n_rows):
        for c in range(n_cols):
            grid[n_header_rows + r][n_header_cols + c] = "1"
    return grid, n_header_rows, n_header_cols


def norm(path):
    return tuple(s.strip().lower() for s in path if s.strip())


def bad_indices(table):
    """Return (bad_cols set, bad_rows set, n_cols, n_rows) for one table."""
    top_paths = list(table.top_paths)
    left_paths = list(table.left_paths)
    if not top_paths or not left_paths:
        return set(), set(), len(top_paths), len(left_paths)
    grid, nhr, nhc = flatten_to_grid(top_paths, left_paths)
    rec_cols = reconstruct_col_paths(grid, nhr, nhc)
    rec_rows = reconstruct_row_paths(grid, nhr, nhc)
    bad_cols = {c for c in range(len(top_paths))
                if c >= len(rec_cols) or norm(top_paths[c]) != norm(rec_cols[c])}
    bad_rows = {r for r in range(len(left_paths))
                if r >= len(rec_rows) or norm(left_paths[r]) != norm(rec_rows[r])}
    return bad_cols, bad_rows, len(top_paths), len(left_paths)


def is_arith(aggr) -> bool:
    return str(aggr).lower() in {"sum", "average", "difference", "divide",
                                 "diff", "div", "proportion", "ratio", "change"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/mnt/d/hart_data/hitab/HiTab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default="results/g2_reconstruct_gold_overlap.json")
    args = ap.parse_args()

    queries, tables = load_queries(args.data_dir, args.split)

    # per-table reconstruction error sets + count of mis-reconstructed data cells
    bad = {}
    tot_data_cells = misrec_data_cells = 0
    for tid, t in tables.items():
        bc, br, ncol, nrow = bad_indices(t)
        bad[tid] = (bc, br)
        tot_data_cells += ncol * nrow
        # a data cell (r,c) is mis-reconstructed iff c in bc or r in br
        misrec_data_cells += ncol * nrow - (ncol - len(bc)) * (nrow - len(br))

    gold_operand_cells = set()  # (tid,r,c) unique
    gold_operand_cells_affected = set()

    def blank_pop():
        return {"n_queries": 0, "queries_affected": 0,
                "operand_cells": 0, "operand_cells_affected": 0,
                "affected_by_row": 0, "affected_by_col": 0}

    pops = {"all_with_gold": blank_pop(), "arith_m>=2": blank_pop()}

    for q in queries:
        if not q.gold_operands or q.gold_table_id not in bad:
            continue
        bc, br = bad[q.gold_table_id]
        affected_ops = 0
        by_row = by_col = 0
        for op in q.gold_operands:
            key = (q.gold_table_id, op.row, op.col)
            gold_operand_cells.add(key)
            hit_col = op.col in bc
            hit_row = op.row in br
            if hit_col or hit_row:
                affected_ops += 1
                gold_operand_cells_affected.add(key)
                by_row += int(hit_row)
                by_col += int(hit_col)
        buckets = ["all_with_gold"]
        if len(q.gold_operands) >= 2 and is_arith(q.aggregation):
            buckets.append("arith_m>=2")
        for b in buckets:
            p = pops[b]
            p["n_queries"] += 1
            p["queries_affected"] += int(affected_ops > 0)
            p["operand_cells"] += len(q.gold_operands)
            p["operand_cells_affected"] += affected_ops
            p["affected_by_row"] += by_row
            p["affected_by_col"] += by_col

    def rate(a, b):
        return round(a / b, 4) if b else None

    for p in pops.values():
        p["query_affected_rate"] = rate(p["queries_affected"], p["n_queries"])
        p["operand_cell_affected_rate"] = rate(p["operand_cells_affected"], p["operand_cells"])

    out = {
        "population": {"split": args.split, "n_tables": len(tables)},
        "error_model": "synthetic flatten->rebuild (matches tree_reconstruct_hitab), boundary known",
        "unique_gold_operand_cells": len(gold_operand_cells),
        "unique_gold_operand_cells_affected": len(gold_operand_cells_affected),
        "unique_gold_operand_affected_rate": rate(len(gold_operand_cells_affected), len(gold_operand_cells)),
        "plan_framing_precision_of_errorset_onto_gold": {
            "misreconstructed_data_cells": misrec_data_cells,
            "total_data_cells": tot_data_cells,
            "share_of_errors_that_are_gold_operands": rate(len(gold_operand_cells_affected), misrec_data_cells),
        },
        "by_population": pops,
    }
    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    print(f"[G2] tables={len(tables)}  unique gold operand cells={len(gold_operand_cells)}")
    print(f"     gold operand cells on a mis-reconstructed path: "
          f"{len(gold_operand_cells_affected)}/{len(gold_operand_cells)} "
          f"= {out['unique_gold_operand_affected_rate']}")
    for name, p in pops.items():
        print(f"  [{name}] n_q={p['n_queries']:<5} "
              f"QUERY-level affected={p['query_affected_rate']}  "
              f"(op-cell affected={p['operand_cell_affected_rate']}; "
              f"by_row={p['affected_by_row']} by_col={p['affected_by_col']})")
    print(f"  PLAN framing: of {misrec_data_cells} mis-reconstructed data cells, "
          f"{out['plan_framing_precision_of_errorset_onto_gold']['share_of_errors_that_are_gold_operands']} are gold operands")
    print(f"\nwrote -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
