#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verbalized-sentence accuracy — are the sentences we index actually correct?

Cell-level verbalization (`rag_agent.serialize.verbalize`) is a deterministic
template over three inputs: the cell VALUE, its ROW header path and its COLUMN
header path. So a sentence is wrong exactly when one of those three is wrong,
which happens when header-tree reconstruction misfires. The existing
reconstruction numbers (`tree_reconstruct_hitab.py`) score *paths*; this script
scores the artifact that actually reaches the index — the sentence.

Method (HiTab dev, the only corpus with gold header trees):

  1. Render each table's GOLD paths + real data into a flat blank-after-first
     grid (the same synthetic flatten used by `tree_reconstruct_hitab.py`, but
     carrying real cell values rather than "1" placeholders).
  2. Run the real pipeline on that grid: guess the header/data boundary, then
     reconstruct row/column paths, then verbalize every nonempty data cell.
  3. Align each produced sentence back to its GRID coordinate and compare it to
     the sentence the gold structure would have produced for that same cell.

Because step 2 guesses the boundary, a table can emit sentences for cells that
are really header cells (spurious) and can miss real data cells (dropped), so
this is scored as precision / recall, not plain accuracy:

  precision = correct sentences / sentences produced
  recall    = correct sentences / gold data cells

Run: PYTHONPATH=. .venv/bin/python scripts/sentence_accuracy_hitab.py --split dev
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_agent.bench.hitab import load_queries
from rag_agent.bench.schema import BenchTable
from rag_agent.reconstruct import (
    guess_n_header_rows,
    reconstruct_col_paths,
    reconstruct_row_paths,
)
from rag_agent.serialize.verbalize import STYLES, _fmt, verbalize_cell
from tree_reconstruct_hitab import size_bucket


def flatten_with_values(table: BenchTable):
    """Blank-after-first header grid carrying the table's REAL data values."""
    top_paths = [table.col_path(c) for c in range(table.n_cols)]
    left_paths = [table.row_path(r) for r in range(table.n_rows)]
    n_header_rows = max((len(p) for p in top_paths), default=0)
    n_header_cols = max((len(p) for p in left_paths), default=0)
    width = n_header_cols + table.n_cols
    height = n_header_rows + table.n_rows
    grid = [["" for _ in range(width)] for _ in range(height)]

    # A merged span belongs to one tree node, so a label is suppressed (blank)
    # only while the FULL prefix (parent path included) repeats — two same-named
    # siblings under different parents are distinct cells in a real dump.
    for c, path in enumerate(top_paths):
        for d in range(n_header_rows):
            label = path[d] if d < len(path) else ""
            if label and (c == 0 or top_paths[c - 1][:d + 1] != path[:d + 1]):
                grid[d][n_header_cols + c] = label

    for r, path in enumerate(left_paths):
        for d in range(n_header_cols):
            label = path[d] if d < len(path) else ""
            if label and (r == 0 or left_paths[r - 1][:d + 1] != path[:d + 1]):
                grid[n_header_rows + r][d] = label

    for r in range(table.n_rows):
        for c in range(table.n_cols):
            # Store the rendered form: a scraped grid carries display strings,
            # so comparing against `_fmt(gold)` keeps float-vs-string
            # stringification out of the error attribution.
            grid[n_header_rows + r][n_header_cols + c] = _fmt(table.cell(r, c))

    return grid, n_header_rows, n_header_cols


def _nonempty(v) -> bool:
    return v is not None and str(v).strip() != ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max-tables", type=int, default=0, help="0 = all")
    ap.add_argument("--known-boundary", action="store_true",
                    help="use the true header/data boundary instead of guessing "
                         "(isolates path reconstruction from boundary detection)")
    ap.add_argument("--out", default="results/sentence_accuracy_hitab.json")
    args = ap.parse_args()

    _, tables = load_queries(args.data_dir, args.split)
    table_ids = list(tables.keys())
    if args.max_tables:
        table_ids = table_ids[: args.max_tables]
    print(f"[pop] HiTab {args.split} tables: {len(table_ids)}")

    # stats[style] -> counters; buckets[style][bucket] -> counters
    def new_counter():
        return {"produced": 0, "gold_cells": 0, "correct": 0,
                "wrong_value": 0, "wrong_row": 0, "wrong_col": 0,
                "spurious": 0, "dropped": 0}

    stats = {s: new_counter() for s in STYLES}
    buckets = {s: defaultdict(new_counter) for s in STYLES}
    boundary_hit = boundary_total = 0
    error_examples = []
    collision = Counter()  # sentences whose (row,col) path pair duplicates another cell's

    for tid in table_ids:
        gt = tables[tid]
        if not gt.n_rows or not gt.n_cols:
            continue
        grid, nhr, nhc = flatten_with_values(gt)

        if args.known_boundary:
            use_nhr = nhr
        else:
            use_nhr = guess_n_header_rows(grid, n_header_cols=nhc)
            boundary_total += 1
            boundary_hit += int(use_nhr == nhr)

        rec_cols = reconstruct_col_paths(grid, use_nhr, nhc)
        rec_rows = reconstruct_row_paths(grid, use_nhr, nhc)
        rec_data = [row[nhc:] for row in grid[use_nhr:]]
        rec = BenchTable(table_id=tid, title=gt.title, data=rec_data,
                         top_paths=rec_cols, left_paths=rec_rows, source="hitab")

        bucket = size_bucket(gt.n_rows, gt.n_cols)
        shift = use_nhr - nhr  # >0: header guessed too deep (real data rows eaten)

        # path-pair collisions inside the produced table (two cells, one address)
        seen_addr = set()

        for style in STYLES:
            st = stats[style]
            bs = buckets[style][bucket]
            bs["n_tables"] = bs.get("n_tables", 0) + 1

            covered = set()
            for r in range(rec.n_rows):
                for c in range(rec.n_cols):
                    if not _nonempty(rec.cell(r, c)):
                        continue
                    st["produced"] += 1
                    bs["produced"] += 1
                    g_row = r + shift  # index into the gold data rows
                    if g_row < 0 or g_row >= gt.n_rows or c >= gt.n_cols:
                        # sentence for something that is not a gold data cell
                        st["spurious"] += 1
                        bs["spurious"] += 1
                        continue
                    covered.add((g_row, c))
                    got = verbalize_cell(rec, r, c, style)
                    want = verbalize_cell(gt, g_row, c, style)
                    if got == want:
                        st["correct"] += 1
                        bs["correct"] += 1
                    else:
                        # attribute the error
                        if _fmt(rec.cell(r, c)) != _fmt(gt.cell(g_row, c)):
                            st["wrong_value"] += 1
                            bs["wrong_value"] += 1
                        if rec.row_path(r) != gt.row_path(g_row):
                            st["wrong_row"] += 1
                            bs["wrong_row"] += 1
                        if rec.col_path(c) != gt.col_path(c):
                            st["wrong_col"] += 1
                            bs["wrong_col"] += 1
                        if style == "long" and len(error_examples) < 12:
                            error_examples.append({
                                "table_id": tid, "cell": [g_row, c],
                                "generated": got, "gold": want,
                            })
                    if style == "long":
                        addr = (tuple(rec.row_path(r)), tuple(rec.col_path(c)))
                        if addr in seen_addr:
                            collision["duplicate_address"] += 1
                        seen_addr.add(addr)

            gold_cells = {(r, c) for r in range(gt.n_rows) for c in range(gt.n_cols)
                          if _nonempty(gt.cell(r, c))}
            st["gold_cells"] += len(gold_cells)
            bs["gold_cells"] += len(gold_cells)
            missed = len(gold_cells - covered)
            st["dropped"] += missed
            bs["dropped"] += missed

    def summarize(c):
        prod, gold = c["produced"], c["gold_cells"]
        p = c["correct"] / prod if prod else None
        r = c["correct"] / gold if gold else None
        f1 = (2 * p * r / (p + r)) if (p and r) else None
        return {
            "sentences_produced": prod,
            "gold_data_cells": gold,
            "exact_correct": c["correct"],
            "precision": round(p, 4) if p is not None else None,
            "recall": round(r, 4) if r is not None else None,
            "f1": round(f1, 4) if f1 else None,
            "err_wrong_value": c["wrong_value"],
            "err_wrong_row_path": c["wrong_row"],
            "err_wrong_col_path": c["wrong_col"],
            "spurious_sentences": c["spurious"],
            "dropped_gold_cells": c["dropped"],
        }

    out = {
        "population": {"name": "hitab_synthetic_flatten_with_values",
                       "split": args.split, "n_tables": len(table_ids)},
        "boundary_mode": "known (ground truth)" if args.known_boundary else "guessed (pipeline)",
        "boundary_guess_accuracy": (round(boundary_hit / boundary_total, 4)
                                    if boundary_total else None),
        "by_style": {s: summarize(stats[s]) for s in STYLES},
        "by_style_by_size_bucket": {
            s: {b: summarize(c) | {"n_tables": c.get("n_tables")}
                for b, c in buckets[s].items()}
            for s in STYLES
        },
        "duplicate_address_sentences": collision["duplicate_address"],
        "error_examples_long_style": error_examples,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"boundary mode : {out['boundary_mode']}")
    if out["boundary_guess_accuracy"] is not None:
        print(f"boundary acc  : {out['boundary_guess_accuracy']}")
    for s in STYLES:
        v = out["by_style"][s]
        print(f"\n[{s}] produced={v['sentences_produced']} gold_cells={v['gold_data_cells']}")
        print(f"  sentence exact  P={v['precision']}  R={v['recall']}  F1={v['f1']}")
        print(f"  errors: value={v['err_wrong_value']} row_path={v['err_wrong_row_path']} "
              f"col_path={v['err_wrong_col_path']} spurious={v['spurious_sentences']} "
              f"dropped={v['dropped_gold_cells']}")
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
