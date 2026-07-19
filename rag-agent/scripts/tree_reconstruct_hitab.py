#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Tree-reconstruction experiment A — synthetic HiTab flatten -> rebuild -> compare to gold.

HiTab hands us the header tree pre-parsed (``top_root``/``left_root``). To
test the "build the tree from a raw 2D grid" step (`rag_agent.reconstruct`)
on something with an exact, known-correct answer, this script does the
opposite of what the real pipeline does: it takes each table's ALREADY-KNOWN
gold header paths (``top_paths`` / ``left_paths``) and renders them into a
synthetic "blank-after-first" grid — the way a merged-cell spreadsheet looks
once copy-pasted into a flat CSV (a repeated header value shows only at the
first column/row of its span, blank elsewhere). It then runs the SAME
reconstruction algorithm used on real scraped tables
(`scripts/tree_reconstruct_multihiertt.py`) on that synthetic grid, and
scores the rebuilt paths token-for-token against the real gold paths — a
clean, ground-truth-verified measurement of the algorithm alone, isolated
from any header/data-boundary guessing (the header block size is given, not
guessed) unless ``--guess-boundary`` is passed.

Run: PYTHONPATH=. python scripts/tree_reconstruct_hitab.py --split dev --max-tables 150
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.bench.hitab import load_queries
from rag_agent.reconstruct import guess_n_header_rows, reconstruct_col_paths, reconstruct_row_paths


def flatten_to_grid(top_paths, left_paths):
    """Render known gold header paths into a synthetic blank-after-first grid."""
    n_header_rows = max((len(p) for p in top_paths), default=0)
    n_header_cols = max((len(p) for p in left_paths), default=0)
    n_cols, n_rows = len(top_paths), len(left_paths)
    width = n_header_cols + n_cols
    height = n_header_rows + n_rows
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

    # Placeholder numeric data cells so guess_n_header_rows has something to
    # trigger on when --guess-boundary is used.
    for r in range(n_rows):
        for c in range(n_cols):
            grid[n_header_rows + r][n_header_cols + c] = "1"

    return grid, n_header_rows, n_header_cols


def norm(path):
    return tuple(s.strip().lower() for s in path if s.strip())


def size_bucket(n_rows: int, n_cols: int) -> str:
    """Matches the lab-meeting claim's stated scope: tree-mapped 'up to 10x10',
    experimenting beyond that. Bucketed by the larger of the two dimensions."""
    d = max(n_rows, n_cols)
    if d <= 10:
        return "<=10x10"
    if d <= 20:
        return "10-20"
    return ">20"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max-tables", type=int, default=150)
    ap.add_argument("--guess-boundary", action="store_true",
                     help="also guess the header/data row boundary instead of using the known one")
    ap.add_argument("--out", default="results/tree_reconstruct_hitab.json")
    args = ap.parse_args()

    _, tables = load_queries(args.data_dir, args.split)
    table_ids = list(tables.keys())[: args.max_tables] if args.max_tables else list(tables.keys())
    print(f"[pop] HiTab tables: {len(table_ids)}")

    from collections import defaultdict
    bucket_stats = defaultdict(lambda: {"col_total": 0, "col_hit": 0, "row_total": 0, "row_hit": 0,
                                         "n_tables": 0})
    boundary_hit = boundary_total = 0
    examples = []

    for tid in table_ids:
        bt = tables[tid]
        top_paths = [bt.col_path(c) for c in range(bt.n_cols)]
        left_paths = [bt.row_path(r) for r in range(bt.n_rows)]
        if not top_paths or not left_paths:
            continue
        grid, nhr, nhc = flatten_to_grid(top_paths, left_paths)

        use_nhr = nhr
        if args.guess_boundary:
            guessed = guess_n_header_rows(grid, n_header_cols=nhc)
            boundary_total += 1
            boundary_hit += int(guessed == nhr)
            use_nhr = guessed

        rec_cols = reconstruct_col_paths(grid, use_nhr, nhc)
        rec_rows = reconstruct_row_paths(grid, use_nhr, nhc)

        b = size_bucket(bt.n_rows, bt.n_cols)
        s = bucket_stats[b]
        s["n_tables"] += 1
        for gold, rec in zip(top_paths, rec_cols):
            s["col_total"] += 1
            s["col_hit"] += int(norm(gold) == norm(rec))
        for gold, rec in zip(left_paths, rec_rows):
            s["row_total"] += 1
            s["row_hit"] += int(norm(gold) == norm(rec))

        if len(examples) < 5:
            examples.append({
                "table_id": tid,
                "gold_col_paths": top_paths[:3],
                "reconstructed_col_paths": rec_cols[:3],
            })

    by_bucket = {}
    col_total = col_hit = row_total = row_hit = 0
    for b, s in bucket_stats.items():
        col_total += s["col_total"]; col_hit += s["col_hit"]
        row_total += s["row_total"]; row_hit += s["row_hit"]
        by_bucket[b] = {
            "n_tables": s["n_tables"],
            "col_path_exact_match": round(s["col_hit"] / s["col_total"], 4) if s["col_total"] else None,
            "row_path_exact_match": round(s["row_hit"] / s["row_total"], 4) if s["row_total"] else None,
        }

    out = {
        "population": {"name": "hitab_synthetic_flatten", "split": args.split, "n_tables": len(table_ids)},
        "boundary_mode": "guessed" if args.guess_boundary else "known (ground truth)",
        "col_path_exact_match": round(col_hit / col_total, 4) if col_total else None,
        "row_path_exact_match": round(row_hit / row_total, 4) if row_total else None,
        "col_paths_scored": col_total,
        "row_paths_scored": row_total,
        "boundary_guess_accuracy": round(boundary_hit / boundary_total, 4) if boundary_total else None,
        "by_size_bucket": by_bucket,
        "examples": examples,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"boundary mode        : {out['boundary_mode']}")
    if out["boundary_guess_accuracy"] is not None:
        print(f"boundary guess acc    : {out['boundary_guess_accuracy']}")
    print(f"col_path_exact_match : {out['col_path_exact_match']}  ({col_hit}/{col_total})")
    print(f"row_path_exact_match : {out['row_path_exact_match']}  ({row_hit}/{row_total})")
    print(f"\nby size bucket (max(rows,cols)):")
    for b in ("<=10x10", "10-20", ">20"):
        if b in by_bucket:
            v = by_bucket[b]
            print(f"  {b:<10} n_tables={v['n_tables']:<4} "
                  f"col_exact={v['col_path_exact_match']}  row_exact={v['row_path_exact_match']}")
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
