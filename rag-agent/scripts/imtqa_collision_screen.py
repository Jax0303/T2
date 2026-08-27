#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Screen IM-TQA for the method's win condition. LLM-free, encoder-free.

Before spending a reader on a fifth corpus, ask the question the collision
statistic already answers on four: how much of the gain is even available here?
``cell_sentence_collision.py`` established that the drop from ``flat`` to a
header path predicts the size of the win (HiTab 69.8 points -> +.32 EM, AIT-QA
20.8 -> +.06, MultiHiertt 16.1 -> +.03), and that predictor called the fourth
corpus before the run. This applies it to the fifth.

IM-TQA (Zheng et al., ACL 2023) ships ``table_type``, so the screen also splits
by it: the method needs multi-level headers, and a corpus that is mostly
``vertical``/``horizontal`` has nothing for a path to repair. That split is the
point of screening rather than guessing.

English values are used (``english_cell_value_list`` is populated), so the
frozen ``bge-small-en`` front-end would apply unchanged.

No titles in this corpus, so S3 == S2 and only ``flat``/``S2`` are reported.

  PYTHONPATH=. python3 scripts/imtqa_collision_screen.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cell_sentence_collision import collisions, render
from imtqa_structural_detector import grids_of
from rag_agent.reconstruct import (guess_n_header_cols, guess_n_header_rows,
                                   reconstruct_col_paths, reconstruct_row_paths)
from rag_agent.runenv import run_env


def corpus_of(tables: list[dict]) -> SimpleNamespace:
    """cell_paths / cell_owner in the shape ``render`` expects.

    Same reconstruction front-end as ``realhitbench_corpus``: the grid is all we
    get, exactly as in deployment. IM-TQA's own ``row_attribute`` labels are NOT
    used -- reading them would measure a structure we would not have at index
    time, which is the mistake condition (1) of the header experiments.
    """
    cell_paths, cell_owner, types = [], [], {}
    for t in tables:
        _, grid = grids_of(t)                      # english
        if not grid or len(grid) < 3 or len(grid[0]) < 2:
            continue
        nhc = guess_n_header_cols(grid)
        nhr = max(1, min(guess_n_header_rows(grid, n_header_cols=nhc), len(grid) - 1))
        nhc = guess_n_header_cols(grid, n_header_rows=nhr)
        nhr = max(1, min(guess_n_header_rows(grid, n_header_cols=nhc), len(grid) - 1))
        cols = reconstruct_col_paths(grid, nhr, n_header_cols=nhc)
        rows = reconstruct_row_paths(grid, nhr, n_header_cols=nhc)
        data = [[str(x).strip() for x in r[nhc:]] for r in grid[nhr:]]
        if not data or not data[0]:
            continue
        tid = t["table_id"]
        types[tid] = t.get("table_type", "?")
        for i, row in enumerate(data):
            rp = list(rows[i]) if i < len(rows) else []
            for j, v in enumerate(row):
                if not str(v).strip():
                    continue
                cp = list(cols[j]) if j < len(cols) else []
                cell_paths.append((list(rp), cp, v))
                cell_owner.append((tid, i, j))
    return SimpleNamespace(cell_paths=cell_paths, cell_owner=cell_owner,
                           title={}, types=types)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/imtqa")
    ap.add_argument("--splits", default="dev_tables.json,test_tables.json")
    ap.add_argument("--out", default="results/imtqa_collision_screen.json")
    args = ap.parse_args()

    tables = []
    for s in args.splits.split(","):
        tables += json.loads((Path(args.data_dir) / s).read_text())
    C = corpus_of(tables)

    out = {"experiment": "IM-TQA collision screen (no LLM, no encoder)",
           "env": run_env(0, "none — no encoder in this diagnostic"),
           "tables": len(C.types), "cells": len(C.cell_paths), "titled": 0.0,
           "by_table_type": {}}
    for scheme in ("flat", "S2"):
        out[scheme] = {"address": collisions(render(C, scheme, False), C.cell_owner),
                       "sentence": collisions(render(C, scheme, True), C.cell_owner)}
    out["path_repairs_points"] = round(
        100 * (out["flat"]["address"]["dup_cell_rate"]
               - out["S2"]["address"]["dup_cell_rate"]), 1)

    idx_by_type = defaultdict(list)
    for k, (tid, *_) in enumerate(C.cell_owner):
        idx_by_type[C.types[tid]].append(k)
    for ttype, idx in sorted(idx_by_type.items()):
        row = {"cells": len(idx),
               "tables": len({C.cell_owner[k][0] for k in idx})}
        for scheme in ("flat", "S2"):
            addr = render(C, scheme, False)
            row[scheme] = collisions([addr[k] for k in idx],
                                     [C.cell_owner[k] for k in idx])["dup_cell_rate"]
        row["repairs_points"] = round(100 * (row["flat"] - row["S2"]), 1)
        out["by_table_type"][ttype] = row

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"IM-TQA {out['tables']} tables / {out['cells']} cells")
    print(f"  flat address dup {out['flat']['address']['dup_cell_rate']:.1%}"
          f"  -> S2 {out['S2']['address']['dup_cell_rate']:.1%}"
          f"   ({out['path_repairs_points']} points repaired)")
    for ttype, row in out["by_table_type"].items():
        print(f"  {ttype:13s} {row['tables']:3d} tables {row['cells']:6d} cells "
              f"flat {row['flat']:.1%} -> S2 {row['S2']:.1%}  "
              f"({row['repairs_points']} pts)")
    print(f"wrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
