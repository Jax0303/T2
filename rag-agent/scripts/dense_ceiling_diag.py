#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Why dense/hybrid retrieval has an OSC ceiling it cannot pass (the killer evidence).

Similarity retrieval ranks cells by query↔cell resemblance. Aggregation operands,
though, include cells the query does NOT resemble — chiefly the **unnamed total row**
(the denominator of a share/ratio) whose header is empty or just "total". This script
quantifies, LLM-free, that these structurally-required cells are unreachable by
similarity, which is the mechanism behind dense full-set completeness plateauing
(~0.86) below 1.0. Structural header-tree enumeration reaches them by construction.

For each query (HiTab dev arith m>=2) we embed the query and EVERY numeric cell's
header lineage ("row-path > col-path"), rank cells by cosine, and record for each gold
operand cell: its similarity rank and whether it sits in a total-like row. Then:
  * share of gold operands that live in total-like rows,
  * similarity rank of total-row operands vs ordinary operands,
  * dense full-set completeness@k (the plateau),
  * of the @50 incompletes (the ceiling), how many are explained by a total-row operand.

Run: PYTHONPATH=. python scripts/dense_ceiling_diag.py --split dev
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from rag_agent.bench.hitab import load_queries
from rag_agent.data.loader import load_table
from rag_agent.retrieve.header_enum import is_total_row, is_ratio_query
from rag_agent.query.operand_decomposer import Embedder
from rag_agent.stores.original_store import build_original_table

ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}
KS = (5, 10, 20, 50)


def numeric_cells(ot, rows, cols):
    return {(r, c) for r in rows for c in cols if ot.cell_num(r, c) is not None}


def cell_text(ot, r, c, with_title=False):
    path = list(ot.row_path(r)) + list(ot.col_path(c))
    text = " > ".join(s for s in path if s) or f"r{r}c{c}"
    if with_title and ot.title:
        text = f"{ot.title} > {text}"
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--out", default="results/dense_ceiling_diag.json")
    ap.add_argument("--with-title", action="store_true",
                     help="prepend table title to each cell's embedded text")
    args = ap.parse_args()

    queries, _ = load_queries(args.data_dir, args.split)
    pop = [q for q in queries
           if (q.aggregation or "none") in ARITH
           and len({(o.row, o.col) for o in q.gold_operands}) >= 2]
    n = len(pop)
    print(f"[pop] arithmetic m>=2: {n}")

    emb = Embedder(args.embed_model, device="cpu")
    ots = {t: build_original_table(load_table(t, args.data_dir))
           for t in {q.gold_table_id for q in pop}}
    # per-table: ordered numeric cell list + cell-text embedding matrix
    cache = {}
    for tid, ot in ots.items():
        cells = sorted(numeric_cells(ot, range(ot.n_rows), range(ot.n_cols)))
        mat = np.asarray(emb.encode([cell_text(ot, r, c, args.with_title) for (r, c) in cells])) \
            if cells else np.zeros((0, 1))
        cache[tid] = (cells, mat)

    gold_total_ranks, gold_normal_ranks = [], []   # similarity ranks (0=best)
    n_gold_cells = n_gold_total = 0
    q_has_total = 0
    reach_total = [0, 0]   # [reachable@50, count]  for total-row gold cells
    reach_normal = [0, 0]
    full_complete = {k: 0 for k in KS}             # dense full-set OSC@k
    incomplete50_total, incomplete50 = 0, 0        # @50 ceiling failures w/ total operand
    ratio_total_dep = [0, 0]                        # ratio queries w/ a total operand

    for q in pop:
        ot = ots[q.gold_table_id]
        cells, mat = cache[q.gold_table_id]
        idx = {cell: i for i, cell in enumerate(cells)}
        qv = np.asarray(emb.encode([q.question])[0])
        order = list(np.argsort(-(mat @ qv))) if len(cells) else []
        rank_of = {cells[pos]: r for r, pos in enumerate(order)}

        gold = {(o.row, o.col) for o in q.gold_operands}
        ranks = []
        has_total = False
        for (r, c) in gold:
            n_gold_cells += 1
            tot = is_total_row(ot, r)
            rk = rank_of.get((r, c))           # None if cell not numeric / absent
            if tot:
                n_gold_total += 1
                has_total = True
                if rk is not None:
                    gold_total_ranks.append(rk)
                    reach_total[1] += 1
                    reach_total[0] += int(rk < 50)
            else:
                if rk is not None:
                    gold_normal_ranks.append(rk)
                    reach_normal[1] += 1
                    reach_normal[0] += int(rk < 50)
            ranks.append(rk if rk is not None else 10 ** 9)
        if has_total:
            q_has_total += 1
        for k in KS:
            if all(r < k for r in ranks):
                full_complete[k] += 1
        if not all(r < 50 for r in ranks):        # @50 incomplete = ceiling failure
            incomplete50 += 1
            if has_total:
                incomplete50_total += 1
        if is_ratio_query(q.question):
            ratio_total_dep[1] += 1
            ratio_total_dep[0] += int(has_total)

    med = lambda xs: round(statistics.median(xs), 1) if xs else None
    out = {
        "population": {"name": "arithmetic_m>=2", "n": n},
        "cell_repr": "table title > header lineage (row-path > col-path)" if args.with_title
                     else "header lineage (row-path > col-path)",
        "with_title": args.with_title, "embed_model": args.embed_model,
        "gold_operand_cells": n_gold_cells,
        "gold_in_total_rows": n_gold_total,
        "pct_gold_in_total_rows": round(n_gold_total / n_gold_cells, 4) if n_gold_cells else 0,
        "pct_queries_with_total_operand": round(q_has_total / n, 4),
        "median_sim_rank_total_operand": med(gold_total_ranks),
        "median_sim_rank_normal_operand": med(gold_normal_ranks),
        "reachable@50_total_operand": round(reach_total[0] / reach_total[1], 4) if reach_total[1] else None,
        "reachable@50_normal_operand": round(reach_normal[0] / reach_normal[1], 4) if reach_normal[1] else None,
        "dense_full_set_complete": {f"@{k}": round(full_complete[k] / n, 4) for k in KS},
        "incomplete@50": incomplete50,
        "incomplete@50_with_total_operand": incomplete50_total,
        "pct_ceiling_failures_explained_by_total": round(incomplete50_total / incomplete50, 4) if incomplete50 else None,
        "ratio_queries": ratio_total_dep[1],
        "ratio_queries_with_total_operand": ratio_total_dep[0],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"\ngold operand cells in total-like rows : {out['pct_gold_in_total_rows']*100:.1f}% "
          f"({n_gold_total}/{n_gold_cells})")
    print(f"queries needing >=1 total-row operand : {out['pct_queries_with_total_operand']*100:.1f}%")
    print(f"\nmedian similarity rank  total-row operand : {out['median_sim_rank_total_operand']}")
    print(f"median similarity rank  normal   operand : {out['median_sim_rank_normal_operand']}")
    print(f"reachable within top-50  total : {out['reachable@50_total_operand']}  "
          f"normal : {out['reachable@50_normal_operand']}")
    print(f"\ndense full-set completeness (the plateau):")
    for k in KS:
        print(f"   @{k:<3} {out['dense_full_set_complete'][f'@{k}']:.3f}")
    print(f"\n@50 ceiling failures explained by a total operand: "
          f"{out['pct_ceiling_failures_explained_by_total']} "
          f"({incomplete50_total}/{incomplete50})")
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
