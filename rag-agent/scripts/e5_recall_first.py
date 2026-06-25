#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""E5 — recall-first union retrieval: how close to OSC=1.0, at what cell cost.

The professor's constraint: operand-set completeness must be ~100%. Precise
enumeration (E2) trades completeness for a small cell set. Here we measure a
completeness/cost ladder, prioritizing OSC and reporting the cell budget so the
trade is explicit:

  A enum_precise      hybrid enumeration, matched rows x matched cols (E2 method)
  B enum_axis_complete  complete the *other* axis: (R x all numeric cols)
                        union (all numeric rows x C) — covers aggregation along
                        whichever axis the query actually named
  C dense_k20         dense top-20 row-chunks' cells (similarity recall ceiling)
  D union(B, dense_k20)  the recall-first system: enumeration + similarity
  E whole_table       all numeric cells (completeness upper bound + max cost)

Each config reports OSC (completeness) and mean cell count, on HiTab dev
arithmetic m>=2 (n≈158). LLM-free (hybrid resolver = row embed + col lexical).
Run:
    PYTHONPATH=. python scripts/e5_recall_first.py --split dev
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.bench.hitab import load_queries
from rag_agent.data.loader import load_table
from rag_agent.eval.operand_set import bin_scope, operand_set_completeness
from rag_agent.query.header_embed_resolver import EmbedResolver
from rag_agent.query.operand_decomposer import Embedder
from rag_agent.retrieve.header_enum import enumerate_scope
from rag_agent.retrieve.operand_retriever import HybridRetriever, retrieve
from rag_agent.serialize import S2, serialize_table

SEED = 42
ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}


def _numeric_cells(ot):
    return {(r, c) for r in range(ot.n_rows) for c in range(ot.n_cols)
            if ot.cell_num(r, c) is not None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--k-dense", type=int, default=20)
    ap.add_argument("--out", default="results/e5_recall_first.json")
    args = ap.parse_args()

    queries, tables = load_queries(args.data_dir, args.split, args.max)
    pop = [q for q in queries
           if (q.aggregation or "none") in ARITH and len(q.gold_operands) >= 2]
    print(f"[pop] arithmetic m>=2: {len(pop)}")

    embedder = Embedder(args.embed_model, device=args.device)
    resolver = EmbedResolver(embedder, row_mode="embed", col_mode="lexical")
    from rag_agent.stores.original_store import build_original_table
    ots, retr = {}, {}
    for tid in {q.gold_table_id for q in pop}:
        ots[tid] = build_original_table(load_table(tid, args.data_dir))
        retr[tid] = HybridRetriever(serialize_table(tables[tid], S2), embedder)

    CONFIGS = ["A_enum_precise", "B_enum_axis_complete", "C_dense_k20",
               "D_union_B_dense", "E_whole_table"]
    recs = []
    for i, q in enumerate(pop):
        ot = ots[q.gold_table_id]
        gold = {(o.row, o.col) for o in q.gold_operands}
        all_num = _numeric_cells(ot)
        all_rows = set(range(ot.n_rows))
        all_cols = set(range(ot.n_cols))

        intent = resolver.resolve(q.question, ot)
        e = enumerate_scope(ot, intent.row_paths, intent.col_paths)
        R, C = e.rows, e.cols

        precise = set(e.cells)
        axis_complete = {(r, c) for r in R for c in all_cols if ot.cell_num(r, c) is not None}
        axis_complete |= {(r, c) for r in all_rows for c in C if ot.cell_num(r, c) is not None}

        res = retrieve(q.question, tables[q.gold_table_id], None, mode="plain",
                       k=args.k_dense, scheme=S2, embedder=embedder, retriever=retr[q.gold_table_id])
        dense = set()
        for ch in res.retrieved:
            for r in ch.rows:
                for c in ch.cols:
                    if ot.cell_num(r, c) is not None:
                        dense.add((r, c))

        sets = {
            "A_enum_precise": precise,
            "B_enum_axis_complete": axis_complete,
            "C_dense_k20": dense,
            "D_union_B_dense": axis_complete | dense,
            "E_whole_table": all_num,
        }
        rec = {"m": len(gold)}
        for name, s in sets.items():
            rec[f"{name}_osc"] = operand_set_completeness(gold, s)
            rec[f"{name}_cells"] = len(s)
        recs.append(rec)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(pop)}")

    n = len(recs)

    def agg(sub):
        ns = len(sub)
        if not ns:
            return {}
        return {c: {"osc": round(sum(r[f"{c}_osc"] for r in sub) / ns, 4),
                    "mean_cells": round(sum(r[f"{c}_cells"] for r in sub) / ns, 1)}
                for c in CONFIGS} | {"n": ns}

    out = {
        "experiment": "E5_recall_first", "split": args.split, "seed": SEED,
        "k_dense": args.k_dense,
        "note": "OSC=completeness (gold subset of retrieved); mean_cells=budget",
        "overall": agg(recs),
        "by_scope": {b: agg([r for r in recs if bin_scope(r["m"]) == b])
                     for b in ("2", "3-4", "5-8", "9+")},
    }
    out["by_scope"] = {b: v for b, v in out["by_scope"].items() if v}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    # pretty print the ladder
    print("\nconfig                  OSC     mean_cells")
    for c in CONFIGS:
        v = out["overall"][c]
        print(f"  {c:<22} {v['osc']:<7} {v['mean_cells']}")
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
