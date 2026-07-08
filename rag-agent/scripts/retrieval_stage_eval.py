#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Consolidated retrieval-stage evaluation (LLM-free).

Reports OSC / row-axis coverage / col-axis coverage / mean cells for the enumeration
family on HiTab dev arithmetic m>=2, so the retrieval contribution is measured
directly (answer generation is a separate, solver-dependent stage; see E7). Backs
`docs/RETRIEVAL_STAGE.md`. Dense / whole-table reference rows come from
`scripts/e7_retrieval_ablation.py --dry-run`.

Run: PYTHONPATH=. python scripts/retrieval_stage_eval.py --split dev
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.bench.hitab import load_queries
from rag_agent.data.loader import load_table
from rag_agent.eval.operand_set import operand_set_completeness
from rag_agent.query.header_embed_resolver import EmbedResolver
from rag_agent.query.operand_decomposer import Embedder
from rag_agent.retrieve.header_enum import enumerate_scope, is_ratio_query
from rag_agent.stores.original_store import build_original_table

ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--cross-encoder", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--out", default="results/retrieval_stage.json")
    args = ap.parse_args()

    queries, _ = load_queries(args.data_dir, args.split)
    pop = [q for q in queries
           if (q.aggregation or "none") in ARITH
           and len({(o.row, o.col) for o in q.gold_operands}) >= 2]
    n = len(pop)
    print(f"[pop] arithmetic m>=2: {n}")

    emb = Embedder(args.embed_model, device="cpu")
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(args.cross_encoder)
    lex = EmbedResolver(emb, row_mode="embed", col_mode="lexical")
    crs = EmbedResolver(emb, row_mode="embed", col_mode="cross_cascade",
                        cross_encoder=ce, top_n_cross=2)
    ots = {t: build_original_table(load_table(t, args.data_dir))
           for t in {q.gold_table_id for q in pop}}

    def measure(label, resolver, **enum_kw):
        osc = rc = cc = cells = 0
        for q in pop:
            ot = ots[q.gold_table_id]
            intent = resolver.resolve(q.question, ot)
            kw = {k: (is_ratio_query(q.question) if v == "ratio" else v)
                  for k, v in enum_kw.items()}
            e = enumerate_scope(ot, intent.row_paths, intent.col_paths, **kw)
            g = q.gold_operands
            osc += operand_set_completeness(g, e.cells)
            rc += int({o.row for o in g} <= e.rows)
            cc += int({o.col for o in g} <= e.cols)
            cells += len(e.cells)
        row = {"config": label, "OSC": round(osc / n, 3), "row_cov": round(rc / n, 3),
               "col_cov": round(cc / n, 3), "mean_cells": round(cells / n, 1)}
        print(json.dumps(row))
        return row

    rows = [
        measure("enum_base", lex),
        measure("enum_treated (lexical col)", lex, add_total_rows="ratio", expand_siblings=True),
        measure("enum_cross (cross-encoder col)", crs, add_total_rows="ratio", expand_siblings=True),
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"population": {"name": "arithmetic_m>=2", "n": n}, "rows": rows}, fh, indent=2)
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
