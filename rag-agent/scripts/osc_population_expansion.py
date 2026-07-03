#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Scope justification: OSC across the FULL question population, not just arith m>=2.

Answers "isn't the problem definition too narrow?" empirically. For every HiTab
dev question whose gold evidence cells are buildable from `linked_cells`
(quantity_link), we measure plain-retrieval OSC@k (BM25 / dense / hybrid) by:

  * population bucket — lookup (aggregation none), arithmetic m>=2 (the paper's
    population), selection/comparison (argmax/pair/topk/... where quantity_link
    is annotated);
  * operand-set size m (1, 2, 3-4, 5-8, 9+).

Expected shape: lookups (m=1) are near-solved by similarity retrieval; OSC decays
monotonically with m — the paper's scope is *where the problem lives*, measured,
not assumed. Also reports gold-coverage per aggregation type, so the exclusion of
unannotated selection questions is a documented data limitation, not a choice.

LLM-free. Run: PYTHONPATH=. /usr/bin/python3 scripts/osc_population_expansion.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from rag_agent.bench.hitab import load_queries
from rag_agent.data.loader import load_table
from rag_agent.eval.operand_set import operand_set_completeness
from rag_agent.retrieve.operand_retriever import HybridRetriever, _tok
from rag_agent.query.operand_decomposer import Embedder
from rag_agent.serialize import S2, serialize_table
from rag_agent.stores.original_store import build_original_table

ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}
SELECT = {"max", "min", "argmax", "argmin", "pair-argmax", "pair-argmin",
          "topk-argmax", "topk-argmin", "kth-argmax", "kth-argmin",
          "greater_than", "less_than"}
KS = (5, 10, 20)
METHODS = ("bm25", "dense", "hybrid")


def bucket_of(q, m):
    agg = q.aggregation or "none"
    if agg == "none":
        return "lookup(m=1)" if m == 1 else "lookup(m>=2)"
    if agg in ARITH:
        return "arith(m=1)" if m == 1 else "arith(m>=2) [paper]"
    if agg in SELECT:
        return "selection/comparison"
    return "other"


def m_band(m):
    if m == 1:
        return "1"
    if m == 2:
        return "2"
    if m <= 4:
        return "3-4"
    if m <= 8:
        return "5-8"
    return "9+"


def numeric_cells(ot, rows, cols):
    return {(r, c) for r in rows for c in cols if ot.cell_num(r, c) is not None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--out", default="results/osc_population_expansion.json")
    args = ap.parse_args()

    queries, tables = load_queries(args.data_dir, args.split)

    # gold-coverage funnel per aggregation type (before filtering)
    cov = defaultdict(lambda: [0, 0])  # agg -> [total, with_gold]
    for q in queries:
        agg = q.aggregation or "none"
        key = ("arith" if agg in ARITH else
               ("selection" if agg in SELECT else agg))
        cov[key][0] += 1
        if q.gold_operands:
            cov[key][1] += 1
    print("gold-evidence coverage by type (n_with_gold/n):")
    for k, (t, g) in sorted(cov.items(), key=lambda kv: -kv[1][0]):
        print(f"  {k:12} {g:5}/{t:5} ({g/t:.2f})")

    pop = [q for q in queries if q.gold_operands]
    print(f"\n[pop] questions with buildable gold evidence: {len(pop)}"
          f" / {len(queries)}")

    emb = Embedder(args.embed_model, device="cpu")
    need = {q.gold_table_id for q in pop}
    print(f"[tables] building retrievers for {len(need)} tables ...")
    retr, ots = {}, {}
    for tid in need:
        retr[tid] = HybridRetriever(serialize_table(tables[tid], S2), emb)
        ots[tid] = build_original_table(load_table(tid, args.data_dir))

    # acc[(bucket|band, method, k)] = [sum_osc, n]
    acc = defaultdict(lambda: [0.0, 0])

    for q in pop:
        ot = ots[q.gold_table_id]
        R = retr[q.gold_table_id]
        gold = q.gold_operands
        m = len({(o.row, o.col) for o in gold})
        bk = bucket_of(q, m)
        band = f"m={m_band(m)}"

        qv = np.asarray(emb.encode([q.question])[0])
        bm25_rank = R._rank(R._bm25.get_scores(_tok(q.question)))
        dense_rank = (R._rank(np.asarray(R._emb) @ qv)
                      if R._emb is not None else bm25_rank)
        fused = {}
        for rank, j in enumerate(bm25_rank):
            fused[j] = fused.get(j, 0.0) + 1.0 / (R.rrf_k + rank)
        for rank, j in enumerate(dense_rank):
            fused[j] = fused.get(j, 0.0) + 1.0 / (R.rrf_k + rank)
        hybrid_rank = sorted(fused, key=lambda j: -fused[j])
        ranks = {"bm25": bm25_rank, "dense": dense_rank, "hybrid": hybrid_rank}

        for meth in METHODS:
            got = set()
            k_done = 0
            for k in KS:
                for i in ranks[meth][k_done:k]:
                    ch = R.chunks[i]
                    got |= {(r, c) for r in ch.rows for c in ch.cols}
                k_done = k
                osc = operand_set_completeness(gold, got)
                for key in (bk, band):
                    a = acc[(key, meth, k)]
                    a[0] += osc
                    a[1] += 1

    out = {"split": args.split, "coverage": {k: {"n": v[0], "with_gold": v[1]}
                                             for k, v in cov.items()},
           "osc_by_bucket": {}, "osc_by_m": {}}
    order_bk = ["lookup(m=1)", "lookup(m>=2)", "arith(m=1)",
                "arith(m>=2) [paper]", "selection/comparison", "other"]
    order_m = ["m=1", "m=2", "m=3-4", "m=5-8", "m=9+"]

    for title, keys, dest in (("population bucket", order_bk, "osc_by_bucket"),
                              ("operand-set size", order_m, "osc_by_m")):
        print(f"\n== OSC@k by {title} (plain retrieval) ==")
        print(f"{'group':22}{'n':>6}" + "".join(
            f"{m}@{k:>2}".rjust(11) for m in ("bm", "de", "hy") for k in KS))
        for key in keys:
            n = acc.get((key, "bm25", KS[0]), [0, 0])[1]
            if not n:
                continue
            row = {"n": n}
            cells = []
            for meth in METHODS:
                for k in KS:
                    s, c = acc[(key, meth, k)]
                    row[f"{meth}@{k}"] = round(s / c, 4)
                    cells.append(f"{s/c:11.3f}")
            out[dest][key] = row
            print(f"{key:22}{n:6}" + "".join(cells))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
