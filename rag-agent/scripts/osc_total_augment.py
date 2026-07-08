#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Beat BM25/dense/hybrid on average OSC by patching their structural blind spot.

The ceiling diagnosis (`dense_ceiling_diag`) showed 76% of similarity-retrieval
completeness failures are unnamed **total rows** the query does not resemble. Those
cells are unreachable by similarity at any budget — but trivially reachable by
*structure*. So we don't replace similarity retrieval; we **augment** it: take any
retriever's top-k rows and union in the table's total-like rows' cells.

For each similarity retriever (BM25-only, dense-only, hybrid RRF) we report OSC and
mean cells at each budget k, plain vs +total-augmentation, on the SAME queries. If the
augmented curve dominates (higher OSC at matched cells) and breaks the plain asymptote,
the structural patch wins where similarity structurally cannot.

Population: HiTab dev arith m>=2 (n=161). LLM-free.
Run: PYTHONPATH=. python scripts/osc_total_augment.py --split dev
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from rag_agent.bench.hitab import load_queries
from rag_agent.data.loader import load_table
from rag_agent.eval.operand_set import operand_set_completeness
from rag_agent.retrieve.header_enum import total_like_rows_hybrid, is_ratio_query
from rag_agent.retrieve.operand_retriever import HybridRetriever, _tok
from rag_agent.query.operand_decomposer import Embedder
from rag_agent.serialize import S2, serialize_table
from rag_agent.stores.original_store import build_original_table

ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}
KS = (5, 10, 20, 40)
# fine plain grid for the per-query cell-matched strict-budget test
PLAIN_GRID = tuple(range(1, 61))


def numeric_cells(ot, rows, cols):
    return {(r, c) for r in rows for c in cols if ot.cell_num(r, c) is not None}


def cells_of_chunks(ot, retriever, idxs):
    out = set()
    for i in idxs:
        ch = retriever.chunks[i]
        out |= numeric_cells(ot, ch.rows, ch.cols)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--ratio-only-aug", action="store_true",
                    help="inject total rows only for ratio/share queries (precision-safe)")
    ap.add_argument("--aug-k", type=int, default=10)
    ap.add_argument("--plain-k", type=int, default=20)
    ap.add_argument("--col-targeted", action="store_true",
                    help="inject total rows only in columns the retrieved rows already touch "
                         "(no-op for S2 row chunks: they span all columns)")
    ap.add_argument("--resolver-cols", action="store_true",
                    help="inject total rows only in the 1-2 columns the cross-encoder column "
                         "resolver picks (cheap, query-targeted denominator)")
    ap.add_argument("--cross-encoder", default="BAAI/bge-reranker-base",
                    help="bge-reranker-base validated best (2026-06-29): all 3 retrievers "
                         "significant on the strict cell-matched test; MiniLM left dense/hybrid n.s.")
    ap.add_argument("--top-n-cross", type=int, default=2,
                    help="columns the cross-encoder column resolver keeps (precision<->recall)")
    ap.add_argument("--total-cols-only", action="store_true",
                    help="restrict the column resolver's candidates to columns that carry a "
                         "total row (removes distractor columns -> higher column precision)")
    ap.add_argument("--out", default="results/osc_total_augment.json")
    args = ap.parse_args()

    queries, tables = load_queries(args.data_dir, args.split)
    pop = [q for q in queries
           if (q.aggregation or "none") in ARITH
           and len({(o.row, o.col) for o in q.gold_operands}) >= 2]
    n = len(pop)
    print(f"[pop] arithmetic m>=2: {n}  ratio_only_aug={args.ratio_only_aug}")

    emb = Embedder(args.embed_model, device="cpu")
    col_resolver = None
    if args.resolver_cols:
        from sentence_transformers import CrossEncoder
        from rag_agent.query.header_embed_resolver import EmbedResolver
        col_resolver = EmbedResolver(emb, col_mode="cross",
                                     cross_encoder=CrossEncoder(args.cross_encoder),
                                     top_n_cross=args.top_n_cross)
    need = {q.gold_table_id for q in pop}
    retr, ots, totals, total_rows = {}, {}, {}, {}
    for tid in need:
        retr[tid] = HybridRetriever(serialize_table(tables[tid], S2), emb)
        ot = build_original_table(load_table(tid, args.data_dir))
        ots[tid] = ot
        tr = total_like_rows_hybrid(ot)
        total_rows[tid] = tr
        totals[tid] = numeric_cells(ot, tr, range(ot.n_cols))

    methods = ("bm25", "dense", "hybrid")
    # acc[method][k] = {"osc_plain":sum, "osc_aug":sum, "cells_plain":sum, "cells_aug":sum}
    acc = {m: {k: dict(osc_p=0.0, osc_a=0.0, c_p=0, c_a=0) for k in KS} for m in methods}
    aug_cells_added = 0
    # matched-budget paired test: aug@AUG_K (fewer cells) vs plain@PLAIN_K (more cells)
    AUG_K, PLAIN_K = args.aug_k, args.plain_k
    paired = {m: [] for m in methods}   # (osc_aug@AUG_K, osc_plain@PLAIN_K) per query
    # strict per-query cell-matched test: aug@AUG_K vs plain@k' where cells(plain@k')>=cells(aug)
    cellmatch = {m: [] for m in methods}   # (osc_aug, osc_plain_matched, cells_aug, kp) per query
    samedepth = {m: [] for m in methods}   # (osc_aug@AUG_K, osc_plain@AUG_K) per query

    for q in pop:
        ot = ots[q.gold_table_id]
        R = retr[q.gold_table_id]
        gold = q.gold_operands
        all_tcells = totals[q.gold_table_id]
        trows = total_rows[q.gold_table_id]
        inject_on = not (args.ratio_only_aug and not is_ratio_query(q.question))
        # query-targeted columns from the cross-encoder column resolver (computed once)
        resolver_cols = None
        if args.resolver_cols:
            col_allowed = {c for (_, c) in all_tcells} if args.total_cols_only else None
            intent = col_resolver.resolve(q.question, ot, col_allowed=col_allowed)
            cidx = set()
            for p in intent.col_paths:
                cidx.update(ot.find_cols_by_header(" > ".join(p)))
            resolver_cols = cidx
            resolver_tcells = {(r, c) for r in trows for c in cidx
                               if ot.cell_num(r, c) is not None}

        def tcells_for(base):
            if not inject_on:
                return set()
            if args.resolver_cols:
                return resolver_tcells
            if args.col_targeted:
                cols = {c for (_, c) in base}
                return {(r, c) for r in trows for c in cols if ot.cell_num(r, c) is not None}
            return all_tcells

        qv = np.asarray(emb.encode([q.question])[0])
        bm25_rank = R._rank(R._bm25.get_scores(_tok(q.question)))
        dense_rank = R._rank(np.asarray(R._emb) @ qv) if R._emb is not None else bm25_rank
        # hybrid = RRF of the two
        fused = {}
        for rank, i in enumerate(bm25_rank):
            fused[i] = fused.get(i, 0.0) + 1.0 / (R.rrf_k + rank)
        for rank, i in enumerate(dense_rank):
            fused[i] = fused.get(i, 0.0) + 1.0 / (R.rrf_k + rank)
        hybrid_rank = sorted(fused, key=lambda i: -fused[i])
        ranks = {"bm25": bm25_rank, "dense": dense_rank, "hybrid": hybrid_rank}

        for m in methods:
            osc_by_k = {}
            for k in KS:
                base = cells_of_chunks(ot, R, ranks[m][:k])
                tcells = tcells_for(base)
                if m == "hybrid" and k == args.aug_k:
                    aug_cells_added += len(tcells - base)
                aug = base | tcells
                ob = operand_set_completeness(gold, base)
                oa = operand_set_completeness(gold, aug)
                osc_by_k[("p", k)] = ob
                osc_by_k[("a", k)] = oa
                acc[m][k]["osc_p"] += ob
                acc[m][k]["osc_a"] += oa
                acc[m][k]["c_p"] += len(base)
                acc[m][k]["c_a"] += len(aug)
            paired[m].append((osc_by_k[("a", AUG_K)], osc_by_k[("p", PLAIN_K)]))
            # strict cell-matched plain: smallest plain k' giving >= aug@AUG_K cells
            base_aug = cells_of_chunks(ot, R, ranks[m][:AUG_K])
            cells_aug_q = len(base_aug | tcells_for(base_aug))
            osc_p_matched, kp_used = None, PLAIN_GRID[-1]
            for kp in PLAIN_GRID:
                cp = cells_of_chunks(ot, R, ranks[m][:kp])
                if len(cp) >= cells_aug_q:
                    osc_p_matched, kp_used = operand_set_completeness(gold, cp), kp
                    break
            if osc_p_matched is None:  # plain maxes out below aug's budget -> give plain top of grid
                cp = cells_of_chunks(ot, R, ranks[m][:PLAIN_GRID[-1]])
                osc_p_matched = operand_set_completeness(gold, cp)
            cellmatch[m].append((osc_by_k[("a", AUG_K)], osc_p_matched, cells_aug_q, kp_used))
            samedepth[m].append((osc_by_k[("a", AUG_K)], osc_by_k[("p", AUG_K)]))

    out = {"population": {"name": "arithmetic_m>=2", "n": n},
           "ratio_only_aug": args.ratio_only_aug,
           "config": {"resolver_cols": args.resolver_cols,
                      "cross_encoder": args.cross_encoder if args.resolver_cols else None,
                      "top_n_cross": args.top_n_cross if args.resolver_cols else None,
                      "total_cols_only": args.total_cols_only},
           "mean_total_cells_injected": round(aug_cells_added / n, 1),
           "methods": {}}
    for m in methods:
        out["methods"][m] = {}
        for k in KS:
            a = acc[m][k]
            out["methods"][m][f"@{k}"] = {
                "osc_plain": round(a["osc_p"] / n, 4),
                "osc_aug": round(a["osc_a"] / n, 4),
                "delta": round((a["osc_a"] - a["osc_p"]) / n, 4),
                "cells_plain": round(a["c_p"] / n, 1),
                "cells_aug": round(a["c_a"] / n, 1),
            }
    # matched-budget paired significance: aug@AUG_K vs plain@PLAIN_K (aug uses fewer cells)
    from scipy.stats import binomtest
    out["matched_budget_test"] = {"aug_k": AUG_K, "plain_k": PLAIN_K, "methods": {}}
    for m in methods:
        pr = paired[m]
        aug_win = sum(1 for a, p in pr if a > p)
        plain_win = sum(1 for a, p in pr if p > a)
        d = (sum(a for a, _ in pr) - sum(p for _, p in pr)) / n
        pv = binomtest(aug_win, aug_win + plain_win, 0.5).pvalue if (aug_win + plain_win) else 1.0
        out["matched_budget_test"]["methods"][m] = {
            "osc_aug": round(sum(a for a, _ in pr) / n, 4),
            "osc_plain": round(sum(p for _, p in pr) / n, 4),
            "delta": round(d, 4), "aug_only": aug_win, "plain_only": plain_win,
            "mcnemar_p": round(float(pv), 5)}

    # same-depth paired test at AUG_K (injection is a strict superset -> hurt is 0 by construction)
    out["same_depth_test"] = {"k": AUG_K, "methods": {}}
    for m in methods:
        sd = samedepth[m]
        flip = sum(1 for a, p in sd if a > p)
        hurt = sum(1 for a, p in sd if p > a)
        pv = binomtest(flip, flip + hurt, 0.5).pvalue if (flip + hurt) else 1.0
        out["same_depth_test"]["methods"][m] = {
            "osc_plain": round(sum(p for _, p in sd) / n, 4),
            "osc_aug": round(sum(a for a, _ in sd) / n, 4),
            "delta": round((sum(a for a, _ in sd) - sum(p for _, p in sd)) / n, 4),
            "flipped": flip, "hurt": hurt, "mcnemar_p": round(float(pv), 6)}

    # strict per-query cell-matched paired test (plain gets >= aug's cell budget)
    out["cell_matched_test"] = {"aug_k": AUG_K, "note": "plain@k' with cells>=aug cells",
                                "methods": {}}
    for m in methods:
        cm = cellmatch[m]
        aug_win = sum(1 for a, p, _, _ in cm if a > p)
        plain_win = sum(1 for a, p, _, _ in cm if p > a)
        d = (sum(a for a, _, _, _ in cm) - sum(p for _, p, _, _ in cm)) / n
        pv = binomtest(aug_win, aug_win + plain_win, 0.5).pvalue if (aug_win + plain_win) else 1.0
        out["cell_matched_test"]["methods"][m] = {
            "osc_aug": round(sum(a for a, _, _, _ in cm) / n, 4),
            "osc_plain_matched": round(sum(p for _, p, _, _ in cm) / n, 4),
            "delta": round(d, 4), "aug_only": aug_win, "plain_only": plain_win,
            "cells_aug": round(sum(c for _, _, c, _ in cm) / n, 1),
            "plain_k_used": round(sum(k for _, _, _, k in cm) / n, 1),
            "mcnemar_p": round(float(pv), 5)}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"\nmean total-cells injected/query: {out['mean_total_cells_injected']}")
    print(f"\n{'method':<8}{'k':>4}{'OSC plain':>11}{'OSC +tot':>10}{'Δ':>8}"
          f"{'cells_p':>9}{'cells_a':>9}")
    for m in methods:
        for k in KS:
            r = out["methods"][m][f"@{k}"]
            print(f"{m:<8}{k:>4}{r['osc_plain']:>11.3f}{r['osc_aug']:>10.3f}"
                  f"{r['delta']:>+8.3f}{r['cells_plain']:>9.1f}{r['cells_aug']:>9.1f}")
    mb = out["matched_budget_test"]
    print(f"\n== matched-budget paired: aug@{mb['aug_k']} (fewer cells) vs plain@{mb['plain_k']} ==")
    print(f"{'method':<8}{'OSC aug':>9}{'OSC plain':>11}{'Δ':>8}{'aug>':>6}{'pl>':>5}{'p':>9}")
    for m in methods:
        t = mb["methods"][m]
        print(f"{m:<8}{t['osc_aug']:>9.3f}{t['osc_plain']:>11.3f}{t['delta']:>+8.3f}"
              f"{t['aug_only']:>6}{t['plain_only']:>5}{t['mcnemar_p']:>9.4f}")
    cm = out["cell_matched_test"]
    print(f"\n== STRICT cell-matched paired: aug@{cm['aug_k']} vs plain@k'(cells>=aug) ==")
    print(f"{'method':<8}{'OSC aug':>9}{'OSC pl':>9}{'Δ':>8}{'aug>':>6}{'pl>':>5}"
          f"{'c_aug':>7}{'pl_k':>6}{'p':>9}")
    for m in methods:
        t = cm["methods"][m]
        print(f"{m:<8}{t['osc_aug']:>9.3f}{t['osc_plain_matched']:>9.3f}{t['delta']:>+8.3f}"
              f"{t['aug_only']:>6}{t['plain_only']:>5}{t['cells_aug']:>7.1f}"
              f"{t['plain_k_used']:>6.1f}{t['mcnemar_p']:>9.4f}")
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
