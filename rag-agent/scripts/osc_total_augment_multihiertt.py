#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""OSC generalization test: frozen total-row injection on MultiHiertt.

Everything tuned on HiTab dev is FROZEN here (bge-reranker-base column resolver,
top-2 columns, injection at k=10, total_like_rows heuristic, seed-free LLM-free
protocol); MultiHiertt is a pure confirmation dataset. Population: arithmetic,
table-only evidence, single-table, m>=2 distinct evidence cells, every evidence
cell value-validated against the dataset's own table_description (~99%).

Reports:
  1. diagnosis-lite — share of gold operands in total-like rows, share of
     queries needing >=1 total-row operand, reach of those operands @10;
  2. plain vs +injection OSC for bm25/dense/hybrid at k in (5,10,20,40);
  3. same_depth_test @10 and STRICT cell_matched_test (the headline test),
     identical to scripts/osc_total_augment.py.

Run: PYTHONPATH=. /usr/bin/python3 scripts/osc_total_augment_multihiertt.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from rag_agent.bench.multihiertt import load_queries
from rag_agent.eval.operand_set import operand_set_completeness
from rag_agent.retrieve.header_enum import total_like_rows, is_ratio_query
from rag_agent.retrieve.operand_retriever import HybridRetriever, _tok
from rag_agent.query.operand_decomposer import Embedder
from rag_agent.serialize import S2, serialize_table
from rag_agent.stores.original_store import OriginalTable

KS = (5, 10, 20, 40)
PLAIN_GRID = tuple(range(1, 61))


def to_original(bt) -> OriginalTable:
    return OriginalTable(table_id=bt.table_id, title=bt.title, data=bt.data,
                         top_paths=bt.top_paths, left_paths=bt.left_paths)


def numeric_cells(ot, rows, cols):
    return {(r, c) for r in rows for c in cols if ot.cell_num(r, c) is not None}


def cells_of_chunks(ot, retriever, idxs):
    out = set()
    for i in idxs:
        ch = retriever.chunks[i]
        out |= numeric_cells(ot, ch.rows, ch.cols)
    return out


def mcnemar_p(w, l):
    from scipy.stats import binomtest
    return float(binomtest(w, w + l, 0.5).pvalue) if (w + l) else 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/multihiertt")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--cross-encoder", default="BAAI/bge-reranker-base")
    ap.add_argument("--top-n-cross", type=int, default=2)
    ap.add_argument("--aug-k", type=int, default=10)
    ap.add_argument("--out", default="results/osc_total_augment_multihiertt.json")
    args = ap.parse_args()

    pop, tables, stats = load_queries(args.data_dir, args.split)
    n = len(pop)
    print(f"[pop] MultiHiertt {args.split} arithmetic single-table m>=2 "
          f"validated: {n}")
    print(f"[gold-fidelity] evidence cells value-matched: "
          f"{stats['cells_matched']}/{stats['cells_checked']} "
          f"({stats['cells_matched']/stats['cells_checked']:.3f})")

    emb = Embedder(args.embed_model, device="cpu")
    from sentence_transformers import CrossEncoder
    from rag_agent.query.header_embed_resolver import EmbedResolver
    col_resolver = EmbedResolver(emb, col_mode="cross",
                                 cross_encoder=CrossEncoder(args.cross_encoder),
                                 top_n_cross=args.top_n_cross)

    retr, ots, trows_by_t = {}, {}, {}
    for tid, bt in tables.items():
        retr[tid] = HybridRetriever(serialize_table(bt, S2), emb)
        ot = to_original(bt)
        ots[tid] = ot
        trows_by_t[tid] = total_like_rows(ot)

    methods = ("bm25", "dense", "hybrid")
    K = args.aug_k
    acc = {m: {k: dict(osc_p=0.0, osc_a=0.0, c_p=0, c_a=0) for k in KS}
           for m in methods}
    samedepth = {m: [] for m in methods}
    cellmatch = {m: [] for m in methods}
    reach = {m: dict(plain=0, inj=0) for m in methods}
    inj_cells_sum = 0

    # diagnosis-lite over gold operands
    n_gold = sum(len({(o.row, o.col) for o in q.gold_operands}) for q in pop)
    n_gold_total = 0
    n_need_total = 0
    n_ratio_q = sum(1 for q in pop if is_ratio_query(q.question))

    for q in pop:
        ot = ots[q.gold_table_id]
        R = retr[q.gold_table_id]
        gold = q.gold_operands
        trows = trows_by_t[q.gold_table_id]
        gold_cells = {(o.row, o.col) for o in gold}
        gold_total = {(r, c) for (r, c) in gold_cells if r in trows}
        n_gold_total += len(gold_total)
        needs_total = bool(gold_total)
        n_need_total += needs_total

        qv = np.asarray(emb.encode([q.question])[0])
        bm25_rank = R._rank(R._bm25.get_scores(_tok(q.question)))
        dense_rank = (R._rank(np.asarray(R._emb) @ qv)
                      if R._emb is not None else bm25_rank)
        fused = {}
        for rank, i in enumerate(bm25_rank):
            fused[i] = fused.get(i, 0.0) + 1.0 / (R.rrf_k + rank)
        for rank, i in enumerate(dense_rank):
            fused[i] = fused.get(i, 0.0) + 1.0 / (R.rrf_k + rank)
        hybrid_rank = sorted(fused, key=lambda i: -fused[i])
        ranks = {"bm25": bm25_rank, "dense": dense_rank, "hybrid": hybrid_rank}

        # frozen column-resolver injection cells
        intent = col_resolver.resolve(q.question, ot, col_allowed=None)
        cidx = set()
        for p in intent.col_paths:
            cidx.update(ot.find_cols_by_header(" > ".join(p)))
        inj_tcells = {(r, c) for r in trows for c in cidx
                      if ot.cell_num(r, c) is not None}

        for m in methods:
            osc_by_k = {}
            for k in KS:
                base = cells_of_chunks(ot, R, ranks[m][:k])
                aug = base | inj_tcells
                if m == "hybrid" and k == K:
                    inj_cells_sum += len(aug - base)
                ob = operand_set_completeness(gold, base)
                oa = operand_set_completeness(gold, aug)
                osc_by_k[("p", k)], osc_by_k[("a", k)] = ob, oa
                acc[m][k]["osc_p"] += ob
                acc[m][k]["osc_a"] += oa
                acc[m][k]["c_p"] += len(base)
                acc[m][k]["c_a"] += len(aug)
            samedepth[m].append((osc_by_k[("a", K)], osc_by_k[("p", K)]))
            base_k = cells_of_chunks(ot, R, ranks[m][:K])
            if needs_total:
                reach[m]["plain"] += gold_total <= base_k
                reach[m]["inj"] += gold_total <= (base_k | inj_tcells)
            cells_aug_q = len(base_k | inj_tcells)
            osc_p_matched, kp_used = None, PLAIN_GRID[-1]
            for kp in PLAIN_GRID:
                cp = cells_of_chunks(ot, R, ranks[m][:kp])
                if len(cp) >= cells_aug_q:
                    osc_p_matched, kp_used = operand_set_completeness(gold, cp), kp
                    break
            if osc_p_matched is None:
                cp = cells_of_chunks(ot, R, ranks[m][:PLAIN_GRID[-1]])
                osc_p_matched = operand_set_completeness(gold, cp)
            cellmatch[m].append((osc_by_k[("a", K)], osc_p_matched,
                                 cells_aug_q, kp_used))

    out = {"population": {"name": "multihiertt_arith_single_table_m>=2",
                          "n": n, "split": args.split, "funnel": stats},
           "config": {"frozen_from": "HiTab dev (osc_total_augment defaults)",
                      "cross_encoder": args.cross_encoder,
                      "top_n_cross": args.top_n_cross, "aug_k": K},
           "diagnosis": {
               "gold_operands": n_gold,
               "gold_operands_in_total_rows": n_gold_total,
               "total_row_share": round(n_gold_total / n_gold, 4),
               "queries_needing_total": n_need_total,
               "queries_needing_total_share": round(n_need_total / n, 4),
               "ratio_cue_queries": n_ratio_q},
           "mean_total_cells_injected": round(inj_cells_sum / n, 1),
           "methods": {}}

    print(f"\n[diag] gold operands in total-like rows: "
          f"{n_gold_total}/{n_gold} ({n_gold_total/n_gold:.3f}); "
          f"queries needing a total-row operand: {n_need_total}/{n} "
          f"({n_need_total/n:.3f}); ratio-cue queries: {n_ratio_q}")
    print(f"mean total-cells injected/query: {inj_cells_sum/n:.1f}\n")

    print(f"{'method':8}{'k':>4}{'OSC plain':>11}{'OSC +tot':>10}{'d':>9}"
          f"{'cells_p':>9}{'cells_a':>9}")
    for m in methods:
        out["methods"][m] = {}
        for k in KS:
            a = acc[m][k]
            out["methods"][m][f"@{k}"] = {
                "osc_plain": round(a["osc_p"] / n, 4),
                "osc_aug": round(a["osc_a"] / n, 4),
                "delta": round((a["osc_a"] - a["osc_p"]) / n, 4),
                "cells_plain": round(a["c_p"] / n, 1),
                "cells_aug": round(a["c_a"] / n, 1)}
            print(f"{m:8}{k:4}{a['osc_p']/n:11.3f}{a['osc_a']/n:10.3f}"
                  f"{(a['osc_a']-a['osc_p'])/n:+9.3f}{a['c_p']/n:9.1f}"
                  f"{a['c_a']/n:9.1f}")

    out["same_depth_test"] = {"k": K, "methods": {}}
    print(f"\n== same-depth paired k={K} ==")
    print(f"{'method':8}{'OSC plain':>10}{'OSC aug':>9}{'d':>9}{'flip':>6}"
          f"{'hurt':>6}{'p':>10}")
    for m in methods:
        sd = samedepth[m]
        flip = sum(1 for a, p in sd if a > p)
        hurt = sum(1 for a, p in sd if p > a)
        pv = mcnemar_p(flip, hurt)
        op_, oa_ = sum(p for _, p in sd) / n, sum(a for a, _ in sd) / n
        out["same_depth_test"]["methods"][m] = {
            "osc_plain": round(op_, 4), "osc_aug": round(oa_, 4),
            "delta": round(oa_ - op_, 4), "flipped": flip, "hurt": hurt,
            "mcnemar_p": round(pv, 6)}
        print(f"{m:8}{op_:10.3f}{oa_:9.3f}{oa_-op_:+9.3f}{flip:6}{hurt:6}"
              f"{pv:10.4f}")

    out["cell_matched_test"] = {"aug_k": K,
                                "note": "plain@k' with cells>=aug cells",
                                "methods": {}}
    print(f"\n== STRICT cell-matched paired: aug@{K} vs plain@k'(cells>=aug) ==")
    print(f"{'method':8}{'OSC aug':>8}{'OSC pl':>8}{'d':>9}{'aug>':>6}{'pl>':>5}"
          f"{'c_aug':>7}{'pl_k':>6}{'p':>10}")
    for m in methods:
        cm = cellmatch[m]
        w = sum(1 for a, p, _, _ in cm if a > p)
        l = sum(1 for a, p, _, _ in cm if p > a)
        pv = mcnemar_p(w, l)
        oa_ = sum(a for a, _, _, _ in cm) / n
        op_ = sum(p for _, p, _, _ in cm) / n
        out["cell_matched_test"]["methods"][m] = {
            "osc_aug": round(oa_, 4), "osc_plain_matched": round(op_, 4),
            "delta": round(oa_ - op_, 4), "aug_only": w, "plain_only": l,
            "cells_aug": round(sum(c for _, _, c, _ in cm) / n, 1),
            "plain_k_used": round(sum(k for _, _, _, k in cm) / n, 1),
            "mcnemar_p": round(pv, 5)}
        print(f"{m:8}{oa_:8.3f}{op_:8.3f}{oa_-op_:+9.3f}{w:6}{l:5}"
              f"{sum(c for _, _, c, _ in cm)/n:7.1f}"
              f"{sum(k for _, _, _, k in cm)/n:6.1f}{pv:10.4f}")

    out["total_operand_reach_at_k"] = {
        "k": K, "n_queries_needing_total": n_need_total,
        "methods": {m: ({kk: round(v / n_need_total, 4)
                         for kk, v in reach[m].items()} if n_need_total else {})
                    for m in methods}}
    if n_need_total:
        print(f"\n== reach of required total-row operands @{K} "
              f"(n={n_need_total}) ==")
        for m in methods:
            print(f"{m:8} plain {reach[m]['plain']/n_need_total:.3f}   "
                  f"injection {reach[m]['inj']/n_need_total:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
