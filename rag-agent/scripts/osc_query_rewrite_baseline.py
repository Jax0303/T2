#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Counterexample check for the structural-unreachability claim (PAPER 5.1b).

The cheapest reviewer objection to "similarity retrieval cannot reach unnamed total
rows by construction" is: *just rewrite the query* — append "total"/"overall" so the
total row's (near-empty) header lexically/semantically matches. If that closes the
gap, injection is merely query expansion in disguise; if it does not, the structural
claim stands on direct evidence.

Arms per retriever (bm25 / dense / hybrid), same population as osc_total_augment:
  plain     — original query
  rewrite   — query + " total overall" (variant `always`; variant `ratio` rewrites
              only ratio/share queries, mirroring when a denominator total is needed)
  injection — frozen config from osc_total_augment (bge-reranker resolver, top-2)

Tests:
  1. rewrite vs plain, same depth k=10 (paired; rewrite is NOT a superset — it can hurt)
  2. injection@10 vs rewrite@k' cell-matched (rewrite gets >= injection's cell budget)
  3. mechanism: reach rate of required total-row operands @10, plain vs rewrite vs inj

Run: PYTHONPATH=. /usr/bin/python3 scripts/osc_query_rewrite_baseline.py --split dev
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
from rag_agent.retrieve.header_enum import total_like_rows, is_ratio_query
from rag_agent.retrieve.operand_retriever import HybridRetriever, _tok
from rag_agent.query.operand_decomposer import Embedder
from rag_agent.serialize import S2, serialize_table
from rag_agent.stores.original_store import build_original_table

ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}
PLAIN_GRID = tuple(range(1, 61))
REWRITE_SUFFIX = " total overall"


def numeric_cells(ot, rows, cols):
    return {(r, c) for r in rows for c in cols if ot.cell_num(r, c) is not None}


def cells_of_chunks(ot, retriever, idxs):
    out = set()
    for i in idxs:
        ch = retriever.chunks[i]
        out |= numeric_cells(ot, ch.rows, ch.cols)
    return out


def rank_query(R, emb, question):
    qv = np.asarray(emb.encode([question])[0])
    bm25_rank = R._rank(R._bm25.get_scores(_tok(question)))
    dense_rank = R._rank(np.asarray(R._emb) @ qv) if R._emb is not None else bm25_rank
    fused = {}
    for rank, i in enumerate(bm25_rank):
        fused[i] = fused.get(i, 0.0) + 1.0 / (R.rrf_k + rank)
    for rank, i in enumerate(dense_rank):
        fused[i] = fused.get(i, 0.0) + 1.0 / (R.rrf_k + rank)
    hybrid_rank = sorted(fused, key=lambda i: -fused[i])
    return {"bm25": bm25_rank, "dense": dense_rank, "hybrid": hybrid_rank}


def mcnemar_p(wins, losses):
    from scipy.stats import binomtest
    return float(binomtest(wins, wins + losses, 0.5).pvalue) if (wins + losses) else 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--cross-encoder", default="BAAI/bge-reranker-base")
    ap.add_argument("--top-n-cross", type=int, default=2)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rewrite-variant", choices=("always", "ratio"), default="always")
    ap.add_argument("--out", default="results/osc_query_rewrite_baseline.json")
    args = ap.parse_args()

    queries, tables = load_queries(args.data_dir, args.split)
    pop = [q for q in queries
           if (q.aggregation or "none") in ARITH
           and len({(o.row, o.col) for o in q.gold_operands}) >= 2]
    n = len(pop)
    print(f"[pop] arithmetic m>=2: {n}  rewrite_variant={args.rewrite_variant}")

    emb = Embedder(args.embed_model, device="cpu")
    from sentence_transformers import CrossEncoder
    from rag_agent.query.header_embed_resolver import EmbedResolver
    col_resolver = EmbedResolver(emb, col_mode="cross",
                                 cross_encoder=CrossEncoder(args.cross_encoder),
                                 top_n_cross=args.top_n_cross)

    need = {q.gold_table_id for q in pop}
    retr, ots, total_rows = {}, {}, {}
    for tid in need:
        retr[tid] = HybridRetriever(serialize_table(tables[tid], S2), emb)
        ot = build_original_table(load_table(tid, args.data_dir))
        ots[tid] = ot
        total_rows[tid] = total_like_rows(ot)

    methods = ("bm25", "dense", "hybrid")
    K = args.k
    # per query per method: (osc_plain@K, osc_rw@K, osc_inj@K, osc_rw_cellmatched, ...)
    rw_vs_plain = {m: [] for m in methods}
    inj_vs_rw = {m: [] for m in methods}       # (osc_inj, osc_rw_matched, cells_inj, kp)
    reach = {m: dict(plain=0, rw=0, inj=0) for m in methods}
    n_total_need = 0
    n_rewritten = 0

    for q in pop:
        ot = ots[q.gold_table_id]
        R = retr[q.gold_table_id]
        gold = q.gold_operands
        trows = total_rows[q.gold_table_id]
        gold_cells = {(o.row, o.col) for o in gold}
        gold_total = {(r, c) for (r, c) in gold_cells if r in trows}
        needs_total = bool(gold_total)
        n_total_need += needs_total

        do_rewrite = args.rewrite_variant == "always" or is_ratio_query(q.question)
        n_rewritten += do_rewrite
        q_rw = q.question + REWRITE_SUFFIX if do_rewrite else q.question

        ranks = rank_query(R, emb, q.question)
        ranks_rw = rank_query(R, emb, q_rw) if do_rewrite else ranks

        # injection cells (frozen resolver config, same as osc_total_augment)
        intent = col_resolver.resolve(q.question, ot, col_allowed=None)
        cidx = set()
        for p in intent.col_paths:
            cidx.update(ot.find_cols_by_header(" > ".join(p)))
        inj_tcells = {(r, c) for r in trows for c in cidx
                      if ot.cell_num(r, c) is not None}

        for m in methods:
            base = cells_of_chunks(ot, R, ranks[m][:K])
            base_rw = cells_of_chunks(ot, R, ranks_rw[m][:K])
            base_inj = base | inj_tcells
            op = operand_set_completeness(gold, base)
            orw = operand_set_completeness(gold, base_rw)
            oinj = operand_set_completeness(gold, base_inj)
            rw_vs_plain[m].append((orw, op))
            if needs_total:
                reach[m]["plain"] += gold_total <= base
                reach[m]["rw"] += gold_total <= base_rw
                reach[m]["inj"] += gold_total <= base_inj
            # cell-matched: rewrite gets >= injection's cell budget (deeper k')
            cells_inj = len(base_inj)
            orw_matched, kp_used = None, PLAIN_GRID[-1]
            for kp in PLAIN_GRID:
                cp = cells_of_chunks(ot, R, ranks_rw[m][:kp])
                if len(cp) >= cells_inj:
                    orw_matched, kp_used = operand_set_completeness(gold, cp), kp
                    break
            if orw_matched is None:
                cp = cells_of_chunks(ot, R, ranks_rw[m][:PLAIN_GRID[-1]])
                orw_matched = operand_set_completeness(gold, cp)
            inj_vs_rw[m].append((oinj, orw_matched, cells_inj, kp_used))

    out = {"population": {"name": "arithmetic_m>=2", "n": n, "split": args.split},
           "config": {"k": K, "rewrite_variant": args.rewrite_variant,
                      "rewrite_suffix": REWRITE_SUFFIX, "n_rewritten": n_rewritten,
                      "cross_encoder": args.cross_encoder,
                      "top_n_cross": args.top_n_cross},
           "n_queries_needing_total": n_total_need}

    out["rewrite_vs_plain_same_depth"] = {"k": K, "methods": {}}
    print(f"\n== rewrite vs plain, same depth k={K} "
          f"(rewritten {n_rewritten}/{n} queries) ==")
    print(f"{'method':8}{'OSC plain':>10}{'OSC rw':>8}{'d':>8}{'rw>':>5}{'pl>':>5}{'p':>9}")
    for m in methods:
        pr = rw_vs_plain[m]
        w = sum(1 for a, p in pr if a > p)
        l = sum(1 for a, p in pr if p > a)
        osc_rw = sum(a for a, _ in pr) / n
        osc_p = sum(p for _, p in pr) / n
        pv = mcnemar_p(w, l)
        out["rewrite_vs_plain_same_depth"]["methods"][m] = {
            "osc_plain": round(osc_p, 4), "osc_rewrite": round(osc_rw, 4),
            "delta": round(osc_rw - osc_p, 4), "rw_only": w, "plain_only": l,
            "mcnemar_p": round(pv, 5)}
        print(f"{m:8}{osc_p:10.3f}{osc_rw:8.3f}{osc_rw-osc_p:+8.3f}{w:5}{l:5}{pv:9.4f}")

    out["injection_vs_rewrite_cell_matched"] = {"k": K, "methods": {}}
    print(f"\n== injection@{K} vs rewrite@k' (cells>=injection) ==")
    print(f"{'method':8}{'OSC inj':>8}{'OSC rw':>8}{'d':>8}{'inj>':>6}{'rw>':>5}"
          f"{'c_inj':>7}{'rw_k':>6}{'p':>9}")
    for m in methods:
        cm = inj_vs_rw[m]
        w = sum(1 for a, p, _, _ in cm if a > p)
        l = sum(1 for a, p, _, _ in cm if p > a)
        osc_i = sum(a for a, _, _, _ in cm) / n
        osc_r = sum(p for _, p, _, _ in cm) / n
        pv = mcnemar_p(w, l)
        out["injection_vs_rewrite_cell_matched"]["methods"][m] = {
            "osc_injection": round(osc_i, 4), "osc_rewrite_matched": round(osc_r, 4),
            "delta": round(osc_i - osc_r, 4), "inj_only": w, "rw_only": l,
            "cells_injection": round(sum(c for _, _, c, _ in cm) / n, 1),
            "rewrite_k_used": round(sum(k for _, _, _, k in cm) / n, 1),
            "mcnemar_p": round(pv, 5)}
        print(f"{m:8}{osc_i:8.3f}{osc_r:8.3f}{osc_i-osc_r:+8.3f}{w:6}{l:5}"
              f"{sum(c for _, _, c, _ in cm)/n:7.1f}{sum(k for _, _, _, k in cm)/n:6.1f}"
              f"{pv:9.4f}")

    out["total_operand_reach_at_k"] = {"k": K, "n_queries_needing_total": n_total_need,
                                       "methods": {}}
    print(f"\n== reach rate of required total-row operands @{K} "
          f"(n={n_total_need} queries needing totals) ==")
    print(f"{'method':8}{'plain':>7}{'rewrite':>9}{'injection':>11}")
    for m in methods:
        r = reach[m]
        row = {kk: round(v / n_total_need, 4) for kk, v in r.items()} if n_total_need else {}
        out["total_operand_reach_at_k"]["methods"][m] = row
        print(f"{m:8}{r['plain']/n_total_need:7.3f}{r['rw']/n_total_need:9.3f}"
              f"{r['inj']/n_total_need:11.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
