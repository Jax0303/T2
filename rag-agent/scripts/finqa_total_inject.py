#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Total-row injection on FinQA: does the HiTab OSC win transfer to financial tables?

Same mechanism as ``osc_total_augment.py`` (HiTab), ported to the registry so it runs
on FinQA's BenchTable objects. For each retriever (bm25 / dense / hybrid) over the gold
table's row-chunks, report operand-set completeness (OSC) plain vs +total-row injection
at budgets k. Population: FinQA validation, gold-operand count m>=2 (FinQA has no
arithmetic agg labels — every query is a "program" — so we select by scope, not label).

Basic all-column injection (no cross-encoder resolver), LLM-free.
Run: HF_HOME=data/hf_cache .venv/bin/python scripts/finqa_total_inject.py --bench finqa
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from rag_agent.bench import registry
from rag_agent.bench.schema import BenchTable
from rag_agent.eval.operand_set import operand_set_completeness
from rag_agent.retrieve.header_enum import total_like_rows
from rag_agent.retrieve.operand_retriever import HybridRetriever, _tok
from rag_agent.query.operand_decomposer import Embedder
from rag_agent.serialize import S2, serialize_table
from rag_agent.stores.original_store import _to_float

KS = (1, 2, 3, 5, 10, 20)

# BenchTable lacks cell_num (HiTab's OriginalTable has it); total_like_rows/numeric_cells
# need it. Add it via the shared _to_float parser so the structural logic is identical.
BenchTable.cell_num = lambda self, r, c: _to_float(self.cell(r, c))  # type: ignore[attr-defined]


def numeric_cells(t, rows, cols):
    return {(r, c) for r in rows for c in cols if t.cell_num(r, c) is not None}


def cells_of_chunks(t, retriever, idxs):
    out = set()
    for i in idxs:
        ch = retriever.chunks[i]
        out |= numeric_cells(t, ch.rows, ch.cols)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="finqa")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--aug-k", type=int, default=10)
    ap.add_argument("--plain-k", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/finqa_total_inject.json")
    args = ap.parse_args()

    queries, tables = registry.load(args.bench)
    pop = [q for q in queries
           if q.gold_table_id in tables
           and len({(o.row, o.col) for o in q.gold_operands}) >= 2]
    if args.limit:
        pop = pop[:args.limit]
    n = len(pop)
    sizes = sorted(tables[q.gold_table_id].n_rows for q in pop)
    med = sizes[len(sizes) // 2] if sizes else 0
    print(f"[pop] {args.bench} m>=2: {n}  gold-table rows: "
          f"min={sizes[0]} median={med} max={sizes[-1]}", flush=True)

    emb = Embedder(args.embed_model, device="cpu")
    need = {q.gold_table_id for q in pop}
    retr, total_rows, totals = {}, {}, {}
    for tid in need:
        t = tables[tid]
        retr[tid] = HybridRetriever(serialize_table(t, S2), emb)
        tr = total_like_rows(t)
        total_rows[tid] = tr
        totals[tid] = numeric_cells(t, tr, range(t.n_cols))

    methods = ("bm25", "dense", "hybrid")
    acc = {m: {k: dict(osc_p=0.0, osc_a=0.0, c_p=0, c_a=0) for k in KS} for m in methods}
    aug_cells_added = 0
    AUG_K, PLAIN_K = args.aug_k, args.plain_k
    paired = {m: [] for m in methods}

    for qi, q in enumerate(pop):
        t = tables[q.gold_table_id]
        R = retr[q.gold_table_id]
        gold = q.gold_operands
        all_tcells = totals[q.gold_table_id]

        qv = np.asarray(emb.encode([q.question])[0])
        bm25_rank = R._rank(R._bm25.get_scores(_tok(q.question)))
        dense_rank = R._rank(np.asarray(R._emb) @ qv) if R._emb is not None else bm25_rank
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
                base = cells_of_chunks(t, R, ranks[m][:k])
                tcells = all_tcells
                if m == "hybrid" and k == AUG_K:
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
        if (qi + 1) % 100 == 0:
            print(f"  {qi+1}/{n}", flush=True)

    out = {"population": {"name": f"{args.bench}_m>=2", "n": n},
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
            "binom_p": round(float(pv), 5)}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"\nmean total-cells injected/query: {out['mean_total_cells_injected']}")
    print(f"\n{'method':<8}{'k':>4}{'OSC plain':>11}{'OSC +tot':>10}{'Δ':>8}"
          f"{'cells_p':>9}{'cells_a':>9}")
    for m in methods:
        for k in KS:
            e = out["methods"][m][f"@{k}"]
            print(f"{m:<8}{k:>4}{e['osc_plain']:>11}{e['osc_aug']:>10}{e['delta']:>+8.4f}"
                  f"{e['cells_plain']:>9}{e['cells_aug']:>9}")
    print(f"\nmatched-budget (aug@{AUG_K} vs plain@{PLAIN_K}):")
    for m in methods:
        e = out["matched_budget_test"]["methods"][m]
        print(f"  {m:<8} aug={e['osc_aug']} plain={e['osc_plain']} Δ={e['delta']:+.4f} "
              f"(aug_only={e['aug_only']} plain_only={e['plain_only']} p={e['binom_p']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
