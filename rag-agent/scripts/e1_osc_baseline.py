#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""E1 (W3) — operand-set completeness collapse curve for the dense baseline.

Tests H1: a dense single-vector cell/chunk retriever's OSC degrades as the
aggregation scope size m grows. We run the repo's ``mode="plain"`` dense
retrieval (the single-vector baseline) over HiTab arithmetic-aggregation queries
and report OSC stratified by scope size m and by retrieval budget k.

Population (per the gold-population decision): arithmetic aggregations
(sum/diff/div/average/range/...) with resolved operands. m>=2 is the primary
OSC population; m==1 is kept as the curve's anchor.

Outputs ``results/e1_osc_baseline.json``. LLM-free.

Run:
    PYTHONPATH=. python scripts/e1_osc_baseline.py --split dev
    PYTHONPATH=. python scripts/e1_osc_baseline.py --split dev --no-dense   # BM25 only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.bench.hitab import load_queries
from rag_agent.eval.operand_set import (
    bin_scope, covered_gold_cells, operand_set_completeness, per_cell_recall,
)
from rag_agent.query.operand_decomposer import Embedder
from rag_agent.retrieve.operand_retriever import HybridRetriever, retrieve
from rag_agent.serialize import S2, serialize_table

SEED = 42
ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}
K_LIST = (1, 3, 5, 10, 20)


def _bootstrap_ci(values, n_boot=2000, seed=SEED):
    if not values:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        s = sum(values[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    return [round(means[int(0.025 * n_boot)], 4), round(means[int(0.975 * n_boot)], 4)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--no-dense", action="store_true", help="BM25-only baseline")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/e1_osc_baseline.json")
    args = ap.parse_args()

    queries, tables = load_queries(args.data_dir, args.split, args.max)
    # arithmetic population with resolved operands
    pop = [q for q in queries
           if (q.aggregation or "none") in ARITH and len(q.gold_operands) >= 1]
    print(f"[pop] arithmetic w/ operands: {len(pop)} "
          f"(m>=2: {sum(1 for q in pop if len(q.gold_operands) >= 2)})")

    embedder = None if args.no_dense else Embedder(args.embed_model, device=args.device)
    print(f"[index] dense={embedder is not None}; building retrievers ...")
    retr = {}
    needed = {q.gold_table_id for q in pop}
    for tid in needed:
        retr[tid] = HybridRetriever(serialize_table(tables[tid], S2), embedder)

    max_k = max(K_LIST)
    # records[k] = list of {osc, per_cell_recall, m}
    records = {k: [] for k in K_LIST}
    per_scope_osc = {k: defaultdict(list) for k in K_LIST}  # k -> m -> [osc]
    use_dense = embedder is not None
    for i, q in enumerate(pop):
        tab = tables[q.gold_table_id]
        res = retrieve(q.question, tab, q.gold_operands, mode="plain",
                       k=max_k, scheme=S2, embedder=embedder, retriever=retr[q.gold_table_id])
        ranked = res.retrieved  # ranked chunks, len<=max_k
        m = len({(o.row, o.col) for o in q.gold_operands})
        for k in K_LIST:
            covered = covered_gold_cells(q.gold_operands, ranked[:k])
            osc = operand_set_completeness(q.gold_operands, covered)
            pcr = per_cell_recall(q.gold_operands, covered)
            records[k].append({"osc": osc, "per_cell_recall": pcr, "m": m})
            per_scope_osc[k][m].append(osc)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(pop)}")

    # --- aggregate ---------------------------------------------------------
    def agg(recs):
        n = len(recs)
        osc = [r["osc"] for r in recs]
        return {"n": n,
                "osc": round(sum(osc) / n, 4) if n else None,
                "osc_ci95": _bootstrap_ci(osc),
                "per_cell_recall": round(sum(r["per_cell_recall"] for r in recs) / n, 4) if n else None}

    by_budget = {str(k): agg(records[k]) for k in K_LIST}
    by_budget_m2 = {str(k): agg([r for r in records[k] if r["m"] >= 2]) for k in K_LIST}

    # OSC vs scope-size bin at each budget
    by_scope = {}
    for k in K_LIST:
        bins = defaultdict(list)
        for r in records[k]:
            bins[bin_scope(r["m"])].append(r["osc"])
        by_scope[str(k)] = {b: {"n": len(v), "osc": round(sum(v) / len(v), 4)}
                            for b, v in sorted(bins.items())}

    # r^k fit: per-cell recall r (from m==1 or overall) vs observed OSC by exact m
    # use largest budget for the cleanest curve
    kmax = max(K_LIST)
    exact = {m: round(sum(v) / len(v), 4) for m, v in sorted(per_scope_osc[kmax].items())}
    # estimate single-cell hit-rate r from per-cell recall at kmax (independence model)
    r_hat = by_budget[str(kmax)]["per_cell_recall"]
    rk_pred = {str(m): round(r_hat ** m, 4) for m in exact} if r_hat else {}

    out = {
        "experiment": "E1_osc_baseline",
        "hypothesis": "H1: dense single-vector OSC collapses as scope size m grows",
        "split": args.split,
        "seed": SEED,
        "dense": use_dense,
        "embed_model": None if args.no_dense else args.embed_model,
        "population": {
            "name": "arithmetic_aggregations",
            "n_total": len(pop),
            "n_m_ge_2": sum(1 for q in pop if len(q.gold_operands) >= 2),
        },
        "osc_by_budget": by_budget,
        "osc_by_budget_m_ge_2": by_budget_m2,
        "osc_by_scope_bin": by_scope,
        "rk_fit_at_kmax": {
            "k": kmax,
            "r_hat_per_cell": r_hat,
            "osc_observed_by_exact_m": exact,
            "osc_rk_predicted": rk_pred,
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: out[k] for k in
                      ("population", "osc_by_budget_m_ge_2", "osc_by_scope_bin", "rk_fit_at_kmax")},
                     indent=2))
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
