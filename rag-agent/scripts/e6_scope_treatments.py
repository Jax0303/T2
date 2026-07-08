#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""E6 (task 2) — structural scope-selection treatments vs the hybrid enumeration.

The row-axis diagnosis (`scripts/diag_row_failures.py`,
`results/diag_row_failures_summary.md`) showed the residual row-axis bottleneck is
*not* sibling selection but **total-row pairing** (68% of failures: share/ratio
queries needing a table-level total the resolver can't name). This experiment
measures the diagnosis-driven treatments, each as a row augmentation over the SAME
hybrid-resolved scope, paired against the un-augmented hybrid enumeration:

  base          : hybrid enumeration (row=embed, col=lexical), no augmentation.
  T_total_all   : + every total-like row, unconditionally.
  T_total_ratio : + total-like rows only when the question reads as a share/ratio.
  T_subtree     : expand each matched row to its full sibling group.
  T_both        : ratio-gated total augmentation + sibling expansion.

Reports per arm: OSC, OSC|decomposition-correct, row/col-axis coverage, mean cells
(precision proxy), and paired ΔOSC vs base (bootstrap 95% CI + McNemar). LLM-free.
Population: HiTab dev arithmetic, distinct-cell scope m≥2.

Run: PYTHONPATH=. python scripts/e6_scope_treatments.py --split dev
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.bench.hitab import load_queries
from rag_agent.data.loader import load_table
from rag_agent.eval.operand_set import (
    covered_gold_cells, operand_set_completeness, per_cell_recall,
)
from rag_agent.query.header_embed_resolver import EmbedResolver
from rag_agent.query.operand_decomposer import Embedder
from rag_agent.retrieve.header_enum import enumerate_scope, is_ratio_query
from rag_agent.retrieve.operand_retriever import HybridRetriever, retrieve
from rag_agent.serialize import S2, serialize_table
from rag_agent.stores.original_store import build_original_table

SEED = 42
ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}

# arm -> enumerate_scope kwargs factory (takes is_ratio flag)
ARMS = {
    "base":          lambda ratio: {},
    "T_total_all":   lambda ratio: {"add_total_rows": True},
    "T_total_ratio": lambda ratio: {"add_total_rows": ratio},
    "T_subtree":     lambda ratio: {"expand_siblings": True},
    "T_both":        lambda ratio: {"add_total_rows": ratio, "expand_siblings": True},
}


def _bootstrap_diff_ci(pairs, n_boot=2000, seed=SEED):
    if not pairs:
        return [float("nan")] * 2
    rng = random.Random(seed)
    n = len(pairs)
    diffs = []
    for _ in range(n_boot):
        s = sum(a - b for a, b in (pairs[rng.randrange(n)] for _ in range(n)))
        diffs.append(s / n)
    diffs.sort()
    return [round(diffs[int(0.025 * n_boot)], 4), round(diffs[int(0.975 * n_boot)], 4)]


def _mcnemar(pairs):
    b = sum(1 for a, base in pairs if a == 1 and base == 0)  # arm win
    c = sum(1 for a, base in pairs if a == 0 and base == 1)  # base win
    return {"arm_only": b, "base_only": c}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/e6_scope_treatments.json")
    ap.add_argument("--dense", action="store_true",
                    help="also compute the dense single-vector baseline (k=5,10) "
                         "and paired ΔOSC of each treatment vs dense k=10")
    args = ap.parse_args()

    queries, tables = load_queries(args.data_dir, args.split, args.max)
    pop = [q for q in queries
           if (q.aggregation or "none") in ARITH
           and len({(o.row, o.col) for o in q.gold_operands}) >= 2]
    print(f"[pop] arithmetic m>=2 (distinct cells): {len(pop)}")

    embedder = Embedder(args.embed_model, device=args.device)
    resolver = EmbedResolver(embedder, row_mode="embed", col_mode="lexical")
    ots = {tid: build_original_table(load_table(tid, args.data_dir))
           for tid in {q.gold_table_id for q in pop}}
    retr = {}
    if args.dense:
        print(f"[dense] building {len(ots)} retrievers ...")
        retr = {tid: HybridRetriever(serialize_table(tables[tid], S2), embedder)
                for tid in ots}

    # per-query, per-arm records
    recs = {a: [] for a in ARMS}
    dense = {"dense_k5": [], "dense_k10": []} if args.dense else {}
    for i, q in enumerate(pop):
        ot = ots[q.gold_table_id]
        gold = q.gold_operands
        gold_rows = {o.row for o in gold}
        gold_cols = {o.col for o in gold}
        ratio = is_ratio_query(q.question)
        intent = resolver.resolve(q.question, ot)
        for arm, kw in ARMS.items():
            enum = enumerate_scope(ot, intent.row_paths, intent.col_paths, **kw(ratio))
            recs[arm].append({
                "osc": operand_set_completeness(gold, enum.cells),
                "pcr": per_cell_recall(gold, enum.cells),
                "cells": len(enum.cells),
                "row_cov": int(gold_rows <= enum.rows),
                "col_cov": int(gold_cols <= enum.cols),
            })
        if args.dense:
            res = retrieve(q.question, tables[q.gold_table_id], gold, mode="plain",
                           k=10, scheme=S2, embedder=embedder,
                           retriever=retr[q.gold_table_id])
            for k, key in ((5, "dense_k5"), (10, "dense_k10")):
                covered = covered_gold_cells(gold, res.retrieved[:k])
                dense[key].append({"osc": operand_set_completeness(gold, covered)})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(pop)}")

    n = len(pop)
    base = recs["base"]
    out = {"experiment": "E6_scope_treatments", "split": args.split, "seed": SEED,
           "population": {"name": "arithmetic_m>=2", "n": n},
           "ratio_query_rate": round(sum(is_ratio_query(q.question) for q in pop) / n, 4),
           "arms": {}}
    for arm, rs in recs.items():
        dec_ok = [r for r in rs if r["row_cov"] and r["col_cov"]]
        pairs = [(rs[i]["osc"], base[i]["osc"]) for i in range(n)]
        out["arms"][arm] = {
            "osc": round(sum(r["osc"] for r in rs) / n, 4),
            "pcr": round(sum(r["pcr"] for r in rs) / n, 4),
            "mean_cells": round(sum(r["cells"] for r in rs) / n, 1),
            "row_cov": round(sum(r["row_cov"] for r in rs) / n, 4),
            "col_cov": round(sum(r["col_cov"] for r in rs) / n, 4),
            "n_decomp_correct": len(dec_ok),
            "osc_given_decomp": round(sum(r["osc"] for r in dec_ok) / len(dec_ok), 4)
                                if dec_ok else None,
            "delta_osc_vs_base": round((sum(r["osc"] for r in rs) - sum(r["osc"] for r in base)) / n, 4),
            "delta_ci95_vs_base": _bootstrap_diff_ci(pairs) if arm != "base" else None,
            "mcnemar_vs_base": _mcnemar(pairs) if arm != "base" else None,
        }
        if args.dense:
            dk10 = dense["dense_k10"]
            dpairs = [(rs[i]["osc"], dk10[i]["osc"]) for i in range(n)]
            out["arms"][arm]["delta_osc_vs_dense_k10"] = round(
                (sum(r["osc"] for r in rs) - sum(r["osc"] for r in dk10)) / n, 4)
            out["arms"][arm]["delta_ci95_vs_dense_k10"] = _bootstrap_diff_ci(dpairs)
            out["arms"][arm]["mcnemar_vs_dense_k10"] = _mcnemar(dpairs)
    if args.dense:
        out["dense_baseline"] = {
            "osc_k5": round(sum(r["osc"] for r in dense["dense_k5"]) / n, 4),
            "osc_k10": round(sum(r["osc"] for r in dense["dense_k10"]) / n, 4),
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    # compact console table
    print(f"\n{'arm':<15}{'OSC':>7}{'ΔvsBase':>9}{'cells':>7}{'rowcov':>8}{'colcov':>8}{'OSC|dec':>9}")
    for arm, a in out["arms"].items():
        dv = a["delta_osc_vs_base"]
        d = ("%+.3f" % dv) if arm != "base" else "  --  "
        print(f"{arm:<15}{a['osc']:>7.3f}{d:>9}{a['mean_cells']:>7.1f}"
              f"{a['row_cov']:>8.3f}{a['col_cov']:>8.3f}{str(a['osc_given_decomp']):>9}")
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
