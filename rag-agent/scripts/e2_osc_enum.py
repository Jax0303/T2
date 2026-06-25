#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""E2 (W4) — header-tree enumeration OSC vs dense baseline (H2).

Paired comparison on the same queries:
  * baseline = dense single-vector retrieval (mode="plain") at budget k.
  * treatment = deterministic header-tree scope enumeration (header_enum).

Reports OSC for both, paired ΔOSC (bootstrap 95% CI + McNemar), the effective
retrieval-set size of each, and the decomposition diagnostic (row-axis / col-axis
coverage rates) so enumeration failure is separated from decomposer failure.

Population: HiTab dev arithmetic aggregations with operands; m>=2 primary (n=158).
LLM-free. Run:
    PYTHONPATH=. python scripts/e2_osc_enum.py --split dev
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.bench.hitab import load_queries
from rag_agent.data.loader import load_table
from rag_agent.eval.operand_set import (
    bin_scope, covered_gold_cells, operand_set_completeness, per_cell_recall,
)
from rag_agent.query.header_path_resolver import resolve_against_table, resolve_intent
from rag_agent.query.header_embed_resolver import EmbedResolver
from rag_agent.query.operand_decomposer import Embedder
from rag_agent.retrieve.header_enum import enumerate_scope
from rag_agent.retrieve.operand_retriever import HybridRetriever, retrieve
from rag_agent.serialize import S2, serialize_table
from rag_agent.stores.original_store import build_original_table

SEED = 42
ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}
BASELINE_K = (5, 10)


def _bootstrap_diff_ci(pairs, n_boot=2000, seed=SEED):
    """Paired bootstrap CI of mean(enum - base)."""
    if not pairs:
        return [float("nan")] * 2
    rng = random.Random(seed)
    n = len(pairs)
    diffs = []
    for _ in range(n_boot):
        s = 0
        for _ in range(n):
            e, b = pairs[rng.randrange(n)]
            s += e - b
        diffs.append(s / n)
    diffs.sort()
    return [round(diffs[int(0.025 * n_boot)], 4), round(diffs[int(0.975 * n_boot)], 4)]


def _mcnemar(pairs):
    """Discordant pairs (enum wins / base wins) + two-sided exact-ish p via b,c."""
    b = sum(1 for e, ba in pairs if e == 1 and ba == 0)  # enum win
    c = sum(1 for e, ba in pairs if e == 0 and ba == 1)  # base win
    return {"enum_only": b, "base_only": c, "n_discordant": b + c}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--llm", default=None,
                    help="W4b: refine decomposition via LLM, e.g. groq:llama-3.1-8b-instant")
    ap.add_argument("--resolver", default="deterministic",
                    choices=["deterministic", "embed", "hybrid"],
                    help="header-path resolver: lexical fuzzy (default), semantic tree-node "
                         "embedding, or hybrid (row=embed, col=lexical)")
    ap.add_argument("--out", default="results/e2_osc_enum.json")
    args = ap.parse_args()

    llm = None
    if args.llm:
        from rag_agent.llm.factory import build_llm
        llm = build_llm(args.llm)
        print(f"[llm] decomposition refinement via {args.llm}")

    queries, tables = load_queries(args.data_dir, args.split, args.max)
    pop = [q for q in queries
           if (q.aggregation or "none") in ARITH and len(q.gold_operands) >= 1]
    print(f"[pop] arithmetic w/ operands: {len(pop)} "
          f"(m>=2: {sum(1 for q in pop if len(q.gold_operands) >= 2)})")

    embedder = None if args.no_dense else Embedder(args.embed_model, device=args.device)
    embed_resolver = None
    if args.resolver in ("embed", "hybrid"):
        if embedder is None:
            ap.error(f"--resolver {args.resolver} requires an embedder (drop --no-dense)")
        if args.resolver == "hybrid":
            embed_resolver = EmbedResolver(embedder, row_mode="embed", col_mode="lexical")
            print("[resolver] hybrid (row=embed, col=lexical)")
        else:
            embed_resolver = EmbedResolver(embedder)
            print("[resolver] semantic tree-node embedding")
    needed = {q.gold_table_id for q in pop}
    print(f"[index] dense={embedder is not None}; building {len(needed)} retrievers + tables ...")
    retr, ots = {}, {}
    for tid in needed:
        retr[tid] = HybridRetriever(serialize_table(tables[tid], S2), embedder)
        raw = load_table(tid, args.data_dir)
        ots[tid] = build_original_table(raw)

    recs = []  # per-query records
    for i, q in enumerate(pop):
        tab = tables[q.gold_table_id]
        ot = ots[q.gold_table_id]
        gold = q.gold_operands
        m = len({(o.row, o.col) for o in gold})
        gold_rows = {o.row for o in gold}
        gold_cols = {o.col for o in gold}

        # --- treatment: enumeration over resolved scope ---
        if embed_resolver is not None:
            intent = embed_resolver.resolve(q.question, ot)        # semantic tree-node
        elif llm is not None:
            intent = resolve_intent(q.question, ot, llm=llm)       # W4b LLM refine
        else:
            intent = resolve_against_table(q.question, ot)         # lexical fuzzy
        enum = enumerate_scope(ot, intent.row_paths, intent.col_paths)
        osc_enum = operand_set_completeness(gold, enum.cells)
        pcr_enum = per_cell_recall(gold, enum.cells)
        row_cov = int(gold_rows <= enum.rows)
        col_cov = int(gold_cols <= enum.cols)

        rec = {
            "query_id": q.query_id, "m": m, "aggregation": q.aggregation,
            "decomp_src": intent.source,
            "osc_enum": osc_enum, "pcr_enum": pcr_enum,
            "enum_cells": len(enum.cells),
            "row_cov": row_cov, "col_cov": col_cov,
            "row_fallback": int(enum.row_fallback), "col_fallback": int(enum.col_fallback),
        }
        # --- baseline: dense plain at each k ---
        res = retrieve(q.question, tab, gold, mode="plain", k=max(BASELINE_K),
                       scheme=S2, embedder=embedder, retriever=retr[q.gold_table_id])
        for k in BASELINE_K:
            covered = covered_gold_cells(gold, res.retrieved[:k])
            rec[f"osc_base_k{k}"] = operand_set_completeness(gold, covered)
        recs.append(rec)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(pop)}")

    # --- aggregate over population subsets ---
    def summarize(subset, label):
        n = len(subset)
        if n == 0:
            return {"n": 0}
        osc_e = sum(r["osc_enum"] for r in subset) / n
        out = {
            "n": n,
            "osc_enum": round(osc_e, 4),
            "pcr_enum": round(sum(r["pcr_enum"] for r in subset) / n, 4),
            "mean_enum_cells": round(sum(r["enum_cells"] for r in subset) / n, 1),
            "row_axis_coverage": round(sum(r["row_cov"] for r in subset) / n, 4),
            "col_axis_coverage": round(sum(r["col_cov"] for r in subset) / n, 4),
            "row_fallback_rate": round(sum(r["row_fallback"] for r in subset) / n, 4),
            "col_fallback_rate": round(sum(r["col_fallback"] for r in subset) / n, 4),
        }
        # OSC | decomposition-correct (both axes cover gold) — isolates enum from decomposer
        dec_ok = [r for r in subset if r["row_cov"] and r["col_cov"]]
        out["n_decomp_correct"] = len(dec_ok)
        out["osc_given_decomp_correct"] = round(
            sum(r["osc_enum"] for r in dec_ok) / len(dec_ok), 4) if dec_ok else None
        for k in BASELINE_K:
            base = sum(r[f"osc_base_k{k}"] for r in subset) / n
            pairs = [(r["osc_enum"], r[f"osc_base_k{k}"]) for r in subset]
            out[f"osc_base_k{k}"] = round(base, 4)
            out[f"delta_osc_vs_k{k}"] = round(osc_e - base, 4)
            out[f"delta_ci95_vs_k{k}"] = _bootstrap_diff_ci(pairs)
            out[f"mcnemar_vs_k{k}"] = _mcnemar(pairs)
        return out

    all_pop = recs
    m2 = [r for r in recs if r["m"] >= 2]
    m1 = [r for r in recs if r["m"] == 1]
    by_scope = {b: summarize([r for r in recs if bin_scope(r["m"]) == b], b)
                for b in ("1", "2", "3-4", "5-8", "9+")}
    by_scope = {b: v for b, v in by_scope.items() if v.get("n")}

    out = {
        "experiment": "E2_osc_enum",
        "hypothesis": "H2: header-tree enumeration raises OSC vs similarity baseline",
        "split": args.split, "seed": SEED, "dense_baseline": embedder is not None,
        "decomp_source": args.llm or "deterministic",
        "population": {"name": "arithmetic_aggregations", "n_total": len(pop),
                       "n_m_ge_2": len(m2)},
        "decomp_source_dist": {s: sum(1 for r in recs if r["decomp_src"] == s)
                               for s in sorted({r["decomp_src"] for r in recs})},
        "primary_m_ge_2": summarize(m2, "m>=2"),
        "anchor_m_eq_1": summarize(m1, "m==1"),
        "all": summarize(all_pop, "all"),
        "by_scope_bin": by_scope,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({"primary_m_ge_2": out["primary_m_ge_2"]}, indent=2))
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
