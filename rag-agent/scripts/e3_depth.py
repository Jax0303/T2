#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""E3 (W5) — causal isolation of header depth d (H2 causality).

Two complementary tests, holding data/vocab/scope fixed:

  (A) Observational: stratify enumeration OSC and the row/col-axis coverage
      (the E2 bottleneck) by header depth d, and by d × scope-size m to
      deconfound depth from scope size.

  (B) Synthetic depth control: leaf-flatten each table to depth 1 (keep only the
      leaf header token of every row/col path, dropping ancestor levels — same
      data, same leaf vocab, no tree), then re-resolve + re-enumerate. Paired
      comparison of row/col coverage and OSC at d_original vs d=1 on the same
      queries isolates the effect of the *tree* (ancestor levels) from domain.

The dense baseline (optional, --dense) is re-run on both the original and the
flattened tables so the baseline's depth-sensitivity can be contrasted with the
enumeration method's.

Population: HiTab dev arithmetic aggregations, m>=2. LLM-free.
Run:
    PYTHONPATH=. python scripts/e3_depth.py --split dev            # enum-only (fast)
    PYTHONPATH=. python scripts/e3_depth.py --split dev --dense    # + dense baseline
"""
from __future__ import annotations

import argparse
import copy
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
    bin_scope, covered_gold_cells, header_depth, operand_set_completeness,
)
from rag_agent.query.header_path_resolver import resolve_against_table
from rag_agent.query.header_embed_resolver import EmbedResolver
from rag_agent.query.operand_decomposer import Embedder
from rag_agent.retrieve.header_enum import enumerate_scope
from rag_agent.retrieve.operand_retriever import HybridRetriever, retrieve
from rag_agent.serialize import S2, serialize_table
from rag_agent.stores.original_store import build_original_table

SEED = 42
ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}


def _bin_depth(d: int) -> str:
    return "d1" if d <= 1 else ("d2" if d == 2 else "d3+")


def leaf_flatten_ot(ot):
    """Return a depth-1 copy: every header path -> [leaf token]. Data unchanged."""
    flat = copy.copy(ot)
    flat.top_paths = [[p[-1]] if p else [] for p in ot.top_paths]
    flat.left_paths = [[p[-1]] if p else [] for p in ot.left_paths]
    flat.top_paths_by_col = {c: [v[-1]] for c, v in ot.top_paths_by_col.items() if v}
    flat.left_paths_by_row = {r: [v[-1]] for r, v in ot.left_paths_by_row.items() if v}
    return flat


def _bench_from_ot(ot, source="hitab"):
    from rag_agent.bench.schema import BenchTable
    return BenchTable(
        table_id=ot.table_id, title=ot.title, data=ot.data,
        top_paths=[ot.col_path(c) for c in range(ot.n_cols)],
        left_paths=[ot.row_path(r) for r in range(ot.n_rows)], source=source)


def enum_record(ot, q, resolve_fn):
    intent = resolve_fn(q.question, ot)
    e = enumerate_scope(ot, intent.row_paths, intent.col_paths)
    gold = q.gold_operands
    return {
        "osc": operand_set_completeness(gold, e.cells),
        "row_cov": int({o.row for o in gold} <= e.rows),
        "col_cov": int({o.col for o in gold} <= e.cols),
        "cells": len(e.cells),
    }


def _agg(recs, keys):
    n = len(recs)
    if not n:
        return {"n": 0}
    out = {"n": n}
    for k in keys:
        out[k] = round(sum(r[k] for r in recs) / n, 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--dense", action="store_true", help="also run dense baseline (orig+flat)")
    ap.add_argument("--resolver", default="deterministic", choices=["deterministic", "embed"])
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/e3_depth.json")
    args = ap.parse_args()

    queries, tables = load_queries(args.data_dir, args.split, args.max)
    pop = [q for q in queries
           if (q.aggregation or "none") in ARITH and len(q.gold_operands) >= 2]
    print(f"[pop] arithmetic m>=2: {len(pop)}")

    need_embedder = args.dense or args.resolver == "embed"
    embedder = Embedder(args.embed_model, device=args.device) if need_embedder else None
    # original and flattened tables share a table_id, so the embed resolver (which
    # caches candidates per table_id) needs a separate instance per variant.
    if args.resolver == "embed":
        _er_o, _er_f = EmbedResolver(embedder), EmbedResolver(embedder)
        resolve_o = _er_o.resolve
        resolve_f = _er_f.resolve
    else:
        resolve_o = resolve_f = resolve_against_table
    ots, ots_flat, retr_o, retr_f = {}, {}, {}, {}
    for tid in {q.gold_table_id for q in pop}:
        ot = build_original_table(load_table(tid, args.data_dir))
        ots[tid] = ot
        ots_flat[tid] = leaf_flatten_ot(ot)
        if args.dense:
            retr_o[tid] = HybridRetriever(serialize_table(tables[tid], S2), embedder)
            retr_f[tid] = HybridRetriever(serialize_table(_bench_from_ot(ots_flat[tid]), S2), embedder)

    recs = []
    for i, q in enumerate(pop):
        ot = ots[q.gold_table_id]
        d = header_depth(ot.top_paths, ot.left_paths)
        ro = enum_record(ot, q, resolve_o)
        rf = enum_record(ots_flat[q.gold_table_id], q, resolve_f)
        rec = {"m": len({(o.row, o.col) for o in q.gold_operands}), "d": d,
               "orig": ro, "flat": rf}
        if args.dense:
            gold = q.gold_operands
            res_o = retrieve(q.question, tables[q.gold_table_id], gold, mode="plain", k=10,
                             scheme=S2, embedder=embedder, retriever=retr_o[q.gold_table_id])
            res_f = retrieve(q.question, _bench_from_ot(ots_flat[q.gold_table_id]), gold,
                             mode="plain", k=10, scheme=S2, embedder=embedder,
                             retriever=retr_f[q.gold_table_id])
            rec["base_orig_osc"] = operand_set_completeness(gold, covered_gold_cells(gold, res_o.retrieved))
            rec["base_flat_osc"] = operand_set_completeness(gold, covered_gold_cells(gold, res_f.retrieved))
        recs.append(rec)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(pop)}")

    keys = ["osc", "row_cov", "col_cov", "cells"]

    # (A) observational: enum (original tables) by depth, and by depth x scope
    by_depth = {}
    for db in ("d1", "d2", "d3+"):
        sub = [r["orig"] for r in recs if _bin_depth(r["d"]) == db]
        by_depth[db] = _agg(sub, keys)
    by_depth_scope = {}
    for db in ("d1", "d2", "d3+"):
        for mb in ("2", "3-4", "5-8", "9+"):
            sub = [r["orig"] for r in recs
                   if _bin_depth(r["d"]) == db and bin_scope(r["m"]) == mb]
            if sub:
                by_depth_scope[f"{db}|m{mb}"] = _agg(sub, keys)

    # (B) synthetic: paired original(d) vs leaf-flattened(d=1)
    n = len(recs)
    paired = {
        "n": n,
        "orig": _agg([r["orig"] for r in recs], keys),
        "flat_d1": _agg([r["flat"] for r in recs], keys),
        "delta_row_cov": round(sum(r["flat"]["row_cov"] - r["orig"]["row_cov"] for r in recs) / n, 4),
        "delta_col_cov": round(sum(r["flat"]["col_cov"] - r["orig"]["col_cov"] for r in recs) / n, 4),
        "delta_osc": round(sum(r["flat"]["osc"] - r["orig"]["osc"] for r in recs) / n, 4),
    }
    # the flatten effect should be largest on originally-deep tables
    paired_by_depth = {}
    for db in ("d2", "d3+"):
        sub = [r for r in recs if _bin_depth(r["d"]) == db]
        if sub:
            ns = len(sub)
            paired_by_depth[db] = {
                "n": ns,
                "orig_row_cov": round(sum(r["orig"]["row_cov"] for r in sub) / ns, 4),
                "flat_row_cov": round(sum(r["flat"]["row_cov"] for r in sub) / ns, 4),
                "orig_osc": round(sum(r["orig"]["osc"] for r in sub) / ns, 4),
                "flat_osc": round(sum(r["flat"]["osc"] for r in sub) / ns, 4),
            }

    out = {
        "experiment": "E3_depth",
        "hypothesis": "H2 causality: header depth d effect on enumeration, isolated from scope size",
        "split": args.split, "seed": SEED, "n": n, "dense": args.dense,
        "A_observational_enum_by_depth": by_depth,
        "A_observational_enum_by_depth_x_scope": by_depth_scope,
        "B_synthetic_leaf_flatten": paired,
        "B_synthetic_by_orig_depth": paired_by_depth,
    }
    if args.dense:
        out["dense_baseline_orig_osc"] = round(sum(r["base_orig_osc"] for r in recs) / n, 4)
        out["dense_baseline_flat_osc"] = round(sum(r["base_flat_osc"] for r in recs) / n, 4)
        out["dense_delta_flat_minus_orig"] = round(
            sum(r["base_flat_osc"] - r["base_orig_osc"] for r in recs) / n, 4)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: out[k] for k in
                      ("A_observational_enum_by_depth", "B_synthetic_leaf_flatten",
                       "B_synthetic_by_orig_depth")}, indent=2))
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
