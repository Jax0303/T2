#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Full lookup pipeline on MultiHiertt RAW tables — the raw-data counterpart of
scripts/pipeline_lookup.py (HiTab).

Same experiment, genuinely raw input: MultiHiertt ships no header tree, so the
S2 arm's header paths are SELF-RECONSTRUCTED (rag_agent.reconstruct), meaning
this run also stresses the reconstruction front-end (row axis is the weak one).

Stages (LLM-free extractive: answer = value of the top-1 retrieved cell):
  Gate 1  find the table   : dense vs hybrid(+BM25) table-chunk retrieval, top-1
  Gate 2  find the cell     : within that table, top-1 cell under {flat, S2}
  Answer                    : that cell's value

Also reports the oracle-table variant (gate-1 solved) to separate the two gates.

SCORER: strict EM via hitab_exact_match (WTQ-style normalisation, 1e-5 float
tolerance) — NOT the ±2% tolerant numeric_match the older MultiHiertt scripts
use for diagnostics. MultiHiertt's official metric is EM/F1; this strict EM is
the literature-aligned choice, deliberately not the lenient one.

Run: PYTHONPATH=. .venv/bin/python scripts/pipeline_lookup_multihiertt.py --max-queries 400
An LLM-reader stage (like scripts/pipeline_lookup_llm.py) is added once GROQ TPD
resets; this LLM-free stage needs no API and runs now.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from rank_bm25 import BM25Okapi

from rag_agent.eval.metrics import hitab_exact_match
from rag_agent.retrieve.encoders import default_encoder, _tokenize
from rag_agent.serialization import serialize
from s3_table_chunk_baseline_multihiertt import (build_tables,
    load_clean_lookup_queries)


def cell_text(rp, cp, v, scheme):
    """flat = leaf row+col labels only; S2 = full row-path > col-path."""
    if scheme == "flat":
        lab = " ".join(x for x in ((rp[-1] if rp else ""), (cp[-1] if cp else "")) if x)
        return f"{lab}: {v}" if lab else str(v)
    path = " > ".join([*rp, *cp])
    return f"{path}: {v}" if path else str(v)


def minmax(a):
    a = np.asarray(a, dtype=np.float32)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-queries", type=int, default=400)
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--out", default="results/pipeline_lookup_multihiertt.json")
    args = ap.parse_args()

    raw_q, docs = load_clean_lookup_queries(args.max_queries)
    pop, tables = build_tables(raw_q, docs)
    n = len(pop)
    print(f"[pop] {n} raw single-cell-lookup queries | table pool={len(tables)} "
          f"(self-reconstructed, no gold tree)", flush=True)

    enc = default_encoder(model_name=args.embed_model)
    tids = list(tables.keys())
    tid_idx = {t: i for i, t in enumerate(tids)}
    tchunks = [serialize(tables[t], scheme="S3", length="medium", granularity="table")[0].text
               for t in tids]
    tvecs = np.asarray(enc.encode(tchunks))
    bm = BM25Okapi([_tokenize(t) for t in tchunks])

    # per-(table,scheme) cell texts, values, vectors
    cache = {}
    def cells(tid, scheme):
        key = (tid, scheme)
        if key in cache:
            return cache[key]
        rt = tables[tid]
        T, vals = [], []
        for i in range(rt.n_rows):
            for j in range(rt.n_cols):
                v = rt.cell(i, j)
                T.append(cell_text(list(rt.row_path(i)), list(rt.col_path(j)), v, scheme))
                vals.append(v)
        vecs = np.asarray(enc.encode(T)) if T else np.zeros((0, 384))
        cache[key] = (vecs, vals)
        return cache[key]

    qvecs = np.asarray(enc.encode([q["question"] for q in pop]))

    def top1_value(tid, qv, scheme):
        vecs, vals = cells(tid, scheme)
        if not len(vals):
            return None
        return vals[int(np.argmax(vecs @ qv))]

    gate1 = {"dense": 0, "hybrid": 0}
    res = {s: {"e2e_dense": 0, "e2e_hybrid": 0, "oracle": 0} for s in ("flat", "S2")}
    for qi, q in enumerate(pop):
        qv = qvecs[qi]
        gold_tid = q["table_id"]
        dn = minmax(tvecs @ qv)
        bmv = minmax(bm.get_scores(_tokenize(q["question"])))
        top_dense = tids[int(np.argmax(dn))]
        top_hybrid = tids[int(np.argmax(0.5 * dn + 0.5 * bmv))]
        gate1["dense"] += int(top_dense == gold_tid)
        gate1["hybrid"] += int(top_hybrid == gold_tid)
        for s in ("flat", "S2"):
            res[s]["e2e_dense"] += int(hitab_exact_match(top1_value(top_dense, qv, s), q["answer"]))
            res[s]["e2e_hybrid"] += int(hitab_exact_match(top1_value(top_hybrid, qv, s), q["answer"]))
            res[s]["oracle"] += int(hitab_exact_match(top1_value(gold_tid, qv, s), q["answer"]))
        if (qi + 1) % 50 == 0:
            print(f"  {qi+1}/{n}", flush=True)

    out = {
        "dataset": "multihiertt_lookup_fullpipeline_RAW",
        "n": n, "table_pool": len(tables),
        "hierarchy_source": "self-reconstructed (no gold tree exists)",
        "scorer": "hitab_exact_match (strict EM, WTQ-style; NOT the ±2% numeric_match)",
        "gate1_table_recall@1": {k: round(v / n, 4) for k, v in gate1.items()},
        "answer_EM": {s: {"end_to_end_dense": round(res[s]["e2e_dense"] / n, 4),
                          "end_to_end_hybrid": round(res[s]["e2e_hybrid"] / n, 4),
                          "oracle_table": round(res[s]["oracle"] / n, 4)}
                      for s in res},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out["gate1_table_recall@1"], indent=2))
    print(json.dumps(out["answer_EM"], indent=2))
    print(f"wrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
