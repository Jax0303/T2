#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Operand-collision experiment under the LITERATURE-STANDARD within-document
pool (the condition every comparison paper actually evaluates in).

Motivation: our open-corpus numbers (42,715-cell pool) cannot be placed next to
MT2Net / FT-RAG / Topo-RAG's 90%-range scores because those systems retrieve
inside the query's own source document (MT2Net: intra-document; FT-RAG: 328
tables). This runs the SAME queries, gold, serializations and score math as
operand_collision_multihiertt.py, but restricts each query's candidate pool to
the cells of its own MultiHiertt document — so the absolute numbers become
directly comparable to the literature, with no change to gold or metrics.

Collision is still real here: a MultiHiertt filing holds ~10 tables with
repeated leaf labels ("total", twin tables), so flat vs S2/S3 stays a fair
within-doc contrast.

Conditions: {flat, S2, S3} x {bm25, dense, hybrid, cross}. "cross" reranks the
ENTIRE doc pool with a cross-encoder (bge-reranker-large) — feasible only in
this setting (pools are a few hundred cells), and by construction every gold
cell is in the pool, so failures are pure ranking failures.

Outputs the standard metric battery (Hit@k / Recall@k / MRR / nDCG@k /
set-EM@k) inline and writes per-operand rank records in the exact schema of
operand_collision_multihiertt.py, so operand_collision_significance.py and
standard_ir_metrics_from_records.py run on them unchanged.

Run: PYTHONPATH=. .venv/bin/python scripts/operand_collision_within_doc.py --max-queries 300
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from rag_agent.retrieve.encoders import _tokenize, default_encoder

from operand_collision_multihiertt import (_minmax, _norm_label, build_corpus,
                                           cell_text, load_population)
from standard_ir_metrics_from_records import (hit_at_k, ndcg_at_k, rr,
                                              recall_at_k, set_em_at_k)

KS = (1, 5, 10, 20, 50)
RETRIEVERS = [("bm25", 0.0), ("dense", 1.0), ("hybrid", 0.5)]


def summarize(per_query: dict) -> dict:
    qs = sorted(per_query)
    n = len(qs)
    out = {"n_queries": n,
           "mrr": round(sum(rr(per_query[q]) for q in qs) / n, 4)}
    for k in KS:
        out[f"hit@{k}"] = round(sum(hit_at_k(per_query[q], k) for q in qs) / n, 4)
        out[f"recall@{k}"] = round(sum(recall_at_k(per_query[q], k) for q in qs) / n, 4)
        out[f"ndcg@{k}"] = round(sum(ndcg_at_k(per_query[q], k) for q in qs) / n, 4)
        out[f"set_em@{k}"] = round(sum(set_em_at_k(per_query[q], k) for q in qs) / n, 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-queries", type=int, default=300)
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--population", default="arith_multi",
                    choices=["arith_multi", "lookup_single"])
    ap.add_argument("--collision-min", type=int, default=5)
    ap.add_argument("--reranker", default="BAAI/bge-reranker-large")
    ap.add_argument("--rerank-max-length", type=int, default=192)
    ap.add_argument("--rerank-batch-size", type=int, default=64)
    ap.add_argument("--no-cross", action="store_true",
                    help="skip the cross-encoder condition (embedding-only run)")
    ap.add_argument("--out", default="results/operand_collision_within_doc.json")
    args = ap.parse_args()

    from rank_bm25 import BM25Okapi

    queries, docs = load_population(args.max_queries, args.population)
    tables, cells, pop = build_corpus(queries, docs)
    n_q = len(pop)
    print(f"[pop] {args.population} queries: {n_q} | corpus: {len(tables)} tables, "
          f"{len(cells)} cells", flush=True)
    if not n_q:
        return 1

    # doc pool: uid -> global cell indices (all tables of that document)
    pool_of = {}
    for gi, c in enumerate(cells):
        pool_of.setdefault(c["table"][0], []).append(gi)
    pool_sizes = [len(pool_of[q["uid"]]) for q in pop]
    print(f"[pool] within-doc cells/query: median {statistics.median(pool_sizes):.0f}, "
          f"min {min(pool_sizes)}, max {max(pool_sizes)}", flush=True)
    # sanity: every gold cell must be in its own doc pool by construction
    for q in pop:
        assert all(cells[g]["table"][0] == q["uid"] for g in q["gold"])

    encoder = default_encoder(model_name=args.embed_model)
    q_vecs = np.asarray(encoder.encode([q["question"] for q in pop]))

    reranker = None
    if not args.no_cross:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(args.reranker, max_length=args.rerank_max_length)

    records, results = [], {}
    for scheme in ("flat", "S2", "S2_shuf", "S3"):
        t0 = time.time()
        texts = [cell_text(c, scheme) for c in cells]
        vecs = np.asarray(encoder.encode(texts))
        # per-doc BM25 (IDF computed inside the pool the query actually sees)
        bm25_of = {uid: BM25Okapi([_tokenize(texts[gi]) for gi in idxs])
                   for uid, idxs in pool_of.items()}

        per_ret = {name: {} for name, _ in RETRIEVERS}
        if reranker is not None:
            per_ret["cross"] = {}
        for qi, q in enumerate(pop):
            idxs = pool_of[q["uid"]]
            gold = {int(g) for g in q["gold"]}
            dn = _minmax(np.asarray([vecs[gi] for gi in idxs]) @ q_vecs[qi])
            bm = _minmax(np.asarray(
                bm25_of[q["uid"]].get_scores(_tokenize(q["question"])), dtype=np.float32))
            orders = {}
            for name, alpha in RETRIEVERS:
                orders[name] = np.argsort(-(alpha * dn + (1.0 - alpha) * bm))
            if reranker is not None:
                pairs = [(q["question"], texts[gi]) for gi in idxs]
                cs = reranker.predict(pairs, batch_size=args.rerank_batch_size,
                                      show_progress_bar=False)
                orders["cross"] = np.argsort(-np.asarray(cs))
            for name, order in orders.items():
                rank_of = {}
                for pos, local in enumerate(order, 1):
                    gi = idxs[int(local)]
                    if gi in gold:
                        rank_of[gi] = pos
                        if len(rank_of) == len(gold):
                            break
                per_ret[name][qi] = [rank_of[g] for g in gold]
                for g in gold:
                    records.append({
                        "scheme": scheme, "retriever": name, "query": qi,
                        "cell": g, "rank": rank_of[g],
                        "colliding": cells[g]["n_tables_with_label"] >= args.collision_min,
                        "total_like": cells[g]["is_total_like"]})

        results[scheme] = {name: summarize(pq) for name, pq in per_ret.items()}
        print(f"\n=== scheme={scheme} ({time.time()-t0:.0f}s) ===")
        for name, s in results[scheme].items():
            print(f"  {name:<7} hit@10={s['hit@10']:.3f} recall@10={s['recall@10']:.3f} "
                  f"mrr={s['mrr']:.3f} ndcg@10={s['ndcg@10']:.3f} "
                  f"set_em@10={s['set_em@10']:.3f} set_em@50={s['set_em@50']:.3f}",
                  flush=True)

    out = {
        "population": {"name": f"multihiertt_{args.population}_within_doc",
                       "n_queries": n_q,
                       "pool": "within-document (all tables of the query's own doc)",
                       "pool_cells_median": statistics.median(pool_sizes),
                       "pool_cells_min": min(pool_sizes),
                       "pool_cells_max": max(pool_sizes),
                       "collision_min_tables": args.collision_min},
        "corpus": {"n_tables": len(tables), "n_cells": len(cells)},
        "embed_model": args.embed_model,
        "encoder": encoder.name,
        "reranker": None if args.no_cross else args.reranker,
        "score_math": "per-query min-max over the DOC pool, alpha*dense+(1-alpha)*bm25; "
                      "cross = cross-encoder over the full doc pool",
        "note": "same queries/gold/serializations as operand_collision_multihiertt.py; "
                "only the candidate pool changes to the literature-standard "
                "within-document setting, so absolute numbers are comparable to "
                "MT2Net/FT-RAG-style intra-document retrieval scores",
        "by_scheme": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    rec_path = str(Path(args.out).with_suffix("")) + "_records.jsonl"
    with open(rec_path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    print(f"\nwrote -> {args.out}  (+ records -> {rec_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
