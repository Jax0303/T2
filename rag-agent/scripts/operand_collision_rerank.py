#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Strong cross-encoder reranker baseline for the operand-collision experiment.

The #1 reviewer attack on the collision result (operand_collision_multihiertt):
"a strong reranker on top of the flat baseline would fix set completeness, so
structural serialization is unnecessary." This script measures exactly that,
on the SAME population / corpus / first-stage math as the n=300 run.

Design — 2x2 over the headline retriever (hybrid, alpha=0.5):
  {flat, S3} x {pool order (no rerank), bge-reranker-large rerank}
  * First stage: hybrid top-``--pool`` candidates per scheme (min-max
    normalized alpha*dense+(1-alpha)*bm25, HybridIndex convention).
  * Rerank: cross-encoder scores (question, cell_text) for every pool
    candidate; final ranking = cross-encoder order. Both conditions see the
    IDENTICAL candidate pool and are scored at the same final k, so any gap is
    pure ranking, and the pool ceiling (set_recall@pool) bounds what ANY
    reranker could achieve — separating "ranking failure" (reranker can fix)
    from "candidate-generation failure" (structural, our (1)/(3) claim).

Metrics: OSC set_recall@k / coverage@k (rag_agent.eval.operand_set), pool
ceiling, colliding vs unique operand median ranks, and inline paired sign
tests for the four decision contrasts. Per-operand records go to *_records
.jsonl in the operand_collision_significance.py format (retriever field =
"hybrid_pool{N}" / "rerank_pool{N}") so the existing test battery reruns as-is.

Run: PYTHONPATH=. python scripts/operand_collision_rerank.py --max-queries 300
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

from rag_agent.eval.operand_set import (coverage_at_k, osc_at_k_summary,
                                        paired_set_recall_flip)
from rag_agent.retrieve.encoders import _tokenize, default_encoder

from operand_collision_multihiertt import (_minmax, build_corpus, cell_text,
                                           load_population)

KS = (10, 20, 50)


def rank_map(order, gold, limit=None):
    """gold cell -> 1-based rank within ``order`` (None if absent/beyond limit)."""
    gold = [int(g) for g in gold]
    rank_of = {g: None for g in gold}
    left = set(gold)
    for pos, idx in enumerate(order, 1):
        if limit is not None and pos > limit:
            break
        ii = int(idx)
        if ii in left:
            rank_of[ii] = pos
            left.discard(ii)
            if not left:
                break
    return rank_of


def operand_strata(all_ranks, cells, collision_min):
    """Median rank / reached@50 for colliding-label vs unique-label operands."""
    coll, uniq = [], []
    for ranks in all_ranks:
        for gi, r in ranks.items():
            (coll if cells[gi]["n_tables_with_label"] >= collision_min else uniq).append(r)
    med = lambda xs: (round(statistics.median([x for x in xs if x is not None]), 1)
                      if any(x is not None for x in xs) else None)
    reach = lambda xs: (round(sum(1 for x in xs if x is not None and x <= 50) / len(xs), 4)
                        if xs else None)
    return {"median_rank": {"colliding_label": med(coll), "unique_label": med(uniq)},
            "reached@50": {"colliding_label": reach(coll), "unique_label": reach(uniq)},
            "n_operands": {"colliding_label": len(coll), "unique_label": len(uniq)}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-queries", type=int, default=300)
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--reranker", default="BAAI/bge-reranker-large")
    ap.add_argument("--pool", type=int, default=100,
                    help="first-stage hybrid candidates fed to the reranker")
    ap.add_argument("--schemes", default="flat,S3")
    ap.add_argument("--alpha", type=float, default=0.5, help="hybrid dense weight")
    ap.add_argument("--collision-min", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--rerank-max-length", type=int, default=192,
                    help="cross-encoder max_length; 512 = no-truncation spotcheck")
    ap.add_argument("--out", default="results/operand_collision_rerank.json")
    args = ap.parse_args()

    from rank_bm25 import BM25Okapi
    from sentence_transformers import CrossEncoder

    queries, docs = load_population(args.max_queries)
    tables, cells, pop = build_corpus(queries, docs)
    n_q = len(pop)
    print(f"[pop] queries: {n_q} | corpus: {len(tables)} tables, {len(cells)} cells",
          flush=True)
    if not n_q:
        print("no evaluable queries — abort")
        return 1

    encoder = default_encoder(model_name=args.embed_model)
    q_texts = [q["question"] for q in pop]
    q_vecs = np.asarray(encoder.encode(q_texts))
    reranker = CrossEncoder(args.reranker, max_length=args.rerank_max_length)

    results, records = {}, []
    cond_ranks = {}                      # (scheme, "hybrid"|"rerank") -> per-query rank maps
    for scheme in args.schemes.split(","):
        t0 = time.time()
        texts = [cell_text(c, scheme) for c in cells]
        vecs = np.asarray(encoder.encode(texts))
        bm25 = BM25Okapi([_tokenize(t) for t in texts])
        print(f"[{scheme}] corpus encoded in {time.time() - t0:.0f}s", flush=True)

        pool_ranks, rr_ranks = [], []
        t0 = time.time()
        for qi, q in enumerate(pop):
            dn = _minmax(vecs @ q_vecs[qi])
            bm = _minmax(np.asarray(bm25.get_scores(_tokenize(q_texts[qi])),
                                    dtype=np.float32))
            combined = args.alpha * dn + (1.0 - args.alpha) * bm
            cand = np.argsort(-combined)[:args.pool]
            pool_ranks.append(rank_map(cand, q["gold"]))

            pairs = [(q_texts[qi], texts[int(i)]) for i in cand]
            scores = reranker.predict(pairs, batch_size=args.batch_size,
                                      show_progress_bar=False)
            reorder = cand[np.argsort(-np.asarray(scores))]
            rr_ranks.append(rank_map(reorder, q["gold"]))
            if (qi + 1) % 25 == 0:
                print(f"  [{scheme}] {qi + 1}/{n_q} queries reranked "
                      f"({(time.time() - t0) / (qi + 1):.1f}s/q)", flush=True)

        conds = {f"hybrid_pool{args.pool}": pool_ranks,
                 f"rerank_pool{args.pool}": rr_ranks}
        cond_ranks[(scheme, "hybrid")] = pool_ranks
        cond_ranks[(scheme, "rerank")] = rr_ranks
        results[scheme] = {}
        for name, ranks in conds.items():
            summ = osc_at_k_summary(ranks, ks=KS)
            summ.update(operand_strata(ranks, cells, args.collision_min))
            results[scheme][name] = summ
            for qi, rmap in enumerate(ranks):
                for gi, r in rmap.items():
                    records.append({
                        "scheme": scheme, "retriever": name, "query": qi,
                        "cell": gi, "rank": r,
                        "colliding": cells[gi]["n_tables_with_label"] >= args.collision_min,
                        "total_like": cells[gi]["is_total_like"],
                        "scope_size": len(pop[qi]["gold"]),
                    })
        results[scheme][f"pool_ceiling@{args.pool}"] = round(
            sum(coverage_at_k(r, args.pool) == 1.0 for r in pool_ranks) / n_q, 4)
        for name, summ in results[scheme].items():
            if isinstance(summ, dict):
                print(f"  {scheme}/{name}: " + " ".join(
                    f"set@{k}={summ[f'set_recall@{k}']:.3f}" for k in KS), flush=True)
        print(f"  {scheme} pool ceiling@{args.pool}: "
              f"{results[scheme][f'pool_ceiling@{args.pool}']:.3f}", flush=True)

    # decision contrasts (exact binomial sign tests, paired per query)
    contrasts = {}
    pairs = [
        ("flat_hybrid->flat_rerank", ("flat", "hybrid"), ("flat", "rerank")),
        ("flat_rerank->S3_hybrid", ("flat", "rerank"), ("S3", "hybrid")),
        ("S3_hybrid->S3_rerank", ("S3", "hybrid"), ("S3", "rerank")),
        ("flat_rerank->S3_rerank", ("flat", "rerank"), ("S3", "rerank")),
    ]
    for label, a, b in pairs:
        if a not in cond_ranks or b not in cond_ranks:
            continue
        contrasts[label] = {f"@{k}": paired_set_recall_flip(cond_ranks[a],
                                                            cond_ranks[b], k)
                            for k in KS}

    out = {
        "population": {"name": "multihiertt_arithmetic_single_table_multi_operand",
                       "n_queries": n_q,
                       "n_gold_operand_cells": sum(len(q["gold"]) for q in pop)},
        "corpus": {"n_tables": len(tables), "n_cells": len(cells)},
        "embed_model": args.embed_model, "reranker": args.reranker,
        "pool": args.pool, "alpha": args.alpha,
        "collision_min_tables": args.collision_min,
        "score_math": "per-query min-max, alpha*dense+(1-alpha)*bm25; "
                      "rerank = cross-encoder order over the same pool",
        "by_scheme": results,
        "contrasts_set_recall_flip": contrasts,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    rec_path = str(Path(args.out).with_suffix("")) + "_records.jsonl"
    with open(rec_path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    print(f"\nwrote -> {args.out}  (+ per-operand ranks -> {rec_path})")

    for label, ks in contrasts.items():
        for k, st in ks.items():
            sig = " *" if (st["p_two_sided"] or 1) < 0.05 else ""
            print(f"  {label}{k}: {st['a_covered']}->{st['b_covered']} "
                  f"gain={st['gain']} loss={st['loss']} p={st['p_two_sided']}{sig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
