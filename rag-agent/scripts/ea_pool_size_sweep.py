#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""E-A — does the reranker help or hurt? Sweep POOL SIZE, hold everything else fixed.

``RESEARCH_STRUCTURE.md`` §2.2 records a contradiction the thesis cannot ship
with: on the open corpus the cross-encoder *lowers* OSC (93→63 successes,
p=5.21e-5), while inside a document pool (median 134 cells) it *raises* it
(flat @50 .8840→.9010). §3.4 E-A resolves it by making pool size the only thing
that moves — same queries, same gold, same serializations, same score math as
``operand_collision_multihiertt.py`` / ``_within_doc.py``.

Pool construction, per query: its gold cells plus distractors drawn from the
full corpus with a seeded RNG, truncated to size N. The distractor order is
drawn ONCE per query, so pools nest (P(50) ⊂ P(100) ⊂ …) and a query's larger
pool is a strict superset of its smaller one — the sweep is monotone by
construction, not by luck. Gold is always inside the pool, so every failure is
a ranking failure, never a recall-of-the-pool failure.

Full-corpus cross-encoding is not attempted: 43k cells × 293 queries is ~12.7M
pairs. The sweep brackets the crossover instead, which is what §2.2 needs.

Run:
    PYTHONPATH=. .venv/bin/python scripts/ea_pool_size_sweep.py --max-queries 300
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from rag_agent.retrieve.encoders import _tokenize, default_encoder
from rag_agent.runenv import run_env

from operand_collision_multihiertt import (_minmax, build_corpus, cell_text,
                                           load_population)
from operand_collision_within_doc import summarize

RETRIEVERS = [("bm25", 0.0), ("dense", 1.0), ("hybrid", 0.5)]


def pool_for(qi: int, gold: list, n_cells: int, size: int, seed: int, cache: dict) -> list:
    """Gold + seeded distractors, truncated to ``size``. Nests across sizes."""
    if qi not in cache:
        rng = random.Random(seed * 1_000_003 + qi)
        g = set(gold)
        # one draw per query, reused for every size -> nested pools
        draw = rng.sample(range(n_cells), min(n_cells, cache["max_size"] + len(g) + 64))
        cache[qi] = [i for i in draw if i not in g]
    return list(gold) + cache[qi][: max(0, size - len(gold))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-queries", type=int, default=300)
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--population", default="arith_multi",
                    choices=["arith_multi", "lookup_single"])
    ap.add_argument("--pool-sizes", type=int, nargs="+", default=[50, 100, 200, 500, 1000])
    ap.add_argument("--schemes", nargs="+", default=["flat", "S3"],
                    help="§2.2's contradiction is stated on flat and S3; add S2 to widen")
    ap.add_argument("--collision-min", type=int, default=5)
    ap.add_argument("--reranker", default="BAAI/bge-reranker-large")
    ap.add_argument("--rerank-max-length", type=int, default=192)
    ap.add_argument("--rerank-batch-size", type=int, default=64)
    ap.add_argument("--no-cross", action="store_true")
    ap.add_argument("--out", default="results/ea_pool_size_sweep.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    env = run_env(args.seed, args.embed_model)

    from rank_bm25 import BM25Okapi

    queries, docs = load_population(args.max_queries, args.population)
    tables, cells, pop = build_corpus(queries, docs)
    n_q, n_cells = len(pop), len(cells)
    print(f"[pop] {args.population} queries: {n_q} | corpus: {len(tables)} tables, "
          f"{n_cells} cells | sizes {args.pool_sizes}", flush=True)
    if not n_q:
        return 1

    sizes = sorted(args.pool_sizes)
    cache = {"max_size": max(sizes)}
    encoder = default_encoder(model_name=args.embed_model)
    q_vecs = np.asarray(encoder.encode([q["question"] for q in pop]))

    reranker = None
    if not args.no_cross:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(args.reranker, max_length=args.rerank_max_length)

    names = [n for n, _ in RETRIEVERS] + ([] if reranker is None else ["cross"])
    results: dict = {}
    records: dict = {s: [] for s in sizes}

    for scheme in args.schemes:
        texts = [cell_text(c, scheme) for c in cells]
        vecs = np.asarray(encoder.encode(texts))
        toks = [_tokenize(t) for t in texts]
        for size in sizes:
            t0 = time.time()
            per_ret = {n: {} for n in names}
            skipped = 0
            for qi, q in enumerate(pop):
                gold = [int(g) for g in q["gold"]]
                if len(gold) > size:                       # pool cannot hold the gold set
                    skipped += 1
                    continue
                idxs = pool_for(qi, gold, n_cells, size, args.seed, cache)
                gset = set(gold)
                dn = _minmax(vecs[idxs] @ q_vecs[qi])
                bm = _minmax(np.asarray(
                    BM25Okapi([toks[i] for i in idxs]).get_scores(_tokenize(q["question"])),
                    dtype=np.float32))
                orders = {n: np.argsort(-(a * dn + (1.0 - a) * bm)) for n, a in RETRIEVERS}
                if reranker is not None:
                    cs = reranker.predict([(q["question"], texts[i]) for i in idxs],
                                          batch_size=args.rerank_batch_size,
                                          show_progress_bar=False)
                    orders["cross"] = np.argsort(-np.asarray(cs))
                for name, order in orders.items():
                    rank_of = {}
                    for pos, local in enumerate(order, 1):
                        gi = idxs[int(local)]
                        if gi in gset:
                            rank_of[gi] = pos
                            if len(rank_of) == len(gset):
                                break
                    per_ret[name][qi] = [rank_of[g] for g in gold]
                    for g in gold:
                        records[size].append({
                            "scheme": scheme, "retriever": name, "query": qi,
                            "cell": g, "rank": rank_of[g],
                            "colliding": cells[g]["n_tables_with_label"] >= args.collision_min,
                            "total_like": cells[g]["is_total_like"]})
            results.setdefault(str(size), {})[scheme] = {
                n: summarize(pq) for n, pq in per_ret.items() if pq}
            print(f"=== pool={size} scheme={scheme} ({time.time()-t0:.0f}s, "
                  f"skipped {skipped}) ===", flush=True)
            for n in names:
                s = results[str(size)][scheme].get(n)
                if s:
                    print(f"  {n:<7} set_em@10={s['set_em@10']:.3f} "
                          f"set_em@50={s['set_em@50']:.3f} recall@10={s['recall@10']:.3f} "
                          f"mrr={s['mrr']:.3f}", flush=True)

    out = {
        "env": env,
        "experiment": "E-A (RESEARCH_STRUCTURE.md §3.4) — pool size is the only "
                      "manipulated variable; resolves the §2.2 reranker contradiction",
        "population": {"name": f"multihiertt_{args.population}_pool_sweep",
                       "n_queries": n_q, "collision_min_tables": args.collision_min},
        "corpus": {"n_tables": len(tables), "n_cells": n_cells},
        "pool": "per query: gold cells + seeded distractors from the full corpus, "
                "truncated to size N; nested across N (P(50) subset of P(100) ...)",
        "pool_sizes": sizes,
        "embed_model": args.embed_model,
        "encoder": encoder.name,
        "reranker": None if args.no_cross else args.reranker,
        "score_math": "per-query min-max, alpha*dense+(1-alpha)*bm25 "
                      "(same convention as operand_collision_multihiertt.py)",
        "by_pool_size": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    stem = str(Path(args.out).with_suffix(""))
    for size in sizes:                       # one file per size: the significance
        with open(f"{stem}_records_p{size}.jsonl", "w") as fh:   # script runs unchanged
            for rec in records[size]:
                fh.write(json.dumps(rec) + "\n")
    print(f"\nwrote -> {args.out}  (+ per-size records -> {stem}_records_p*.jsonl)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
