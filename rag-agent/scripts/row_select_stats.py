#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Row-selection significance test: cross-encoder vs embed, per-query paired.

``row_select_bench.py`` reports aggregate row-recall@k. This asks whether the
cross-encoder's lift over the embedding matcher (the production row default) is
real or noise, with a **per-query paired** test (McNemar on the @k hit indicator).
Mirrors the column-axis ``compare_runs.py`` significance pattern, applied to the
row axis so "retriever improvement" is backed by a p-value, not just a delta.

McNemar: discordant pairs only. b = embed hit & cross miss, c = embed miss &
cross hit. Exact two-sided binomial p over (b, c). Population: HiTab dev
arithmetic m>=2 (same as the bench). LLM-free.

Run: PYTHONPATH=. python scripts/row_select_stats.py --split dev --k 2 \
        --cross-encoder cross-encoder/ms-marco-MiniLM-L-6-v2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import binomtest

from rag_agent.bench.hitab import load_queries
from rag_agent.data.loader import load_table
from rag_agent.query.header_path_resolver import _distinct_paths, _rank_paths, extract_target_terms
from rag_agent.query.header_embed_resolver import _node_candidates
from rag_agent.query.operand_decomposer import Embedder
from rag_agent.stores.original_store import build_original_table

ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}


def rows_of(table, paths):
    out = set()
    for p in paths:
        out.update(table.find_rows_by_header(" > ".join(p)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--cross-encoder", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--out", default="results/row_select_stats.json")
    args = ap.parse_args()

    queries, _ = load_queries(args.data_dir, args.split)
    pop = [q for q in queries
           if (q.aggregation or "none") in ARITH
           and len({(o.row, o.col) for o in q.gold_operands}) >= 2]
    n = len(pop)
    print(f"[pop] arithmetic m>=2: {n}  |  k={args.k}  ce={args.cross_encoder}")

    emb = Embedder(args.embed_model, device="cpu")
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(args.cross_encoder)
    ots = {t: build_original_table(load_table(t, args.data_dir))
           for t in {q.gold_table_id for q in pop}}
    cache = {}
    for tid, ot in ots.items():
        cands = _node_candidates(_distinct_paths(ot, "row"))
        mat = np.asarray(emb.encode([" > ".join(c) for c in cands])) if cands else np.zeros((0, 1))
        cache[tid] = (cands, mat)

    k = args.k
    embed_hits, cross_hits = [], []
    for q in pop:
        ot = ots[q.gold_table_id]
        gold = {o.row for o in q.gold_operands}
        cands, mat = cache[q.gold_table_id]
        if cands:
            qv = np.asarray(emb.encode([q.question])[0])
            e_rank = [cands[i] for i in np.argsort(-(mat @ qv))[:k]]
            cs = ce.predict([(q.question, " > ".join(c)) for c in cands])
            c_rank = [cands[i] for i in sorted(range(len(cands)), key=lambda i: -float(cs[i]))[:k]]
        else:
            e_rank = c_rank = []
        embed_hits.append(gold <= rows_of(ot, e_rank))
        cross_hits.append(gold <= rows_of(ot, c_rank))

    eh, ch = np.array(embed_hits), np.array(cross_hits)
    both = int(np.sum(eh & ch))
    b = int(np.sum(eh & ~ch))     # embed hit, cross miss
    c = int(np.sum(~eh & ch))     # embed miss, cross hit (cross wins)
    neither = int(np.sum(~eh & ~ch))
    res = binomtest(c, b + c, 0.5, alternative="two-sided") if (b + c) else None
    p = res.pvalue if res else float("nan")

    out = {
        "population": {"name": "arithmetic_m>=2", "n": n},
        "metric": f"row_recall@{k} hit", "cross_encoder": args.cross_encoder,
        "embed_acc": round(float(eh.mean()), 3), "cross_acc": round(float(ch.mean()), 3),
        "delta": round(float(ch.mean() - eh.mean()), 3),
        "contingency": {"both_hit": both, "embed_only": b, "cross_only": c, "neither": neither},
        "mcnemar_exact_binom_p": round(float(p), 5),
        "discordant": b + c,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"\nembed  acc@{k} = {out['embed_acc']:.3f}")
    print(f"cross  acc@{k} = {out['cross_acc']:.3f}   (Δ = {out['delta']:+.3f})")
    print(f"\ncontingency (b=embed-only={b}, c=cross-only={c}, both={both}, neither={neither})")
    print(f"McNemar exact binomial p = {p:.5f}   {'(significant @0.05)' if p < 0.05 else '(n.s.)'}")
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
