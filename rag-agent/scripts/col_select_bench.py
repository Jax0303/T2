#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Column-selection benchmark (the column axis as a schema-linking problem).

OSC rewards the trivial whole-axis dump, hiding column-selection quality. We instead
measure column selection directly: **col-recall@k** = fraction of queries whose gold
column(s) are all within the top-k columns a selector returns. This isolates "pick
the right column under a small budget" — the real column bottleneck — and compares
matchers fairly (lexical vs bi-encoder vs cross-encoder).

Population: HiTab dev arithmetic m>=2. LLM-free.
Run: PYTHONPATH=. python scripts/col_select_bench.py --split dev
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from rag_agent.bench.hitab import load_queries
from rag_agent.data.loader import load_table
from rag_agent.query.header_path_resolver import _distinct_paths, extract_target_terms
from rag_agent.query.header_embed_resolver import _node_candidates
from rag_agent.query.operand_decomposer import Embedder
from rag_agent.stores.original_store import build_original_table

ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}
KS = (1, 2, 3)


def cols_of(table, paths):
    out = set()
    for p in paths:
        out.update(table.find_cols_by_header(" > ".join(p)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--cross-encoder", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--out", default="results/col_select_bench.json")
    args = ap.parse_args()

    queries, _ = load_queries(args.data_dir, args.split)
    pop = [q for q in queries
           if (q.aggregation or "none") in ARITH
           and len({(o.row, o.col) for o in q.gold_operands}) >= 2]
    n = len(pop)
    print(f"[pop] arithmetic m>=2: {n}")

    emb = Embedder(args.embed_model, device="cpu")
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(args.cross_encoder)
    ots = {t: build_original_table(load_table(t, args.data_dir))
           for t in {q.gold_table_id for q in pop}}

    # per-selector ranked column-node lists.
    # NOTE: rank_lexical must score the same `cands` (ancestor-node candidates)
    # that embed/cross rank over — not re-derive a leaf-only candidate set via
    # _rank_paths — or the comparison is biased: a selector that can name a short
    # ancestor node covers many gold columns in one top-k slot, while a
    # leaf-restricted lexical selector needs one slot per gold column regardless
    # of match quality.
    def rank_lexical(q, ot, cands, mat):
        if not cands:
            return []
        query_str = " ".join(extract_target_terms(q.question))
        scored = []
        for c in cands:
            s = ot._fuzzy_score(query_str, c)
            if s > 0:
                scored.append((s, " > ".join(c), c))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [c for _, _, c in scored[:max(KS)]]

    def rank_embed(q, ot, cands, mat):
        if not cands:
            return []
        qv = np.asarray(emb.encode([q.question])[0])
        order = np.argsort(-(mat @ qv))[:max(KS)]
        return [cands[i] for i in order]

    def rank_cross(q, ot, cands, mat):
        if not cands:
            return []
        scores = ce.predict([(q.question, " > ".join(c)) for c in cands])
        order = sorted(range(len(cands)), key=lambda i: -float(scores[i]))[:max(KS)]
        return [cands[i] for i in order]

    selectors = {"lexical": rank_lexical, "embed": rank_embed, "cross": rank_cross}
    hits = {s: {k: 0 for k in KS} for s in selectors}

    # cache col candidates + embeddings per table
    cache = {}
    for tid, ot in ots.items():
        cands = _node_candidates(_distinct_paths(ot, "col"))
        mat = np.asarray(emb.encode([" > ".join(c) for c in cands])) if cands else np.zeros((0, 1))
        cache[tid] = (cands, mat)

    for q in pop:
        ot = ots[q.gold_table_id]
        gold_cols = {o.col for o in q.gold_operands}
        cands, mat = cache[q.gold_table_id]
        for s, fn in selectors.items():
            ranked = fn(q, ot, cands, mat)
            for k in KS:
                if gold_cols <= cols_of(ot, ranked[:k]):
                    hits[s][k] += 1

    out = {"population": {"name": "arithmetic_m>=2", "n": n}, "metric": "col_recall@k",
           "selectors": {s: {f"@{k}": round(hits[s][k] / n, 3) for k in KS} for s in selectors}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n{'selector':<10}" + "".join(f"  col-recall@{k}" for k in KS))
    for s in selectors:
        print(f"{s:<10}" + "".join(f"{out['selectors'][s][f'@{k}']:>14.3f}" for k in KS))
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
