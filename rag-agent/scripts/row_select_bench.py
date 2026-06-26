#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Row-selection benchmark (the row axis as a node-resolution problem).

Mirror of ``col_select_bench.py`` for the **row** axis. The measured bottleneck is
header-path decomposition, localized to rows; the production row resolver uses an
*embedding* node matcher (E3: row entities benefit from semantic matching). This
asks the unmeasured question: does a **cross-encoder** (query x header joint
attention) pick the right row scope-node better than the embedding matcher — the
same lift it gave the column axis?

**row-recall@k** = fraction of queries whose gold row(s) are all within the rows
covered by the top-k row scope-nodes a selector returns. A parent node expands to
all its child rows (``find_rows_by_header``), so one correct scope node can cover a
whole aggregation. This is the row analogue of col-recall@k and handles
many-row aggregations via node-level (not leaf-level) selection.

Population: HiTab dev arithmetic m>=2. LLM-free.
Run: PYTHONPATH=. python scripts/row_select_bench.py --split dev
     PYTHONPATH=. python scripts/row_select_bench.py --cross-encoder BAAI/bge-reranker-base \
         --out results/row_select_bench_bge.json
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
from rag_agent.query.header_path_resolver import (
    _distinct_paths, _rank_paths, extract_target_terms,
)
from rag_agent.query.header_embed_resolver import _node_candidates
from rag_agent.query.operand_decomposer import Embedder
from rag_agent.stores.original_store import build_original_table

ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}
KS = (1, 2, 3, 4)


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
    ap.add_argument("--out", default="results/row_select_bench.json")
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

    # per-selector ranked row-node lists
    def rank_lexical(q, ot, cands, mat):
        return _rank_paths(ot, extract_target_terms(q.question), "row", max(KS))

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

    # cache row candidates + embeddings per table
    cache = {}
    for tid, ot in ots.items():
        cands = _node_candidates(_distinct_paths(ot, "row"))
        mat = np.asarray(emb.encode([" > ".join(c) for c in cands])) if cands else np.zeros((0, 1))
        cache[tid] = (cands, mat)

    for q in pop:
        ot = ots[q.gold_table_id]
        gold_rows = {o.row for o in q.gold_operands}
        cands, mat = cache[q.gold_table_id]
        for s, fn in selectors.items():
            ranked = fn(q, ot, cands, mat)
            for k in KS:
                if gold_rows <= rows_of(ot, ranked[:k]):
                    hits[s][k] += 1

    out = {"population": {"name": "arithmetic_m>=2", "n": n}, "metric": "row_recall@k",
           "cross_encoder": args.cross_encoder,
           "selectors": {s: {f"@{k}": round(hits[s][k] / n, 3) for k in KS} for s in selectors}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n{'selector':<10}" + "".join(f"  row-recall@{k}" for k in KS))
    for s in selectors:
        print(f"{s:<10}" + "".join(f"{out['selectors'][s][f'@{k}']:>14.3f}" for k in KS))
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
