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
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from rag_agent.bench.hitab import load_queries
from rag_agent.data.loader import load_table
from rag_agent.query.header_path_resolver import _distinct_paths, extract_target_terms
from rag_agent.query.header_embed_resolver import _node_candidates
from rag_agent.query.operand_decomposer import Embedder
from rag_agent.stores.original_store import build_original_table

ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}
KS = (1, 2, 3, 4)


class TableCands(NamedTuple):
    """One table's candidate nodes and the embedding of each text rendering."""
    cands: list          # ancestor-node candidate paths
    mat: np.ndarray      # rows aligned to `cands`: ">"-joined path
    cap: np.ndarray      # rows aligned to `cands`: caption sentence


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

    # per-selector ranked row-node lists.
    # NOTE: rank_lexical must score the same `cands` (ancestor-node candidates)
    # that embed/cross rank over — not re-derive a leaf-only candidate set via
    # _rank_paths — or the comparison is biased: a selector that can name a short
    # ancestor node covers many gold rows in one top-k slot, while a
    # leaf-restricted lexical selector needs one slot per gold row regardless of
    # match quality.
    # Every selector takes (query, table, TableCands, query vector) so the
    # dispatch below is uniform; each ignores the parts it does not need.
    def rank_lexical(q, ot, tc, qv):
        if not tc.cands:
            return []
        query_str = " ".join(extract_target_terms(q.question))
        scored = []
        for c in tc.cands:
            s = ot._fuzzy_score(query_str, c)
            if s > 0:
                scored.append((s, " > ".join(c), c))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [c for _, _, c in scored[:max(KS)]]

    def rank_embed(q, ot, tc, qv):
        if not tc.cands:
            return []
        order = np.argsort(-(tc.mat @ qv))[:max(KS)]
        return [tc.cands[i] for i in order]

    def rank_cross(q, ot, tc, qv):
        if not tc.cands:
            return []
        scores = ce.predict([(q.question, " > ".join(c)) for c in tc.cands])
        order = sorted(range(len(tc.cands)), key=lambda i: -float(scores[i]))[:max(KS)]
        return [tc.cands[i] for i in order]

    # caption: rank a candidate row node by a sentence-shaped rendering instead
    # of the bare ">"-joined path, to test whether sentence text embeds closer
    # to a natural-language question (the S3 hypothesis).
    # CAVEAT: `_cap_text` below is a local approximation, NOT the deployed
    # rendering. The operand retriever indexes cells with S2
    # (`operand_retrieval.py` -> `header_path.serialize`), and the S3 renderer
    # that does exist (`serialization/caption.py::_cell_sentence`, "long")
    # formats differently. Do not read this arm as "what the retriever sees".
    def rank_caption(q, ot, tc, qv):
        if not tc.cands:
            return []
        order = np.argsort(-(tc.cap @ qv))[:max(KS)]
        return [tc.cands[i] for i in order]

    selectors = {"lexical": rank_lexical, "embed": rank_embed, "cross": rank_cross,
                 "caption": rank_caption}
    hits = {s: {k: 0 for k in KS} for s in selectors}

    def _cap_text(title, path):
        body = " > ".join(path)
        return f"In the table '{title}', among {body}, the value." if title else f"Among {body}, the value."

    # Per-table row candidates + embeddings. `mat` embeds the bare ">"-joined
    # path (embed selector), `cap` the caption sentence (caption selector).
    # Both are encoded in ONE call over all tables: at ~21 candidates per table
    # a per-table encode is a batch the model never fills.
    tids = list(ots)
    cand_lists = [_node_candidates(_distinct_paths(ots[t], "row")) for t in tids]
    path_texts = [" > ".join(c) for cands in cand_lists for c in cands]
    cap_texts = [_cap_text(ots[t].title, c)
                 for t, cands in zip(tids, cand_lists) for c in cands]
    path_mat = np.asarray(emb.encode(path_texts)) if path_texts else np.zeros((0, 1))
    cap_mat = np.asarray(emb.encode(cap_texts)) if cap_texts else np.zeros((0, 1))

    cache, off = {}, 0
    for tid, cands in zip(tids, cand_lists):
        cache[tid] = TableCands(cands, path_mat[off:off + len(cands)],
                                cap_mat[off:off + len(cands)])
        off += len(cands)

    # One batched encode for the whole query set: embed and caption both rank
    # against this vector, so encoding per-selector meant embedding each
    # question twice, each time as a batch of one.
    qvecs = np.asarray(emb.encode([q.question for q in pop]))

    for qi, q in enumerate(pop):
        ot = ots[q.gold_table_id]
        gold_rows = {o.row for o in q.gold_operands}
        tc = cache[q.gold_table_id]
        for s, fn in selectors.items():
            ranked = fn(q, ot, tc, qvecs[qi])
            for k in KS:
                if gold_rows <= rows_of(ot, ranked[:k]):
                    hits[s][k] += 1

    out = {"population": {"name": "arithmetic_m>=2", "n": n}, "metric": "row_recall@k",
           "embed_model": args.embed_model, "cross_encoder": args.cross_encoder,
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
