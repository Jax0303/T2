#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Which way of ranking TABLES from cell scores finds the gold table best?

`--max-context-tables` admits tables in cell-rank order, and
`results/context_table_cap_verdict.json` showed that is exactly top-N by
`table_rank_cellvote` -- a name that oversells what it does. That field is
FIRST-OCCURRENCE order: a table's rank is wherever its single best cell landed.
One lucky cell outranks a table with five good ones.

The cap experiment ended with reading gains and recall losses cancelling, and
its decision table named the ranker as the thing to fix. So before any reader
runs, this measures gold-table recall under real aggregations:

  first        the current rule -- position of the table's best cell
  sum_topK     sum of the table's K best cell scores
  mean_topK    their mean (rewards a table whose good cells are ALL good)
  count_topK   how many of the table's cells are inside the global top K

No LLM. The encoder pass is cached, so this is minutes, not hours.

    PYTHONPATH=.:scripts .venv/bin/python scripts/table_ranker_sweep.py --dataset hitab
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from rag_agent.retrieve.hybrid_index import HybridIndex
from rag_agent.serialization.base import Chunk
import corpus_dump_vs_cell as M


def rankers(scores: np.ndarray, owner_idx: np.ndarray, n_tables: int) -> dict:
    """tid_index -> rank (1-based) for the gold table, under each rule."""
    order = np.argsort(-scores)
    out = {}

    first = {}
    for p in order:
        t = int(owner_idx[p])
        if t not in first:
            first[t] = len(first)
    out["first"] = {t: r + 1 for t, r in first.items()}

    by_t = defaultdict(list)
    for p in order[:20000]:                 # the tail cannot enter a top-N anyway
        by_t[int(owner_idx[p])].append(scores[p])
    for K in (3, 5, 10):
        for name, fn in (("sum", np.sum), ("mean", np.mean)):
            agg = {t: float(fn(v[:K])) for t, v in by_t.items()}
            ranked = sorted(agg, key=lambda t: -agg[t])
            out[f"{name}_top{K}"] = {t: i + 1 for i, t in enumerate(ranked)}
    for K in (50, 200):
        cnt = defaultdict(int)
        for p in order[:K]:
            cnt[int(owner_idx[p])] += 1
        # ties broken by the best cell, so this can only differ from `first`
        # where a table actually has more cells in the pool
        ranked = sorted(cnt, key=lambda t: (-cnt[t], out["first"].get(t, 10 ** 9)))
        out[f"count_top{K}"] = {t: i + 1 for i, t in enumerate(ranked)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hitab",
                    choices=["hitab", "aitqa", "realhitbench", "multihiertt"])
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--population", default="hitab_dev_lookup_all")
    ap.add_argument("--cell-scheme", default="S2")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--cache-dir", default=".cache/corpus_dump_vs_cell")
    ap.add_argument("--mh-queries", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.dataset == "hitab":
        C = M.hitab_corpus(args.data_dir, args.split, args.population)
    elif args.dataset == "aitqa":
        C = M.aitqa_corpus()
    elif args.dataset == "realhitbench":
        C = M.realhitbench_corpus()
    else:
        C = M.multihiertt_corpus(args.mh_queries, args.seed)

    if args.cell_scheme in ("S3", "S3c", "mt2net"):
        from rag_agent.serialization.caption import caption_sentence
        from rag_agent.serialization.templates import (MT2NET, STRUCTURAL,
                                                       STRUCTURAL_COMPACT)
        tmpl = {"mt2net": MT2NET, "S3c": STRUCTURAL_COMPACT}.get(
            args.cell_scheme, STRUCTURAL)
        C.cell_text[:] = [caption_sentence(C.title.get(t, ""), rp, cp, v, template=tmpl)
                          for (rp, cp, v), (t, _i, _j)
                          in zip(C.cell_paths, C.cell_owner)]

    tix = {t: i for i, t in enumerate(C.tids)}
    owner_idx = np.array([tix[t] for t, _i, _j in C.cell_owner])
    chunks = [Chunk(table_id=t, chunk_id=f"c::{t}::{i}:{j}", text=x,
                    scheme=args.cell_scheme, kind="cell")
              for x, (t, i, j) in zip(C.cell_text, C.cell_owner)]
    enc = M._CachedEncoder(M.default_encoder(model_name=args.embed_model),
                           args.cache_dir,
                           f"{args.dataset}_{args.split}_{args.embed_model}_{args.cell_scheme}")
    ix = HybridIndex(chunks, encoder=enc, alpha=0.5)
    print(f"[corpus] {len(C.tids)} tables / {len(C.cell_text)} cells / "
          f"{len(C.queries)} queries | scheme {args.cell_scheme}", flush=True)

    hits = defaultdict(lambda: defaultdict(int))
    n = 0
    for q in C.queries:
        gold = q["gold_table"]
        if gold not in tix:
            continue
        n += 1
        sc = ix._dense_scores(q["question"])          # dense, as every run uses
        for name, ranks in rankers(sc, owner_idx, len(C.tids)).items():
            r = ranks.get(tix[gold], 10 ** 9)
            for k in (1, 2, 3, 4, 5, 10):
                hits[name][k] += int(r <= k)
    names = ["first"] + [x for x in hits if x != "first"]
    print(f"\n{'ranker':14}" + "".join(f"{'@'+str(k):>8}" for k in (1,2,3,4,5,10)))
    out = {"dataset": args.dataset, "cell_scheme": args.cell_scheme, "n": n,
           "recall": {}}
    for name in names:
        row = {f"@{k}": round(hits[name][k] / n, 4) for k in (1, 2, 3, 4, 5, 10)}
        out["recall"][name] = row
        mark = "  <- 현행" if name == "first" else ""
        print(f"{name:14}" + "".join(f"{row['@'+str(k)]:8.3f}"
                                     for k in (1,2,3,4,5,10)) + mark)
    p = Path(args.out or f"results/table_ranker_sweep_{args.dataset}_{args.cell_scheme}.json")
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
