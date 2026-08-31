#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Phase 3b -- Recall at a matched TOKEN budget, not a matched k.

Recall@10 across chunking policies is not a fair comparison: a P1_fixed_1024
chunk costs ~40x what a P4_path_cell sentence costs, so @10 hands one policy
40x the context of the other. This measures each policy's mean chunk length and
compares them at the k each one can actually afford out of the same budget:

    k(policy, B) = floor(B / mean_chunk_tokens(policy))

Chunk length is MEASURED over the whole corpus chunk set with the repo's
``Budget`` (the embed model's wordpiece tokenizer, the same counter every other
budget number in this repo uses), never estimated.

No stratification: every gold cell counts, including the ones that landed in no
chunk at all -- those are recall 0 by construction and their count is reported.

  PYTHONPATH=. .venv/bin/python analysis/token_equiv_recall.py \
      --dataset hitab --population hitab_dev_lookup_all
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "analysis"))

import numpy as np                                                   # noqa: E402
import corpus_dump_vs_cell as cdv                                    # noqa: E402
from baseline_comparison_llm import Budget                           # noqa: E402
from header_path_coverage import header_path, load_corpus            # noqa: E402
from rag_agent.retrieve.encoders import default_encoder              # noqa: E402
from rag_agent.retrieve.hybrid_index import HybridIndex, _minmax     # noqa: E402
from stratified_recall import POLICIES, build_chunks                 # noqa: E402

BUDGETS = (2560, 5120, 10240)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="hitab",
                    choices=["hitab", "multihiertt", "aitqa", "realhitbench"])
    ap.add_argument("--policy", default="all")
    ap.add_argument("--cell-scheme", default="S3c",
                    choices=["S2", "S3", "S3c", "mt2net"])
    ap.add_argument("--retriever", default="hybrid", choices=list(cdv.ALPHA))
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--population", default="hitab_dev_lookup_all")
    ap.add_argument("--rhb-question-types", nargs="*", default=[])
    ap.add_argument("--rhb-em-only", action="store_true")
    ap.add_argument("--mh-queries", type=int, default=400)
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--budgets", type=int, nargs="*", default=list(BUDGETS))
    ap.add_argument("--cache-dir", default=".cache/corpus_dump_vs_cell")
    ap.add_argument("--out-dir", default="results/hpc/token_equiv")
    a = ap.parse_args()

    pols = POLICIES if a.policy == "all" else tuple(
        p.strip() for p in a.policy.split(",") if p.strip())
    bad = set(pols) - set(POLICIES)
    if bad:
        ap.error(f"unknown policy {sorted(bad)}; pick from {POLICIES}")

    t0 = time.time()
    C = load_corpus(a)
    pop = C.queries[:a.max_queries] if a.max_queries else C.queries
    print(f"[corpus] {len(C.tids)} tables / {len(C.cell_owner)} cells | "
          f"[pop] {len(pop)} queries | {time.time() - t0:.0f}s", flush=True)

    bud = Budget(a.embed_model)
    enc = cdv._CachedEncoder(default_encoder(model_name=a.embed_model),
                             a.cache_dir, f"{a.dataset}_{a.split}_{a.embed_model}")
    by_table = defaultdict(dict)
    for n, (t, i, j) in enumerate(C.cell_owner):
        by_table[t][(i, j)] = n

    outdir = Path(a.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    for policy in pols:
        t0 = time.time()
        chunks, owner = build_chunks(C, policy, bud, by_table, a.cell_scheme)
        toks = np.array([bud.count(c.text) for c in chunks], dtype=np.int64)
        holds = defaultdict(list)
        for n, (tid, cells) in enumerate(owner):
            for (i, j) in cells:
                holds[(tid, i, j)].append(n)
        ks = {B: max(1, int(B // (toks.mean() if toks.mean() else 1)))
              for B in a.budgets}
        kmax = max(ks.values())
        ix = HybridIndex(chunks, encoder=enc, alpha=0.5)
        print(f"  [{policy}] {len(chunks)} chunks, mean {toks.mean():.2f} tok, "
              f"k={ks}, index {time.time() - t0:.0f}s", flush=True)

        def scores(question):
            bm = ix._bm25_scores(question)
            if a.alpha == 0.0:
                return bm
            dn = ix._dense_scores(question)
            if a.alpha == 1.0:
                return dn
            return a.alpha * _minmax(dn) + (1 - a.alpha) * _minmax(bm)

        rows, t0 = [], time.time()
        for n_q, q in enumerate(pop, 1):
            gold = [g for g in sorted(q["gold_cells"]) if g[0] in by_table
                    and (g[1], g[2]) in by_table[g[0]]]
            if not gold:
                continue
            s = scores(q["question"])
            top = np.argpartition(-s, min(kmax, len(s) - 1))[:kmax]
            top = top[np.argsort(-s[top])]
            rank_of = {int(p): r for r, p in enumerate(top)}
            for (tid, i, j) in gold:
                rp, cp, v = C.cell_paths[by_table[tid][(i, j)]]
                if not header_path(rp, cp):
                    continue                  # the cell IS a header; excluded
                cs = holds.get((tid, i, j), [])
                best = (min(rank_of.get(c, 10 ** 9) for c in cs) if cs else 10 ** 9)
                rows.append({"query_id": str(q["query_id"]), "table_id": tid,
                             "gold_row": i, "gold_col": j,
                             "in_chunk": int(bool(cs)), "rank": best})
            if n_q % 200 == 0:
                print(f"    {n_q}/{len(pop)}  {time.time() - t0:.0f}s", flush=True)

        res = {"dataset": a.dataset, "policy": policy,
               "population": a.population if a.dataset == "hitab" else a.dataset,
               "retriever": a.retriever, "alpha": a.alpha,
               "cell_scheme": a.cell_scheme,
               "n_chunks": len(chunks),
               "mean_chunk_tokens": float(toks.mean()),
               "median_chunk_tokens": float(np.median(toks)),
               "total_chunk_tokens": int(toks.sum()),
               "n_gold_cells": len(rows),
               "n_not_in_any_chunk": sum(1 for r in rows if not r["in_chunk"]),
               "budgets": {}}
        for B, k in ks.items():
            res["budgets"][str(B)] = {
                "k": k, "cost_tokens": float(k * toks.mean()),
                "recall": (sum(1 for r in rows if r["rank"] < k) / len(rows)
                           if rows else None)}
        (outdir / f"{a.dataset}_{policy}_tokeq.json").write_text(
            json.dumps(res, indent=2, ensure_ascii=False))
        with open(outdir / f"{a.dataset}_{policy}_cells.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["query_id", "table_id", "gold_row", "gold_col",
                        "in_chunk", "rank"])
            for r in rows:
                w.writerow([r["query_id"], r["table_id"], r["gold_row"],
                            r["gold_col"], r["in_chunk"], r["rank"]])
        print(f"  [{policy}] {json.dumps(res['budgets'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
