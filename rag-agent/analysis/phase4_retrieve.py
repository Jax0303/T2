#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Phase 4 Task B -- re-derive retrieval for sample_294, then fill greedily.

Phase 3 stored only each gold cell's rank, never the retrieved chunks, so the
reader records cannot be filled from those files. Retrieval is deterministic, so
this re-runs it with the SAME index, alpha and chunk builder and checks every
re-derived gold rank against the stored one (B1). Only then does it measure what
a 4096-Qwen-token context actually holds (B2).

  PYTHONPATH=. .venv/bin/python analysis/phase4_retrieve.py
"""
from __future__ import annotations

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
from qwen_equiv_k import (MODEL, REV, B_READER, args_for, count_all,  # noqa: E402
                          prompt_text)
from rag_agent.retrieve.encoders import default_encoder              # noqa: E402
from rag_agent.retrieve.hybrid_index import HybridIndex, _minmax     # noqa: E402
from stratified_recall import build_chunks                           # noqa: E402
from transformers import AutoTokenizer                               # noqa: E402

ALPHA = 0.7          # CLAUDE.md 7; the value Phase 3 ran
SCHEME = "S3c"
EMBED = "BAAI/bge-small-en-v1.5"
TOPN = 200
POLICIES = ("P1_fixed_512", "P4_path_cell")
# pool -> (dataset, population, split-tag used for the encoder cache)
POOLS = {"hitab_lookup": ("hitab", "hitab_dev_lookup_all", "dev"),
         "hitab_arith": ("hitab", "hitab_dev_corpus_arith", "dev"),
         "aitqa": ("aitqa", "aitqa", "dev"),
         "rhb_fact": ("realhitbench", "rhb_lookup_all", "dev"),
         "rhb_num": ("realhitbench", "rhb_lookup_all", "dev")}
SAMPLE = Path(sys.argv[1] if len(sys.argv) > 1
              else "results/phase4/sample_294.csv")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "results/phase4/retrieval_294")
# Phase 3b's stored ranks; ranks at or past that run's kmax were written as 1e9
PH3 = Path("results/hpc/token_equiv")


def phase3_ranks(dataset, policy):
    """(query_id, tid, i, j) -> rank, plus the kmax that run could see."""
    f = PH3 / f"{dataset}_{policy}_cells.csv"
    if not f.exists():
        return None, None
    d = {(r["query_id"], r["table_id"], int(r["gold_row"]), int(r["gold_col"])):
         int(r["rank"]) for r in csv.DictReader(open(f))}
    j = json.load(open(PH3 / f"{dataset}_{policy}_tokeq.json"))
    kmax = max(v["k"] for v in j["budgets"].values())
    return d, kmax


def main() -> int:
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    bud = Budget(EMBED)
    sample = list(csv.DictReader(open(SAMPLE)))
    by_pool = defaultdict(list)
    for r in sample:
        by_pool[r["pool"]].append(r)
    OUT.mkdir(parents=True, exist_ok=True)

    corpora, qmap = {}, {}
    for pool, (ds, pop, _) in POOLS.items():
        if (ds, pop) not in corpora:
            corpora[(ds, pop)] = load_corpus(args_for(ds, pop))
        qmap[pool] = {str(q["query_id"]): q for q in corpora[(ds, pop)].queries}

    b1 = []          # (policy, pool, n_cells, n_compared, n_mismatch, n_skipped)
    b2 = []
    for policy in POLICIES:
        for ds, pop_key in (("hitab", "hitab_dev_lookup_all"),
                            ("aitqa", "aitqa"),
                            ("realhitbench", "rhb_lookup_all")):
            C = corpora[(ds, pop_key)]
            by_table = defaultdict(dict)
            for n, (t, i, j) in enumerate(C.cell_owner):
                by_table[t][(i, j)] = n
            t0 = time.time()
            chunks, owner = build_chunks(C, policy, bud, by_table, SCHEME)
            holds = defaultdict(list)
            for n, (tid, cells) in enumerate(owner):
                for (i, j) in cells:
                    holds[(tid, i, j)].append(n)
            enc = cdv._CachedEncoder(default_encoder(model_name=EMBED),
                                     ".cache/corpus_dump_vs_cell",
                                     f"{ds}_dev_{EMBED}")
            ix = HybridIndex(chunks, encoder=enc, alpha=0.5)
            qt = count_all(tok, [c.text + "\n" for c in chunks])
            stored, kmax3 = phase3_ranks(ds, policy)
            print(f"[{policy}/{ds}] {len(chunks)} chunks, index {time.time()-t0:.0f}s,"
                  f" phase3 kmax={kmax3}", flush=True)

            for pool in [p for p, v in POOLS.items() if v[0] == ds]:
                rows = by_pool[pool]
                if not rows:
                    continue
                fh = open(OUT / f"{policy}_{pool}.jsonl", "w")
                n_cmp = n_mis = n_skip = 0
                mis_ex = []
                used, ptoks, hit, ncells, multi = [], [], [], 0, 0
                for r in rows:
                    qid = str(r["query_id"])
                    q = qmap[pool][qid]
                    bm = ix._bm25_scores(q["question"])
                    dn = ix._dense_scores(q["question"])
                    s3 = ALPHA * _minmax(dn) + (1 - ALPHA) * _minmax(bm)
                    s = s3
                    n_top = min(TOPN, len(s))
                    top = np.argpartition(-s, n_top - 1)[:n_top]
                    top = top[np.argsort(-s[top])]
                    rank_of = {int(p): i for i, p in enumerate(top)}
                    # B1 compares against Phase 3, so it has to rank the way
                    # Phase 3 did: np.argsort is not stable, and tied scores
                    # (they happen -- two chunks with identical hybrid score)
                    # come out in a different order when argpartition is handed
                    # a different kmax. Rank once more at Phase 3's own kmax.
                    if kmax3:
                        n3 = min(kmax3, len(s3))
                        t3 = np.argpartition(-s3, n3 - 1)[:n3]
                        t3 = t3[np.argsort(-s3[t3])]
                        rank3 = {int(p): i for i, p in enumerate(t3)}
                    else:
                        rank3 = {}

                    # --- greedy fill under the reader's tokenizer ---
                    ov = len(tok(prompt_text(tok, "", q["question"]),
                                 add_special_tokens=False)["input_ids"])
                    cum, keep = ov, []
                    for p in top:
                        if cum + int(qt[p]) > B_READER:
                            break
                        cum += int(qt[p])
                        keep.append(int(p))
                    ctx = "\n".join(chunks[p].text for p in keep)
                    ptok = len(tok(prompt_text(tok, ctx, q["question"]),
                                   add_special_tokens=False)["input_ids"])
                    used.append(len(keep))
                    ptoks.append(ptok)
                    kept = set(keep)

                    gold = [g for g in json.loads(r["gold_cells"])
                            if g[0] in by_table and (g[1], g[2]) in by_table[g[0]]]
                    gold = [g for g in gold if header_path(
                        *C.cell_paths[by_table[g[0]][(g[1], g[2])]][:2])]
                    if len(gold) > 1:
                        multi += 1
                    for (tid, i, j) in gold:
                        ncells += 1
                        cs = holds.get((tid, i, j), [])
                        hit.append(int(any(c in kept for c in cs)))
                        best = min((rank_of.get(c, 10 ** 9) for c in cs),
                                   default=10 ** 9)
                        best3 = min((rank3.get(c, 10 ** 9) for c in cs),
                                    default=10 ** 9)
                        key = (qid, str(tid), i, j)
                        if stored is None or key not in stored:
                            n_skip += 1
                            continue
                        n_cmp += 1
                        old = stored[key]
                        # ranks past the Phase 3 run's kmax were stored as 1e9;
                        # both must agree on "outside", not on the exact rank
                        ok = (old == best3) if old < kmax3 else (best3 >= kmax3)
                        if not ok:
                            n_mis += 1
                            if len(mis_ex) < 5:
                                mis_ex.append({"key": key, "phase3": old,
                                               "now": best3, "top200": best,
                                               "kmax3": kmax3})
                    fh.write(json.dumps({
                        "query_id": qid, "pool": pool, "dataset": ds,
                        "policy": policy, "question": q["question"],
                        "gold_cells": gold, "overhead_tokens": ov,
                        "n_chunks_used": len(keep), "prompt_tokens": ptok,
                        "topk": [{"rank": i, "table_id": chunks[p].table_id,
                                  "chunk_id": chunks[p].chunk_id,
                                  "cells": sorted(list(c) for c in owner[p][1]),
                                  "text": chunks[p].text,
                                  "qwen_tokens": int(qt[p]),
                                  "used": int(p) in kept}
                                 for i, p in enumerate(top)]}) + "\n")
                fh.close()
                b1.append((policy, pool, ncells, n_cmp, n_mis, n_skip, mis_ex))
                u, pt = np.array(used), np.array(ptoks)
                b2.append((policy, pool, len(rows), ncells, multi,
                           u.mean(), np.median(u), u.min(), u.max(),
                           pt.mean(), np.median(pt), pt.max(),
                           float(np.mean(hit)) if hit else None))
                print(f"  {pool:<13} cells={ncells:<4} cmp={n_cmp:<4} "
                      f"mismatch={n_mis:<4} skip={n_skip:<4} "
                      f"used mean={u.mean():.1f} recall={np.mean(hit):.4f}",
                      flush=True)

    Path("results/phase4").mkdir(parents=True, exist_ok=True)
    json.dump({"b1": [{"policy": p, "pool": q, "n_gold_cells": c,
                       "n_compared": m, "n_mismatch": x, "n_skipped": s,
                       "examples": e} for p, q, c, m, x, s, e in b1],
               "b2": [dict(zip(("policy", "pool", "n_queries", "n_gold_cells",
                                "n_queries_multi_gold", "used_mean", "used_median",
                                "used_min", "used_max", "ptok_mean", "ptok_median",
                                "ptok_max", "recall"), map(
                   lambda v: float(v) if isinstance(v, np.generic) else v, r)))
                      for r in b2]},
              open(f"results/phase4/taskB_{OUT.name}.json", "w"), indent=2)
    print(f"\n-> results/phase4/taskB_{OUT.name}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
