"""How far is cell retrieval from 100%, and what is actually stopping it?

``OSC`` in every run on disk is recall *inside a 512-token budget*, which admits
about twelve cell sentences -- so it is recall@12 wearing a different name. That
conflates two failures a fix has to tell apart:

  RANKING   the gold cell is ranked below the cut. More budget cannot help if
            it is at rank 4,000; a better retriever must move it.
  CAPACITY  the gold cell is ranked well above the cut but the budget ran out.
            No retriever change helps; the context has to hold more.

This sweeps k and alpha over the SAME index the runs use and reports recall@k
for a ladder, so the two are separable and the ceiling is a measured number
rather than an argument. It also reports what no ranking can ever fix: gold
cells whose sentence is byte-identical to another cell's, and gold cells that
are not in the index at all.

Retrieval only -- no reader, no GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from corpus_dump_vs_cell import (ALPHA, FROZEN_POP, RHB_POPS, _CachedEncoder,
                                 aitqa_corpus, hitab_corpus,
                                 realhitbench_corpus)
from error_worksheet import render_cells
from baseline_comparison_llm import Budget
from rag_agent.retrieve.encoders import default_encoder, default_prefixes
from rag_agent.retrieve.hybrid_index import HybridIndex, _minmax
from rag_agent.serialization.base import Chunk

K_LADDER = (1, 5, 12, 25, 50, 100, 250, 500, 1000, 5000)


def fills_budget(order, tok, budget: int, gold_pos, last_rank: int) -> int:
    """Does the driver's fill loop admit EVERY gold cell within ``budget``?

    Replays ``corpus_dump_vs_cell``'s loop exactly, including the part that is
    easy to get wrong: a cell that does not fit is SKIPPED, not a stop, and the
    loop only breaks once the corpus's smallest cell cannot fit either. A prefix
    ``cumsum`` would model a stop and undercount every budget.

    Scans no further than the worst-ranked gold cell -- past that the answer is
    already 0 -- which is what makes the ladder affordable on 58k cells.
    """
    used, mn, want = 0, int(tok.min()), set(gold_pos)
    for r, pos in enumerate(order):
        if r > last_rank:
            return 0
        n = int(tok[pos])
        if used + n > budget:
            if used + mn > budget:
                return 0
            continue
        used += n
        want.discard(int(pos))
        if not want:
            return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="hitab",
                    choices=["hitab", "aitqa", "realhitbench"])
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--population", default="hitab_dev_lookup_all")
    ap.add_argument("--cell-scheme", default="S3c")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--cache-dir", default=".cache/corpus_dump_vs_cell")
    ap.add_argument("--alphas", default="0,0.5,0.7,1.0",
                    help="dense weight; 1.0 is what every committed run used")
    ap.add_argument("--budgets", default="512,1024,2048,4096")
    ap.add_argument("--query-prefix", default="auto",
                    choices=["auto", "none"],
                    help="auto = the instruction the embed model was TRAINED "
                         "with, measured beside the bare query the driver sends")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.dataset == "hitab":
        C = hitab_corpus(args.data_dir, args.split, args.population)
        pop_name = args.population
    elif args.dataset == "aitqa":
        C = aitqa_corpus()
        pop_name = FROZEN_POP["aitqa"]
    else:
        pop_name = (args.population if args.population in RHB_POPS
                    else FROZEN_POP["realhitbench"])
        C = realhitbench_corpus(pin=pop_name, **RHB_POPS[pop_name])
    render_cells(C, args.cell_scheme, False)

    inner = default_encoder(model_name=args.embed_model)
    enc = _CachedEncoder(inner, args.cache_dir,
                         f"{args.dataset}_{args.split}_{args.embed_model}")
    chunks = [Chunk(table_id=t, chunk_id=f"c::{t}::{i}:{j}", text=txt,
                    scheme=args.cell_scheme, kind="cell")
              for txt, (t, i, j) in zip(C.cell_text, C.cell_owner)]
    ix = HybridIndex(chunks, encoder=enc, alpha=0.5)
    bud = Budget(args.embed_model)
    tok = np.array([bud.count(c.text) for c in chunks], dtype=np.int32)
    pos_of_cell = {k: n for n, k in enumerate(C.cell_owner)}
    n_cells = len(chunks)

    # what NO ranking can fix -------------------------------------------------
    text_count = Counter(C.cell_text)
    missing, ambiguous = 0, 0
    for q in C.queries:
        gp = [pos_of_cell.get(g) for g in q["gold_cells"]]
        if any(p is None for p in gp):
            missing += 1
            continue
        if any(text_count[C.cell_text[p]] > 1 for p in gp):
            ambiguous += 1
    hard = {"queries": len(C.queries),
            "gold_cell_not_indexed": missing,
            "gold_cell_text_is_duplicated": ambiguous,
            "duplicated_share": round(ambiguous / max(len(C.queries), 1), 4),
            "note": "a duplicated gold sentence can still be retrieved -- the "
                    "twin scores identically -- but nothing in the query can "
                    "prefer the right one, so its rank is a coin flip"}

    prefixes = {"none": ""}
    if args.query_prefix == "auto":
        qpre = default_prefixes(inner.name)[0]
        if qpre:
            prefixes["trained"] = qpre

    alphas = [float(a) for a in args.alphas.split(",")]
    budgets = [int(b) for b in args.budgets.split(",")]
    out = {"dataset": args.dataset, "population": pop_name,
           "cell_scheme": args.cell_scheme, "embed_model": args.embed_model,
           "encoder_name": inner.name, "n_cells": n_cells,
           "n_queries": len(C.queries), "hard_limits": hard, "sweeps": {}}

    for pname, pfx in prefixes.items():
        # scores are computed ONCE per query and re-fused per alpha, so the
        # ladder costs one pass no matter how many alphas are asked for
        per_alpha = {a: {"ranks": [], "budget_hit": {b: [] for b in budgets}}
                     for a in alphas}
        for n, q in enumerate(C.queries, 1):
            gp = [pos_of_cell[g] for g in q["gold_cells"] if g in pos_of_cell]
            if not gp:
                continue
            bm = ix._bm25_scores(q["question"])
            dn = ix._dense_scores(pfx + q["question"])
            for a in alphas:
                fused = (dn if a == 1.0 else bm if a == 0.0 else
                         a * _minmax(dn) + (1 - a) * _minmax(bm))
                order = np.argsort(-fused, kind="stable")
                rank = np.empty(n_cells, dtype=np.int64)
                rank[order] = np.arange(n_cells)
                last = max(int(rank[p]) for p in gp)
                per_alpha[a]["ranks"].append(last + 1)
                for b in budgets:
                    per_alpha[a]["budget_hit"][b].append(
                        fills_budget(order, tok, b, gp, last))
            if n % 100 == 0:
                print(f"  [{pname}] {n}/{len(C.queries)}", flush=True)
        for a in alphas:
            r = np.array(per_alpha[a]["ranks"])
            out["sweeps"][f"{pname}|alpha={a}"] = {
                "n": int(r.size),
                "recall_at_k": {str(k): round(float((r <= k).mean()), 4)
                                for k in K_LADDER},
                "gold_rank": {"p50": float(np.percentile(r, 50)),
                              "p90": float(np.percentile(r, 90)),
                              "p99": float(np.percentile(r, 99)),
                              "max": int(r.max())},
                "osc_at_budget": {str(b): round(float(np.mean(v)), 4)
                                  for b, v in per_alpha[a]["budget_hit"].items()},
            }

    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"[write] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
