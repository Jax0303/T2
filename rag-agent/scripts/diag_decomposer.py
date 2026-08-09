#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""What is the query decomposer actually capable of?

`decompose_operands` is upstream of everything that has been losing: the
operand-targeted retriever spends its budget per operand, and the completeness
verdict asks whether each operand was found. Both lose to baselines that do not
decompose at all. Before fixing either, measure the thing they share.

Three questions, all against gold operand cells (HiTab `linked_cells`):

  ceiling      -- do the decomposed paths, resolved to cells, CONTAIN the whole
                  gold operand set? This is the best any downstream retriever
                  could do with this decomposition, before retrieval error.
  cardinality  -- `max_rows` x `max_cols` caps the operand count at 4 by
                  construction. How many gold sets are larger than the cap?
  addressing   -- of the gold cells, how many does some decomposed path name,
                  and how much of what the paths name is not gold at all?

Run: PYTHONPATH=. .venv/bin/python scripts/diag_decomposer.py --split dev
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.bench.hitab import load_queries
from rag_agent.data.loader import load_table
from rag_agent.retrieve.encoders import default_encoder
from rag_agent.retrieve.operand_retrieval import decompose_operands
from rag_agent.runenv import run_env
from rag_agent.stores.original_store import build_original_table

ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}
_NORM = re.compile(r"[^a-z0-9]+")


def norm(s) -> str:
    return _NORM.sub(" ", str(s).lower()).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--max-rows", type=int, default=2)
    ap.add_argument("--max-cols", type=int, default=2)
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    env = run_env(args.seed, args.embed_model)
    out_path = args.out or f"results/diag_decomposer_{args.split}.json"

    queries, _ = load_queries(args.data_dir, args.split)
    pop = [q for q in queries
           if (q.aggregation or "none") in ARITH and len(q.gold_operands) >= 1]
    if args.max:
        pop = pop[: args.max]
    print(f"[pop] hitab {args.split} arith w/ operands: {len(pop)}", flush=True)

    enc = default_encoder(model_name=args.embed_model)
    from rag_agent.query.header_embed_resolver import EmbedResolver
    resolver = EmbedResolver(enc)

    ots, recs = {}, []
    for n, q in enumerate(pop, 1):
        tid = q.gold_table_id
        if tid not in ots:
            try:
                ots[tid] = build_original_table(load_table(tid, args.data_dir))
            except Exception:
                ots[tid] = None
        ot = ots[tid]
        if ot is None:
            continue
        gold = {(o.row, o.col) for o in q.gold_operands}
        ops = decompose_operands(q.question, ot, encoder=resolver,
                                 max_rows=args.max_rows, max_cols=args.max_cols)

        # resolve each decomposed path to the cells whose own path carries it
        named = set()
        for op in ops:
            want = [norm(s) for s in op.header_path if norm(s)]
            if not want:
                continue
            for r_ in range(ot.n_rows):
                rp = " ".join(norm(s) for s in ot.row_path(r_))
                for c_ in range(ot.n_cols):
                    cp = " ".join(norm(s) for s in ot.col_path(c_))
                    have = rp + " " + cp
                    if all(w in have for w in want):
                        named.add((r_, c_))

        hit = len(gold & named)
        recs.append({
            "m": len(gold),
            "n_ops": len(ops),
            "ceiling": int(gold <= named),
            "gold_named": hit,
            "n_named": len(named),
            "over_cap": int(len(gold) > args.max_rows * args.max_cols),
        })
        if n % 25 == 0:
            print(f"  {n}/{len(pop)}", flush=True)

    def block(rs):
        n = len(rs)
        by_m = Counter(x["m"] for x in rs)
        return {
            "n": n,
            "ceiling_rate": round(mean(x["ceiling"] for x in rs), 4),
            "gold_cell_coverage": round(
                sum(x["gold_named"] for x in rs) / sum(x["m"] for x in rs), 4),
            "mean_cells_named": round(mean(x["n_named"] for x in rs), 2),
            "mean_gold_m": round(mean(x["m"] for x in rs), 2),
            "mean_operands_found": round(mean(x["n_ops"] for x in rs), 2),
            "share_gold_larger_than_operand_cap": round(
                mean(x["over_cap"] for x in rs), 4),
            "gold_m_histogram": {str(k): v for k, v in sorted(by_m.items())},
            "ceiling_by_over_cap": {
                "within_cap": round(mean([x["ceiling"] for x in rs
                                          if not x["over_cap"]] or [0]), 4),
                "over_cap": round(mean([x["ceiling"] for x in rs
                                        if x["over_cap"]] or [0]), 4),
            },
        }

    m2 = [x for x in recs if x["m"] >= 2]
    out = {
        "env": env,
        "leg": "decomposer capability: can the decomposed header paths address "
               "the gold operand set at all, before any retrieval happens?",
        "population": {"name": "hitab_arith_with_operands", "split": args.split,
                       "n": len(recs), "n_m_ge_2": len(m2)},
        "operand_cap": args.max_rows * args.max_cols,
        "all": block(recs),
        "m_ge_2": block(m2) if m2 else {},
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["m_ge_2"] or out["all"], indent=2))
    print(f"\nwrote -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
