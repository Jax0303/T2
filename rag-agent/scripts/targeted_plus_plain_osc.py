#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Can operand targeting add anything to plain similarity at equal budget?

`inject_osc_matched.py` established that `OperandTargetedRetriever` LOSES to a
plain whole-question similarity ranking cut to the same cell count
(delta_base_vs_matched -.180/-.137/-.130 at k=5/10/20, p<=9.4e-4). The
mechanism is visible in `retrieve()`: it spends k slots PER OPERAND, so every
operand whose decomposition missed burns its whole share, while plain
similarity spends the same budget on question-wide relevance.

That is an argument for combining them, not for dropping targeting: the two
rankings fail on different queries. This measures the union under the repo's
per-query budget rule -- round-robin interleave of the targeted ranking and the
plain ranking, deduplicated, cut to exactly the cell count targeting alone
would have spent. If targeting carries any signal plain similarity lacks, the
merged arm beats plain at that count; if it carries none, this is where it dies.

Run:
    PYTHONPATH=. .venv/bin/python scripts/targeted_plus_plain_osc.py --split dev
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.bench.hitab import load_queries
from rag_agent.data.loader import load_table
from rag_agent.eval.operand_set import operand_set_completeness
from rag_agent.retrieve.encoders import default_encoder
from rag_agent.retrieve.operand_retrieval import OperandTargetedRetriever
from rag_agent.runenv import run_env
from rag_agent.stores.original_store import build_original_table

ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}


def mcnemar_p(b: int, c: int) -> float:
    from scipy.stats import binomtest
    n = b + c
    return binomtest(b, n, 0.5).pvalue if n else 1.0


def cells_of(chunks):
    return [(c.row_index, c.col_index) for c in chunks if c.row_index is not None]


def interleave(a, b, n):
    """Round-robin merge of two ranked cell lists, deduplicated, cut to n."""
    out, seen = [], set()
    for i in range(max(len(a), len(b))):
        for src in (a, b):
            if i < len(src) and src[i] not in seen and len(out) < n:
                seen.add(src[i])
                out.append(src[i])
        if len(out) >= n:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--deep", type=int, default=120)
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    env = run_env(args.seed, args.embed_model)
    out_path = args.out or f"results/targeted_plus_plain_{args.split}.json"

    queries, _ = load_queries(args.data_dir, args.split)
    pop = [q for q in queries
           if (q.aggregation or "none") in ARITH and len(q.gold_operands) >= 1]
    if args.max:
        pop = pop[: args.max]
    print(f"[pop] hitab {args.split} arith w/ operands: {len(pop)}", flush=True)

    enc = default_encoder(model_name=args.embed_model)
    r = OperandTargetedRetriever(encoder=enc, scheme="S3",
                                 caption_template="structural", embed_resolver=True)

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
        deep = cells_of([h.chunk for h in
                         r.index_table(ot).search(q.question, k=args.deep)])
        rec = {"m": len({(o.row, o.col) for o in q.gold_operands})}
        for k in args.ks:
            tgt = cells_of([h.chunk for h in r.retrieve(q.question, ot, k=k).retrieved])
            n_cells = len(tgt)
            rec[k] = {
                "targeted": operand_set_completeness(q.gold_operands, tgt),
                "plain": operand_set_completeness(q.gold_operands, deep[:n_cells]),
                "merged": operand_set_completeness(
                    q.gold_operands, interleave(tgt, deep, n_cells)),
                "n_cells": n_cells,
            }
        recs.append(rec)
        if n % 25 == 0:
            print(f"  {n}/{len(pop)}", flush=True)

    def block(rs):
        out = {}
        for k in args.ks:
            m_ = lambda f: round(mean(x[k][f] for x in rs), 4)
            pair = lambda t, c: (sum(1 for x in rs if x[k][t] > x[k][c]),
                                 sum(1 for x in rs if x[k][c] > x[k][t]))
            b_mp, c_mp = pair("merged", "plain")
            b_mt, c_mt = pair("merged", "targeted")
            b_tp, c_tp = pair("targeted", "plain")
            out[str(k)] = {
                "osc_targeted": m_("targeted"), "osc_plain": m_("plain"),
                "osc_merged": m_("merged"), "cells": m_("n_cells"),
                "delta_merged_minus_plain": round(m_("merged") - m_("plain"), 4),
                "delta_targeted_minus_plain": round(m_("targeted") - m_("plain"), 4),
                "merged_only_vs_plain": b_mp, "plain_only_vs_merged": c_mp,
                "p_merged_vs_plain": round(mcnemar_p(b_mp, c_mp), 6),
                "merged_only_vs_targeted": b_mt, "targeted_only_vs_merged": c_mt,
                "p_merged_vs_targeted": round(mcnemar_p(b_mt, c_mt), 6),
                "targeted_only_vs_plain": b_tp, "plain_only_vs_targeted": c_tp,
                "p_targeted_vs_plain": round(mcnemar_p(b_tp, c_tp), 6),
            }
        return out

    m2 = [x for x in recs if x["m"] >= 2]
    out = {
        "env": env,
        "leg": "operand-targeted retrieval merged with plain whole-question "
               "similarity, both cut to the cell count targeting alone spends",
        "population": {"name": "hitab_arith_with_operands", "split": args.split,
                       "n": len(recs), "n_m_ge_2": len(m2)},
        "index_unit": {"scheme": "S3", "granularity": "cell", "resolver": "embed"},
        "control": "per-query budget-matched: every arm gets exactly the "
                   "targeted arm's cell count (deep pool k=%d)" % args.deep,
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
