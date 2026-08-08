#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Was operand targeting's budget-matched loss a small-pool artifact?

`inject_osc_matched.py` found that inside ONE table, at equal cell budget, plain
whole-question similarity beats operand-targeted retrieval (-.18/-.14/-.13 at
k=5/10/20, p<=9.4e-4) and that aggregate-row injection loses too. The obvious
defence is stage separation: a single HiTab table is a ~200-cell pool, small
enough that one query vector finds everything, and decomposition only earns its
keep when the pool is a corpus.

That defence names pool size as the variable, so this varies pool size and holds
everything else fixed -- the same design as E-A, and the same queries, gold,
index unit (S3 cell, long) and resolver (embed) as the single-table run. Only
the candidate pool changes: from the query's own table to every table in the
population, indexed together.

The table gate is given to BOTH arms (operands are resolved against the gold
table), so this measures the cell-retrieval stage alone. Cells from other tables
can never cover a gold operand, so distractors cost both arms budget equally.

Run:
    PYTHONPATH=. .venv/bin/python scripts/pool_scale_matched.py --split dev
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
from rag_agent.eval.operand_set import operand_set_completeness, per_cell_recall
from rag_agent.retrieve.encoders import default_encoder
from rag_agent.retrieve.hybrid_index import HybridIndex
from rag_agent.retrieve.operand_retrieval import OperandTargetedRetriever
from rag_agent.runenv import run_env
from rag_agent.serialization import caption as s3
from rag_agent.stores.original_store import build_original_table

ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}


def mcnemar_p(b: int, c: int) -> float:
    from scipy.stats import binomtest
    n = b + c
    return binomtest(b, n, 0.5).pvalue if n else 1.0


def cells_of(chunks, table_id):
    """(row, col) of the chunks that came from ``table_id`` -- cells of other
    tables spend budget but cannot cover a gold operand."""
    return [(c.row_index, c.col_index) for c in chunks
            if c.table_id == table_id and c.row_index is not None]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--deep", type=int, default=400,
                    help="plain ranked pool, cut per query to the treatment's size")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--distractor-split", default="",
                    help="also index tables from this split as distractors "
                         "(third point on the pool axis)")
    ap.add_argument("--distractor-tables", type=int, default=800)
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    env = run_env(args.seed, args.embed_model)
    out_path = args.out or f"results/pool_scale_matched_{args.split}.json"

    queries, _ = load_queries(args.data_dir, args.split)
    pop = [q for q in queries
           if (q.aggregation or "none") in ARITH and len(q.gold_operands) >= 1]
    if args.max:
        pop = pop[: args.max]

    ots = {}
    for q in pop:
        if q.gold_table_id not in ots:
            try:
                ots[q.gold_table_id] = build_original_table(
                    load_table(q.gold_table_id, args.data_dir))
            except Exception:
                ots[q.gold_table_id] = None
    pop = [q for q in pop if ots.get(q.gold_table_id) is not None]
    print(f"[pop] hitab {args.split} arith w/ operands: {len(pop)} "
          f"over {len(set(q.gold_table_id for q in pop))} tables", flush=True)

    enc = default_encoder(model_name=args.embed_model)
    # the treatment's resolver + per-table index (single-table condition), reused
    # only for its decomposition -- searching happens on the corpus index below
    r = OperandTargetedRetriever(encoder=enc, scheme="S3", caption_template="structural",
                                 embed_resolver=True)

    # corpus pool: every table of the population, one index
    chunks = []
    for tid, ot in ots.items():
        if ot is not None:
            chunks.extend(s3.serialize(ot, granularity="cell", template="structural"))
    n_gold_tables = len([o for o in ots.values() if o])
    n_distractor = 0
    if args.distractor_split:
        import random
        dq, _ = load_queries(args.data_dir, args.distractor_split)
        dtids = sorted({q.gold_table_id for q in dq} - set(ots))
        random.Random(args.seed).shuffle(dtids)
        for tid in dtids[: args.distractor_tables]:
            try:
                dot = build_original_table(load_table(tid, args.data_dir))
            except Exception:
                continue
            chunks.extend(s3.serialize(dot, granularity="cell", template="structural"))
            n_distractor += 1
        print(f"[pool] +{n_distractor} distractor tables from "
              f"{args.distractor_split}", flush=True)
    print(f"[pool] corpus index: {len(chunks)} cells", flush=True)
    corpus = HybridIndex(chunks, encoder=enc)

    recs = []
    for n, q in enumerate(pop, 1):
        ot = ots[q.gold_table_id]
        tid = ot.table_id
        # decomposition: same operands the single-table run used
        ops = r.retrieve(q.question, ot, k=1).operands
        # kept as chunks: the budget cut must happen BEFORE the own-table filter,
        # or the control silently gets a free table gate on its ranked list
        deep = [h.chunk for h in corpus.search(q.question, k=args.deep)]
        rec = {"m": len({(o.row, o.col) for o in q.gold_operands}),
               "n_ops": len(ops)}
        for k in args.ks:
            seen, tgt = set(), []
            for op in ops:
                for h in corpus.search(op.query_text, k=k):
                    if h.chunk.chunk_id not in seen:
                        seen.add(h.chunk.chunk_id)
                        tgt.append(h.chunk)
            tgt_cells = cells_of(tgt, tid)
            # budget is the WHOLE retrieved set (distractor cells included);
            # the control gets the same number of cells from one ranked list
            budget = len(tgt)
            plain_cells = cells_of(deep[:budget], tid)
            rec[k] = {
                "tgt": operand_set_completeness(q.gold_operands, tgt_cells),
                "plain": operand_set_completeness(q.gold_operands, plain_cells),
                "pcr_tgt": per_cell_recall(q.gold_operands, tgt_cells),
                "pcr_plain": per_cell_recall(q.gold_operands, plain_cells),
                "budget": budget,
                "n_tgt_own_table": len(tgt_cells),
                "n_plain_own_table": len(plain_cells),
            }
        recs.append(rec)
        if n % 25 == 0:
            print(f"  {n}/{len(pop)}", flush=True)

    def block(rs):
        out = {}
        for k in args.ks:
            m_ = lambda f: round(mean(r_[k][f] for r_ in rs), 4)
            b = sum(1 for r_ in rs if r_[k]["tgt"] > r_[k]["plain"])
            c = sum(1 for r_ in rs if r_[k]["plain"] > r_[k]["tgt"])
            out[str(k)] = {
                "osc_targeted": m_("tgt"), "osc_plain_matched": m_("plain"),
                "delta": round(m_("tgt") - m_("plain"), 4),
                "pcr_targeted": m_("pcr_tgt"), "pcr_plain": m_("pcr_plain"),
                "budget_cells": m_("budget"),
                "own_table_cells_targeted": m_("n_tgt_own_table"),
                "own_table_cells_plain": m_("n_plain_own_table"),
                "b_targeted_only": b, "c_plain_only": c,
                "p_mcnemar": round(mcnemar_p(b, c), 6),
            }
        return out

    m2 = [r_ for r_ in recs if r_["m"] >= 2]
    out = {
        "env": env,
        "leg": "operand-targeted vs plain similarity at EQUAL cell budget, "
               "candidate pool = every table in the population (corpus scale). "
               "Companion to inject_osc_matched.py, which is the 1-table pool.",
        "population": {"name": "hitab_arith_with_operands", "split": args.split,
                       "n": len(recs), "n_m_ge_2": len(m2),
                       "pool_cells": len(chunks),
                       "pool_tables": n_gold_tables + n_distractor,
                       "gold_tables": n_gold_tables,
                       "distractor_tables": n_distractor,
                       "distractor_split": args.distractor_split or None},
        "index_unit": {"scheme": "S3", "length": "long", "granularity": "cell",
                       "resolver": "embed"},
        "control": "per-query budget-matched: one whole-question ranked list over "
                   "the same corpus index, cut to the targeted arm's chunk count "
                   "(deep pool k=%d)" % args.deep,
        "all": block(recs),
        "m_ge_2": block(m2) if m2 else {},
        "note": "The table gate is given to both arms (operands resolved against "
                "the gold table); only the search pool grows.",
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["m_ge_2"] or out["all"], indent=2))
    print(f"\nwrote -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
