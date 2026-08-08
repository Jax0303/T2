#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Does structural aggregate-row injection raise OSC, or just spend more budget?

The row-axis diagnosis (`results/diag_row_failures.json`) puts 42 of 62 row-axis
failures in one bucket: **total pairing** — share/ratio questions whose
denominator is a table-level total row no header text names, so no resolver can
bind it. Injection (`inject_structural=True`) is the only treatment that moves
that bucket (`e6_scope_treatments`: row_cov .615 -> .845, decomposition-correct
67 -> 96 of 161), and OSC|decomposition-correct is 1.00, so decomposition is the
whole remaining bottleneck.

But injection adds cells, and `e6` only ever compared it to a *fixed* dense k=10
under S2 row chunks — a different budget in a different unit, not the per-query
control this repo requires of any arm that grows the evidence set (scope-
enumeration union: +.140 raw -> +.014, p=.55). So the treatment was neither
confirmed nor properly killed.

This closes that: same pipeline as the surviving resolver claim
(`resolver_osc_matched.py`), S3 cell index, embed resolver, and a control that
gives plain similarity ranking **exactly the injected arm's cell count**.

Run:
    PYTHONPATH=. .venv/bin/python scripts/inject_osc_matched.py --split dev
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--deep", type=int, default=120,
                    help="plain ranked pool, cut per query to the injected arm's size")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    env = run_env(args.seed, args.embed_model)
    out_path = args.out or f"results/inject_osc_matched_{args.split}.json"

    queries, _ = load_queries(args.data_dir, args.split)
    pop = [q for q in queries
           if (q.aggregation or "none") in ARITH and len(q.gold_operands) >= 1]
    if args.max:
        pop = pop[: args.max]
    print(f"[pop] hitab {args.split} arith w/ operands: {len(pop)}", flush=True)

    enc = default_encoder(model_name=args.embed_model)
    r = OperandTargetedRetriever(encoder=enc, scheme="S3", caption_length="long",
                                 embed_resolver=True)

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
        # neutral control ranking: whole-question similarity over the same index
        deep = cells_of([h.chunk for h in r.index_table(ot).search(q.question,
                                                                  k=args.deep)])
        rec = {"m": len({(o.row, o.col) for o in q.gold_operands})}
        for k in args.ks:
            base = cells_of([h.chunk for h in r.retrieve(q.question, ot, k=k).retrieved])
            inj = cells_of([h.chunk for h in
                            r.retrieve(q.question, ot, k=k,
                                       inject_structural=True).retrieved])
            rec[k] = {
                "base": operand_set_completeness(q.gold_operands, base),
                "inj": operand_set_completeness(q.gold_operands, inj),
                # plain ranking, allowed exactly the injected arm's cell count
                "matched": operand_set_completeness(q.gold_operands, deep[: len(inj)]),
                # same control at the *un-injected* arm's cell count: does
                # operand targeting beat plain similarity on equal budget at all?
                "matched_base": operand_set_completeness(q.gold_operands,
                                                         deep[: len(base)]),
                "pcr_base": per_cell_recall(q.gold_operands, base),
                "pcr_inj": per_cell_recall(q.gold_operands, inj),
                "n_base": len(base), "n_inj": len(inj), "n_deep": len(deep),
            }
        recs.append(rec)
        if n % 25 == 0:
            print(f"  {n}/{len(pop)}", flush=True)

    def block(rs):
        out = {}
        for k in args.ks:
            m_ = lambda f: round(mean(r_[k][f] for r_ in rs), 4)
            pair = lambda t, c: (sum(1 for r_ in rs if r_[k][t] > r_[k][c]),
                                 sum(1 for r_ in rs if r_[k][c] > r_[k][t]))
            b_m, c_m = pair("inj", "matched")
            b_b, c_b = pair("inj", "base")
            b_t, c_t = pair("base", "matched_base")
            out[str(k)] = {
                "osc_base": m_("base"), "osc_inj": m_("inj"),
                "osc_matched": m_("matched"),
                "osc_matched_base_budget": m_("matched_base"),
                "delta_base_vs_matched": round(m_("base") - m_("matched_base"), 4),
                "b_base_only_vs_matched": b_t, "c_matched_only_vs_base": c_t,
                "p_mcnemar_base_vs_matched": round(mcnemar_p(b_t, c_t), 6),
                "delta_vs_matched": round(m_("inj") - m_("matched"), 4),
                "delta_vs_base": round(m_("inj") - m_("base"), 4),
                "pcr_base": m_("pcr_base"), "pcr_inj": m_("pcr_inj"),
                "cells_base": m_("n_base"), "cells_inj": m_("n_inj"),
                "cells_deep_pool": m_("n_deep"),
                "b_inj_only_vs_matched": b_m, "c_matched_only": c_m,
                "p_mcnemar_vs_matched": round(mcnemar_p(b_m, c_m), 6),
                "b_inj_only_vs_base": b_b, "c_base_only": c_b,
                "p_mcnemar_vs_base": round(mcnemar_p(b_b, c_b), 6),
            }
        return out

    m2 = [r_ for r_ in recs if r_["m"] >= 2]
    out = {
        "env": env,
        "leg": "structural aggregate-row injection (inject_structural=True) in "
               "OperandTargetedRetriever, vs the same pipeline without it and "
               "vs a per-query budget-matched plain-similarity control",
        "population": {"name": "hitab_arith_with_operands", "split": args.split,
                       "n": len(recs), "n_m_ge_2": len(m2)},
        "index_unit": {"scheme": "S3", "length": "long", "granularity": "cell",
                       "resolver": "embed"},
        "control": "per-query budget-matched: plain similarity ranking over the "
                   "same S3 index, cut to the injected arm's exact cell count "
                   "(deep pool k=%d)" % args.deep,
        "all": block(recs),
        "m_ge_2": block(m2) if m2 else {},
        "note": "Quote delta_vs_matched. delta_vs_base includes the evidence-set "
                "growth injection causes and is not a claim.",
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["m_ge_2"] or out["all"], indent=2))
    print(f"\nwrote -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
