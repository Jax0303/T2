#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Does the semantic header-path resolver raise OSC, or just spend more budget?

The operand-targeted retrieval path (:class:`OperandTargetedRetriever`, the
pipeline the invention disclosure describes) decomposes a query into header-path
operands via :func:`resolve_against_table`, whose lexical scorer keeps only paths
scoring above 0 — dropping every header that shares no token with the question.
Measured on HiTab dev that is 28.0% of the gold rows and 31.3% of the gold
columns, and nothing is structurally out of reach: every gold row/column has a
real, non-empty header path (``reachable`` 1.000).

:class:`~rag_agent.query.header_embed_resolver.EmbedResolver` already fixes this
for the *enumeration* scripts; this path was simply never wired to it. The wiring
is ``OperandTargetedRetriever(embed_resolver=True)``.

The catch, and why this script exists: a better resolver returns more *distinct*
header paths, so the retriever runs more per-operand searches and its evidence
set grows (~12.0 vs 10.1 cells at k=5). A raw before/after therefore credits the
resolver for a budget change. The control is per query: give the lexical arm a
deep ranked list and cut it to exactly the treatment arm's cell count. A scope
enumeration arm tried earlier died on precisely this control (+.140 raw ->
+.014, p=.55), so it is applied here before any number is quoted.

Run:
    PYTHONPATH=. .venv/bin/python scripts/resolver_osc_matched.py --split dev
    PYTHONPATH=. .venv/bin/python scripts/resolver_osc_matched.py --split train
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
from rag_agent.stores.original_store import build_original_table

ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}


def mcnemar_p(b: int, c: int) -> float:
    from scipy.stats import binomtest
    n = b + c
    return binomtest(b, n, 0.5).pvalue if n else 1.0


def cells_of(res):
    return [(rc.chunk.row_index, rc.chunk.col_index) for rc in res.retrieved
            if rc.chunk.row_index is not None]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--deep", type=int, default=80,
                    help="lexical ranked pool, cut per query to the treatment's size")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out_path = args.out or f"results/resolver_osc_matched_{args.split}.json"

    queries, _ = load_queries(args.data_dir, args.split)
    pop = [q for q in queries
           if (q.aggregation or "none") in ARITH and len(q.gold_operands) >= 1]
    if args.max:
        pop = pop[: args.max]
    print(f"[pop] hitab {args.split} arith w/ operands: {len(pop)}", flush=True)

    enc = default_encoder(model_name=args.embed_model)
    lex = OperandTargetedRetriever(encoder=enc, scheme="S3", caption_length="long")
    emb = OperandTargetedRetriever(encoder=enc, scheme="S3", caption_length="long",
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
        deep = cells_of(lex.retrieve(q.question, ot, k=args.deep))
        rec = {"m": len({(o.row, o.col) for o in q.gold_operands})}
        for k in args.ks:
            e = cells_of(emb.retrieve(q.question, ot, k=k))
            l = cells_of(lex.retrieve(q.question, ot, k=k))
            rec[k] = {
                "emb": operand_set_completeness(q.gold_operands, e),
                "lex": operand_set_completeness(q.gold_operands, l),
                # lexical ranking, allowed exactly the treatment's cell count
                "matched": operand_set_completeness(q.gold_operands, deep[: len(e)]),
                "pcr_emb": per_cell_recall(q.gold_operands, e),
                "pcr_lex": per_cell_recall(q.gold_operands, l),
                "n_emb": len(e), "n_lex": len(l),
            }
        recs.append(rec)
        if n % 100 == 0:
            print(f"  {n}/{len(pop)}", flush=True)

    def block(rs):
        out = {}
        for k in args.ks:
            b = sum(1 for r in rs if r[k]["emb"] > r[k]["matched"])
            c = sum(1 for r in rs if r[k]["matched"] > r[k]["emb"])
            m_ = lambda f: round(mean(r[k][f] for r in rs), 4)
            out[str(k)] = {
                "osc_lex": m_("lex"), "osc_emb": m_("emb"),
                "osc_lex_budget_matched": m_("matched"),
                "delta_vs_matched": round(m_("emb") - m_("matched"), 4),
                "pcr_lex": m_("pcr_lex"), "pcr_emb": m_("pcr_emb"),
                "cells_lex": m_("n_lex"), "cells_emb": m_("n_emb"),
                "b_emb_only": b, "c_matched_only": c,
                "p_mcnemar": round(mcnemar_p(b, c), 6),
            }
        return out

    m2 = [r for r in recs if r["m"] >= 2]
    out = {
        "leg": "semantic header-path resolver (EmbedResolver) wired into "
               "OperandTargetedRetriever, vs the lexical scorer",
        "population": {"name": "hitab_arith_with_operands", "split": args.split,
                       "n": len(recs), "n_m_ge_2": len(m2)},
        "index_unit": {"scheme": "S3", "length": "long", "granularity": "cell"},
        "control": "per-query budget-matched: lexical ranking cut to the "
                   "treatment's exact cell count (deep pool k=%d)" % args.deep,
        "all": block(recs),
        "m_ge_2": block(m2) if m2 else {},
        "note": "Quote delta_vs_matched, not osc_emb - osc_lex: the latter "
                "includes the evidence-set growth a better resolver causes.",
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
