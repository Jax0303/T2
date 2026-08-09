#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Completeness as a RUN-TIME control signal, not a score written down after.

The claimed differentiator is that an incomplete verdict re-enters retrieval
with a wider budget, so completeness steers execution instead of grading it.
Nothing measured that yet: every completeness number in this repo is computed
against gold, after the fact, and could not have driven anything.

This closes it. The verdict is GOLD-FREE -- it asks whether every operand the
decomposer names has a retrieved cell whose header path matches it, which is
information a live system has. Gold is used only to score what the policy did.

Arms, all on the same ranked pool so only the stopping rule differs:
  fixed@n   -- always spend n cells
  loop      -- start at the first rung, escalate while the verdict says
               incomplete, stop at the first rung it accepts

The control that matters is not "does the loop beat the smallest rung" (of
course it does, it spends more) but "does the loop beat a FIXED budget that
spends the same cells on average". That is the repo's standing rule for any arm
that grows the evidence set, and it is what killed scope enumeration and
aggregate-row injection.

Run:
    PYTHONPATH=. .venv/bin/python scripts/completeness_fallback_loop.py --split dev
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from bisect import bisect_left
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.bench.hitab import load_queries
from rag_agent.data.loader import load_table
from rag_agent.eval.operand_set import operand_set_completeness
from rag_agent.retrieve.encoders import default_encoder
from rag_agent.retrieve.operand_retrieval import (OperandTargetedRetriever,
                                                  decompose_operands)
from rag_agent.runenv import run_env
from rag_agent.stores.original_store import build_original_table

ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}
_NORM = re.compile(r"[^a-z0-9]+")


def mcnemar_p(b: int, c: int) -> float:
    from scipy.stats import binomtest
    n = b + c
    return binomtest(b, n, 0.5).pvalue if n else 1.0


def norm(s) -> str:
    return _NORM.sub(" ", str(s).lower()).strip()


def covered(operand, chunks) -> bool:
    """Is there a retrieved cell whose header path carries every segment of
    this operand's target path? The live system can ask this; gold is not
    involved."""
    want = [norm(s) for s in operand.header_path if norm(s)]
    if not want:
        return True
    for ch in chunks:
        have = " ".join(norm(s) for p in (ch.header_paths or []) for s in p)
        if all(w in have for w in want):
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--rungs", type=int, nargs="+",
                    default=[4, 8, 12, 16, 24, 32, 48, 64])
    ap.add_argument("--margins", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="rungs to climb PAST the first the verdict accepts. "
                         "An adaptive policy is a curve, not a point: margin "
                         "buys completeness with cells, and the only fair "
                         "comparison is against the fixed-budget curve.")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    env = run_env(args.seed, args.embed_model)
    out_path = args.out or f"results/completeness_fallback_{args.split}.json"
    rungs = sorted(args.rungs)

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
        index = r.index_table(ot)
        if r.embed_resolver and r._resolver is None:
            from rag_agent.query.header_embed_resolver import EmbedResolver
            r._resolver = EmbedResolver(r.encoder)
        ranked = [h.chunk for h in index.search(q.question, k=max(rungs))]
        operands = decompose_operands(q.question, ot, encoder=r._resolver)

        def cells(cs):
            return [(c.row_index, c.col_index) for c in cs if c.row_index is not None]

        rec = {"m": len({(o.row, o.col) for o in q.gold_operands}),
               "n_operands_found": len(operands), "fixed": {}, "verdict": {}}
        for n_ in rungs:
            got = ranked[:n_]
            rec["fixed"][n_] = {
                "osc": operand_set_completeness(q.gold_operands, cells(got)),
                "cells": len(cells(got)),
            }
            rec["verdict"][n_] = all(covered(o, got) for o in operands)

        # the policy: climb while the verdict rejects, then `margin` rungs past
        # the first it accepts
        first = len(rungs) - 1
        for i_, n_ in enumerate(rungs):
            if rec["verdict"][n_]:
                first = i_
                break
        rec["loop"] = {}
        for mg in args.margins:
            n_ = rungs[min(first + mg, len(rungs) - 1)]
            rec["loop"][mg] = {"rung": n_, **rec["fixed"][n_]}
        recs.append(rec)
        if n % 25 == 0:
            print(f"  {n}/{len(pop)}", flush=True)

    def block(rs):
        by_rung = {n_: {"osc": mean(x["fixed"][n_]["osc"] for x in rs),
                        "cells": mean(x["fixed"][n_]["cells"] for x in rs)}
                   for n_ in rungs}
        order = sorted(by_rung, key=lambda n_: by_rung[n_]["cells"])
        xs = [by_rung[n_]["cells"] for n_ in order]
        ys = [by_rung[n_]["osc"] for n_ in order]

        def fixed_osc_at(cells: float) -> float:
            """What the fixed-budget curve buys at this spend (linear between
            rungs). Comparing an adaptive policy to the nearest RUNG instead of
            to the curve is how a coarse ladder manufactures a saving."""
            if cells <= xs[0]:
                return ys[0]
            if cells >= xs[-1]:
                return ys[-1]
            i = bisect_left(xs, cells)
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            return y0 + (y1 - y0) * (cells - x0) / (x1 - x0) if x1 > x0 else y0

        curve = []
        for mg in args.margins:
            lc = mean(x["loop"][mg]["cells"] for x in rs)
            lo = mean(x["loop"][mg]["osc"] for x in rs)
            near = min(order, key=lambda n_: abs(by_rung[n_]["cells"] - lc))
            b = sum(1 for x in rs if x["loop"][mg]["osc"] > x["fixed"][near]["osc"])
            c = sum(1 for x in rs if x["fixed"][near]["osc"] > x["loop"][mg]["osc"])
            curve.append({
                "margin": mg, "osc": round(lo, 4), "cells": round(lc, 2),
                "fixed_curve_osc_at_same_spend": round(fixed_osc_at(lc), 4),
                "delta_vs_curve": round(lo - fixed_osc_at(lc), 4),
                "nearest_fixed_rung": near,
                "loop_only": b, "fixed_only": c,
                "p_mcnemar_vs_nearest_rung": round(mcnemar_p(b, c), 6),
                "rung_hist": {str(n_): sum(1 for x in rs
                                           if x["loop"][mg]["rung"] == n_)
                              for n_ in rungs},
            })

        # verdict quality at every rung: does it know incompleteness when it
        # sees it? (fires = verdict rejects; bad = the set really is incomplete)
        vq = {}
        for n_ in rungs:
            tp = sum(1 for x in rs if not x["verdict"][n_] and x["fixed"][n_]["osc"] < 1.0)
            fp = sum(1 for x in rs if not x["verdict"][n_] and x["fixed"][n_]["osc"] >= 1.0)
            fn = sum(1 for x in rs if x["verdict"][n_] and x["fixed"][n_]["osc"] < 1.0)
            vq[str(n_)] = {
                "fired": tp + fp, "tp": tp, "fp": fp, "fn": fn,
                "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                "recall": round(tp / (tp + fn), 4) if tp + fn else None,
            }
        return {
            "n": len(rs),
            "fixed_curve": [{"rung": n_, "cells": round(by_rung[n_]["cells"], 2),
                             "osc": round(by_rung[n_]["osc"], 4)} for n_ in order],
            "loop_curve": curve,
            "verdict_quality_by_rung": vq,
        }

    m2 = [x for x in recs if x["m"] >= 2]
    out = {
        "env": env,
        "leg": "gold-free completeness verdict driving budget escalation, "
               "against a fixed budget matched on mean cells spent",
        "population": {"name": "hitab_arith_with_operands", "split": args.split,
                       "n": len(recs), "n_m_ge_2": len(m2)},
        "rungs": rungs,
        "verdict": "every decomposed operand has a retrieved cell whose header "
                   "path carries all of its segments (no gold)",
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
