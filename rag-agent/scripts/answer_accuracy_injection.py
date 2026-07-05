#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""H6 payoff: does total-row injection's OSC gain convert into ANSWER ACCURACY?

§5.10 shows structural total-row injection lifts operand-set completeness (OSC) over
similarity retrieval. The open question (paper marks it pending) is whether that
retrieval-side completeness gain actually raises the FINAL answer accuracy at a fixed
solver. This runs the paired test:

  * baseline  = dense top-k row-chunks                      -> solver -> numeric answer
  * treatment = dense top-k  UNION  total-like row-chunks   -> solver -> numeric answer

Same queries, same solver (Groq llama-3.1-8b-instant, codegen), identical except the
injected total rows. Reports OSC (base vs treat) AND answer accuracy (base vs treat),
with the correctness cross-tab + McNemar (queries flipped wrong->right by injection).

Population: HiTab dev arithmetic m>=2 (the §5.10 population).
Run: GROQ_API_KEY=... .venv/bin/python scripts/answer_accuracy_injection.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from rag_agent.bench.hitab import load_queries
from rag_agent.bench.schema import BenchTable
from rag_agent.eval.operand_set import operand_set_completeness
from rag_agent.generate.answerer import answer, evaluate_answer
from rag_agent.llm.groq_llm import GroqLLM
from rag_agent.retrieve.header_enum import total_like_rows
from rag_agent.retrieve.operand_retriever import HybridRetriever, _tok
from rag_agent.query.operand_decomposer import Embedder
from rag_agent.serialize import S2, serialize_table
from rag_agent.stores.original_store import _to_float

ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}
BenchTable.cell_num = lambda self, r, c: _to_float(self.cell(r, c))  # type: ignore[attr-defined]


def numeric_cells(t, rows, cols):
    return {(r, c) for r in rows for c in cols if t.cell_num(r, c) is not None}


def cells_of_chunks(t, chunks):
    out = set()
    for ch in chunks:
        out |= numeric_cells(t, ch.rows, ch.cols)
    return out


def mcnemar_p(b, c):
    """Exact binomial McNemar on discordant pairs b (treat-only) vs c (base-only)."""
    from scipy.stats import binomtest
    n = b + c
    return binomtest(b, n, 0.5).pvalue if n else 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--retriever", default="dense", choices=["dense", "bm25", "hybrid"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--solver-model", default="llama-3.1-8b-instant")
    ap.add_argument("--mode", default="codegen", choices=["codegen", "direct"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/answer_accuracy_injection.json")
    ap.add_argument("--records", default="results/answer_accuracy_injection_records.jsonl")
    args = ap.parse_args()

    queries, tables = load_queries(args.data_dir, args.split)
    pop = [q for q in queries
           if (q.aggregation or "none") in ARITH
           and q.gold_table_id in tables
           and len({(o.row, o.col) for o in q.gold_operands}) >= 2]
    if args.limit:
        pop = pop[:args.limit]
    n = len(pop)
    print(f"[pop] HiTab {args.split} arith m>=2: {n}  retriever={args.retriever} k={args.k} "
          f"solver={args.solver_model}", flush=True)

    emb = Embedder(args.embed_model, device="cpu")
    llm = GroqLLM(model_name=args.solver_model)

    need = {q.gold_table_id for q in pop}
    retr, total_chunk_idx = {}, {}
    for tid in need:
        t = tables[tid]
        R = HybridRetriever(serialize_table(t, S2), emb)
        retr[tid] = R
        trows = total_like_rows(t)
        # S2 chunks are per-row; a chunk covering any total-like row is a "total chunk"
        total_chunk_idx[tid] = [i for i, ch in enumerate(R.chunks)
                                if set(ch.rows) & trows]

    osc_b = osc_t = acc_b = acc_t = 0.0
    both_right = base_only = treat_only = both_wrong = 0
    inj_added_cells = 0
    recs = []
    t0 = time.time()

    for qi, q in enumerate(pop):
        t = tables[q.gold_table_id]
        R = retr[q.gold_table_id]
        gold = q.gold_operands

        qv = np.asarray(emb.encode([q.question])[0])
        bm25_rank = R._rank(R._bm25.get_scores(_tok(q.question)))
        dense_rank = R._rank(np.asarray(R._emb) @ qv) if R._emb is not None else bm25_rank
        if args.retriever == "bm25":
            order = bm25_rank
        elif args.retriever == "dense":
            order = dense_rank
        else:
            fused = {}
            for rank, i in enumerate(bm25_rank):
                fused[i] = fused.get(i, 0.0) + 1.0 / (R.rrf_k + rank)
            for rank, i in enumerate(dense_rank):
                fused[i] = fused.get(i, 0.0) + 1.0 / (R.rrf_k + rank)
            order = sorted(fused, key=lambda i: -fused[i])

        base_idx = list(order[:args.k])
        base_chunks = [R.chunks[i] for i in base_idx]
        extra_idx = [i for i in total_chunk_idx[q.gold_table_id] if i not in set(base_idx)]
        treat_chunks = base_chunks + [R.chunks[i] for i in extra_idx]

        base_cells = cells_of_chunks(t, base_chunks)
        treat_cells = cells_of_chunks(t, treat_chunks)
        ob = operand_set_completeness(gold, base_cells)
        ot = operand_set_completeness(gold, treat_cells)
        osc_b += ob
        osc_t += ot
        inj_added_cells += len(treat_cells - base_cells)

        rb = answer(q.question, base_chunks, llm, mode=args.mode)
        rt = answer(q.question, treat_chunks, llm, mode=args.mode)
        cb = evaluate_answer(rb.answer, q.answer)
        ct = evaluate_answer(rt.answer, q.answer)
        acc_b += int(cb)
        acc_t += int(ct)
        if cb and ct:
            both_right += 1
        elif cb and not ct:
            base_only += 1
        elif ct and not cb:
            treat_only += 1
        else:
            both_wrong += 1

        recs.append({"qid": q.query_id, "osc_base": ob, "osc_treat": ot,
                     "correct_base": cb, "correct_treat": ct,
                     "pred_base": rb.answer, "pred_treat": rt.answer, "gold": q.answer})
        if (qi + 1) % 4 == 0:
            print(f"  {qi+1}/{n}  acc_b={acc_b/(qi+1):.3f} acc_t={acc_t/(qi+1):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    p_acc = mcnemar_p(treat_only, base_only)
    out = {
        "population": {"name": "hitab_dev_arith_m>=2", "n": n},
        "retriever": args.retriever, "k": args.k, "solver": f"groq:{args.solver_model}",
        "mode": args.mode,
        "osc": {"base": round(osc_b / n, 4), "treat": round(osc_t / n, 4),
                "delta": round((osc_t - osc_b) / n, 4)},
        "answer_accuracy": {"base": round(acc_b / n, 4), "treat": round(acc_t / n, 4),
                            "delta": round((acc_t - acc_b) / n, 4)},
        "correctness_crosstab": {"both_right": both_right, "treat_only": treat_only,
                                 "base_only": base_only, "both_wrong": both_wrong},
        "mcnemar_p_accuracy": round(float(p_acc), 5),
        "mean_injected_cells": round(inj_added_cells / n, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    with open(args.records, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")

    print("\n=== RESULT ===")
    print(f"OSC      base {out['osc']['base']}  ->  treat {out['osc']['treat']}  "
          f"(Δ {out['osc']['delta']:+.4f})")
    print(f"Accuracy base {out['answer_accuracy']['base']}  ->  treat "
          f"{out['answer_accuracy']['treat']}  (Δ {out['answer_accuracy']['delta']:+.4f})")
    print(f"flipped wrong->right: {treat_only}   right->wrong: {base_only}   "
          f"McNemar p={out['mcnemar_p_accuracy']}")
    print(f"mean injected cells/query: {out['mean_injected_cells']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
