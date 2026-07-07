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
    ap.add_argument("--codegen-max-tokens", type=int, default=160,
                    help="completion cap for codegen; reasoning models (gpt-oss, "
                         "qwen3) burn tokens on hidden reasoning first -> use ~1024")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--flips-first", action="store_true",
                    help="run OSC-flip queries (base incomplete -> treat complete) first, "
                         "so a daily-token cutoff still yields the informative subset")
    ap.add_argument("--resume", action="store_true",
                    help="skip qids already present in --records (append mode)")
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
    # long 429 backoff: 70b TPM is 12k and codegen calls are ~2k tokens, so
    # per-minute throttling is expected; only a *daily* quota should abort the run
    llm = GroqLLM(model_name=args.solver_model, retry_on_429=8)

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

    # ---- LLM-free pass: retrieval contexts + OSC for every query ---------
    prep = []
    for q in pop:
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
        prep.append({"q": q, "base_chunks": base_chunks, "treat_chunks": treat_chunks,
                     "ob": ob, "ot": ot, "inj": len(treat_cells - base_cells)})

    n_flip = sum(1 for p in prep if p["ot"] > p["ob"])
    if args.flips_first:  # informative queries first, stable order within groups
        prep.sort(key=lambda p: -(p["ot"] - p["ob"]))
        print(f"[order] flips-first: {n_flip} OSC-flip queries run before the rest",
              flush=True)

    done = {}
    if args.resume and Path(args.records).exists():
        with open(args.records) as fh:
            for line in fh:
                r = json.loads(line)
                done[r["qid"]] = r
        print(f"[resume] {len(done)} qids already recorded -> skipped", flush=True)

    # ---- solver pass: incremental append; a daily-quota cutoff keeps progress ----
    Path(args.records).parent.mkdir(parents=True, exist_ok=True)
    rec_fh = open(args.records, "a" if args.resume else "w")
    t0, n_run, cutoff = time.time(), 0, None
    for qi, p in enumerate(prep):
        q = p["q"]
        if q.query_id in done:
            continue
        try:
            rb = answer(q.question, p["base_chunks"], llm, mode=args.mode,
                        codegen_max_tokens=args.codegen_max_tokens)
            rt = answer(q.question, p["treat_chunks"], llm, mode=args.mode,
                        codegen_max_tokens=args.codegen_max_tokens)
        except Exception as e:  # e.g. Groq daily-token quota -> keep what we have
            cutoff = f"{type(e).__name__}: {e}"
            print(f"\n[cutoff] solver failed at {qi+1}/{n}: {cutoff}", flush=True)
            break
        cb = evaluate_answer(rb.answer, q.answer)
        ct = evaluate_answer(rt.answer, q.answer)
        rec = {"qid": q.query_id, "osc_base": p["ob"], "osc_treat": p["ot"],
               "correct_base": cb, "correct_treat": ct,
               "pred_base": rb.answer, "pred_treat": rt.answer, "gold": q.answer}
        done[q.query_id] = rec
        rec_fh.write(json.dumps(rec) + "\n")
        rec_fh.flush()
        n_run += 1
        if n_run % 4 == 0:
            d = [r for r in done.values()]
            print(f"  {len(done)}/{n}  acc_b={sum(r['correct_base'] for r in d)/len(d):.3f} "
                  f"acc_t={sum(r['correct_treat'] for r in d)/len(d):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    rec_fh.close()

    # ---- aggregate over every recorded query (this run + resumed) ---------
    recs = [done[p["q"].query_id] for p in prep if p["q"].query_id in done]
    ne = len(recs)
    if not ne:
        print("no evaluations recorded (quota exhausted immediately?)")
        return 1
    acc_b = sum(r["correct_base"] for r in recs)
    acc_t = sum(r["correct_treat"] for r in recs)
    both_right = sum(1 for r in recs if r["correct_base"] and r["correct_treat"])
    base_only = sum(1 for r in recs if r["correct_base"] and not r["correct_treat"])
    treat_only = sum(1 for r in recs if r["correct_treat"] and not r["correct_base"])
    both_wrong = ne - both_right - base_only - treat_only
    flips = [r for r in recs if r["osc_treat"] > r["osc_base"]]

    p_acc = mcnemar_p(treat_only, base_only)
    n_nonflip_evaluated = ne - len(flips)
    # With --flips-first, a quota cutoff can leave the evaluated set entirely
    # (or mostly) drawn from the OSC-flip subset — queries pre-selected because
    # treatment can only help there. Flag this so the top-level answer_accuracy
    # block is never quoted as a random-population estimate when it is really
    # conditioned on the flip criterion (see osc_flip_subset for that number).
    flip_only_caveat = (
        f"evaluated set is {len(flips)}/{ne} OSC-flip queries (non-flip evaluated="
        f"{n_nonflip_evaluated}) — NOT a representative sample of the population; "
        "answer_accuracy here is conditioned on OSC having flipped, which can only "
        "help treatment. Do not report it as a population-level accuracy lift."
        if n_nonflip_evaluated == 0 and flips else None
    )
    out = {
        "population": {"name": "hitab_dev_arith_m>=2", "n": n,
                       "n_evaluated": ne, "n_osc_flips_total": n_flip,
                       "n_osc_flips_evaluated": len(flips),
                       "n_nonflip_evaluated": n_nonflip_evaluated,
                       "flips_first": args.flips_first, "cutoff": cutoff},
        "retriever": args.retriever, "k": args.k, "solver": f"groq:{args.solver_model}",
        "mode": args.mode,
        "osc": {"base": round(sum(p["ob"] for p in prep) / n, 4),
                "treat": round(sum(p["ot"] for p in prep) / n, 4),
                "delta": round(sum(p["ot"] - p["ob"] for p in prep) / n, 4),
                "note": "OSC over full population (LLM-free)"},
        "answer_accuracy": {"base": round(acc_b / ne, 4), "treat": round(acc_t / ne, 4),
                            "delta": round((acc_t - acc_b) / ne, 4),
                            "caveat": flip_only_caveat},
        "correctness_crosstab": {"both_right": both_right, "treat_only": treat_only,
                                 "base_only": base_only, "both_wrong": both_wrong},
        "mcnemar_p_accuracy": round(float(p_acc), 5),
        "osc_flip_subset": {
            "n": len(flips),
            "acc_base": round(sum(r["correct_base"] for r in flips) / len(flips), 4)
            if flips else None,
            "acc_treat": round(sum(r["correct_treat"] for r in flips) / len(flips), 4)
            if flips else None,
            "treat_only": sum(1 for r in flips
                              if r["correct_treat"] and not r["correct_base"]),
            "base_only": sum(1 for r in flips
                             if r["correct_base"] and not r["correct_treat"])},
        "mean_injected_cells": round(sum(p["inj"] for p in prep) / n, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    print("\n=== RESULT ===")
    print(f"evaluated {ne}/{n} (cutoff={cutoff})")
    print(f"OSC      base {out['osc']['base']}  ->  treat {out['osc']['treat']}  "
          f"(Δ {out['osc']['delta']:+.4f})  [full pop, LLM-free]")
    if flip_only_caveat:
        print(f"** WARNING: {flip_only_caveat} **")
    print(f"Accuracy base {out['answer_accuracy']['base']}  ->  treat "
          f"{out['answer_accuracy']['treat']}  (Δ {out['answer_accuracy']['delta']:+.4f})")
    print(f"flipped wrong->right: {treat_only}   right->wrong: {base_only}   "
          f"McNemar p={out['mcnemar_p_accuracy']}")
    fs = out["osc_flip_subset"]
    print(f"OSC-flip subset ({fs['n']} evaluated): acc {fs['acc_base']} -> {fs['acc_treat']}"
          f"   treat_only={fs['treat_only']} base_only={fs['base_only']}")
    print(f"mean injected cells/query: {out['mean_injected_cells']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
