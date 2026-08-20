#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""End-to-end cascade: retrieve the TABLE first, then the cells, then answer.

Every other answer leg in this repo hands the retriever the gold table, so table
selection is an oracle and the number is a within-doc number. This leg closes the
loop the way a deployed system faces it:

    query -> [stage 1: rank all corpus tables, keep top-M]
          -> [stage 2: operand-targeted cell search INSIDE those tables]
          -> reader -> answer

The corpus is every distinct table referenced by the split's queries, so stage 1
has real distractors. Stage 2 uses the fixed decomposer (embedding resolver is now
the OperandTargetedRetriever default). ``--oracle-table`` bypasses stage 1 and
searches the gold table only, giving the within-doc ceiling in the same harness —
the gap between the two is exactly what table retrieval costs.

Reported per run:
  * ``table_hit@M``     -- stage 1: fraction of queries whose gold table is kept.
                           This is the new bottleneck (cf. NEXT.md §3).
  * ``osc``             -- operand-set completeness, scored ONLY on gold-table
                           cells (a distractor cell can never be a gold operand).
  * ``em`` / ``num``    -- answer exact match (official) / lenient numeric.

The reader defaults to a local Qwen (no Groq quota). A 4-bit 7B reader answers
lookups but caps arithmetic, so on the multi-operand population read ``table_hit``
and ``osc`` for the retrieval story and switch to a computing reader
(``--solver-model openai/gpt-oss-120b``) for the answer number.

Run (HiTab, top-1 table, local reader):
  .venv/bin/python scripts/answer_accuracy_cascade.py --top-m 1 --mode direct

Compare against the oracle-table ceiling:
  .venv/bin/python scripts/answer_accuracy_cascade.py --oracle-table --mode direct
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.bench.hitab import load_queries
from rag_agent.data.loader import load_table
from rag_agent.eval.metrics import hitab_exact_match
from rag_agent.eval.operand_set import operand_set_completeness
from rag_agent.generate.answerer import answer, evaluate_answer
from rag_agent.llm.factory import build_llm
from rag_agent.retrieve.cascade import TableIndex, cascade_retrieve
from rag_agent.retrieve.encoders import default_encoder
from rag_agent.retrieve.operand_retrieval import OperandTargetedRetriever
from rag_agent.stores.original_store import build_original_table

ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}


def cells_of_in_table(res, gold_table_id):
    """Retrieved (row, col) cells that live in the gold table.

    Operand-set completeness matches gold operands (which are cells of the gold
    table) against retrieved cells by coordinate. Across multiple tables a bare
    (row, col) from a distractor could spuriously "cover" a gold coordinate, so
    completeness is scored only over gold-table cells. If stage 1 dropped the
    gold table, this is empty and OSC is 0 — which is the honest outcome.
    """
    return [(rc.chunk.row_index, rc.chunk.col_index) for rc in res.retrieved
            if rc.chunk.row_index is not None and rc.chunk.table_id == gold_table_id]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--top-m", type=int, default=1,
                    help="stage 1: how many tables to keep per query")
    ap.add_argument("--oracle-table", action="store_true",
                    help="skip stage 1; search the gold table only (within-doc ceiling)")
    ap.add_argument("--min-operands", type=int, default=1,
                    help="1 keeps lookups (a local reader can answer these)")
    ap.add_argument("--arith-only", action="store_true",
                    help="restrict to arithmetic questions (needs a computing reader)")
    ap.add_argument("--solver-model", default="local:Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--mode", default="direct", choices=["codegen", "direct"])
    ap.add_argument("--codegen-max-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default="results/answer_accuracy_cascade.json")
    ap.add_argument("--records", default="results/answer_accuracy_cascade_records.jsonl")
    args = ap.parse_args()

    queries, _ = load_queries(args.data_dir, args.split)

    # Corpus = every distinct table any query in the split references. This is the
    # distractor pool stage 1 must rank the gold table above.
    corpus_ids = sorted({q.gold_table_id for q in queries})
    tables = {}
    for tid in corpus_ids:
        try:
            tables[tid] = build_original_table(load_table(tid, args.data_dir))
        except Exception:
            pass
    print(f"[corpus] {len(tables)}/{len(corpus_ids)} tables loaded", flush=True)

    def keep(q):
        n_ops = len({(o.row, o.col) for o in q.gold_operands})
        if n_ops < args.min_operands:
            return False
        if args.arith_only and (q.aggregation or "none") not in ARITH:
            return False
        return q.gold_table_id in tables

    pop = [q for q in queries if keep(q)]
    if args.limit:
        pop = pop[: args.limit]
    mode_str = "oracle-table" if args.oracle_table else f"top-{args.top_m}"
    print(f"[pop] {args.split} n={len(pop)}  stage1={mode_str}  k={args.k}  "
          f"solver={args.solver_model}", flush=True)

    enc = default_encoder(model_name=args.embed_model)
    # Stage 2: fixed decomposer (embedding resolver is the constructor default).
    cell_retriever = OperandTargetedRetriever(encoder=enc, scheme="S3")

    tindex = None
    if not args.oracle_table:
        print("[stage1] building whole-table index over the corpus...", flush=True)
        tindex = TableIndex(enc).build(tables)

    # ---- retrieval pass (no LLM): stage1 hit + OSC + context per query --------
    prep, table_hits = [], 0
    for q in pop:
        gtid = q.gold_table_id
        if args.oracle_table:
            r = cell_retriever.retrieve(q.question, tables[gtid], k=args.k)
            hit, chosen = True, [gtid]
            retrieved = r.retrieved
        else:
            cr = cascade_retrieve(q.question, tindex, tables, cell_retriever,
                                  m=args.top_m, k=args.k, gold_table_id=gtid)
            hit, chosen, retrieved = cr.gold_table_hit, cr.table_ids, cr.retrieved
        table_hits += int(hit)
        cells = [(rc.chunk.row_index, rc.chunk.col_index) for rc in retrieved
                 if rc.chunk.row_index is not None and rc.chunk.table_id == gtid]
        prep.append({"q": q, "chunks": [rc.chunk for rc in retrieved],
                     "table_hit": int(hit), "chosen": chosen,
                     "osc": operand_set_completeness(q.gold_operands, cells)})

    n = len(prep)
    table_hit_rate = round(table_hits / n, 4) if n else 0.0
    osc_mean = round(sum(p["osc"] for p in prep) / n, 4) if n else 0.0
    print(f"[stage1] table_hit@{args.top_m}={table_hit_rate}   "
          f"[stage2] osc={osc_mean}", flush=True)

    # ---- answer pass ---------------------------------------------------------
    done = {}
    if args.resume and Path(args.records).exists():
        with open(args.records) as fh:
            for line in fh:
                r = json.loads(line)
                done[r["qid"]] = r
        print(f"[resume] {len(done)} qids skipped", flush=True)

    llm = build_llm(args.solver_model if ":" in args.solver_model
                    else f"groq:{args.solver_model}", retry_on_429=8)
    Path(args.records).parent.mkdir(parents=True, exist_ok=True)
    rec_fh = open(args.records, "a" if args.resume else "w")
    t0, cutoff = time.time(), None
    for qi, p in enumerate(prep):
        q = p["q"]
        if q.query_id in done:
            continue
        try:
            a = answer(q.question, p["chunks"], llm, mode=args.mode,
                       codegen_max_tokens=args.codegen_max_tokens)
        except Exception as e:
            cutoff = f"{type(e).__name__}: {e}"
            print(f"\n[cutoff] solver failed at {qi+1}/{len(prep)}: {cutoff}", flush=True)
            break
        rec = {
            "qid": q.query_id, "table_id": q.gold_table_id,
            "table_hit": p["table_hit"], "chosen": p["chosen"], "osc": p["osc"],
            "pred": a.answer, "gold": q.answer,
            "em": int(hitab_exact_match(a.answer, q.answer)),
            "num": int(evaluate_answer(a.answer, q.answer)),
            "trunc": a.context_truncated,
        }
        done[q.query_id] = rec
        rec_fh.write(json.dumps(rec) + "\n")
        rec_fh.flush()
        if len(done) % 10 == 0:
            print(f"  {len(done)}  ({time.time()-t0:.0f}s)", flush=True)
    rec_fh.close()

    rs = list(done.values())
    m = len(rs)
    out = {
        "leg": "cascade: table retrieval -> cell retrieval -> answer",
        "population": {"split": args.split, "n": m, "n_planned": len(prep),
                       "min_operands": args.min_operands, "arith_only": args.arith_only},
        "config": {"stage1": mode_str, "top_m": args.top_m, "k": args.k,
                   "solver": llm.name, "mode": args.mode,
                   "index_unit": "S3/cell", "corpus_tables": len(tables),
                   "decomposer": "embedding" if cell_retriever.embed_resolver else "lexical"},
        "cutoff": cutoff,
        # scored on the answered subset
        "table_hit_rate": round(sum(r["table_hit"] for r in rs) / m, 4) if m else None,
        "osc": round(sum(r["osc"] for r in rs) / m, 4) if m else None,
        "em": round(sum(r["em"] for r in rs) / m, 4) if m else None,
        "num": round(sum(r["num"] for r in rs) / m, 4) if m else None,
        "n_truncated": sum(r["trunc"] for r in rs),
        # scored on the full planned population (stage 1 is LLM-free, so complete)
        "table_hit_rate_full": table_hit_rate,
        "osc_full": osc_mean,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in
                      ("table_hit_rate_full", "osc_full", "em", "num", "cutoff")}, indent=2))
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
