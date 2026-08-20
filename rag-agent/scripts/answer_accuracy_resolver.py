#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Does the embedding resolver's OSC gain convert into ANSWER ACCURACY?

The header-path resolver ranks a table's real header paths against the question.
Its lexical scorer drops every path scoring 0 — any header sharing no token with
the question — which measured as 28-31% of the gold rows/columns on HiTab.
Ranking the same candidates by embedding cosine instead lifts axis accuracy
(.285 -> .537 both axes, dev n=214) and operand-set completeness (+.031..+.048
over a per-query budget-matched control, train n=1006, all p<.015).

Retrieval-side gains do not automatically convert: total-row injection lifted
OSC but converted to answer accuracy only on a capable reader (gpt-oss-120b
+10.5pp) and not at all on llama-3.1-8b. So this runs the paired answer test.

  * baseline  = OperandTargetedRetriever, lexical resolver   -> solver -> answer
  * treatment = same retriever, embed_resolver=True          -> solver -> answer

Same queries, same solver, same k, same index unit; only the resolver differs.
Scored with the dataset's official ``hitab_exact_match`` (the number comparable
to other papers) alongside the repo's lenient numeric diagnostic.

⚠ The evidence set is NOT size-matched here: a better resolver yields more
distinct header paths, so it searches more often (|E| ~21.9 vs 17.5 at k=10).
The budget-matched retrieval control lives in the OSC leg; this leg answers the
narrower question "does the pipeline as configured answer better".

The ``[osc]`` line this prints is the LLM-free evidence: operand-set completeness
for the lexical (base) vs embedding (treat) decomposer, computed before any reader
is called. That is where the decomposer fix shows regardless of reader ability. A
local 4-bit 7B reader answers lookups but caps arithmetic (see the artifact), so on
the multi-operand population its answer EM is low for BOTH arms; read the ``osc``
block for the retrieval effect and switch to a computing reader for the answer EM.

Run (local reader, no quota):
  .venv/bin/python scripts/answer_accuracy_resolver.py --mode direct

Run (hosted reader that can compute, for the answer-EM number):
  GROQ_API_KEY=... .venv/bin/python scripts/answer_accuracy_resolver.py \
      --solver-model openai/gpt-oss-120b --codegen-max-tokens 1024 --flips-first
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
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--min-operands", type=int, default=2)
    # Local reader by default (LocalQwenLLM loads 4-bit on CUDA out of the box),
    # so the leg runs without the Groq daily-token quota. A capable hosted reader
    # (openai/gpt-oss-120b) is still selectable for the arithmetic population.
    ap.add_argument("--solver-model", default="local:Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--mode", default="codegen", choices=["codegen", "direct"])
    ap.add_argument("--codegen-max-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--flips-first", action="store_true",
                    help="run OSC-flip queries (lex incomplete -> emb complete) "
                         "first, so a daily-token cutoff still yields the "
                         "informative subset")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default="results/answer_accuracy_resolver.json")
    ap.add_argument("--records",
                    default="results/answer_accuracy_resolver_records.jsonl")
    args = ap.parse_args()

    queries, _ = load_queries(args.data_dir, args.split)
    pop = [q for q in queries
           if (q.aggregation or "none") in ARITH
           and len({(o.row, o.col) for o in q.gold_operands}) >= args.min_operands]
    if args.limit:
        pop = pop[: args.limit]
    print(f"[pop] HiTab {args.split} arith m>={args.min_operands}: {len(pop)}  "
          f"k={args.k} solver={args.solver_model}", flush=True)

    enc = default_encoder(model_name=args.embed_model)
    # caption_template="structural" is the default and is byte-identical to the
    # caption_length="long" this script was written against.
    # base = lexical decomposer (pinned explicit: the embedding resolver is now
    # the constructor default, and base is the lexical baseline of this A/B).
    base = OperandTargetedRetriever(encoder=enc, scheme="S3", embed_resolver=False)
    treat = OperandTargetedRetriever(encoder=enc, scheme="S3", embed_resolver=True)

    # ---- LLM-free pass: contexts + OSC for every query --------------------
    ots, prep = {}, []
    for q in pop:
        tid = q.gold_table_id
        if tid not in ots:
            try:
                ots[tid] = build_original_table(load_table(tid, args.data_dir))
            except Exception:
                ots[tid] = None
        ot = ots[tid]
        if ot is None:
            continue
        rb = base.retrieve(q.question, ot, k=args.k)
        rt = treat.retrieve(q.question, ot, k=args.k)
        prep.append({
            "q": q,
            "base_chunks": [rc.chunk for rc in rb.retrieved],
            "treat_chunks": [rc.chunk for rc in rt.retrieved],
            "ob": operand_set_completeness(q.gold_operands, cells_of(rb)),
            "ot": operand_set_completeness(q.gold_operands, cells_of(rt)),
        })

    n_flip = sum(1 for p in prep if p["ot"] > p["ob"])
    print(f"[osc] lex={sum(p['ob'] for p in prep)/len(prep):.4f}  "
          f"emb={sum(p['ot'] for p in prep)/len(prep):.4f}  flips={n_flip}",
          flush=True)
    if args.flips_first:
        prep.sort(key=lambda p: -(p["ot"] - p["ob"]))

    done = {}
    if args.resume and Path(args.records).exists():
        with open(args.records) as fh:
            for line in fh:
                r = json.loads(line)
                done[r["qid"]] = r
        print(f"[resume] {len(done)} qids skipped", flush=True)

    # A bare model name stays Groq so existing --resume records keep their
    # solver; "openai:claude-sonnet-5" reaches any OpenAI-compatible provider.
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
            ab = answer(q.question, p["base_chunks"], llm, mode=args.mode,
                        codegen_max_tokens=args.codegen_max_tokens)
            at = answer(q.question, p["treat_chunks"], llm, mode=args.mode,
                        codegen_max_tokens=args.codegen_max_tokens)
        except Exception as e:  # daily-token quota -> keep what we have
            cutoff = f"{type(e).__name__}: {e}"
            print(f"\n[cutoff] solver failed at {qi+1}/{len(prep)}: {cutoff}",
                  flush=True)
            break
        rec = {
            "qid": q.query_id, "table_id": q.gold_table_id,
            # raw prediction + gold, so a scorer change can be re-applied
            # offline instead of paying for the answers again
            "pred_lex": ab.answer, "pred_emb": at.answer, "gold": q.answer,
            "osc_lex": p["ob"], "osc_emb": p["ot"],
            "em_lex": int(hitab_exact_match(ab.answer, q.answer)),
            "em_emb": int(hitab_exact_match(at.answer, q.answer)),
            "num_lex": int(evaluate_answer(ab.answer, q.answer)),
            "num_emb": int(evaluate_answer(at.answer, q.answer)),
            "trunc_lex": ab.context_truncated, "trunc_emb": at.context_truncated,
        }
        done[q.query_id] = rec
        rec_fh.write(json.dumps(rec) + "\n")
        rec_fh.flush()
        if len(done) % 10 == 0:
            print(f"  {len(done)}  ({time.time()-t0:.0f}s)", flush=True)
    rec_fh.close()

    rs = list(done.values())
    n = len(rs)
    out = {
        "leg": "embedding header-path resolver -> answer accuracy",
        "population": {"name": f"hitab_arith_m_ge_{args.min_operands}",
                       "split": args.split, "n": n, "n_planned": len(prep)},
        "config": {"k": args.k, "solver": llm.name, "mode": args.mode,
                   "index_unit": "S3/long/cell", "flips_first": args.flips_first},
        "cutoff": cutoff,
        "n_truncated": {"lex": sum(r["trunc_lex"] for r in rs),
                        "emb": sum(r["trunc_emb"] for r in rs)},
        "note": "Evidence set is not size-matched (a better resolver searches "
                "more distinct header paths); the budget-matched control is in "
                "the OSC leg. flips_first biases a PARTIAL run's delta upward — "
                "quote a partial run's delta only with n and n_planned.",
    }
    for metric in ("osc", "em", "num"):
        lx, eb = f"{metric}_lex", f"{metric}_emb"
        b = sum(1 for r in rs if r[eb] > r[lx])
        c = sum(1 for r in rs if r[lx] > r[eb])
        out[metric] = {
            "lex": round(sum(r[lx] for r in rs) / n, 4) if n else None,
            "emb": round(sum(r[eb] for r in rs) / n, 4) if n else None,
            "b_emb_only": b, "c_lex_only": c,
            "p_mcnemar": round(mcnemar_p(b, c), 6),
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
