#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""E2E: does the OSC win of total-row injection convert to answer accuracy?

Two arms, identical prompt format and solver; ONLY the retrieved cell set differs:
  plain  — numeric cells of the top-k hybrid (BM25+dense RRF) row-chunks
  inject — plain ∪ total-like-row cells in the 2 resolver-picked columns
           (the frozen §5.10 config: bge-reranker-base, top-2, k=10)

Any accuracy delta is attributable to the injected cells. Analysis reports the
overall paired delta AND the flipped-OSC subgroup (queries where injection turns
OSC 0→1) — the causal path predicts the gain concentrates there, diluted by the
~85% of queries whose cell sets barely change (identical contexts are cached and
cost one call).

Rate-limit hygiene: per-query records stream to a JSONL checkpoint (resume on
rerun), calls are paced with --sleep, oversize contexts are recorded not crashed.

Run: PYTHONPATH=. /usr/bin/python3 scripts/e2e_injection_accuracy.py \
       --llm groq:llama-3.3-70b-versatile --sleep 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_dotenv():
    here = Path(__file__).resolve().parent.parent
    for env in (here / ".env", here.parent / ".env"):
        if env.is_file():
            for line in env.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

import numpy as np

from rag_agent.bench.hitab import load_queries
from rag_agent.bench.schema import Chunk
from rag_agent.data.loader import load_table
from rag_agent.eval.operand_set import operand_set_completeness
from rag_agent.generate import answer as gen_answer, evaluate_answer
from rag_agent.retrieve.header_enum import total_like_rows
from rag_agent.retrieve.operand_retriever import HybridRetriever, _tok
from rag_agent.query.operand_decomposer import Embedder
from rag_agent.serialize import S2, serialize_table
from rag_agent.stores.original_store import build_original_table

ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}
ARMS = ("plain", "inject")


def numeric_cells(ot, rows, cols):
    return {(r, c) for r in rows for c in cols if ot.cell_num(r, c) is not None}


def cells_of_chunks(ot, retriever, idxs):
    out = set()
    for i in idxs:
        ch = retriever.chunks[i]
        out |= numeric_cells(ot, ch.rows, ch.cols)
    return out


def ctx_text(ot, cells) -> str:
    lines = []
    for (r, c) in sorted(cells):
        path = list(ot.row_path(r)) + list(ot.col_path(c))
        head = " > ".join(s for s in path if s) or f"r{r}c{c}"
        lines.append(f"{head} = {ot.cell(r, c)}")
    return "\n".join(lines)


def mcnemar_p(w, l):
    from scipy.stats import binomtest
    return float(binomtest(w, w + l, 0.5).pvalue) if (w + l) else 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--llm", default="groq:llama-3.3-70b-versatile")
    ap.add_argument("--mode", default="codegen", choices=["codegen", "direct"])
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--cross-encoder", default="BAAI/bge-reranker-base")
    ap.add_argument("--top-n-cross", type=int, default=2)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--sleep", type=float, default=4.0,
                    help="seconds between LLM calls (free-tier TPM pacing)")
    ap.add_argument("--max-ctx-tokens", type=int, default=4500)
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--out", default="results/e2e_injection_accuracy.json")
    ap.add_argument("--checkpoint", default="results/e2e_injection_records.jsonl")
    args = ap.parse_args()

    queries, tables = load_queries(args.data_dir, args.split)
    pop = [q for q in queries
           if (q.aggregation or "none") in ARITH
           and len({(o.row, o.col) for o in q.gold_operands}) >= 2]
    if args.max:
        pop = pop[:args.max]
    n = len(pop)
    print(f"[pop] arithmetic m>=2: {n}  llm={args.llm} mode={args.mode} "
          f"k={args.k} sleep={args.sleep}s")

    from rag_agent.llm.factory import build_llm
    llm = build_llm(args.llm)

    emb = Embedder(args.embed_model, device="cpu")
    from sentence_transformers import CrossEncoder
    from rag_agent.query.header_embed_resolver import EmbedResolver
    col_resolver = EmbedResolver(emb, col_mode="cross",
                                 cross_encoder=CrossEncoder(args.cross_encoder),
                                 top_n_cross=args.top_n_cross)

    need = {q.gold_table_id for q in pop}
    retr, ots, trows_by_t = {}, {}, {}
    for tid in need:
        retr[tid] = HybridRetriever(serialize_table(tables[tid], S2), emb)
        ot = build_original_table(load_table(tid, args.data_dir))
        ots[tid] = ot
        trows_by_t[tid] = total_like_rows(ot)

    cp = Path(args.checkpoint)
    done = {}
    if cp.is_file():
        for line in cp.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["qid"]] = r
        print(f"[resume] {len(done)} records from {cp}")

    # LLM-free pre-pass: cell sets + OSC per query, so (a) the informative
    # flipped-OSC queries run FIRST (daily-quota exhaustion then costs only
    # diluted-subgroup coverage) and (b) the loop below reuses them.
    pre = {}
    for q in pop:
        ot = ots[q.gold_table_id]
        R = retr[q.gold_table_id]
        qv = np.asarray(emb.encode([q.question])[0])
        bm25_rank = R._rank(R._bm25.get_scores(_tok(q.question)))
        dense_rank = (R._rank(np.asarray(R._emb) @ qv)
                      if R._emb is not None else bm25_rank)
        fused = {}
        for rank, j in enumerate(bm25_rank):
            fused[j] = fused.get(j, 0.0) + 1.0 / (R.rrf_k + rank)
        for rank, j in enumerate(dense_rank):
            fused[j] = fused.get(j, 0.0) + 1.0 / (R.rrf_k + rank)
        hybrid_rank = sorted(fused, key=lambda j: -fused[j])
        base = cells_of_chunks(ot, R, hybrid_rank[:args.k])
        intent = col_resolver.resolve(q.question, ot, col_allowed=None)
        cidx = set()
        for p in intent.col_paths:
            cidx.update(ot.find_cols_by_header(" > ".join(p)))
        inj = base | {(r, c) for r in trows_by_t[q.gold_table_id] for c in cidx
                      if ot.cell_num(r, c) is not None}
        pre[q.query_id] = (base, inj,
                           operand_set_completeness(q.gold_operands, base),
                           operand_set_completeness(q.gold_operands, inj))
    pop.sort(key=lambda q: -(pre[q.query_id][3] - pre[q.query_id][2]))
    n_flip = sum(1 for q in pop if pre[q.query_id][3] > pre[q.query_id][2])
    print(f"[order] {n_flip} flipped-OSC queries scheduled first")

    ans_cache = {}

    def solve(question, ctx, gold_answer):
        if len(ctx) // 3 > args.max_ctx_tokens:
            return None, "oversize", False
        key = hashlib.sha1((question + "␟" + ctx).encode()).hexdigest()
        if key in ans_cache:
            ans = ans_cache[key]
            return ans, ("correct" if evaluate_answer(ans, gold_answer)
                         else "wrong"), False
        try:
            res = gen_answer(question, [Chunk(table_id="t", chunk_id="ctx",
                                              text=ctx)], llm, mode=args.mode)
            ans_cache[key] = res.answer
        except Exception as e:
            print(f"  [error] {str(e)[:120]}", flush=True)
            return None, "error", True
        ans = ans_cache[key]
        return ans, ("correct" if evaluate_answer(ans, gold_answer)
                     else "wrong"), True

    recs = list(done.values())
    with open(cp, "a") as cp_fh:
        for i, q in enumerate(pop):
            if q.query_id in done:
                continue
            ot = ots[q.gold_table_id]
            base, inj, osc_p, osc_i = pre[q.query_id]

            rec = {"qid": q.query_id, "aggregation": q.aggregation,
                   "osc_plain": osc_p, "osc_inject": osc_i,
                   "cells_plain": len(base), "cells_inject": len(inj)}
            called = False
            for arm, cells in (("plain", base), ("inject", inj)):
                ans, bkt, called_now = solve(q.question, ctx_text(ot, cells),
                                             q.answer)
                rec[f"{arm}_answer"] = (ans if isinstance(ans, (int, float, str))
                                        else str(ans))
                rec[f"{arm}_bucket"] = bkt
                rec[f"{arm}_correct"] = int(bkt == "correct")
                called = called or called_now
            cp_fh.write(json.dumps(rec) + "\n")
            cp_fh.flush()
            recs.append(rec)
            done[q.query_id] = rec
            if (i + 1) % 10 == 0:
                c_p = sum(r["plain_correct"] for r in recs)
                c_i = sum(r["inject_correct"] for r in recs)
                print(f"[{i+1}/{n}] acc plain {c_p}/{len(recs)} "
                      f"inject {c_i}/{len(recs)}", flush=True)
            if called:
                time.sleep(args.sleep)

    n = len(recs)
    out = {"population": {"name": "arithmetic_m>=2", "n": n,
                          "split": args.split},
           "config": {"llm": args.llm, "mode": args.mode, "k": args.k,
                      "cross_encoder": args.cross_encoder,
                      "top_n_cross": args.top_n_cross}}

    def summarize(rs, label):
        m = len(rs)
        if not m:
            return {"n": 0}
        answered = [r for r in rs if r["plain_bucket"] in ("correct", "wrong")
                    and r["inject_bucket"] in ("correct", "wrong")]
        w = sum(1 for r in answered
                if r["inject_correct"] > r["plain_correct"])
        l = sum(1 for r in answered
                if r["plain_correct"] > r["inject_correct"])
        d = {"n": m, "n_both_answered": len(answered),
             "acc_plain": round(sum(r["plain_correct"] for r in rs) / m, 4),
             "acc_inject": round(sum(r["inject_correct"] for r in rs) / m, 4),
             "osc_plain": round(sum(r["osc_plain"] for r in rs) / m, 4),
             "osc_inject": round(sum(r["osc_inject"] for r in rs) / m, 4),
             "inject_only_correct": w, "plain_only_correct": l,
             "mcnemar_p": round(mcnemar_p(w, l), 5)}
        print(f"{label:22} n={m:4}  acc {d['acc_plain']:.3f}->"
              f"{d['acc_inject']:.3f}  flips +{w}/-{l}  p={d['mcnemar_p']}")
        return d

    print()
    out["overall"] = summarize(recs, "overall")
    flipped = [r for r in recs if r["osc_inject"] > r["osc_plain"]]
    unchanged = [r for r in recs if r["osc_inject"] == r["osc_plain"]]
    out["osc_flipped_subgroup"] = summarize(flipped, "OSC flipped (0->1)")
    out["osc_unchanged_subgroup"] = summarize(unchanged, "OSC unchanged")
    n_err = sum(1 for r in recs
                if "error" in (r["plain_bucket"], r["inject_bucket"]))
    n_over = sum(1 for r in recs
                 if "oversize" in (r["plain_bucket"], r["inject_bucket"]))
    out["failures"] = {"error": n_err, "oversize": n_over}
    print(f"failures: error={n_err} oversize={n_over}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
