#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Full lookup pipeline WITH an LLM reader — the framework showcase.

Both retrieval gates are held at their solved state so the number isolates the
CONTRIBUTION (cell:sentence 1:1 verbalization -> LLM answer):

  * Gate 1 (find the table): oracle here. Justified because HiTab table
    retrieval already reaches recall@20 = 1.00 (scripts gate-1 measurement),
    so table-finding is not the variable under test.
  * Gate 2 + answer: within that table, the top-k cell sentences are retrieved
    under each serialization and handed to the LLM, which reads them and
    answers. Same encoder / LLM / k for both arms — only the serialization of
    each cell differs:
        flat : leaf row+col labels only (baseline serialization)
        S2   : full row-path > col-path per cell (the student's method)

Scored with the official ``hitab_exact_match``. Resumable: each (query, arm)
result is appended to a jsonl and skipped on re-run, so a GROQ TPD cutoff
mid-run costs no repeated tokens — rerun with --resume after the daily reset.

Run: PYTHONPATH=. .venv/bin/python scripts/pipeline_lookup_llm.py \
        --n 150 --topk 8 --model openai/gpt-oss-120b --resume
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from rag_agent.bench.hitab import load_queries
from rag_agent.eval.metrics import hitab_exact_match
from rag_agent.generate.answerer import _DIRECT_SYS
from rag_agent.llm.groq_llm import GroqLLM
from rag_agent.retrieve.encoders import default_encoder
from point3_reconstruction_cost import build_table_paths, cell_text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default="results/pipeline_lookup_llm.json")
    ap.add_argument("--resume", action="store_true",
                    help="skip (query,arm) pairs already in the records jsonl")
    args = ap.parse_args()
    rec_path = Path(str(Path(args.out).with_suffix("")) + "_records.jsonl")

    queries, tables = load_queries(args.data_dir, args.split)
    raw_dir = Path(args.data_dir) / "data/tables/raw"
    paths = {}
    for tid, bt in tables.items():
        p = raw_dir / f"{tid}.json"
        if not p.exists():
            continue
        try:
            raw = json.load(open(p))
        except Exception:
            continue
        pt = build_table_paths(raw, bt)
        if pt is not None:
            paths[tid] = pt

    pop = []
    for q in queries:
        pt = paths.get(q.gold_table_id)
        if pt is None:
            continue
        ops = [(op.row, op.col) for op in q.gold_operands]
        if len(ops) != 1:
            continue
        r, c = ops[0]
        if not (0 <= r < pt["n_r"] and 0 <= c < pt["n_c"]):
            continue
        pop.append(q)
    random.Random(0).shuffle(pop)
    pop = pop[: args.n]

    done = {}
    if args.resume and rec_path.exists():
        for line in open(rec_path):
            r = json.loads(line)
            done[(r["query_id"], r["arm"])] = r["correct"]
        print(f"[resume] {len(done)} (query,arm) results already recorded", flush=True)
    print(f"[pop] {len(pop)} lookup queries | model={args.model} | top-{args.topk}", flush=True)

    enc = default_encoder(model_name="BAAI/bge-small-en-v1.5")
    llm = GroqLLM(model_name=args.model)

    def topk_sentences(tid, qv, scheme, k):
        pt, bt = paths[tid], tables[tid]
        T = []
        for i in range(pt["n_r"]):
            for j in range(pt["n_c"]):
                v = bt.data[i][j]
                T.append(cell_text(pt["gold_rp"][i], pt["gold_cp"][j], v,
                                   "flat" if scheme == "flat" else "S2"))
        vecs = np.asarray(enc.encode(T))
        order = np.argsort(-(vecs @ qv))[: k]
        return [T[o] for o in order]

    rec_fh = open(rec_path, "a")
    n_done = 0
    try:
        for q in pop:
            qv = np.asarray(enc.encode([q.question])[0])
            for s in ("flat", "S2"):
                if (q.query_id, s) in done:
                    continue
                ctx = "\n".join(topk_sentences(q.gold_table_id, qv, s, args.topk))
                user = f"ROWS:\n{ctx}\n\nQUESTION: {q.question}\n\nAnswer:"
                raw = llm.complete(system=_DIRECT_SYS, user=user, max_tokens=512)
                if not raw and getattr(llm, "last_finish_reason", None) == "length":
                    raw = llm.complete(system=_DIRECT_SYS, user=user, max_tokens=1024)
                ok = bool(hitab_exact_match(raw, q.answer))
                done[(q.query_id, s)] = ok
                rec_fh.write(json.dumps({"query_id": q.query_id, "arm": s,
                                         "correct": ok, "pred": raw[:120]}) + "\n")
                rec_fh.flush()
            n_done += 1
            if n_done % 25 == 0:
                cur = {s: sum(v for (qid, a), v in done.items() if a == s) for s in ("flat", "S2")}
                tot = {s: sum(1 for (qid, a) in done if a == s) for s in ("flat", "S2")}
                print(f"  {n_done}/{len(pop)}  "
                      f"flat={cur['flat']}/{tot['flat']} S2={cur['S2']}/{tot['S2']}", flush=True)
    except RuntimeError as e:
        print(f"[stopped] {str(e)[:120]} ... progress saved, rerun with --resume", flush=True)
    finally:
        rec_fh.close()

    # aggregate whatever is recorded
    agg = {s: [v for (qid, a), v in done.items() if a == s] for s in ("flat", "S2")}
    complete = [q.query_id for q in pop
                if (q.query_id, "flat") in done and (q.query_id, "S2") in done]
    out = {
        "dataset": "hitab_lookup_llm_reader", "model": args.model, "topk": args.topk,
        "table": "oracle (gate-1 solved: HiTab table recall@20=1.0)",
        "scorer": "hitab_exact_match",
        "n_complete_both_arms": len(complete),
        "answer_EM": {s: (round(sum(agg[s]) / len(agg[s]), 4) if agg[s] else None)
                      for s in agg},
        "n_scored": {s: len(agg[s]) for s in agg},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
