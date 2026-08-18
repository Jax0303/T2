#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Which retriever finds the answer cell, and what does the header path buy each one?

Two axes on one population, so the serialization gain can be read per retriever
instead of being reported against a single fixed one:

  serialization (what a cell is indexed as)
    flat      : leaf row + leaf col label only        -- the baseline RAG unit
    S2_recon  : full row-path > col-path, headers RECONSTRUCTED from the grid
                -- what production actually indexes
    S2_gold   : the same with the gold header tree    -- structure-perfect ceiling

  retriever (how the query scores that unit)
    bm25   : lexical only            (HybridIndex alpha=0)
    dense  : bi-encoder only         (alpha=1, bge-small-en-v1.5)
    hybrid : min-max weighted fusion (alpha=0.5, the repo default)
    cross  : dense top-``--cross-pool`` re-scored by bge-reranker-large, the
             standard two-stage production pattern

Population, pool and gold are those of ``manual_sentence_ceiling.py`` /
``pipeline_lookup_llm.py``: HiTab dev, single gold operand, seed-0 shuffle, first
--n, table retrieval oracle, pool = every cell of the gold table. One gold cell
per query, so recall@k is just ``rank <= k`` and MRR is 1/rank.

``S2_gold - S2_recon`` is the price of header reconstruction; ``S2_recon - flat``
is what the method buys on real data, per retriever. With --llm the same
orderings are handed to the reader so the price is also in answer EM.

--oracle-cell pins the gold cell at the head of the context for every query, so
cell recall is 1.0 BY CONSTRUCTION and the arms differ only in how the cell reads.
Answer EM across serializations is only a reader comparison under this flag:
without it the arms are handed different contexts (S2 .50 vs flat .31 recall@1),
so their EM gap is mostly retrieval and says nothing about the reader.

  # pass 1: all 12 retrieval cells (loads the reranker)
  PYTHONPATH=. .venv/bin/python scripts/cell_retrieval_matrix.py --no-llm --resume
  # pass 2: answer leg on the dense column (no reranker -> GPU free for the reader)
  PYTHONPATH=. .venv/bin/python scripts/cell_retrieval_matrix.py --resume \
      --retrievers dense --model local:Qwen/Qwen2.5-7B-Instruct
  # pass 3: reader-only leg, retrieval held at 100%
  PYTHONPATH=. .venv/bin/python scripts/cell_retrieval_matrix.py --resume \
      --retrievers dense --oracle-cell --out results/cell_retrieval_oracle.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from rag_agent.eval.metrics import hitab_exact_match
from rag_agent.generate.answerer import _DIRECT_SYS
from rag_agent.llm.factory import build_llm
from rag_agent.retrieve.encoders import default_encoder
from rag_agent.retrieve.hybrid_index import HybridIndex
from rag_agent.runenv import guard_resume, run_env
from rag_agent.serialization.base import Chunk
from manual_sentence_ceiling import build_population
from point3_reconstruction_cost import cell_text

SERIALIZATIONS = ("flat", "S2_recon", "S2_gold")
RETRIEVERS = ("bm25", "dense", "hybrid", "cross")
ALPHA = {"bm25": 0.0, "dense": 1.0, "hybrid": 0.5, "cross": 1.0}
KS = (1, 5, 8, 20)


def context_order(order, gold_idx, k, oracle):
    """The k unit indices handed to the reader, gold pinned first when oracle."""
    if not oracle:
        return order[:k]
    return [gold_idx] + [i for i in order if i != gold_idx][: k - 1]


def units(bt, pt, scheme):
    """(texts, gold-cell index map) for one table under one serialization."""
    rp = pt["gold_rp"] if scheme != "S2_recon" else pt["rec_rp"]
    cp = pt["gold_cp"] if scheme != "S2_recon" else pt["rec_cp"]
    txt, idx = [], {}
    for i in range(pt["n_r"]):
        for j in range(pt["n_c"]):
            idx[(i, j)] = len(txt)
            txt.append(cell_text(rp[i], cp[j], bt.data[i][j],
                                 "flat" if scheme == "flat" else "S2"))
    return txt, idx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--topk", type=int, default=8, help="context size for the reader")
    ap.add_argument("--cross-pool", type=int, default=50,
                    help="dense candidates the cross-encoder re-scores")
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--serializations", default=",".join(SERIALIZATIONS))
    ap.add_argument("--retrievers", default=",".join(RETRIEVERS))
    ap.add_argument("--model", default="local:Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--reranker", default="BAAI/bge-reranker-large")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--oracle-cell", action="store_true",
                    help="pin the gold cell first in the reader context "
                         "(recall=1.0 by construction; isolates the reader)")
    ap.add_argument("--out", default="results/cell_retrieval_matrix.json")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force-resume", action="store_true",
                    help="append even though the records file was written under\n"
                         "a different reader/seed/population")
    args = ap.parse_args()

    schemes = [s for s in args.serializations.split(",") if s]
    retrievers = [r for r in args.retrievers.split(",") if r]
    env = run_env(0, "BAAI/bge-small-en-v1.5")

    pop, tables, paths = build_population(args.data_dir, args.split, args.n)
    print(f"[pop] {len(pop)} single-operand lookup queries | "
          f"{len(schemes)}x{len(retrievers)} cells", flush=True)

    enc = default_encoder(model_name="BAAI/bge-small-en-v1.5")
    ce = None
    if "cross" in retrievers:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(args.reranker)
    llm = None if args.no_llm else build_llm(args.model)

    rec_path = Path(str(Path(args.out).with_suffix("")) + "_records.jsonl")
    done = {}
    if args.resume and rec_path.exists():
        for line in open(rec_path):
            r = json.loads(line)
            done[(r["query_id"], r["scheme"], r["retriever"])] = r
        print(f"[resume] {len(done)} records already on disk", flush=True)

    guard_resume(rec_path, env, reader=(None if llm is None else llm.name),
                 population="hitab_dev_lookup_single", force=args.force_resume)
    rec_fh = open(rec_path, "a")
    try:
        for n_done, q in enumerate(pop, 1):
            bt, pt = tables[q.gold_table_id], paths[q.gold_table_id]
            gold = (q.gold_operands[0].row, q.gold_operands[0].col)
            for scheme in schemes:
                todo = [r for r in retrievers
                        if (q.query_id, scheme, r) not in done
                        or (llm is not None and "correct" not in done[(q.query_id, scheme, r)])]
                if not todo:
                    continue
                txt, idx = units(bt, pt, scheme)
                gi = idx[gold]
                chunks = [Chunk(table_id=bt.table_id, chunk_id=f"{bt.table_id}::{n}",
                                text=t, scheme=scheme, kind="cell") for n, t in enumerate(txt)]
                # one index per (table, scheme): both backends are built once and
                # every retriever below is a different read of the same scores
                index = HybridIndex(chunks, encoder=enc, alpha=0.5)
                bm = index._bm25_scores(q.question)
                dn = index._dense_scores(q.question)
                from rag_agent.retrieve.hybrid_index import _minmax
                for r in todo:
                    if r == "cross":
                        pool = np.argsort(-dn)[: args.cross_pool]
                        ce_scores = ce.predict([(q.question, txt[i]) for i in pool])
                        order = list(pool[np.argsort(-np.asarray(ce_scores))])
                        # candidates the first stage never proposed keep their
                        # dense order behind the reranked head
                        seen = set(int(i) for i in pool)
                        order = order + [i for i in np.argsort(-dn) if int(i) not in seen]
                    else:
                        a = ALPHA[r]
                        order = list(np.argsort(-(a * _minmax(dn) + (1 - a) * _minmax(bm))))
                    rank = int(order.index(gi)) + 1
                    rec = {"query_id": q.query_id, "scheme": scheme, "retriever": r,
                           "rank": rank, "pool": len(txt)}
                    if llm is not None:
                        ctx_idx = context_order(order, gi, args.topk, args.oracle_cell)
                        rec["oracle_cell"] = bool(args.oracle_cell)
                        ctx = "\n".join(txt[i] for i in ctx_idx)
                        user = f"ROWS:\n{ctx}\n\nQUESTION: {q.question}\n\nAnswer:"
                        raw = llm.complete(system=_DIRECT_SYS, user=user, max_tokens=512)
                        rec["correct"] = bool(hitab_exact_match(raw, q.answer))
                        rec["pred"] = raw[:120]
                    done[(q.query_id, scheme, r)] = rec
                    rec_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    rec_fh.flush()
                index.close()
            if n_done % 10 == 0:
                print(f"  {n_done}/{len(pop)}", flush=True)
    except RuntimeError as e:
        print(f"[stopped] {str(e)[:160]} ... progress saved, rerun with --resume", flush=True)
    finally:
        rec_fh.close()

    def cell(scheme, r):
        rows = [v for (_, s, x), v in done.items() if s == scheme and x == r]
        if not rows:
            return None
        ranks = [v["rank"] for v in rows]
        em = [v["correct"] for v in rows if "correct" in v]
        return {"n": len(rows),
                **{f"recall@{k}": round(sum(x <= k for x in ranks) / len(ranks), 4) for k in KS},
                "MRR": round(sum(1 / x for x in ranks) / len(ranks), 4),
                "median_rank": float(np.median(ranks)),
                "answer_em": (round(sum(em) / len(em), 4) if em else None),
                "n_answered": len(em)}

    matrix = {s: {r: cell(s, r) for r in RETRIEVERS} for s in SERIALIZATIONS}

    def delta(a, b, r, key):
        x, y = matrix[a][r], matrix[b][r]
        if not x or not y or x[key] is None or y[key] is None:
            return None
        return round(x[key] - y[key], 4)

    # paired significance on the two contrasts the matrix exists to make, per
    # retriever: what the header path buys, and what reconstruction error costs
    from manual_sentence_ceiling import mcnemar
    paired = {}
    for a, b in (("S2_recon", "flat"), ("S2_gold", "S2_recon")):
        for r in RETRIEVERS:
            for key, metric in (("rank", "recall@1"), ("correct", "answer_em")):
                va = {q: v[key] for (q, s, x), v in done.items()
                      if s == a and x == r and key in v}
                vb = {q: v[key] for (q, s, x), v in done.items()
                      if s == b and x == r and key in v}
                shared = sorted(set(va) & set(vb))
                if not shared:
                    continue
                f = (lambda z: z == 1) if key == "rank" else bool
                paired[f"{metric}:{a}_vs_{b}:{r}"] = mcnemar([f(va[q]) for q in shared],
                                                             [f(vb[q]) for q in shared])

    out = {
        "experiment": "cell retrieval: retriever x serialization",
        "paired_tests": paired,
        "population": f"HiTab {args.split}, single gold operand, seed-0 shuffle, n={len(pop)}",
        "table": "oracle (gate-1 solved: HiTab table recall@20=1.0)",
        "pool": "every cell of the gold table",
        "reranker": args.reranker, "cross_pool": args.cross_pool,
        "reader": None if llm is None else llm.name, "topk": args.topk, "env": env,
        "oracle_cell": bool(args.oracle_cell),
        "matrix": matrix,
        "delta_S2recon_minus_flat": {
            r: {k: delta("S2_recon", "flat", r, k)
                for k in ("recall@1", "recall@8", "MRR", "answer_em")} for r in RETRIEVERS},
        "delta_S2gold_minus_S2recon": {
            r: {k: delta("S2_gold", "S2_recon", r, k)
                for k in ("recall@1", "recall@8", "MRR", "answer_em")} for r in RETRIEVERS},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    hdr = f"{'':10}" + "".join(f"{r:>12}" for r in RETRIEVERS)
    print("\nrecall@1" + "\n" + hdr)
    for s in SERIALIZATIONS:
        print(f"{s:10}" + "".join(
            f"{(matrix[s][r]['recall@1'] if matrix[s][r] else float('nan')):>12.4f}"
            for r in RETRIEVERS))
    print(f"wrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
