#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Literature-standard IR metrics for the verbalized cell-sentence retriever.

Replaces the ad-hoc ``cell_hit@k`` in verbalize_retrieval_eval.py with the
exact metric set the rest of the paper reports (prof's rule: metrics must
match the comparison literature) — Recall@k, Hit@k, MRR, nDCG@k, set-EM@k —
computed from each gold operand cell's 1-based rank in the full sentence
ranking, reusing the metric/significance functions of
standard_ir_metrics_from_records.py verbatim.

Conditions are the sentence-length ablation (sent_short/medium/long); the
fulltable baselines have no cell granularity, so cell-level metrics do not
exist for them (table-level R@k/MRR for all five conditions live in
verbalize_retrieval_eval.py results). Paired tests: sent_long vs the other
two styles, Wilcoxon for continuous metrics, exact binomial flip for binary.

Retrievers (``--retrievers``), matching operand_collision_within_doc.py so the
two legs are read on the same axis:

  dense   — bi-encoder cosine over the whole split pool (the original arm)
  bm25    — BM25Okapi over the same pool, indexed on the BARE sentence (the
            bge query instruction is an embedder prompt; feeding one constant
            into every document only skews the lexical statistics)
  hybrid  — 0.5*dense + 0.5*bm25 after per-query min-max, alpha as in
            operand_collision_*.py
  cross   — TWO-STAGE, because this pool is 67k sentences and a cross-encoder
            has no precomputable representation: hybrid supplies the top
            ``--rerank-depth`` (default 100), bge-reranker-large reorders those,
            and a gold cell that stage 1 never surfaced KEEPS ITS HYBRID RANK.
            Reranking can only reorder what retrieval found, so crediting an
            unsurfaced cell with a reranked position would be fiction. This is
            also why MT2Net's retriever is a stage-2 object (RESULTS.md J-3):
            same architecture class, same per-query cost, different supervision.

Usage:
  .venv/bin/python scripts/verbalize_standard_ir_metrics.py                # full dev
  .venv/bin/python scripts/verbalize_standard_ir_metrics.py --n 50        # smoke
  .venv/bin/python scripts/verbalize_standard_ir_metrics.py \
      --retrievers dense,bm25,hybrid,cross
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rag_agent.bench.hitab import load_queries  # noqa: E402
from rag_agent.retrieve.encoders import _tokenize  # noqa: E402
from rag_agent.serialize.verbalize import verbalize_table, STYLES  # noqa: E402
from operand_collision_multihiertt import _minmax  # noqa: E402
from standard_ir_metrics_from_records import (  # noqa: E402
    summarize, paired_tests)
import standard_ir_metrics_from_records as sim  # noqa: E402

SEED = 42
MODEL = "BAAI/bge-small-en-v1.5"
ALPHAS = {"bm25": 0.0, "dense": 1.0, "hybrid": 0.5}  # as operand_collision_*.py
QUERY_INSTR = "Represent this sentence for searching relevant passages: "
CACHE_DIR = ROOT / "data" / "verbalize_cache"
KS = (10, 20, 50)
sim.KS = KS  # summarize/paired_tests read the module-level constant


def build_sentence_corpus(tables: dict, style: str):
    texts, keys = [], []  # keys: (table_id, row, col)
    for tid in sorted(tables):
        for ch in verbalize_table(tables[tid], style):
            texts.append(ch.text)
            keys.append((tid, ch.rows[0], ch.cols[0]))
    return texts, keys


def encode_cached(enc, tag: str, texts):
    path = CACHE_DIR / f"{tag.replace('/', '_')}.npy"
    if path.exists():
        vecs = np.load(path)
        if len(vecs) == len(texts):
            return vecs
    t0 = time.time()
    vecs = enc.encode(texts)
    print(f"    encoded {len(texts)} in {time.time()-t0:.0f}s")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, vecs)
    return vecs


def gold_cell_ranks(qvecs, cvecs, keys, queries, block=64, bm25=None,
                    alphas=(("dense", 1.0),), reranker=None, texts=None,
                    rerank_depth=100, rerank_batch=64):
    """Per retriever, per query: 1-based rank of each gold operand cell's
    sentence in the full ranking (None if the cell has no sentence).

    ``bm25`` is a prebuilt ``BM25Okapi`` over ``texts``; when absent only the
    alphas that need no lexical side (dense) are computable.
    """
    key2idx = {k: i for i, k in enumerate(keys)}
    out = {name: {} for name, _ in alphas}
    if reranker is not None:
        out["cross"] = {}
    for s in range(0, len(qvecs), block):
        sims = qvecs[s:s + block] @ cvecs.T
        for bi in range(sims.shape[0]):
            q = queries[s + bi]
            dn = _minmax(sims[bi].astype(np.float32))
            bm = (_minmax(np.asarray(bm25.get_scores(_tokenize(q.question)),
                                     dtype=np.float32))
                  if bm25 is not None else None)
            gold_idx = [key2idx.get((q.gold_table_id, op.row, op.col))
                        for op in q.gold_operands]

            scores = {}
            for name, a in alphas:
                scores[name] = dn if a == 1.0 or bm is None else \
                    (a * dn + (1.0 - a) * bm)
            for name, sc in scores.items():
                # rank of i = 1 + count of strictly higher scores
                out[name][q.query_id] = [None if i is None
                                         else int((sc > sc[i]).sum()) + 1
                                         for i in gold_idx]

            if reranker is not None:
                base = scores["hybrid"]
                # stage 1 hands over a fixed-depth candidate list; everything
                # below it keeps the rank stage 1 gave it.
                cand = np.argpartition(-base, rerank_depth)[:rerank_depth]
                cand = cand[np.argsort(-base[cand])]
                cs = reranker.predict([(q.question, texts[int(i)]) for i in cand],
                                      batch_size=rerank_batch,
                                      show_progress_bar=False)
                new_rank = {int(cand[p]): pos for pos, p in
                            enumerate(np.argsort(-np.asarray(cs)), 1)}
                base_rank = out["hybrid"][q.query_id]
                out["cross"][q.query_id] = [
                    None if i is None else new_rank.get(i, br)
                    for i, br in zip(gold_idx, base_rank)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default=MODEL)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--retrievers", default="dense",
                    help="comma list of bm25,dense,hybrid,cross ('cross' implies hybrid)")
    ap.add_argument("--reranker", default="BAAI/bge-reranker-large")
    ap.add_argument("--rerank-depth", type=int, default=100)
    ap.add_argument("--rerank-max-length", type=int, default=192)
    ap.add_argument("--rerank-batch-size", type=int, default=64)
    ap.add_argument("--styles", default=",".join(STYLES))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rets = [r.strip() for r in args.retrievers.split(",") if r.strip()]
    want_cross = "cross" in rets
    if want_cross and "hybrid" not in rets:
        rets.append("hybrid")  # cross reranks hybrid's candidate list
    alphas = [(r, ALPHAS[r]) for r in rets if r in ALPHAS]
    need_bm25 = any(a < 1.0 for _, a in alphas)
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]

    queries, tables = load_queries("data/hitab", args.split)
    queries = [q for q in queries if q.gold_operands]
    if args.n:
        import random
        random.Random(SEED).shuffle(queries)
        queries = queries[:args.n]
    print(f"operand queries={len(queries)} table_pool={len(tables)}")

    from rag_agent.retrieve.encoders import SentenceTransformerEncoder
    enc = SentenceTransformerEncoder(model_name=args.encoder, batch_size=256)
    qvecs = enc.encode([QUERY_INSTR + q.question for q in queries])

    reranker = None
    if want_cross:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(args.reranker, max_length=args.rerank_max_length)

    enc_tag = args.encoder.replace("/", "_")
    report = {"config": {"encoder": args.encoder, "split": args.split,
                         "n_operand_queries": len(queries),
                         "table_pool": len(tables), "ks": list(KS),
                         "retrievers": rets, "styles": styles,
                         "reranker": args.reranker if want_cross else None,
                         "rerank_depth": args.rerank_depth if want_cross else None,
                         "rerank_note": ("two-stage: hybrid top-depth reranked; a gold "
                                         "cell stage 1 never surfaced keeps its hybrid "
                                         "rank") if want_cross else None,
                         "metric_defs": "standard_ir_metrics_from_records.py"},
              "by_condition": {}, "paired_vs_sent_long": {}}

    per_cond = {}
    for style in styles:
        texts, keys = build_sentence_corpus(tables, style)
        cvecs = encode_cached(enc, f"{args.split}_sent_{style}_{enc_tag}", texts)
        bm25 = None
        if need_bm25:
            from rank_bm25 import BM25Okapi
            t0 = time.time()
            bm25 = BM25Okapi([_tokenize(t) for t in texts])
            print(f"    bm25 index {len(texts)} docs in {time.time()-t0:.0f}s")
        t0 = time.time()
        by_ret = gold_cell_ranks(qvecs, cvecs, keys, queries, bm25=bm25,
                                 alphas=alphas, reranker=reranker, texts=texts,
                                 rerank_depth=args.rerank_depth,
                                 rerank_batch=args.rerank_batch_size)
        print(f"    ranked {len(queries)} queries in {time.time()-t0:.0f}s")
        for ret, pq in by_ret.items():
            cond = f"sent_{style}/{ret}"
            per_cond[cond] = pq
            report["by_condition"][cond] = summarize(pq)
            print(f"[{cond}]", report["by_condition"][cond], flush=True)

    # the length ladder is read within a retriever, so long/X is the reference
    for cond in list(per_cond):
        style, _, ret = cond.partition("/")
        if style == "sent_long":
            continue
        ref = f"sent_long/{ret}"
        if ref in per_cond:
            report["paired_vs_sent_long"][f"{cond}->{ref}"] = paired_tests(
                per_cond[cond], per_cond[ref])

    out = Path(args.out) if args.out else \
        ROOT / "results" / f"verbalize_standard_ir_{args.split}_{enc_tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print("saved →", out)

    hdr = (f"{'condition':<22} {'MRR':>6} " +
           " ".join(f"{m}@{k:<3}" for k in KS
                    for m in ("hit", "R", "nDCG", "EM")))
    print(hdr)
    for cond, m in report["by_condition"].items():
        cells = " ".join(
            f"{m[f'{name}@{k}']:>6.3f}" for k in KS
            for name in ("hit", "recall", "ndcg", "set_em"))
        print(f"{cond:<22} {m['mrr']:>6.3f} {cells}")


if __name__ == "__main__":
    main()
