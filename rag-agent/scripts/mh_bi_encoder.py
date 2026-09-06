#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""MultiHiertt 정면 대결 — 우리 bi-encoder(셀 문장, 1회 학습, ANN) vs MT2Net 검색기(크로스인코더, 저장소 재구현).

모집단·풀·gold·분할은 `scripts/mt2net_retriever_baseline.py` 와 **바이트 동일**: 처음 --eval-queries 개가 평가
(within-doc 풀), 그 뒤 --train-queries 개가 학습이며 문서 uid 를 공유하지 않는다. 학습 = (질문, gold 셀 문장) +
같은 문서 negative 4개, CachedMNRL, p1 레시피. 평가 = 문서 풀 안에서 hybrid α 정렬, set_em@k / recall@k / hit@k.
MT2Net 쪽 기록: results/mt2net_retriever_baseline.json (set_em@10 .8624 / @20 .9497, n=298).

  PYTHONPATH=.:scripts .venv/bin/python scripts/mh_bi_encoder.py --scheme S3 --out results/mh/bi_S3.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                                    # noqa: E402
from operand_collision_multihiertt import _minmax, build_corpus, cell_text, load_population  # noqa: E402
from mt2net_retriever_baseline import pools, summarize                # noqa: E402
from rag_agent.retrieve.encoders import _tokenize                     # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-queries", type=int, default=300)
    ap.add_argument("--train-queries", type=int, default=1361)
    ap.add_argument("--scheme", default="S3")
    ap.add_argument("--base", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.5, 0.8, 1.0])
    ap.add_argument("--neg", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-model", default="models/bge-base-mh-cell")
    ap.add_argument("--out", default="results/mh/bi_S3.json")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    import torch
    from datasets import Dataset
    from rank_bm25 import BM25Okapi
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments)
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss

    total = a.eval_queries + a.train_queries
    queries, docs = load_population(total, "arith_multi")
    ev_q, tr_q = queries[:a.eval_queries], queries[a.eval_queries:]
    ev_uids = {q["uid"] for q in ev_q}
    tr_q = [q for q in tr_q if q["uid"] not in ev_uids]
    _, ev_cells, ev_pop = build_corpus(ev_q, {q["uid"]: docs[q["uid"]] for q in ev_q})
    _, tr_cells, tr_pop = build_corpus(tr_q, {q["uid"]: docs[q["uid"]] for q in tr_q})
    print(f"[eval ] {len(ev_pop)} queries | {len(ev_cells)} cells  [train] {len(tr_pop)} | {len(tr_cells)}", flush=True)

    tr_texts = [cell_text(c, a.scheme) for c in tr_cells]
    tr_pools = pools(tr_pop, tr_cells)
    rows = []
    for q in tr_pop:
        gold = {int(g) for g in q["gold"]}
        pool = [gi for gi in tr_pools[q["uid"]] if gi not in gold]
        for g in sorted(gold):
            negs = rng.sample(pool, min(a.neg, len(pool)))
            if len(negs) < a.neg:
                continue
            rows.append({"anchor": q["question"], "positive": tr_texts[g],
                         **{f"negative_{i + 1}": tr_texts[n] for i, n in enumerate(negs)}})
    print(f"[pairs] {len(rows)} rows", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(a.base, device=dev)
    t0 = time.time()
    targs = SentenceTransformerTrainingArguments(
        output_dir=f"{a.save_model}-ckpt", num_train_epochs=a.epochs, per_device_train_batch_size=32,
        learning_rate=2e-5, warmup_ratio=0.1, fp16=(dev == "cuda"), seed=a.seed,
        logging_steps=50, save_strategy="no", report_to=[])
    SentenceTransformerTrainer(model=model, args=targs, train_dataset=Dataset.from_list(rows),
                               loss=CachedMultipleNegativesRankingLoss(model, mini_batch_size=16)).train()
    model.save(a.save_model)
    train_s = round(time.time() - t0, 1)

    ev_texts = [cell_text(c, a.scheme) for c in ev_cells]
    ev_pools = pools(ev_pop, ev_cells)
    E = model.encode(ev_texts, batch_size=256, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    Q = model.encode([q["question"] for q in ev_pop], batch_size=256, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False)
    bm_of = {uid: BM25Okapi([_tokenize(ev_texts[gi]) for gi in idxs]) for uid, idxs in ev_pools.items()}
    out = {"population": {"name": "multihiertt_arith_multi_within_doc", "n_queries": len(ev_pop)},
           "train": {"n_queries": len(tr_pop), "n_rows": len(rows), "neg": a.neg, "epochs": a.epochs,
                     "train_s": train_s, "base": a.base, "disjoint_from_eval": "by document uid"},
           "scheme": a.scheme, "by_alpha": {}}
    for alpha in a.alphas:
        per_query = {}
        for qi, q in enumerate(ev_pop):
            idxs = ev_pools[q["uid"]]
            dn = E[idxs] @ Q[qi]
            bm = np.asarray(bm_of[q["uid"]].get_scores(_tokenize(q["question"])), dtype=np.float32)
            sc = dn if alpha == 1.0 else bm if alpha == 0.0 else alpha * _minmax(dn) + (1 - alpha) * _minmax(bm)
            order = [idxs[int(l)] for l in np.argsort(-sc)]
            gold = {int(g) for g in q["gold"]}
            rank_of = {gi: r for r, gi in enumerate(order, 1) if gi in gold}
            per_query[qi] = [rank_of.get(g) for g in gold]
        out["by_alpha"][str(alpha)] = summarize(per_query)
        print(f"  alpha={alpha}: " + " ".join(f"{k}={v}" for k, v in out["by_alpha"][str(alpha)].items()
                                             if k.startswith(("set_em", "recall@10", "hit@1"))), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
