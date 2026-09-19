#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""MT2Net(Zhao 2022)의 실제 채점 메커니즘 — 질문+후보 문장을 BERT류 pairwise 분류기로
채점 — 을 HiTab에서 학습한다. PREREG-2026-09-20-mt2net-reranker-hitab.md.

`git show 575ff88~1:rag-agent/scripts/train_cell_reranker.py`의 학습 루프를 재사용하되,
negative 채굴을 (삭제된) 파인튜닝 retriever 대신 기성품 하이브리드(default_encoder+BM25,
α=0.7)로 바꾸고, `--corpus gold`와 동일하게 질의 자신의 표 안에서만 채굴한다.

  PYTHONPATH=.:scripts .venv/bin/python scripts/train_mt2net_reranker_hitab.py
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="BAAI/bge-reranker-base")
    ap.add_argument("--out", default="models/mt2net-reranker-hitab")
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--eval-tables", default="results/mt2net_reranker_hitab/eval_tables.json",
                    help="이 표들이 학습 데이터에 섞이면 즉시 assert 실패")
    ap.add_argument("--neg-per-query", type=int, default=7)
    ap.add_argument("--mine-topk", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=224)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    import numpy as np
    import torch
    from datasets import Dataset
    from sentence_transformers.cross_encoder import CrossEncoder
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
    from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments

    from retrieval_accuracy import PAGE_TITLES, build_corpus, load_queries
    from rag_agent.retrieve.encoders import default_encoder
    from rag_agent.retrieve.hybrid_index import _minmax, _tokenize
    from rag_agent.retrieve.sparse_bm25 import SparseBM25

    random.seed(a.seed)
    ALPHA = 0.7

    eval_tables = set(json.loads(Path(a.eval_tables).read_text()))
    assert eval_tables, f"{a.eval_tables} is empty — refusing to train without a known eval exclusion set"

    tabs: dict = {}
    queries = load_queries(a.data_dir, "train", tabs)
    queries = [q for q in queries if not q["excluded"] and q["gold"]]
    tids = sorted({q["table_id"] for q in queries})
    assert not (set(tids) & eval_tables), (
        "train tables leak into the eval population — HiTab train/test should be "
        "table-disjoint by construction; this must never fire")
    print(f"[data] {len(queries)} train queries (gold present) over {len(tids)} tables, "
          f"eval-table overlap=0 (checked)", flush=True)

    page_titles = json.loads(PAGE_TITLES.read_text()) if PAGE_TITLES.exists() else {}
    t0 = time.time()
    texts, covers, _is_row, unit_tids, _grid = build_corpus(
        a.data_dir, tids, "mt2net", "cell", page_titles)
    unit_tid_arr = np.array(unit_tids)
    print(f"[corpus] {len(tids)} tables / {len(texts)} cells ({time.time() - t0:.0f}s)", flush=True)
    pos_of = {next(iter(c)): i for i, c in enumerate(covers)}
    assert len(pos_of) == len(covers), "duplicate cell coordinate in train corpus"

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = default_encoder()
    bm = SparseBM25(_tokenize(t) for t in texts)
    t0 = time.time()
    E = enc.encode(texts)
    print(f"[dense] encoded {len(texts)} cells with {enc.name} in {time.time() - t0:.0f}s", flush=True)

    rows, n_pos = [], 0
    t0 = time.time()
    for qi, q in enumerate(queries, 1):
        sel = np.flatnonzero(unit_tid_arr == q["table_id"])
        if len(sel) == 0:
            continue
        gold_units = {pos_of[g] for g in q["gold"] if g in pos_of}
        if not gold_units:
            continue
        s_sparse = bm.get_scores(_tokenize(q["question"]))[sel]
        d = E[sel] @ enc.encode_query([q["question"]])[0].astype(np.float32)
        s = ALPHA * _minmax(d) + (1 - ALPHA) * _minmax(s_sparse)
        order = sel[np.argsort(-s, kind="stable")]
        negs = [int(p) for p in order if p not in gold_units][:a.neg_per_query]
        for g in sorted(gold_units):
            rows.append({"query": q["question"], "cell": texts[g], "label": 1.0})
            n_pos += 1
        for p in negs:
            rows.append({"query": q["question"], "cell": texts[p], "label": 0.0})
        if qi % 1000 == 0:
            print(f"[mine] {qi}/{len(queries)} ({time.time() - t0:.0f}s)", flush=True)
    del E
    if dev == "cuda":
        torch.cuda.empty_cache()

    random.shuffle(rows)
    print(f"[data] {len(rows)} pairs ({n_pos} positive) from {len(queries)} queries", flush=True)
    if a.dry_run:
        return 0

    model = CrossEncoder(a.base, num_labels=1, max_length=a.max_length, device=dev)
    ds = Dataset.from_list(rows)
    args = CrossEncoderTrainingArguments(
        output_dir=f"{a.out}-ckpt", num_train_epochs=a.epochs, per_device_train_batch_size=a.batch_size,
        learning_rate=a.lr, warmup_ratio=0.1, fp16=torch.cuda.is_available(), seed=a.seed,
        logging_steps=100, save_strategy="no", report_to=[])
    t0 = time.time()
    CrossEncoderTrainer(model=model, args=args, train_dataset=ds,
                        loss=BinaryCrossEntropyLoss(model)).train()
    train_s = time.time() - t0
    Path(a.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out)
    json.dump({"base": a.base, "negatives": "hybrid(default_encoder+BM25,a=0.7) gold-corpus top-k, gold removed",
               "n_pairs": len(rows), "n_pos": n_pos, "neg_per_query": a.neg_per_query,
               "mine_topk": a.mine_topk, "epochs": a.epochs, "batch_size": a.batch_size,
               "max_length": a.max_length, "lr": a.lr, "seed": a.seed,
               "n_train_queries": len(queries), "n_train_tables": len(tids),
               "train_seconds": round(train_s, 1)},
              open(Path(a.out) / "train_meta.json", "w"), indent=1)
    print(f"[saved] {a.out} (train {train_s:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
