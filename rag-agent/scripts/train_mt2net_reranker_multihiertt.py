#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""MT2Net(Zhao 2022)의 실제 채점 메커니즘(BERT pairwise 분류기)을 MultiHiertt에서
문서 범위로 학습한다. PREREG-2026-09-20-mt2net-reranker-multihiertt.md.

`train_mt2net_reranker_hitab.py`의 학습 루프를 재사용하되, 표 단위가 아니라
문서(uid) 단위로 후보 풀을 좁힌다(MultiHiertt는 문서 하나에 표가 여러 개 있어
MT2Net의 "문서 범위" 전제가 실제로 성립함 — HiTab은 표 540개 중 537개가
문서당 표 1개뿐이라 성립하지 않았음, 그래서 이 arm이 없었다).

학습: MultiHiertt validation split(338질의/338문서, table-only). 평가는
mh_arms.py가 쓰는 train split(§2, 2908건) — 두 분할의 문서 집합이 겹치지
않음을 코드에서 확인한다(교집합 0).

  PYTHONPATH=.:scripts .venv/bin/python scripts/train_mt2net_reranker_multihiertt.py
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
    ap.add_argument("--out", default="models/mt2net-reranker-multihiertt")
    ap.add_argument("--neg-per-query", type=int, default=7)
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

    from mh_arms import build_tables, load_population, mt2net_units, resolve_gold
    from rag_agent.retrieve.encoders import default_encoder
    from rag_agent.retrieve.hybrid_index import _minmax, _tokenize
    from rag_agent.retrieve.sparse_bm25 import SparseBM25

    random.seed(a.seed)
    ALPHA = 0.7

    t0 = time.time()
    queries, docs, skipped = load_population("validation")
    _eval_queries, eval_docs, _ = load_population("train")
    assert not (set(docs) & set(eval_docs)), (
        "validation docs leak into the train-split eval population — "
        "this must never fire (official HF splits should be doc-disjoint)")
    print(f"[data] {len(queries)} validation queries / {len(docs)} docs, "
          f"eval-doc overlap=0 (checked against {len(eval_docs)} train docs), "
          f"skipped={dict(skipped)} ({time.time() - t0:.0f}s)", flush=True)

    tables, hdr = build_tables(docs, header_rule="v1", label_rule="none")
    texts, covers, _is_row, unit_tids, _grid = mt2net_units(tables, docs, form="desc")
    uid_arr = np.array([t.split("::")[0] for t in unit_tids])
    print(f"[corpus] {len(tables)} tables / {len(texts)} mt2net_desc units "
          f"({time.time() - t0:.0f}s)", flush=True)

    live = {(tid, i, j) for tid, tab in tables.items()
            for i, row in enumerate(tab.table.data)
            for j, v in enumerate(row) if str(v).strip()}
    queries = resolve_gold(queries, tables, hdr, live)
    queries = [q for q in queries if not q["excluded"] and q["gold"]]
    pos_of = {}
    for i, c in enumerate(covers):
        if c:
            pos_of.setdefault(next(iter(c)), i)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = default_encoder()
    bm = SparseBM25(_tokenize(t) for t in texts)
    t0 = time.time()
    E = enc.encode(texts)
    print(f"[dense] encoded {len(texts)} units with {enc.name} in {time.time() - t0:.0f}s", flush=True)

    rows, n_pos = [], 0
    t0 = time.time()
    for qi, q in enumerate(queries, 1):
        sel = np.flatnonzero(uid_arr == q["uid"])
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
        if qi % 100 == 0:
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
        logging_steps=50, save_strategy="no", report_to=[])
    t0 = time.time()
    CrossEncoderTrainer(model=model, args=args, train_dataset=ds,
                        loss=BinaryCrossEntropyLoss(model)).train()
    train_s = time.time() - t0
    Path(a.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out)
    json.dump({"base": a.base, "negatives": "hybrid(default_encoder+BM25,a=0.7) doc-scope top-k, gold removed",
               "n_pairs": len(rows), "n_pos": n_pos, "neg_per_query": a.neg_per_query,
               "epochs": a.epochs, "batch_size": a.batch_size,
               "max_length": a.max_length, "lr": a.lr, "seed": a.seed,
               "n_train_queries": len(queries), "n_train_docs": len(docs),
               "train_split": "validation", "eval_split": "train",
               "train_seconds": round(train_s, 1)},
              open(Path(a.out) / "train_meta.json", "w"), indent=1)
    print(f"[saved] {a.out} (train {train_s:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
