#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""HiTab train 위에서 셀 검색용 bi-encoder를 미세조정한다.

왜 이 형태인가. 2026-08-25에 실패 유형을 갈랐다: gold 셀의 주소가 코퍼스 전체에서
유일한데도 1등은 47%뿐이고 중앙 순위가 2등이다. 못 찾는 게 아니라 **형제 셀과 1~2등을
가르지 못한다** -- `2013`을 `2012`와, `35 to 44 years`를 `45 to 54 years`와.
그래서 hard negative는 무작위 셀이 아니라 **같은 표의 다른 셀**이어야 한다. 그것이
실제로 gold를 밀어내고 있는 것들이다.

누출 방지: 학습은 `populations/hitab_train_lookup_all.txt`(n=3,667)에서만 나오고,
HiTab의 train/dev/test는 표를 하나도 공유하지 않는다(각각 2,519 / 540 / 538, 교집합 0).
dev는 평가용, test는 손대지 않는다.

쿼리 접두어: BGE v1.5는 쿼리 쪽에만 instruction을 달고 학습됐다. 학습과 추론이
어긋나면 미세조정이 그 어긋남을 배우므로, `--query-prefix`로 추론과 같은 설정을 준다.

  PYTHONPATH=. python3 scripts/finetune_cell_encoder.py --out models/bge-cell-ft
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--out", default="models/bge-cell-ft")
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--population", default="hitab_train_lookup_all")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--neg-per-query", type=int, default=4,
                    help="같은 표에서 뽑는 hard negative 수")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--query-prefix", default="auto",
                    help="'auto' = default_prefixes(base)의 쿼리 접두어, 'none' = 없음")
    args = ap.parse_args()

    import torch
    from sentence_transformers import (SentenceTransformer,
                                       SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments)
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from datasets import Dataset

    from corpus_dump_vs_cell import hitab_corpus
    from rag_agent.retrieve.encoders import default_prefixes
    from rag_agent.serialization.caption import caption_sentence
    from rag_agent.serialization.templates import STRUCTURAL_COMPACT

    random.seed(args.seed)
    qpre = (default_prefixes(args.base)[0] if args.query_prefix == "auto"
            else ("" if args.query_prefix == "none" else args.query_prefix))
    print(f"[cfg] base={args.base} qpre={qpre!r} neg={args.neg_per_query}", flush=True)

    C = hitab_corpus(args.data_dir, "train", args.population)
    C.cell_text[:] = [caption_sentence(C.title.get(t, ""), *C.cell_paths[n],
                                       template=STRUCTURAL_COMPACT)
                      for n, (t, i, j) in enumerate(C.cell_owner)]
    pos_of = {k: n for n, k in enumerate(C.cell_owner)}
    by_table: dict[str, list[int]] = {}
    for n, (t, i, j) in enumerate(C.cell_owner):
        by_table.setdefault(t, []).append(n)

    rows = []
    for q in C.queries:
        g = [pos_of[k] for k in q["gold_cells"] if k in pos_of]
        if len(g) != 1:
            continue
        g = g[0]
        # 같은 표의 다른 셀 -- 실제로 gold를 밀어내는 것들
        sibs = [p for p in by_table[C.cell_owner[g][0]] if p != g]
        if not sibs:
            continue
        negs = random.sample(sibs, min(args.neg_per_query, len(sibs)))
        row = {"anchor": qpre + q["question"], "positive": C.cell_text[g]}
        for i, p in enumerate(negs):
            row[f"negative_{i+1}"] = C.cell_text[p]
        if len(negs) == args.neg_per_query:      # 열 수가 고른 행만 (Dataset 요구)
            rows.append(row)
    print(f"[data] {len(rows)} triplets from {len(C.queries)} queries "
          f"({len(by_table)} tables)", flush=True)

    ds = Dataset.from_list(rows)
    model = SentenceTransformer(args.base,
                                device="cuda" if torch.cuda.is_available() else "cpu")
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=16)
    targs = SentenceTransformerTrainingArguments(
        output_dir=f"{args.out}-ckpt", num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, learning_rate=args.lr,
        warmup_ratio=0.1, fp16=torch.cuda.is_available(), seed=args.seed,
        logging_steps=50, save_strategy="no", report_to=[])
    SentenceTransformerTrainer(model=model, args=targs,
                               train_dataset=ds, loss=loss).train()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    json.dump({"base": args.base, "population": args.population,
               "n_triplets": len(rows), "neg_per_query": args.neg_per_query,
               "epochs": args.epochs, "batch_size": args.batch_size,
               "lr": args.lr, "seed": args.seed, "query_prefix": qpre,
               "negatives": "same-table siblings",
               "leakage": "train/dev/test share zero tables and zero queries"},
              open(Path(args.out) / "train_meta.json", "w"), indent=1)
    print(f"[saved] {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
