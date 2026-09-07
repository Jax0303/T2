#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""JSONL (anchor, positive, negative_1..n) 로 bi-encoder 를 미세조정한다 — finetune_cell_encoder.py 와 같은
손실·하이퍼파라미터, 데이터만 파일에서 읽는다 (헤더 범위 문장 pairs, analysis/scope_bench.py pairs).

  PYTHONPATH=. .venv/bin/python scripts/finetune_from_pairs.py --pairs results/scope/train_pairs.jsonl --out models/bge-base-scope-p2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--base", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    import torch
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments)
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    rows = [json.loads(l) for l in open(a.pairs)]
    n_neg = max(len([k for k in r if k.startswith("negative_")]) for r in rows)
    rows = [r for r in rows if len([k for k in r if k.startswith("negative_")]) == n_neg]
    print(f"[data] {len(rows)} rows, {n_neg} negatives each", flush=True)
    model = SentenceTransformer(a.base, device="cuda" if torch.cuda.is_available() else "cpu")
    # 범위 문장은 셀 문장보다 길다 — bge-base 최대 512 토큰까지 그대로 쓴다
    model.max_seq_length = 512
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=8)
    targs = SentenceTransformerTrainingArguments(
        output_dir=f"{a.out}-ckpt", num_train_epochs=a.epochs, per_device_train_batch_size=a.batch_size,
        learning_rate=a.lr, warmup_ratio=0.1, fp16=torch.cuda.is_available(), seed=a.seed,
        logging_steps=50, save_strategy="no", report_to=[])
    SentenceTransformerTrainer(model=model, args=targs, train_dataset=Dataset.from_list(rows), loss=loss).train()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(a.out)
    json.dump({"base": a.base, "pairs": a.pairs, "n_rows": len(rows), "n_neg": n_neg, "epochs": a.epochs,
               "batch_size": a.batch_size, "lr": a.lr, "seed": a.seed, "max_seq_length": 512},
              open(Path(a.out) / "train_meta.json", "w"), indent=1)
    print(f"[saved] {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
