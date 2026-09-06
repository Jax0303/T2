#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""검색 상위 후보 중 정답 셀을 고르는 크로스인코더를 **이 도메인에서** 학습한다 (E3).

기성품 bge-reranker 는 파인튜닝된 bi-encoder 위에서 해롭다(hit@1 .7193 → .6181). 원인은
표면형 — 기성품은 `제목 > 행경로 > 열경로: 값` 문장을 본 적이 없고, 정답을 밀어내는 것은
같은 표의 형제 셀이다. 그래서 negative 를 무작위가 아니라 **1단계 검색기(p0)가 실제로 상위에
올린 오답 셀**에서 뽑는다. 학습은 train fit 풀(표 기준 분할)만, 평가는 dev/test.

  PYTHONPATH=.:scripts .venv/bin/python scripts/train_cell_reranker.py \
      --retriever models/bge-base-cell-ft-p0 --out models/ce-cell-p0
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
    ap.add_argument("--base", default="BAAI/bge-reranker-base")
    ap.add_argument("--retriever", default="models/bge-base-cell-ft-p0",
                    help="negative 를 채굴할 1단계 인코더 (dense 만, 학습 코퍼스 전체)")
    ap.add_argument("--out", default="models/ce-cell-p0")
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--population", default="hitab_train_fit_alltypes")
    ap.add_argument("--title-mode", default="page")
    ap.add_argument("--neg-per-query", type=int, default=7, help="dense top-k 에서 뽑는 오답 수")
    ap.add_argument("--mine-topk", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=224)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.cross_encoder import CrossEncoder
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
    from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
    from datasets import Dataset
    from corpus_dump_vs_cell import hitab_corpus
    from rag_agent.serialization.caption import caption_sentence, effective_titles
    from rag_agent.serialization.templates import STRUCTURAL_COMPACT

    random.seed(a.seed)
    C = hitab_corpus(a.data_dir, "train", a.population)
    pt = json.load(open("results/tableconf/totto_page_titles.json")) if a.title_mode == "page" else None
    ti = effective_titles(C.tids, C.title, C.cell_owner, C.cell_paths, a.title_mode, page_titles=pt)
    texts = [caption_sentence(ti[t], *C.cell_paths[n], template=STRUCTURAL_COMPACT)
             for n, (t, _i, _j) in enumerate(C.cell_owner)]
    pos_of = {k: n for n, k in enumerate(C.cell_owner)}
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # --- mine: the retriever's own top-k, dense only (cheap, and it is what it gets wrong) ---
    enc = SentenceTransformer(a.retriever, device=dev)
    print(f"[mine] encoding {len(texts)} cells with {a.retriever}", flush=True)
    E = enc.encode(texts, batch_size=256, convert_to_tensor=True, normalize_embeddings=True,
                   show_progress_bar=False).half()
    qs = [q for q in C.queries if any(k in pos_of for k in q["gold_cells"])]
    rows, n_pos = [], 0
    for s in range(0, len(qs), 256):
        chunk = qs[s:s + 256]
        Q = enc.encode([q["question"] for q in chunk], batch_size=256, convert_to_tensor=True,
                       normalize_embeddings=True, show_progress_bar=False).half()
        idx = (Q @ E.T).topk(a.mine_topk, dim=1).indices.cpu().tolist()
        for q, top in zip(chunk, idx):
            gold = {pos_of[k] for k in q["gold_cells"] if k in pos_of}
            negs = [p for p in top if p not in gold][:a.neg_per_query]
            for g in sorted(gold):
                rows.append({"query": q["question"], "cell": texts[g], "label": 1.0}); n_pos += 1
            for p in negs:
                rows.append({"query": q["question"], "cell": texts[p], "label": 0.0})
    del E, enc
    torch.cuda.empty_cache()
    random.shuffle(rows)
    print(f"[data] {len(rows)} pairs ({n_pos} positive) from {len(qs)} queries", flush=True)
    if a.dry_run:
        return 0

    model = CrossEncoder(a.base, num_labels=1, max_length=a.max_length, device=dev)
    ds = Dataset.from_list(rows)
    args = CrossEncoderTrainingArguments(
        output_dir=f"{a.out}-ckpt", num_train_epochs=a.epochs, per_device_train_batch_size=a.batch_size,
        learning_rate=a.lr, warmup_ratio=0.1, fp16=torch.cuda.is_available(), seed=a.seed,
        logging_steps=100, save_strategy="no", report_to=[])
    CrossEncoderTrainer(model=model, args=args, train_dataset=ds,
                        loss=BinaryCrossEntropyLoss(model)).train()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out)
    json.dump({"base": a.base, "retriever": a.retriever, "population": a.population,
               "n_pairs": len(rows), "n_pos": n_pos, "neg_per_query": a.neg_per_query,
               "mine_topk": a.mine_topk, "epochs": a.epochs, "batch_size": a.batch_size,
               "max_length": a.max_length, "lr": a.lr, "seed": a.seed, "title_mode": a.title_mode,
               "negatives": "retriever's dense top-k on the train corpus, gold removed"},
              open(Path(a.out) / "train_meta.json", "w"), indent=1)
    print(f"[saved] {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
