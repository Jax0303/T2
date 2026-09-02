#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""HiTab train 위에서 셀 검색용 bi-encoder를 미세조정한다.

왜 이 형태인가. 2026-08-25에 실패 유형을 갈랐다: gold 셀의 주소가 코퍼스 전체에서
유일한데도 1등은 47%뿐이고 중앙 순위가 2등이다. 못 찾는 게 아니라 **형제 셀과 1~2등을
가르지 못한다** -- `2013`을 `2012`와, `35 to 44 years`를 `45 to 54 years`와.
그래서 hard negative는 무작위 셀이 아니라 **같은 표의 다른 셀**이어야 한다. 그것이
실제로 gold를 밀어내고 있는 것들이다.

그 진단은 여전히 맞지만 구간이 좁다. `results/rank/STAGE1_BOARD.md` §B의 더 최근
측정은 setEM@10 실패의 **94.3%가 다른 표 셀**이라고 말한다. 둘은 모순이 아니라 서로
다른 구간이다 -- 형제는 gold가 2~3위일 때의 경쟁자, 다른 표 셀은 gold가 10위 밖으로
밀렸을 때의 경쟁자다. setEM@10 .8373 -> .9는 **후자**를 고치는 일이므로 negative에
현재 색인에서 채굴한 고득점 타표 셀이 들어가야 한다: `--neg-cross`.

채굴은 dense 전용이다. 평가 색인은 hybrid(alpha=0.7)지만 3,667개 질의에 대해
35만 셀 BM25를 돌리는 비용이 채굴 이득에 비해 크고, 파인튜닝이 실제로 움직이는
것은 코사인 쪽이다. 이 근사는 train_meta.json에 `negatives`로 기록된다.

누출 방지: 학습은 `populations/hitab_train_lookup_all.txt`(n=3,667)에서만 나오고,
HiTab의 train/dev/test는 표를 하나도 공유하지 않는다(각각 2,519 / 540 / 538, 교집합 0).
dev는 평가용, test는 손대지 않는다.

쿼리 접두어: BGE v1.5는 쿼리 쪽에만 instruction을 달고 학습됐다. 학습과 추론이
어긋나면 미세조정이 그 어긋남을 배우므로, `--query-prefix`로 추론과 같은 설정을 준다.
**평가기 `analysis/cell_rank_dump.py`는 질의에 접두어를 붙이지 않는다** -- 그래서
기본값이 `none`이다. 접두어 자체는 이 저장소에서 이미 무효로 측정됐다
(HiTab .789->.788, AIT .550->.552). 바꿔 달려면 평가도 같이 바꿔야 한다.

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


def model_for_mining(base: str):
    """The index the negatives are mined FROM is the one being replaced: the
    off-the-shelf base encoder, which is what produced .5157 R@1. Mining with a
    half-trained model would chase its own errors."""
    from sentence_transformers import SentenceTransformer
    import torch
    return SentenceTransformer(base,
                               device="cuda" if torch.cuda.is_available() else "cpu")


def mine_cross_table(model, C, usable, qpre, n_cross, topk, batch):
    """For each (query, gold) pick the ``n_cross`` best-scoring cells that sit in
    a DIFFERENT table. Returns a list aligned with ``usable``.

    These are the competitors §B measured: 94.3% of what outranks a gold cell in
    a setEM@10 failure comes from another table, and no sibling negative teaches
    the encoder anything about them.
    """
    import torch
    if n_cross <= 0:
        return [[] for _ in usable]
    tid_of = {t: k for k, t in enumerate(C.tids)}
    cell_tid = torch.tensor([tid_of[t] for t, _i, _j in C.cell_owner])
    dev = model.device
    print(f"[mine] encoding {len(C.cell_text)} cells", flush=True)
    E = model.encode(C.cell_text, batch_size=256, convert_to_tensor=True,
                     normalize_embeddings=True, show_progress_bar=False).to(dev).half()
    cell_tid = cell_tid.to(dev)
    out = []
    for s in range(0, len(usable), batch):
        chunk = usable[s:s + batch]
        Q = model.encode([qpre + q["question"] for q, _g, _gs in chunk], batch_size=256,
                         convert_to_tensor=True, normalize_embeddings=True,
                         show_progress_bar=False).to(dev).half()
        idx = (Q @ E.T).topk(topk, dim=1).indices
        for r, (_q, g, gs) in enumerate(chunk):
            want = cell_tid[g]
            keep = [int(p) for p in idx[r]
                    if cell_tid[p] != want and int(p) not in gs][:n_cross]
            out.append(keep)
        del Q, idx
        print(f"[mine] {min(s + batch, len(usable))}/{len(usable)}", flush=True)
    del E
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return out


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
                    help="쿼리당 hard negative 총 수")
    ap.add_argument("--neg-cross", type=int, default=2,
                    help="그중 다른 표에서 채굴하는 수 (나머지는 같은 표 형제). "
                         "0이면 형제만 -- 2026-08-25 이전 동작")
    ap.add_argument("--mine-topk", type=int, default=50,
                    help="채굴 후보로 볼 dense 상위 셀 수")
    ap.add_argument("--mine-batch", type=int, default=256, help="채굴 쿼리 배치")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--multi-gold", default="skip", choices=["skip", "one", "all"],
                    help="gold 셀이 2개 이상인 질문 처리. 'skip'=버림(현행), "
                         "'one'=질문당 1행 (positive는 seed로 고른 gold 1개), "
                         "'all'=(질문, gold셀)마다 1행. "
                         "PREREG-2026-09-02-train-expand.md")
    ap.add_argument("--dry-run", action="store_true",
                    help="mine and count triplets, then stop before training")
    ap.add_argument("--query-prefix", default="none",
                    help="'none' = 접두어 없음 (기본, 평가기와 일치), "
                         "'auto' = default_prefixes(base)의 쿼리 접두어")
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
    n_cross = min(args.neg_cross, args.neg_per_query)
    n_sib = args.neg_per_query - n_cross
    print(f"[cfg] base={args.base} qpre={qpre!r} neg={args.neg_per_query} "
          f"(sib={n_sib} cross={n_cross})", flush=True)

    C = hitab_corpus(args.data_dir, "train", args.population)
    C.cell_text[:] = [caption_sentence(C.title.get(t, ""), *C.cell_paths[n],
                                       template=STRUCTURAL_COMPACT)
                      for n, (t, i, j) in enumerate(C.cell_owner)]
    pos_of = {k: n for n, k in enumerate(C.cell_owner)}
    by_table: dict[str, list[int]] = {}
    for n, (t, i, j) in enumerate(C.cell_owner):
        by_table.setdefault(t, []).append(n)

    # usable[n] = (query, positive cell, 그 질문의 gold 셀 전부)
    # gold 전체를 들고 다니는 이유: negative에서 **모든** gold를 빼야 한다. 하나만
    # 빼면 다중 gold 질문에서 다른 정답 칸을 오답으로 가르친다.
    usable = []
    for q in C.queries:
        g = [pos_of[k] for k in q["gold_cells"] if k in pos_of]
        if not g:
            continue
        gs = set(g)
        if not any(p not in gs for p in by_table[C.cell_owner[g[0]][0]]):
            continue
        if len(g) == 1:
            usable.append((q, g[0], gs))
        elif args.multi_gold == "one":
            usable.append((q, random.Random(args.seed).choice(sorted(g)), gs))
        elif args.multi_gold == "all":
            usable += [(q, p, gs) for p in sorted(g)]

    cross = mine_cross_table(model_for_mining(args.base), C, usable, qpre,
                             n_cross, args.mine_topk, args.mine_batch)

    rows = []
    for n, (q, g, gs) in enumerate(usable):
        # 같은 표의 다른 셀 -- gold가 2~3위일 때 밀어내는 것들. 그 질문의 gold는 전부 뺀다.
        sibs = [p for p in by_table[C.cell_owner[g][0]] if p not in gs]
        # a query whose top-k holds fewer than n_cross other-table cells gets the
        # shortfall back as siblings rather than being dropped: the row keeps its
        # negative count, only the mix moves. Dropping cost 367 of 3,667 queries.
        want_sib = args.neg_per_query - len(cross[n])
        negs = random.sample(sibs, min(want_sib, len(sibs))) + cross[n]
        row = {"anchor": qpre + q["question"], "positive": C.cell_text[g]}
        for i, p in enumerate(negs):
            row[f"negative_{i+1}"] = C.cell_text[p]
        if len(negs) == args.neg_per_query:      # 열 수가 고른 행만 (Dataset 요구)
            rows.append(row)
    print(f"[data] {len(rows)} triplets from {len(C.queries)} queries "
          f"({len(usable)} usable, {len(by_table)} tables); dropped "
          f"{len(usable) - len(rows)} for too few negatives", flush=True)
    if args.dry_run:
        return 0

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
               "neg_cross": n_cross, "mine_topk": args.mine_topk,
               "multi_gold": args.multi_gold,
               "negatives": (f"{n_sib} same-table siblings + {n_cross} mined "
                             f"other-table cells (dense-only, top-{args.mine_topk} "
                             f"of the base index)" if n_cross else
                             "same-table siblings"),
               "leakage": "train/dev/test share zero tables and zero queries"},
              open(Path(args.out) / "train_meta.json", "w"), indent=1)
    print(f"[saved] {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
