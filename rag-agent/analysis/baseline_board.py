#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""검색 방식 비교 — 같은 질의·같은 gold·같은 지표(all-covered@k)로 색인 단위와 검색기를 나란히.

  bm25_cell      S3c 셀 문장, BM25 만                       (어휘 검색)
  dense_cell     S3c 셀 문장, 기성품 bge-base dense 만        (일반 dense retriever)
  hybrid_cell    S3c 셀 문장, 기성품 bge-base hybrid α        (기성품 arm)
  flat_cell      잎 헤더만 붙인 셀 문장, 기성품 hybrid         (헤더 경로 없는 셀 = 구조 무시 ablation)
  row            행 청크(행 경로 + 열 잎 라벨:값), 기성품 hybrid (LangChain/LlamaIndex 류 행 청킹 관행)
  table          표 청크(제목 + 헤더 라벨), 기성품 hybrid       (표 단위 검색, TARGET 류)
  ours           S3c 셀 문장, 파인튜닝 인코더 hybrid α          (본 방법)

all-covered@k = 상위 k 청크가 담는 셀의 합집합이 gold 셀을 전부 덮으면 1. 행·표 청크는 셀 여러 개를
한 번에 넘기므로 청크당 평균 셀 수를 같이 적는다 (같은 k 가 같은 양이 아니다).

  PYTHONPATH=.:scripts:analysis .venv/bin/python analysis/baseline_board.py --split dev \
      --population hitab_dev_lookup_all --gold-file results/audit2/hitab_dev_lookup_all_gold.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "analysis"))

import numpy as np                                                    # noqa: E402
import corpus_dump_vs_cell as cdv                                     # noqa: E402
from cell_rank_dump import cell_texts                                 # noqa: E402
from header_path_coverage import load_corpus                          # noqa: E402
from rag_agent.retrieve.encoders import default_encoder               # noqa: E402
from rag_agent.retrieve.hybrid_index import HybridIndex, _minmax      # noqa: E402
from rag_agent.serialization.base import Chunk                        # noqa: E402

KS = (1, 3, 5, 10, 20, 50)


def scores(ix, question, alpha):
    bm = ix._bm25_scores(question)
    if alpha == 0.0:
        return bm
    dn = ix._dense_scores(question)
    return dn if alpha == 1.0 else alpha * _minmax(dn) + (1 - alpha) * _minmax(bm)


def covered_at(order, covers, gold):
    """rank (0-based) at which the union of chunk cells first covers gold; 1e9 if never."""
    need = set(gold)
    for r, p in enumerate(order):
        need -= covers[p]
        if not need:
            return r
        if r > 5000:
            break
    return 10 ** 9


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--population", required=True)
    ap.add_argument("--gold-file", required=True)
    ap.add_argument("--stock", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--ours", default="models/bge-base-cell-ft-p0")
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--title-mode", default="page")
    ap.add_argument("--cache-dir", default=".cache/corpus_dump_vs_cell")
    ap.add_argument("--out-dir", default="results/baselines")
    a = ap.parse_args()
    a.dataset, a.data_dir, a.seed, a.rhb_question_types, a.rhb_em_only = "hitab", "data/hitab", 42, [], False
    C = load_corpus(a)
    G = json.load(open(a.gold_file))
    pop = [q | {"gold_cells": {tuple(x) for x in G[q["query_id"]]}} for q in C.queries if q["query_id"] in G]
    print(f"[corpus] {len(C.tids)} tables / {len(C.cell_owner)} cells | [pop] {len(pop)} audited queries", flush=True)

    cell_cov = [{o} for o in C.cell_owner]
    row_cells = defaultdict(set)
    for o in C.cell_owner:
        row_cells[(o[0], o[1])].add(o)
    row_cov = [row_cells[ro] for ro in C.row_owner]
    tab_cells = defaultdict(set)
    for o in C.cell_owner:
        tab_cells[o[0]].add(o)
    tab_cov = [tab_cells[t] for t in C.tids]

    def enc(model):
        return cdv._CachedEncoder(default_encoder(model_name=model), a.cache_dir, f"hitab_{a.split}_{model}")

    s3c = cell_texts(C, "S3c", a.title_mode)
    arms = [  # name, texts, covers, encoder model, alpha
        ("bm25_cell", s3c, cell_cov, None, 0.0),
        ("dense_cell", s3c, cell_cov, a.stock, 1.0),
        ("hybrid_cell", s3c, cell_cov, a.stock, a.alpha),
        ("flat_cell", C.flat_text, cell_cov, a.stock, a.alpha),
        ("row", C.row_text, row_cov, a.stock, a.alpha),
        ("table", [C.table_text[t] for t in C.tids], tab_cov, a.stock, a.alpha),
        ("ours", s3c, cell_cov, a.ours, a.alpha),
    ]
    out = {}
    for name, texts, covers, model, alpha in arms:
        t0 = time.time()
        chunks = [Chunk(table_id="", chunk_id=str(n), text=x, scheme=name, kind="x") for n, x in enumerate(texts)]
        ix = HybridIndex(chunks, encoder=(cdv._NoDense() if model is None else enc(model)), alpha=0.5)
        ranks = []
        for q in pop:
            order = np.argsort(-scores(ix, q["question"], alpha))
            ranks.append(covered_at(order, covers, q["gold_cells"]))
        n = len(ranks)
        out[name] = {"n": n, "chunks": len(texts),
                     "cells_per_chunk": round(sum(len(c) for c in covers) / len(covers), 2),
                     **{f"@{k}": round(sum(r < k for r in ranks) / n, 4) for k in KS},
                     "mrr": round(sum(1 / (r + 1) for r in ranks) / n, 4)}
        print(f"  {name:<12} " + " ".join(f"@{k}={out[name][f'@{k}']:.4f}" for k in KS)
              + f"  ({time.time() - t0:.0f}s)", flush=True)
        ix.close()
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    tag = f"{a.population}_clean"
    json.dump(out, open(od / f"{tag}.json", "w"), indent=1)
    L = [f"## `{tag}` (n={len(pop)}, split={a.split}, α={a.alpha}, 제목 {a.title_mode})", "",
         "| 방식 | 청크 수 | 셀/청크 | @1 | @3 | @5 | @10 | @20 | @50 | MRR |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in out.items():
        L.append(f"| `{name}` | {r['chunks']} | {r['cells_per_chunk']} | " + " | ".join(f"{r[f'@{k}']:.4f}" for k in KS) + f" | {r['mrr']:.4f} |")
    (od / f"{tag}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
