#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""셀 recall 과 표 recall 을 같은 런에서 나란히 재고, 계층 깊이로 가른다.

`results/cleantable/clean_expand_dev.json` 에서 빈칸 없는 표의 질의가 오히려 두 배
어렵다는 것이 나왔다 (R@1 .2812 대 .574). 빈칸이 없다는 것은 병합 헤더가 없다는 뜻이고
곧 계층이 없다는 뜻이다 -- 빈칸 0 인 표는 열 경로 깊이가 정확히 1.00 이다.

여기서 묻는 것: **그 손해가 표를 못 찾아서인가, 표 안에서 칸을 못 골라서인가.**
표 recall 은 비슷한데 셀 recall 만 떨어진다면 후자이고, 그것은 "계층 경로가 셀을
구별해 준다" 는 뜻이 된다.

  셀 recall@k = gold 셀이 상위 k 셀 안에 있는가
  표 recall@k = gold 표가 상위 k 셀이 속한 표들 안에 있는가

⚠️ 사전등록 없는 탐색. 인코더는 기성품 `BAAI/bge-base-en-v1.5` 이고 `p0` 가 아니다
   (이 머신에 모델 파일이 없다). 저장소의 p0 수치와 절대값을 짝지어 인용하지 말 것.

  python3 analysis/cell_vs_table_recall.py --split dev
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "analysis")]

import numpy as np                                                    # noqa: E402
import corpus_dump_vs_cell as cdv                                     # noqa: E402
from header_path_coverage import load_corpus                          # noqa: E402
from clean_table_expand import is_full                                # noqa: E402
from cell_rank_dump import cell_texts                                 # noqa: E402
from rag_agent.retrieve.encoders import default_encoder               # noqa: E402
from rag_agent.retrieve.hybrid_index import HybridIndex, _minmax      # noqa: E402
from rag_agent.serialization.base import Chunk                        # noqa: E402

KS = (1, 5, 10, 50)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--embed-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--cache-dir", default="results/embed_cache")
    ap.add_argument("--out-dir", default="results/cleantable")
    a = ap.parse_args()
    a.dataset, a.cell_scheme, a.data_dir = "hitab", "S3c", "data/hitab"
    a.population = f"hitab_{a.split}_lookup_all"
    a.title_mode, a.seed, a.max_queries, a.mh_queries = "page", 42, 0, 400
    a.rhb_question_types, a.rhb_em_only = [], False

    C = load_corpus(a)
    full = {t for t in C.tids if is_full(C, t)}
    txt = cell_texts(C, "S3c", "page")
    owner = np.array([C.tids.index(t) for t, _i, _j in C.cell_owner])
    chunks = [Chunk(table_id=t, chunk_id=f"c::{n}", text=txt[n],
                    scheme="S3c", kind="cell")
              for n, (t, _i, _j) in enumerate(C.cell_owner)]
    enc = cdv._CachedEncoder(default_encoder(model_name=a.embed_model),
                             a.cache_dir, f"hitab_{a.split}_{a.embed_model}")
    ix = HybridIndex(chunks, encoder=enc, alpha=0.5)
    pos = {c: n for n, c in enumerate(C.cell_owner)}
    tix = {t: k for k, t in enumerate(C.tids)}

    # 질의별 계층 깊이는 gold 표의 셀 열 경로 깊이 평균으로 잡는다
    depth = {}
    for t in C.tids:
        ds = [len(C.cell_paths[n][1]) for n, (tt, _i, _j)
              in enumerate(C.cell_owner) if tt == t]
        depth[t] = st.mean(ds) if ds else 0.0

    rows, t0 = [], time.time()
    for q in C.queries:
        gold = [pos[c] for c in q["gold_cells"] if c in pos]
        if not gold:
            continue
        bm = ix._bm25_scores(q["question"])
        dn = ix._dense_scores(q["question"])
        s = (bm if a.alpha == 0 else dn if a.alpha == 1
             else a.alpha * _minmax(dn) + (1 - a.alpha) * _minmax(bm))
        order = np.argsort(-s)
        gt = tix[q["gold_table"]]
        r = {"query_id": q["query_id"], "m": len(gold),
             "gold_table": q["gold_table"],
             "빈칸0표": q["gold_table"] in full,
             "열경로깊이": round(depth[q["gold_table"]], 2)}
        for k in KS:
            top = order[:k]
            r[f"셀recall@{k}"] = int(all(g in set(top.tolist()) for g in gold))
            r[f"표recall@{k}"] = int(gt in set(owner[top].tolist()))
        rows.append(r)
    print(f"[{a.split}] {len(rows)}건 | {time.time()-t0:.0f}s", flush=True)

    def agg(sub):
        d = {"n": len(sub)}
        for k in KS:
            d[f"셀@{k}"] = round(sum(r[f"셀recall@{k}"] for r in sub) / len(sub), 4)
        for k in KS:
            d[f"표@{k}"] = round(sum(r[f"표recall@{k}"] for r in sub) / len(sub), 4)
        return d

    groups = {
        "전체": rows,
        "빈칸0 표 (평면)": [r for r in rows if r["빈칸0표"]],
        "나머지 표": [r for r in rows if not r["빈칸0표"]],
        "열경로 깊이 = 1.0": [r for r in rows if r["열경로깊이"] <= 1.0],
        "열경로 깊이 1.0~2.0": [r for r in rows if 1.0 < r["열경로깊이"] <= 2.0],
        "열경로 깊이 > 2.0": [r for r in rows if r["열경로깊이"] > 2.0],
    }
    board = {k: agg(v) for k, v in groups.items() if v}
    out = {
        "무엇": "셀 recall 과 표 recall 을 같은 런에서 나란히, 계층 깊이로 가름",
        "주의": ["사전등록 없는 탐색. arm 을 고르지 않았다.",
                 f"인코더는 기성품 {a.embed_model}. p0 가 아니다 — "
                 "저장소 p0 수치와 절대값을 짝지어 인용하지 말 것.",
                 "셀 recall@k 는 gold 셀이 **전부** 상위 k 안인가 (m=1 이면 통상 R@k).",
                 "표 recall@k 는 gold 표가 상위 k 셀이 속한 표들 안인가."],
        "설정": {"split": a.split, "alpha": a.alpha, "encoder": a.embed_model,
                 "cell_scheme": "S3c", "title_mode": "page",
                 "셀": len(chunks), "표": len(C.tids), "질의": len(rows)},
        "보드": board, "records": rows,
    }
    d = Path(a.out_dir); d.mkdir(parents=True, exist_ok=True)
    p = d / f"cell_vs_table_recall_{a.split}.json"
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[출력] {p}\n")
    hdr = f"{'그룹':22}{'n':>5}" + "".join(f"{'셀@'+str(k):>9}" for k in KS) \
        + "".join(f"{'표@'+str(k):>9}" for k in KS)
    print(hdr); print("-" * len(hdr))
    for k, v in board.items():
        print(f"{k:22}{v['n']:>5}" + "".join(f"{v['셀@'+str(x)]:>9.4f}" for x in KS)
              + "".join(f"{v['표@'+str(x)]:>9.4f}" for x in KS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
