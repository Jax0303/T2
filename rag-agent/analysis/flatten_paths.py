#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""헤더 경로를 평탄화하면 검색이 달라지는가 — 같은 질의, 같은 gold, 직렬화만 바꿈.

`results/cleantable/cell_vs_table_recall_dev.json` 에서 열 경로 깊이에 따라 표 recall
이 크게 갈렸다 (@1: 깊이1.0 .7606 / 1~2 .9557 / >2 .8645). 다만 그것은 **서로 다른 표**
끼리의 비교라 표 크기·주제와 교란돼 있다. 여기서는 **같은 표를 다르게 직렬화**해서
그 교란을 없앤다.

  A 원본   `doncaster rovers > 1976-77` // `total > goals`
  B 평탄화 `doncaster rovers 1976-77`   // `total goals`      -- 구분자만 제거
  C 잎만   `1976-77`                    // `goals`            -- 조상 라벨 제거

B 는 단어가 A 와 **완전히 같다**. BM25 는 ` > ` 를 토큰으로 보지 않으므로 B-A 차이는
사실상 dense 쪽 문자열 차이뿐이다 -- 거의 null 을 예상한다. C 는 진짜 정보 손실이다.

  B ≈ A, C 만 하락  -> 중요한 것은 헤더의 **단어**이지 계층 구조가 아니다
  B 도 하락         -> 구조 표시 자체가 신호다

⚠️ 사전등록 없는 탐색. 인코더는 기성품 `BAAI/bge-base-en-v1.5` (p0 는 이 머신에 없다).

  python3 analysis/flatten_paths.py --split dev
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "analysis")]

import numpy as np                                                    # noqa: E402
import corpus_dump_vs_cell as cdv                                     # noqa: E402
from header_path_coverage import load_corpus                          # noqa: E402
from cell_rank_dump import PAGE_TITLES, TEMPLATE                      # noqa: E402
from rag_agent.retrieve.encoders import default_encoder               # noqa: E402
from rag_agent.retrieve.hybrid_index import HybridIndex, _minmax      # noqa: E402
from rag_agent.serialization.base import Chunk                        # noqa: E402
from rag_agent.serialization.caption import (caption_sentence,        # noqa: E402
                                             effective_titles)

KS = (1, 5, 10, 50)


def variant(rp, cp, how):
    """경로를 arm 에 맞게 바꾼다. 값과 제목은 건드리지 않는다."""
    rp = [str(s) for s in rp if str(s).strip()]
    cp = [str(s) for s in cp if str(s).strip()]
    if how == "orig":
        return rp, cp
    if how == "flat":                       # 여러 겹을 한 겹으로 뭉갠다
        return ([" ".join(rp)] if rp else []), ([" ".join(cp)] if cp else [])
    if how == "leaf":                       # 조상 버리고 잎만
        return rp[-1:], cp[-1:]
    raise ValueError(how)


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
    pt = json.load(open(PAGE_TITLES))
    ti = effective_titles(C.tids, C.title, C.cell_owner, C.cell_paths,
                          "page", page_titles=pt)
    pos = {c: n for n, c in enumerate(C.cell_owner)}
    tix = {t: k for k, t in enumerate(C.tids)}
    owner = np.array([tix[t] for t, _i, _j in C.cell_owner])
    base = cdv._CachedEncoder(default_encoder(model_name=a.embed_model),
                              a.cache_dir, f"hitab_{a.split}_{a.embed_model}")

    arms, samples = {}, {}
    for how in ("orig", "flat", "leaf"):
        txt = []
        for n, (t, _i, _j) in enumerate(C.cell_owner):
            rp, cp, v = C.cell_paths[n]
            r2, c2 = variant(rp, cp, how)
            txt.append(caption_sentence(ti[t], r2, c2, v,
                                        template=TEMPLATE["S3c"]))
        samples[how] = txt[pos[C.cell_owner[0]]] if C.cell_owner else ""
        samples[how] = txt[0]
        enc = cdv._CachedEncoder(default_encoder(model_name=a.embed_model),
                                 a.cache_dir,
                                 f"hitab_{a.split}_{a.embed_model}_{how}")
        ix = HybridIndex([Chunk(table_id=t, chunk_id=f"c::{n}", text=txt[n],
                                scheme="S3c", kind="cell")
                          for n, (t, _i, _j) in enumerate(C.cell_owner)],
                         encoder=enc, alpha=0.5)
        t0, rows = time.time(), []
        for q in C.queries:
            gold = [pos[c] for c in q["gold_cells"] if c in pos]
            if not gold:
                continue
            bm = ix._bm25_scores(q["question"])
            dn = ix._dense_scores(q["question"])
            s = a.alpha * _minmax(dn) + (1 - a.alpha) * _minmax(bm)
            order = np.argsort(-s)
            gt = tix[q["gold_table"]]
            r = {"query_id": q["query_id"]}
            for k in KS:
                top = set(order[:k].tolist())
                r[f"셀@{k}"] = int(all(g in top for g in gold))
                r[f"표@{k}"] = int(gt in set(owner[order[:k]].tolist()))
            rows.append(r)
        arms[how] = rows
        print(f"  {how:5} {time.time()-t0:.0f}s", flush=True)

    def mcnemar(x, y):
        from math import comb
        u = sum(1 for p, q in zip(x, y) if p and not q)
        v = sum(1 for p, q in zip(x, y) if q and not p)
        n = u + v
        if n == 0:
            return u, v, 1.0
        return u, v, min(1.0, 2 * sum(comb(n, i)
                                      for i in range(min(u, v) + 1)) / 2 ** n)

    board = {h: {"n": len(v), **{k: round(sum(r[k] for r in v) / len(v), 4)
                                 for k in v[0] if k != "query_id"}}
             for h, v in arms.items()}
    tests = {}
    for how in ("flat", "leaf"):
        for k in (f"셀@1", f"셀@10", f"표@1"):
            u, w, p = mcnemar([r[k] for r in arms[how]],
                              [r[k] for r in arms["orig"]])
            tests[f"{how} vs orig · {k}"] = {f"{how}만 맞음": u,
                                             "orig만 맞음": w, "p": round(p, 5)}
    out = {"무엇": "헤더 경로 직렬화 3형태 대조 (원본 / 평탄화 / 잎만)",
           "주의": ["사전등록 없는 탐색. arm 을 고르지 않았다.",
                    f"인코더는 기성품 {a.embed_model} 이고 p0 가 아니다.",
                    "flat 은 orig 과 단어가 완전히 같다 (구분자 ' > ' 만 제거). "
                    "BM25 는 그 차이를 못 보므로 dense 쪽 문자열 차이만 남는다."],
           "설정": {"split": a.split, "alpha": a.alpha, "encoder": a.embed_model,
                    "질의": len(arms["orig"]), "셀": len(C.cell_owner)},
           "문장예시": samples, "보드": board, "페어드검정": tests,
           "records": arms}
    d = Path(a.out_dir); d.mkdir(parents=True, exist_ok=True)
    p = d / f"flatten_paths_{a.split}.json"
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[출력] {p}\n")
    for h, s in samples.items():
        print(f"  [{h}] {s[:130]}")
    cols = [k for k in board["orig"] if k != "n"]
    print("\n" + f"{'arm':7}{'n':>5}" + "".join(f"{c:>9}" for c in cols))
    for h, v in board.items():
        print(f"{h:7}{v['n']:>5}" + "".join(f"{v[c]:>9.4f}" for c in cols))
    print()
    for k, v in tests.items():
        print(f"  {k:26} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
