#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""꽉 찬 표(모든 칸에 값이 있는 표) 100개 위에서 우리 방법을 그대로 잰다.

꽉 찬 표 = 헤더에도 데이터에도 빈칸이 하나도 없는 표. 병합 셀이 없다는 뜻이고,
열 경로 깊이가 대개 1 인 평면 표다. dev 에는 16 개뿐이라 train/dev/test 를 전부
뒤져 100 개 이상을 모은다.

**인코더가 기성품 `BAAI/bge-base-en-v1.5` 라 train 표를 써도 학습 누출이 없다**
(`models/bge-base-cell-ft-p0` 는 이 머신에 없다). 대신 절대값을 저장소의 p0 수치와
짝지어 인용하면 안 된다.

방법은 파이프라인 그대로다 -- S3c 셀 문장 + ToTTo 페이지 제목 + BM25/dense 하이브리드.
바뀌는 것은 코퍼스가 꽉 찬 표만이라는 것뿐이다.

⚠️ EM 은 못 잰다. 리더(Qwen2.5-7B 4-bit)가 GPU 를 요구하고 이 머신에는 없다.
   검색 지표만 낸다. EM 은 집 머신에서 같은 모집단으로 돌려야 한다.

  python3 analysis/full_tables_eval.py
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


def collect(split, want_full):
    a = argparse.Namespace(
        dataset="hitab", cell_scheme="S3c", data_dir="data/hitab",
        population=f"hitab_{split}_lookup_all", split=split, title_mode="page",
        seed=42, max_queries=0, mh_queries=400,
        rhb_question_types=[], rhb_em_only=False)
    C = load_corpus(a)
    full = {t for t in C.tids if is_full(C, t)}
    keep = full if want_full else set(C.tids)
    txt = cell_texts(C, "S3c", "page")
    pos = {c: n for n, c in enumerate(C.cell_owner)}
    cells, gcell = [], {}
    for n, (t, i, j) in enumerate(C.cell_owner):
        if t in keep:
            gcell[n] = len(cells)
            cells.append((t, txt[n]))
    qs = []
    for q in C.queries:
        if q["gold_table"] not in full:
            continue
        g = [gcell[pos[c]] for c in q["gold_cells"] if c in pos and pos[c] in gcell]
        if g:
            qs.append({"query_id": q["query_id"], "split": split,
                       "question": q["question"], "gold_table": q["gold_table"],
                       "gold": g})
    depth = {t: st.mean([len(C.cell_paths[n][1]) for n, (tt, _i, _j)
                         in enumerate(C.cell_owner) if tt == t] or [0])
             for t in full}
    size = {t: C.shape[t][0] * C.shape[t][1] for t in full}
    return cells, qs, full, depth, size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--embed-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--cache-dir", default="results/embed_cache")
    ap.add_argument("--out", default="results/cleantable/full_tables_eval.json")
    a = ap.parse_args()

    cells, qs, tabs, depth, size = [], [], {}, {}, {}
    for sp in ("train", "dev", "test"):
        t0 = time.time()
        c, q, f, d, s = collect(sp, True)
        off = len(cells)
        cells += c
        for x in q:
            x["gold"] = [g + off for g in x["gold"]]
        qs += q
        tabs[sp] = sorted(f); depth.update(d); size.update(s)
        print(f"[{sp}] 꽉 찬 표 {len(f)}개 · 셀 {len(c)} · 질의 {len(q)} "
              f"| {time.time()-t0:.0f}s", flush=True)

    ntab = sum(len(v) for v in tabs.values())
    print(f"\n합계: 표 {ntab}개 · 셀 {len(cells)} · 질의 {len(qs)}", flush=True)

    tids = sorted({t for t, _x in cells})
    tix = {t: k for k, t in enumerate(tids)}
    owner = np.array([tix[t] for t, _x in cells])
    enc = cdv._CachedEncoder(default_encoder(model_name=a.embed_model),
                             a.cache_dir, f"hitab_fulltables_{a.embed_model}")
    ix = HybridIndex([Chunk(table_id=t, chunk_id=f"c::{n}", text=x,
                            scheme="S3c", kind="cell")
                      for n, (t, x) in enumerate(cells)], encoder=enc, alpha=0.5)

    rows, t0 = [], time.time()
    for q in qs:
        bm = ix._bm25_scores(q["question"])
        dn = ix._dense_scores(q["question"])
        s = a.alpha * _minmax(dn) + (1 - a.alpha) * _minmax(bm)
        order = np.argsort(-s)
        gt = tix[q["gold_table"]]
        rank = {int(v): r for r, v in enumerate(order)}
        rs = sorted(rank[g] for g in q["gold"])
        r = {"query_id": q["query_id"], "split": q["split"], "m": len(q["gold"]),
             "gold_rank": rs[-1] + 1, "표크기": size[q["gold_table"]],
             "열경로깊이": round(depth[q["gold_table"]], 2)}
        for k in KS:
            top = set(order[:k].tolist())
            r[f"셀@{k}"] = int(all(g in top for g in q["gold"]))
            r[f"표@{k}"] = int(gt in set(owner[order[:k]].tolist()))
        r["MRR"] = 1.0 / (rs[0] + 1)
        rows.append(r)
    print(f"[검색] {len(rows)}질의 | {time.time()-t0:.0f}s", flush=True)

    def agg(sub):
        if not sub:
            return None
        d = {"n": len(sub)}
        for k in KS:
            d[f"셀@{k}"] = round(sum(r[f"셀@{k}"] for r in sub) / len(sub), 4)
        for k in KS:
            d[f"표@{k}"] = round(sum(r[f"표@{k}"] for r in sub) / len(sub), 4)
        d["MRR"] = round(sum(r["MRR"] for r in sub) / len(sub), 4)
        return d

    groups = {"전체": rows}
    for sp in ("train", "dev", "test"):
        groups[f"  {sp}"] = [r for r in rows if r["split"] == sp]
    med = st.median([r["표크기"] for r in rows])
    groups[f"표 작은 쪽 (셀<={med:.0f})"] = [r for r in rows if r["표크기"] <= med]
    groups[f"표 큰 쪽 (셀>{med:.0f})"] = [r for r in rows if r["표크기"] > med]

    out = {
        "무엇": "꽉 찬 표(모든 칸에 값)만으로 코퍼스를 만들고 파이프라인 그대로 검색",
        "꽉_찬_표란": "헤더·데이터 격자에 빈칸이 하나도 없는 표. 병합 셀이 없다.",
        "방법": {"색인 단위": "셀", "셀 문장": "S3c", "제목": "ToTTo page title",
                 "검색": f"BM25/dense 하이브리드 alpha={a.alpha}",
                 "인코더": a.embed_model},
        "주의": [
            "EM 은 재지 않았다 — 리더가 GPU 를 요구하고 이 머신에는 없다. 검색 지표만이다.",
            f"인코더가 기성품 {a.embed_model} 이라 train 표를 써도 학습 누출은 없다. "
            "대신 저장소의 p0 수치와 절대값을 짝지어 인용하지 말 것.",
            "코퍼스가 꽉 찬 표만이라 셀 수가 훨씬 적다. 방해 셀이 적으므로 "
            "전체 코퍼스 수치보다 유리하게 나온다 — 표 성질의 효과가 아니다.",
            "사전등록 없는 탐색. arm 을 고르지 않았다.",
        ],
        "규모": {"표": ntab, "셀": len(cells), "질의": len(rows),
                 "split별 표": {k: len(v) for k, v in tabs.items()},
                 "표 크기 중앙(셀)": med,
                 "열경로깊이 중앙": st.median([r["열경로깊이"] for r in rows])},
        "표목록": tabs,
        "보드": {k: agg(v) for k, v in groups.items() if v},
        "records": rows,
    }
    p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[출력] {p}\n")
    cols = [f"셀@{k}" for k in KS] + [f"표@{k}" for k in KS] + ["MRR"]
    hdr = f"{'그룹':24}{'n':>5}" + "".join(f"{c:>9}" for c in cols)
    print(hdr); print("-" * len(hdr))
    for k, v in out["보드"].items():
        print(f"{k:24}{v['n']:>5}" + "".join(f"{v[c]:>9.4f}" for c in cols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
