#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""빈칸 없는 표만 모은 코퍼스에서, 1차 top-1 셀 문장을 쿼리에 붙여 재검색한다.

두 가지를 한 번에 본다.

1. **표 구조가 원인인가.** dev 표 424 개 중 헤더·데이터에 빈칸이 하나도 없는 표는
   16 개뿐이다 (test 는 따로 센다). 그 표들만 남긴 코퍼스에서 검색이 얼마나 달라지는가.
2. **쿼리 확장이 통하는가.** 1차 검색 top-1 셀 문장을 질문 뒤에 붙여 다시 검색한다
   (pseudo-relevance feedback). 1 등이 맞으면 강화되고 틀리면 더 끌려갈 수 있다.

⚠️ **사전등록 없는 탐색이다.** 예측을 걸지 않았고 이 결과로 arm 을 고르지 않는다.
⚠️ **인코더가 `p0` 가 아니다.** `models/bge-base-cell-ft-p0` 는 이 머신에 없다
   (`.gitignore`). 기성품 `BAAI/bge-base-en-v1.5` 로 돌리므로 절대값을 저장소의
   p0 수치(dev R@1 .7193)와 **짝지어 인용하면 안 된다**. arm 사이의 상대 비교만 유효하다.
⚠️ **alpha 를 재조정하지 않았다.** p0 확정값 0.8 을 그대로 썼다.
⚠️ **EM 은 재지 않는다.** 리더가 GPU 를 요구하고 이 머신에는 없다. 검색 지표만 낸다.

  python3 analysis/clean_table_expand.py --split dev
  python3 analysis/clean_table_expand.py --split test
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
from intable_failures_xlsx import layout, md_parts                    # noqa: E402
from cell_rank_dump import cell_texts                                 # noqa: E402
from rag_agent.retrieve.encoders import default_encoder               # noqa: E402
from rag_agent.retrieve.hybrid_index import HybridIndex, _minmax      # noqa: E402
from rag_agent.serialization.base import Chunk                        # noqa: E402


def is_full(C, t):
    """헤더에도 데이터에도 빈칸이 하나도 없는 표인가.

    병합 셀은 md 격자에서 빈칸으로 보인다 -- 경로는 헤더 트리가 채우지만 여기서는
    **격자에 글자가 있는가**를 묻는다. 그것이 '빈 공간 없는 표'의 정의다.
    """
    lines, sep, left = layout(C, t)
    nr, nc = C.shape[t]
    for i in range(sep):
        if any(not x.strip() for x in md_parts(lines[i])):
            return False
    for r in range(nr):
        if sep + 1 + r >= len(lines):
            return False
        row = md_parts(lines[sep + 1 + r])
        for c in range(nc):
            if left + c >= len(row) or not str(row[left + c]).strip():
                return False
    return True


def run(C, tids, pop, enc, alpha, expand):
    """tids 로 좁힌 코퍼스에서 pop 을 검색한다. expand 면 2 패스."""
    keep = [n for n, (t, _i, _j) in enumerate(C.cell_owner) if t in tids]
    txt = cell_texts(C, "S3c", "page")
    chunks = [Chunk(table_id=C.cell_owner[n][0], chunk_id=f"c::{n}",
                    text=txt[n], scheme="S3c", kind="cell") for n in keep]
    ix = HybridIndex(chunks, encoder=enc, alpha=0.5)
    at = {c: k for k, c in enumerate(keep)}          # 전역 셀 -> 코퍼스 위치

    def score(q):
        bm = ix._bm25_scores(q)
        if alpha == 0.0:
            return bm
        dn = ix._dense_scores(q)
        return dn if alpha == 1.0 else alpha * _minmax(dn) + (1 - alpha) * _minmax(bm)

    pos = {c: n for n, c in enumerate(C.cell_owner)}
    out = []
    for q in pop:
        gold = [at[pos[c]] for c in q["gold_cells"] if c in pos and pos[c] in at]
        if not gold:
            continue
        s = score(q["question"])
        if expand:
            top = int(np.argmax(s))
            s = score(q["question"] + " " + chunks[top].text)
        order = np.argsort(-s)
        rank = {int(v): r for r, v in enumerate(order)}
        rs = sorted(rank[g] for g in gold)
        out.append({"query_id": q["query_id"], "m": len(gold), "ranks": rs,
                    "r@1": int(rs[0] == 0), "setEM@10": int(rs[-1] < 10),
                    "top1_text": chunks[int(order[0])].text})
    return out


def mcnemar(a, b):
    """페어드 이분 비교. (a만 맞음, b만 맞음, 양측 p)."""
    from math import comb
    x = sum(1 for u, v in zip(a, b) if u and not v)
    y = sum(1 for u, v in zip(a, b) if v and not u)
    n = x + y
    if n == 0:
        return x, y, 1.0
    k = min(x, y)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return x, y, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--embed-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--cache-dir", default="results/embed_cache")
    ap.add_argument("--out-dir", default="results/cleantable")
    a = ap.parse_args()
    a.dataset, a.cell_scheme, a.data_dir = "hitab", "S3c", "data/hitab"
    a.population = f"hitab_{a.split}_lookup_all"
    a.title_mode, a.seed, a.max_queries, a.mh_queries = "page", 42, 0, 400
    a.rhb_question_types, a.rhb_em_only = [], False

    t0 = time.time()
    C = load_corpus(a)
    full = {t for t in C.tids if is_full(C, t)}
    pop = [q for q in C.queries if q["gold_table"] in full]
    print(f"[{a.split}] 표 {len(C.tids)} 중 빈칸 0 인 표 {len(full)} · "
          f"거기 걸린 질의 {len(pop)} / {len(C.queries)} | {time.time()-t0:.0f}s",
          flush=True)
    if not pop:
        return 1

    enc = cdv._CachedEncoder(default_encoder(model_name=a.embed_model),
                             a.cache_dir, f"hitab_{a.split}_{a.embed_model}")

    arms = {}
    for corp, tids in (("clean", full), ("full", set(C.tids))):
        for mode in ("plain", "expand"):
            t = time.time()
            arms[f"{corp}/{mode}"] = run(C, tids, pop, enc, a.alpha,
                                         mode == "expand")
            print(f"  {corp}/{mode:6} {time.time()-t:.0f}s", flush=True)

    rows, keys = [], list(arms)
    for k in keys:
        v = arms[k]
        rows.append({"arm": k, "n": len(v),
                     "R@1": round(sum(r["r@1"] for r in v) / len(v), 4),
                     "setEM@10": round(sum(r["setEM@10"] for r in v) / len(v), 4)})
    tests = {}
    for corp in ("clean", "full"):
        p, e = arms[f"{corp}/plain"], arms[f"{corp}/expand"]
        for met in ("r@1", "setEM@10"):
            x, y, pv = mcnemar([r[met] for r in e], [r[met] for r in p])
            tests[f"{corp} {met} expand vs plain"] = {
                "확장만 맞음": x, "평소만 맞음": y, "p": round(pv, 4)}
    for met in ("r@1", "setEM@10"):
        x, y, pv = mcnemar([r[met] for r in arms["clean/plain"]],
                           [r[met] for r in arms["full/plain"]])
        tests[f"{met} clean vs full (plain)"] = {
            "clean만 맞음": x, "full만 맞음": y, "p": round(pv, 4)}

    out = {
        "무엇": "빈칸 없는 표 코퍼스 + 쿼리 확장(1차 top-1 셀 문장 이어붙이기)",
        "주의": [
            "사전등록 없는 탐색. 이 결과로 arm 을 고르지 않았다.",
            f"인코더는 기성품 {a.embed_model} 이다. models/bge-base-cell-ft-p0 는 "
            "이 머신에 없다. 저장소의 p0 수치와 절대값을 짝지어 인용하지 말 것.",
            f"alpha={a.alpha} 를 재조정하지 않았다 (p0 확정값을 그대로 썼다).",
            "EM 은 재지 않았다 — 리더가 GPU 를 요구하고 이 머신에는 없다.",
            "clean 코퍼스는 셀 수가 훨씬 적어 방해 셀이 적다. clean 대 full 비교는 "
            "그만큼 유리하게 나온다 — 구조 효과가 아니다.",
        ],
        "설정": {"split": a.split, "alpha": a.alpha, "encoder": a.embed_model,
                 "cell_scheme": "S3c", "title_mode": "page",
                 "표_전체": len(C.tids), "표_빈칸0": len(full),
                 "질의_전체": len(C.queries), "질의_모집단": len(pop),
                 "셀_전체": len(C.cell_owner),
                 "셀_clean": sum(1 for t, _i, _j in C.cell_owner if t in full)},
        "빈칸0_표": sorted(full),
        "보드": rows,
        "페어드검정": tests,
        "records": {k: [{kk: vv for kk, vv in r.items() if kk != "top1_text"}
                        for r in v] for k, v in arms.items()},
    }
    d = Path(a.out_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"clean_expand_{a.split}.json"
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[출력] {p}")
    for r in rows:
        print(f"  {r['arm']:14} n={r['n']:3}  R@1={r['R@1']:.4f}  "
              f"setEM@10={r['setEM@10']:.4f}")
    for k, v in tests.items():
        print(f"  {k:36} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
