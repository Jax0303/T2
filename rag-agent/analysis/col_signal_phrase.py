#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""표 내부 실패에서, 질문이 정답 칸 쪽으로 기우는가 오답 칸 쪽으로 기우는가.

`col_signal.py` 의 단어 일치판은 동의어를 못 잡는다. 그렇다고 단어 코사인으로
바꿀 수도 없다 -- 보정해 보면 `2013`~`2014` (.795) 가 `female`~`women` (.738)
보다 높아서, 임계값이 동의어와 **구별자**를 못 가른다.

그래서 임계값을 버리고 **비교**만 한다: 정답 칸과 1등 칸의 헤더 경로 차집합을
각각 한 구절로 묶고, 질문과의 코사인을 견준다.

  sim(질문, 정답에만 있는 구절)  vs  sim(질문, 오답에만 있는 구절)

정답 쪽이 크면 질문 안에 신호가 있는데 검색기가 못 쓴 것이고, 오답 쪽이 크면
질문이 실제로 그 칸을 가리키고 있어 검색으로는 못 고친다.

판정기는 검색에 쓴 인코더가 아니라 **기성품 bge-base** 다. 문장 전체가 아니라
**차이나는 구절만** 보므로 검색기가 한 계산과 같지 않다 -- 검색기는 공유하는
90% 에 압도당하지만 여기서는 그 부분이 빠진다.

  python analysis/col_signal_phrase.py <...>_above.jsonl [--out PATH]
"""
from __future__ import annotations

import json
import sys


def segs(path):
    return [str(s) for s in (path or []) if str(s).strip()]


def main(argv):
    src = argv[0]
    out = argv[argv.index("--out") + 1] if "--out" in argv else \
        "results/colsig/col_signal_phrase.json"
    recs = [json.loads(l) for l in open(src)]

    cases = []
    for r in recs:
        top = r.get("top_above") or []
        if not top or top[0]["table_id"] != r["gold_table"]:
            continue
        w = top[0]
        g = set(segs(r["gold_row_path"]) + segs(r["gold_col_path"]))
        v = set(segs(w["row_path"]) + segs(w["col_path"]))
        g_only, w_only = sorted(g - v), sorted(v - g)
        cases.append((r, w, g_only, w_only))

    texts, keyed = [], []
    for r, w, g_only, w_only in cases:
        keyed.append((len(texts), len(texts) + 1, len(texts) + 2))
        texts += [r["question"], " ".join(g_only), " ".join(w_only)]

    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("BAAI/bge-base-en-v1.5")
    E = m.encode(texts, batch_size=128, normalize_embeddings=True,
                 show_progress_bar=False)

    rows, n_g, n_w, n_none = [], 0, 0, 0
    for (r, w, g_only, w_only), (qi, gi, wi) in zip(cases, keyed):
        if not g_only and not w_only:
            n_none += 1
            kind, margin = "경로가 완전히 동일 (값만 다름)", 0.0
        else:
            sg = float(E[qi] @ E[gi]) if g_only else -1.0
            sw = float(E[qi] @ E[wi]) if w_only else -1.0
            margin = sg - sw
            kind = "질문이 정답 쪽에 더 가깝다" if margin > 0 else \
                   "질문이 오답 쪽에 더 가깝다"
            n_g += margin > 0
            n_w += margin <= 0
        rows.append(dict(query_id=r["query_id"], kind=kind,
                         margin=round(margin, 4), question=r["question"],
                         gold_only=g_only, win_only=w_only,
                         relation=w["relation"]))

    n = len(rows)
    print(f"1등이 정답 표 안인 실패: {n}건\n")
    print(f"  {n_g:>4}  ({n_g/n:.3f})  질문이 정답 쪽에 더 가깝다  <- 신호 있음")
    print(f"  {n_w:>4}  ({n_w/n:.3f})  질문이 오답 쪽에 더 가깝다  <- 검색으로 불가")
    print(f"  {n_none:>4}  ({n_none/n:.3f})  경로가 완전히 동일")

    mg = sorted(abs(x["margin"]) for x in rows if x["margin"])
    for q in (0.25, 0.5, 0.75):
        print(f"  |margin| {int(q*100)}분위 = {mg[int(len(mg)*q)]:.3f}", end="")
    print(f"\n  |margin| < 0.02 인 애매한 건: "
          f"{sum(1 for x in mg if x < 0.02)}/{len(mg)}")

    json.dump(dict(source=src, n=n, gold_side=n_g, win_side=n_w,
                   identical=n_none, rows=rows),
              open(out, "w"), ensure_ascii=False, indent=1)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
