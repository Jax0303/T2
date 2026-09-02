#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""표 내부 실패에서, 질문이 정답 칸을 가릴 단서를 실제로 담고 있는가.

`--dump-above` 덤프를 읽어 **1등이 정답 표 안**인 실패만 남기고, 정답 칸과 그
1등 칸의 헤더 경로 **차집합**을 낸다. 그 차집합 단어가 질문에 있으면 신호가
존재하는 것이고(=검색기가 못 쓴 것), 없으면 어떤 검색기도 못 가른다.

`--tau T` 를 주면 문자열 일치 대신 **의미 일치**로 센다: 차집합 단어와 질문
단어의 코사인이 T 이상이면 "질문에 있다"로 친다 (female/women 류). 판정기는
검색에 쓴 인코더가 아니라 **기성품 bge-base**다 -- 실패한 모델에게 "신호가
있었냐"를 묻지 않기 위해서다.

**τ=1.0 은 문자열 일치와 같아야 한다** (산술 통제). 다르면 버그다.

  python analysis/col_signal.py results/sibling/<...>_above.jsonl [--tau 0.85]
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter

STOP = set("""a an the of in on at to for and or is are was were be by with from
as that this these those it its total all other than per each any no not""".split())


def toks(s):
    return {w for w in re.split(r"[^a-z0-9]+", str(s).lower()) if w and w not in STOP}


def path_toks(path):
    out = set()
    for seg in path or []:
        out |= toks(seg)
    return out


def semantic_hit(need, have, tau, sim):
    """need 중 하나라도 have 안의 단어와 코사인 >= tau 면 True."""
    return any(sim(a, b) >= tau for a in need for b in have)


def main(path, tau=None, out=None):
    recs = [json.loads(l) for l in open(path)]
    sim = _exact_sim
    if tau is not None and tau < 1.0:
        sim = _build_sim(recs)
    rows, cnt = [], Counter()
    for r in recs:
        top = r.get("top_above") or []
        if not top or top[0]["table_id"] != r["gold_table"]:
            continue                      # 1등이 딴 표면 이 질문의 축이 아니다
        w, q = top[0], toks(r["question"])
        gold = path_toks(r["gold_row_path"]) | path_toks(r["gold_col_path"])
        win = path_toks(w["row_path"]) | path_toks(w["col_path"])
        g_only, w_only = gold - win, win - gold
        if tau is None:
            hit_g, hit_w = bool(g_only & q), bool(w_only & q)
        else:
            hit_g = semantic_hit(g_only, q, tau, sim)
            hit_w = semantic_hit(w_only, q, tau, sim)
        if not g_only and not w_only:
            k = "경로가 완전히 동일 (값만 다름)"
        elif hit_g and not hit_w:
            k = "질문이 정답 쪽만 지목"
        elif hit_g and hit_w:
            k = "질문이 양쪽 다 지목"
        elif not hit_g and hit_w:
            k = "질문이 오답 쪽만 지목"
        else:
            k = "질문에 구별 단서 없음"
        cnt[k] += 1
        rows.append(dict(query_id=r["query_id"], kind=k, question=r["question"],
                         gold_only=sorted(g_only), win_only=sorted(w_only),
                         relation=w["relation"]))
    n = len(rows)
    print(f"1등이 정답 표 안인 실패: {n}건\n")
    for k, v in cnt.most_common():
        print(f"  {v:>4}  ({v/n:.3f})  {k}")
    sig = cnt["질문이 정답 쪽만 지목"]
    print(f"\n신호가 확실히 있는데 진 것: {sig}/{n} = {sig/n:.3f}")
    out = out or "results/sibling/col_signal.json"
    json.dump(dict(source=path, tau=tau, n=n, counts=dict(cnt), rows=rows),
              open(out, "w"), ensure_ascii=False, indent=1)
    print(f"-> {out}")
    return 0


def _exact_sim(a, b):
    return 1.0 if a == b else 0.0


def _build_sim(recs):
    """기성품 bge-base 로 단어 임베딩을 만들고 코사인 룩업을 돌려준다."""
    import numpy as np
    vocab = set()
    for r in recs:
        vocab |= toks(r["question"])
        vocab |= path_toks(r["gold_row_path"]) | path_toks(r["gold_col_path"])
        for w in (r.get("top_above") or []):
            vocab |= path_toks(w["row_path"]) | path_toks(w["col_path"])
    vocab = sorted(vocab)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("BAAI/bge-base-en-v1.5")
    E = m.encode(vocab, batch_size=256, normalize_embeddings=True,
                 show_progress_bar=False)
    idx = {w: i for i, w in enumerate(vocab)}
    print(f"[sim] {len(vocab)}개 단어 임베딩", flush=True)

    def sim(a, b):
        return float(E[idx[a]] @ E[idx[b]]) if a in idx and b in idx else 0.0
    return sim


if __name__ == "__main__":
    ap = sys.argv[1:]
    src = ap[0]
    tau = float(ap[ap.index("--tau") + 1]) if "--tau" in ap else None
    out = ap[ap.index("--out") + 1] if "--out" in ap else None
    sys.exit(main(src, tau, out))
