#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""왜 틀렸나 — 검색 실패, 문맥은 맞았는데 틀린 답변, 정답만 줘도 틀린 답변.

세 갈래를 한 파일에서 낸다. 셋을 같은 질의 집합 위에서 봐야
"검색 정확도 != 답변 정확도"의 몫이 어디로 가는지가 더해서 맞는다.

  A. 검색 실패        gold 셀이 상위 20에 못 든 질의. 표를 못 찾은 것인지,
                      표는 찾고 셀을 놓친 것인지 가른다 (후자는 표 우선 2단계로 잡힌다).
  B. 문맥은 맞았는데   정답 셀이 문맥에 있는데 리더가 틀린 질의. 예측값이 문맥의
     답이 틀린 것      *다른 셀* 값인지(distractor), 계산 실패인지, 없는 값인지 가른다.
  C. 정답만 줘도       gold 셀만 넣었는데 틀린 질의 — 검색으로는 절대 못 고치는 몫.
     틀린 것

  PYTHONPATH=.:scripts .venv/bin/python analysis/failure_anatomy.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from hashlib import md5
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                                                     # noqa: E402

D = ROOT / "results/retrieval_accuracy"
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def norm(v) -> str:
    """Compare two renderings of one number/label."""
    s = str(v).strip().lower().replace(",", "").replace("%", "").replace("$", "")
    try:
        f = float(s)
        return f"{f:g}"
    except ValueError:
        return " ".join(s.split())


def nums_in(text) -> set:
    return {norm(m.group(0)) for m in NUM.finditer(str(text))}


def load(name):
    f = D / name
    return [json.loads(l) for l in f.open()] if f.exists() else []


# --------------------------------------------------------------------------
def section_a(recs, args):
    """검색 실패: 표를 못 찾았나, 표 안에서 셀을 놓쳤나."""
    fails = [r for r in recs if r.get("mode") == "all" and r.get("correct") == 0]
    print(f"\n{'='*74}\nA. 검색 실패 {len(fails)}건 / 주지표 "
          f"{sum(1 for r in recs if r.get('mode') == 'all')}건\n{'='*74}")
    by_table = Counter(r["gold_table_in_context"] for r in fails)
    print(f"  gold 표가 문맥에 아예 없음 : {by_table[0]:3}건 "
          f"({by_table[0]/len(fails):.1%})  <- 표를 못 찾은 것")
    print(f"  gold 표는 문맥에 있음      : {by_table[1]:3}건 "
          f"({by_table[1]/len(fails):.1%})  <- 표는 찾고 셀을 놓친 것")
    print(f"\n  m(정답 셀 개수)별:", dict(sorted(Counter(r["m"] for r in fails).items())))
    agg = Counter((r.get("aggregation") or "none") for r in fails)
    tot = Counter((r.get("aggregation") or "none") for r in recs if r.get("mode") == "all")
    print("\n  질문 유형별 실패율 (실패/전체):")
    for k, v in agg.most_common(8):
        print(f"    {k:14} {v:3}/{tot[k]:4} = {v/tot[k]:.3f}")
    if args.no_rank:
        return fails
    # gold 셀의 실제 순위 — 코퍼스를 다시 세워 재계산
    from retrieval_accuracy import build_corpus, load_queries, PAGE_TITLES
    from rag_agent.retrieve.encoders import default_encoder
    from rag_agent.retrieve.hybrid_index import _minmax, _tokenize
    from rag_agent.retrieve.sparse_bm25 import SparseBM25
    pt = json.loads(PAGE_TITLES.read_text()); tabs = {}
    qs = {q["query_id"]: q for q in load_queries(args.data_dir, args.split, tabs)}
    tids = sorted({q["table_id"] for q in qs.values()})
    texts, covers, *_ = build_corpus(args.data_dir, tids, "s3c", "cell", pt)
    owner = [next(iter(c)) for c in covers]
    pos_of = {c: i for i, c in enumerate(owner)}
    by_tab = defaultdict(list)
    for i, (t, _r, _c) in enumerate(owner):
        by_tab[t].append(i)
    bm = SparseBM25(_tokenize(t) for t in texts)
    enc = default_encoder(model_name=args.embed_model)
    key = md5("\x00".join(texts).encode()).hexdigest()[:16]
    emb = np.load(ROOT / f".cache/retrieval_accuracy/"
                         f"{args.embed_model.replace('/', '_')}_{len(texts)}_{key}.npy")
    print(f"\n  [순위 재계산] {len(texts)} 단위", flush=True)
    band = Counter(); in_tab_band = Counter()
    for n, r in enumerate(fails):
        q = qs[r["query_id"]]
        b = bm.get_scores(_tokenize(q["question"]))
        d = emb @ enc.encode_query([q["question"]])[0].astype(np.float32)
        s = 0.7 * _minmax(d) + 0.3 * _minmax(b)
        order = np.argsort(-s)
        rank = np.empty(len(s), dtype=np.int64); rank[order] = np.arange(len(order))
        gp = [pos_of[g] for g in q["gold"] if g in pos_of]
        worst = int(max(rank[p] for p in gp))
        for lo, hi, lab in [(20,30,"21~30"),(30,50,"31~50"),(50,100,"51~100"),
                            (100,500,"101~500"),(500,10**9,"501~")]:
            if lo <= worst < hi:
                band[lab] += 1; break
        # 표를 맞혔다고 가정했을 때(오라클 표 게이팅) 표 안에서의 순위
        tgold = q["gold_table_id"] if "gold_table_id" in q else r["table_id"]
        idx = by_tab.get(tgold, [])
        if idx:
            inside = sorted(idx, key=lambda p: rank[p])
            at = {p: i for i, p in enumerate(inside)}
            w2 = max(at[p] for p in gp if p in at) if any(p in at for p in gp) else 10**9
            in_tab_band["<20" if w2 < 20 else ">=20"] += 1
        if (n + 1) % 50 == 0:
            print(f"    {n+1}/{len(fails)}", flush=True)
    print("\n  gold 셀의 실제 최악 순위:")
    for lab in ("21~30","31~50","51~100","101~500","501~"):
        print(f"    {lab:>8} : {band[lab]:3}건")
    print(f"\n  표를 맞혔다고 가정하면(오라클 표 게이팅) 그 표 안에서 상위 20 안에 드는가:")
    print(f"    든다   : {in_tab_band['<20']:3}건  <- 표 우선 2단계 검색으로 회수 가능한 몫")
    print(f"    안 든다: {in_tab_band['>=20']:3}건  <- 표를 맞혀도 못 잡는 몫")
    return fails


# --------------------------------------------------------------------------
def classify(pred, gold, ctx_text, m, agg):
    """예측이 왜 틀렸나."""
    p = norm(pred)
    g = {norm(x) for x in (gold if isinstance(gold, list) else [gold])}
    if not str(pred).strip():
        return "빈 응답"
    pnums = nums_in(pred)
    cnums = nums_in(ctx_text)
    if p in g or (pnums and pnums <= g):
        return "채점기 형식 차이"          # 값은 맞는데 EM이 안 붙음
    # 반올림/자릿수
    for gv in g:
        try:
            if abs(float(p) - float(gv)) <= 0.011 * max(abs(float(gv)), 1e-9):
                return "반올림·자릿수"
        except (ValueError, TypeError):
            pass
    if pnums and pnums <= cnums:
        return "문맥의 다른 셀 값을 읽음"    # distractor
    if (agg or "none") != "none" or m >= 2:
        return "계산 실패"
    if not pnums:
        return "숫자 아닌 응답"
    return "문맥에 없는 값(환각)"


def section_b(recs, ans, title, only_retrieval_ok):
    byid = {r["query_id"]: r for r in recs}
    sel = [a for a in ans if a["mode"] == "all" and not a["answer_correct"]
           and (a["retrieval_correct"] == 1 if only_retrieval_ok else True)]
    tot = [a for a in ans if a["mode"] == "all"
           and (a["retrieval_correct"] == 1 if only_retrieval_ok else True)]
    print(f"\n{'='*74}\n{title}\n{'='*74}")
    print(f"  대상 {len(tot)}건 중 오답 {len(sel)}건 (정답률 "
          f"{1 - len(sel)/len(tot):.4f})")
    cls = Counter(); ex = defaultdict(list)
    for a in sel:
        r = byid.get(a["query_id"], {})
        ctx = " ".join(r.get("context") or [])
        c = classify(a["pred"], a["answer"], ctx, r.get("m", 1), a.get("aggregation"))
        cls[c] += 1
        if len(ex[c]) < 3:
            ex[c].append((r.get("question", "")[:58], a["answer"], a["pred"][:38].replace("\n", " ")))
    for k, v in cls.most_common():
        print(f"\n  {k}  {v}건 ({v/len(sel):.1%})")
        for q, g, p in ex[k]:
            print(f"      Q {q}")
            print(f"        gold={g}   pred={p!r}")
    return cls, len(sel), len(tot)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--embed-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--no-rank", action="store_true", help="순위 재계산 생략(빠름)")
    args = ap.parse_args()

    recs = [r for r in load("t_s3c_hybrid_records.jsonl") if "correct" in r]
    ret = load("t_s3c_hybrid_answer_retrieved.jsonl")
    gold = load("t_s3c_hybrid_answer_gold.jsonl")

    section_a(recs, args)
    section_b(recs, ret, "B. 검색은 성공했는데 답이 틀린 질의 (distractor 몫)", True)
    section_b(recs, gold, "C. 정답 셀만 줬는데 틀린 질의 (리더 천장 — 검색으로 못 고침)", False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
