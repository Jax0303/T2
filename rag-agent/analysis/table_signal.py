#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""표를 잘못 고른 실패는 왜 나는가 -- 질문이 자기 표를 부르는가.

`col_signal_phrase.py` 가 **표 안**에서 한 계산을 **표 사이**로 옮긴 것이다. 그쪽이
정답 칸과 1등 칸의 헤더 경로 차집합을 봤다면, 여기서는 gold 표의 셀과 1등 표의 셀
사이의 차집합(제목 + 행 경로 + 열 경로)을 보고 질문이 어느 쪽에 가까운지 견준다.

판정기는 검색에 쓴 인코더가 아니라 **기성품 bge-base** 이고 임계값이 없다. 계측 셋을
서로 독립적으로 두고 교차 확인한다:

  A. 구절 코사인 -- 질문이 정답 표 쪽인가 오답 표 쪽인가 (임계값 없음)
  B. 단어 일치   -- 같은 질문을 불용어 뺀 토큰 교집합으로 다시 분류 (임베딩 없음)
  C. 대조군      -- 성공한 질의와 실패한 질의의 `gold 제목 단어 겹침` 을 견준다

C 가 이 진단의 근거다. A/B 는 실패 안에서의 구성이라 "그래서 실패의 원인이냐"를
말할 수 없지만, C 는 성공/실패를 가르므로 원인 주장을 지탱한다.

⚠️ **`--split test` 는 A/B 를 못 돌린다.** 이긴 표를 알려면 `_above.jsonl` 덤프가
필요한데 test 는 랭크 파일만 있다. test 에서는 분포와 회수 상한만 나온다.

⚠️ **사전등록 없는 사후 분석이다. arm 을 고르지 않는다.**

  PYTHONPATH=. .venv/bin/python analysis/table_signal.py --split dev
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "analysis")]

from scipy.stats import chi2_contingency, mannwhitneyu           # noqa: E402

import corpus_dump_vs_cell as cdv                                # noqa: E402
from rag_agent.serialization.caption import effective_titles     # noqa: E402

PAGE_TITLES = "results/tableconf/totto_page_titles.json"
RANKS = "results/p0_confirm/hitab_{split}_lookup_all_S3c_page_hybrid0.8_tp0.0_ranks.jsonl"
ABOVE = "results/colsig/hitab_dev_lookup_all_S3c_page_hybrid0.8_above.jsonl"
JUDGE = "BAAI/bge-base-en-v1.5"

WORD = re.compile(r"[a-z0-9]+")
tok = lambda s: set(WORD.findall(str(s).lower()))                # noqa: E731
# 질문의 골격 단어. 표를 특정하지 않으므로 겹침 계산에서 뺀다.
STOP = tok("the of a an in to for and or by is are was were how many what which "
           "total all with on at from that did do does number percentage")


def segs(path):
    return [str(s) for s in (path or []) if str(s).strip()]


def jac(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


def load(split):
    C = cdv.hitab_corpus("data/hitab", split, f"hitab_{split}_lookup_all")
    pt = json.load(open(PAGE_TITLES))
    ti = effective_titles(C.tids, C.title, C.cell_owner, C.cell_paths, "page",
                          page_titles=pt)
    recs = [json.loads(l) for l in open(RANKS.format(split=split))]
    return C, ti, recs


def distribution(recs, ti, split, out):
    """표 오인의 분포와, 표만 고쳤을 때의 회수 상한."""
    miss = [q for q in recs if q["table_rank_cellvote"] != 1]
    n, m = len(recs), len(miss)
    d = collections.Counter()
    for q in miss:
        r = q["table_rank_cellvote"]
        d["2등" if r == 2 else "3등" if r == 3 else "4~10등" if r <= 10 else "11등+"] += 1
    # 표를 공짜로 맞다고 쳐도 gold 셀이 표 안에서 1등이어야 R@1 이 된다
    recover = sum(1 for q in miss if max(q["ranks_in_table"]) < 1)
    totto = sum(1 for q in miss if "totto" in q["gold_table"])
    pop_totto = sum(1 for q in recs if "totto" in q["gold_table"])
    shared = collections.Counter(v.strip().lower() for v in ti.values())
    dup = {t for t, v in ti.items() if shared[v.strip().lower()] > 1}
    dup_pop = sum(1 for q in recs if q["gold_table"] in dup)
    dup_miss = sum(1 for q in miss if q["gold_table"] in dup)

    print(f"\n=== [{split}] n={n} · 표 오인 {m} ({m / n:.4f})")
    for k in ("2등", "3등", "4~10등", "11등+"):
        print(f"  gold 표 {k:>7}: {d[k]:>3} ({d[k] / m:.3f})")
    print(f"  표를 공짜로 맞다고 쳐도 셀 1등이 되는 건 {recover}/{m} ({recover / m:.3f})")
    print(f"  gold 표가 ToTTo 출신 {totto}/{m} | 모집단 {pop_totto}/{n}")
    print(f"  ToTTo 질의 오인율 {totto / pop_totto:.4f} / "
          f"통계표 질의 오인율 {(m - totto) / (n - pop_totto):.4f}")
    print(f"  page 제목이 겹치는 표 {len(dup)}/{len(ti)} · 그런 질의 {dup_pop}/{n} "
          f"→ 오인 {dup_miss}")
    out[split] = {"n": n, "miss": m, "rank_dist": dict(d), "recoverable": recover,
                  "miss_totto": totto, "pop_totto": pop_totto,
                  "dup_title_tables": len(dup), "dup_title_queries": dup_pop,
                  "dup_title_miss": dup_miss}
    return miss


def specificity(recs, ti, split, out):
    """C. 성공/실패 대조 -- 질문이 gold 표 제목의 단어를 쓰는가."""
    C = out["_corpus"]
    qtext = {}
    for q in C.queries:
        d = q if isinstance(q, dict) else q.__dict__
        qtext[d.get("query_id") or d.get("qid")] = d.get("question") or d.get("text")
    grp = {"성공(표 1등)": [], "실패(표 오인)": []}
    for r in recs:
        t = tok(ti[r["gold_table"]]) - STOP
        qt = tok(qtext[r["query_id"]]) - STOP
        key = "성공(표 1등)" if r["table_rank_cellvote"] == 1 else "실패(표 오인)"
        grp[key].append((len(tok(qtext[r["query_id"]])), len(t & qt)))

    print(f"\n=== [{split}] C. 질문이 자기 표를 이름으로 부르는가")
    print(f"{'':<14}{'n':>5}{'질문 토큰 중앙':>16}{'제목 단어 겹침 0':>18}")
    rows = {}
    for k, v in grp.items():
        z = sum(1 for x in v if x[1] == 0)
        rows[k] = {"n": len(v), "q_tokens_median": st.median(x[0] for x in v),
                   "zero_overlap": z, "zero_rate": round(z / len(v), 4)}
        print(f"{k:<14}{len(v):>5}{st.median(x[0] for x in v):>16.0f}"
              f"{z / len(v):>18.3f}")
    a, b = grp["성공(표 1등)"], grp["실패(표 오인)"]
    p_len = mannwhitneyu([x[0] for x in a], [x[0] for x in b]).pvalue
    za, zb = rows["성공(표 1등)"]["zero_overlap"], rows["실패(표 오인)"]["zero_overlap"]
    p_zero = chi2_contingency([[za, len(a) - za], [zb, len(b) - zb]]).pvalue
    print(f"  질문 길이  Mann-Whitney p = {p_len:.4f}   ← 유의하지 않아야 한다")
    print(f"  겹침 0     chi2 p = {p_zero:.3g}")
    out[split]["specificity"] = {**rows, "p_len": float(p_len), "p_zero": float(p_zero)}


def winners(miss, ti, out):
    """dev 전용. 이긴 표는 무엇이고, 질문은 어느 쪽을 가리키는가 (A + B)."""
    ab = {r["query_id"]: r for r in map(json.loads, open(ABOVE))}
    cases = []
    for q in miss:
        a = ab.get(q["query_id"])
        w = next((t for t in (a or {}).get("top_above", [])
                  if t["relation"] == "other_table"), None)
        if w:
            cases.append((a, q["gold_table"], w))
    n = len(cases)

    # 이긴 표는 아무 표가 아니라 같은 보고서의 형제 표다 -- 무작위 표를 대조로 둔다
    import random
    rnd = random.Random(0)
    tids = list(ti)
    tj, ctrl, rowm, colm, both = [], [], 0, 0, 0
    for a, g, w in cases:
        tj.append(jac(tok(ti[g]), tok(ti[w["table_id"]])))
        ctrl.append(jac(tok(ti[g]), tok(ti[rnd.choice(tids)])))
        r = w["row_path"] == a["gold_row_path"]
        c = w["col_path"] == a["gold_col_path"]
        rowm += r
        colm += c
        both += r and c
    print(f"\n=== [dev] 이긴 표는 무엇인가  n={n}")
    print(f"  제목 토큰 자카드  gold vs 이긴 표 : 중앙 {st.median(tj):.3f}")
    print(f"                    gold vs 무작위 표: 중앙 {st.median(ctrl):.3f}  ← 대조")
    print(f"  이긴 셀의 행 경로가 gold 와 완전 동일 {rowm}/{n} ({rowm / n:.3f})")
    print(f"  열 경로 동일 {colm}/{n} · 행·열 둘 다 {both}/{n}  ← 제목과 값만 다른 문장")

    # A. 구절 코사인. 두 셀 문장의 차집합만 보므로 검색기가 한 계산이 아니다.
    texts, keyed = [], []
    for a, g, w in cases:
        G = set([ti[g]] + segs(a["gold_row_path"]) + segs(a["gold_col_path"]))
        W = set([ti[w["table_id"]]] + segs(w["row_path"]) + segs(w["col_path"]))
        keyed.append((len(texts), len(texts) + 1, len(texts) + 2, sorted(G - W),
                      sorted(W - G)))
        texts += [a["question"], " ".join(sorted(G - W)), " ".join(sorted(W - G))]
    from sentence_transformers import SentenceTransformer
    E = SentenceTransformer(JUDGE).encode(texts, batch_size=64,
                                          normalize_embeddings=True,
                                          show_progress_bar=False)
    rows, ng, nw, nn = [], 0, 0, 0
    for (a, g, w), (qi, gi, wi, go, wo) in zip(cases, keyed):
        if not go and not wo:
            nn += 1
            kind, margin = "구별 내용 없음", 0.0
        else:
            sg = float(E[qi] @ E[gi]) if go else -1.0
            sw = float(E[qi] @ E[wi]) if wo else -1.0
            margin = sg - sw
            kind = "정답 표 쪽" if margin > 0 else "오답 표 쪽"
            ng += margin > 0
            nw += margin <= 0
        rows.append(dict(query_id=a["query_id"], kind=kind, margin=round(margin, 4),
                         question=a["question"], gold_title=ti[g],
                         win_title=ti[w["table_id"]], gold_only=go, win_only=wo))
    mgs = sorted(abs(r["margin"]) for r in rows if r["margin"])
    print(f"\n=== [dev] A. 질문은 어느 표를 지목하는가 (구절 코사인, 임계값 없음)")
    print(f"  {ng:>3} ({ng / n:.3f})  질문이 정답 표 쪽에 가깝다  ← 신호 있음")
    print(f"  {nw:>3} ({nw / n:.3f})  질문이 오답 표 쪽에 가깝다  ← 검색으로 불가")
    print(f"  {nn:>3} ({nn / n:.3f})  구별 내용 자체가 없음")
    print(f"  |margin| 중앙 {st.median(mgs):.3f} · <0.02 애매 "
          f"{sum(1 for x in mgs if x < 0.02)}/{len(mgs)}")

    # B. 같은 질문을 단어로 다시 분류한다 -- 임베딩과 독립인 두 번째 계측
    kinds = collections.Counter()
    for r in rows:
        q = tok(r["question"]) - STOP
        go = (tok(" ".join(r["gold_only"])) - STOP) & q
        wo = (tok(" ".join(r["win_only"])) - STOP) & q
        kinds["① 구별 단서가 아예 없다" if not go and not wo else
              "② 오답 표 쪽 단어만 쓴다" if wo and not go else
              "③ 정답 표 쪽 단어만 쓴다" if go and not wo else
              "④ 양쪽 단어를 다 쓴다"] += 1
    print(f"\n=== [dev] B. 같은 실패를 단어로 다시 분류 (임베딩 없음)")
    for k in sorted(kinds):
        print(f"  {kinds[k]:>3} ({kinds[k] / n:.3f})  {k}")

    # 회수 상한 -- 두 조건은 겹치지 않는다
    at = {q["query_id"]: q for q in miss}
    A = [r for r in rows if r["kind"] == "정답 표 쪽"]
    B = [q for q in miss if max(q["ranks_in_table"]) < 1]
    AB2 = [r for r in A if max(at[r["query_id"]]["ranks_in_table"]) < 1]
    n_pop = out["dev"]["n"]
    print(f"\n=== [dev] 표 축에서 실제로 회수 가능한 상한")
    print(f"  A 표를 고칠 신호가 있다        {len(A)} ({len(A) / n:.3f})")
    print(f"  B 표만 고치면 셀도 1등이 된다  {len(B)} ({len(B) / n:.3f})")
    print(f"  A∩B                            {len(AB2)} ({len(AB2) / n:.3f})")
    print(f"  → R@1 에 주는 값 +{len(AB2)}/{n_pop} = +{len(AB2) / n_pop:.4f}")

    out["dev"]["winners"] = {
        "n": n, "title_jaccard_median": round(st.median(tj), 4),
        "title_jaccard_random_median": round(st.median(ctrl), 4),
        "same_row_path": rowm, "same_col_path": colm, "same_both": both,
        "gold_side": ng, "win_side": nw, "no_content": nn,
        "margin_median": round(st.median(mgs), 4),
        "ambiguous_lt_002": sum(1 for x in mgs if x < 0.02),
        "word_kinds": dict(kinds), "A": len(A), "B": len(B), "A_and_B": len(AB2),
        "r_at_1_headroom": round(len(AB2) / n_pop, 4), "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test", "both"])
    ap.add_argument("--out", default="results/tabsig/table_signal.json")
    a = ap.parse_args()

    out = {}
    splits = ["dev", "test"] if a.split == "both" else [a.split]
    for s in splits:
        C, ti, recs = load(s)
        out["_corpus"] = C
        miss = distribution(recs, ti, s, out)
        specificity(recs, ti, s, out)
        if s == "dev":
            winners(miss, ti, out)
    del out["_corpus"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
