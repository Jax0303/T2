"""2026-09-28 MultiHiertt 셀 1개 그룹: 1등 적중이 산술 > 조회인 차이가 "질문이 정답 칸의 행·열 이름을 전부 부르는가"로
설명되는지. 사후 분석, 새 검색 없음. 1등 판정·버전·입력은 rank1.py 와 같다(처음 버전 = kladder correct_at['1'],
최종 버전 = doc.context[0] 이 정답 칸 문장과 같음).
열 이름을 전부 부름 = 정답 칸 열 경로의 단어(식별자 'Table k'/'row N'/'column N' 원소 제외, _tokenize − sklearn 불용어)가
  전부 질문 단어 안에 있음. 행도 같은 식. 2026-09-23 사후 분석(메모리 기록, 스크립트 없음)과 같은 정의를 의도했다 —
  처음 버전에서 그때의 조회 85/211, 산술 46/71 이 재현되는지 출력한다.
층 = (열 전부 부름, 행 전부 부름) 4칸. 층마다 조회 대 산술 1등 Fisher, 층 통합은 Mantel–Haenszel 공통 OR 과 CMH 검정(연속성 보정 없음).
실행: HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=. .venv/bin/python results/lookup_vs_arith_20260928/why_rank1.py
"""
import json
import re
import sys
from pathlib import Path

from scipy.stats import chi2, fisher_exact
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

sys.path.insert(0, str(Path(__file__).parent))
from rank1 import ROOT, OUT, jl, is_header, gold_paths, ma, build_corpus   # noqa: E402
from rag_agent.retrieve.encoders import _tokenize                          # noqa: E402

words = lambda s: set(_tokenize(s)) - ENGLISH_STOP_WORDS
IDENT = re.compile(r"^(Table \d+|row \d+|column \d+)$")
# 값 조건 = 질문이 칸을 이름이 아니라 값 비교로 고르는 표현(사후 정규식, 정밀도는 표본 30개 눈으로만 확인)
VC = re.compile(r"\b(greatest|largest|highest|most|lowest|least|smallest|biggest|maximum|minimum|max|min|second|"
                r"exceed\w*|greater than|less than|more than|lower than|higher than|larger than|smaller than|"
                r"in the range|between)\b", re.I)


def named(q, path):
    w = words(" ".join(x for x in path if not IDENT.match(x)))
    return bool(w) and w <= words(q)


def rows_for(version, docs, queries):
    t, h = ma.build_tables(docs, version, "none")
    qs = gold_paths(queries, t, h)
    out = []
    if version == "v1":
        src = "results/mh_arms/mh_train_cell_hv1_none_doc_kladder_records.jsonl"
        r1 = lambda r, g: r["correct_at"]["doc"]["1"]
    else:
        src = "results/rerun_20260926/mh/mh_train_s3c_records.jsonl"
        texts, covers, *_ = build_corpus("", sorted(t), "s3c", "cell", {}, 1000, "leaf", "values", 200, None,
                                         load=lambda tid, _d: t.get(tid))
        text_of = {next(iter(c)): x for x, c in zip(texts, covers)}
        r1 = lambda r, g: int(r["doc"]["context"][0] == text_of[g])
    for r in jl(src):
        if "excluded" in r or r["layer"] not in ("lookup_m1", "arith_m1"):
            continue
        (g,) = qs[r["query_id"]]["gold"]
        tab = t[g[0]].table
        rp, cp = tab.row_path(g[1]), tab.col_path(g[2])
        kind = "산술" if r["layer"] == "arith_m1" else (
            "조회_답이_머리글" if is_header([r["answer"]], [*rp, *cp]) else "조회_나머지")
        out.append({"query_id": r["query_id"], "kind": kind, "rank1": r1(r, g),
                    "col_named": named(r["question"], cp), "row_named": named(r["question"], rp),
                    "value_condition": bool(VC.search(r["question"]))})
    return out


def rate(xs):
    return {"n": len(xs), "rank1_hit": sum(x["rank1"] for x in xs),
            "rank1": round(sum(x["rank1"] for x in xs) / len(xs), 4) if xs else None}


def cmh(tables):
    """tables: [[a, b], [c, d]] 층별 (산술 적중, 산술 실패; 조회 적중, 조회 실패)."""
    num = den = s = v = 0.0
    for (a, b), (c, d) in tables:
        n = a + b + c + d
        if n < 2:
            continue
        num += a * d / n
        den += b * c / n
        s += a - (a + b) * (a + c) / n
        v += (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1))
    return {"mh_odds_ratio_산술_대_조회": round(num / den, 3) if den else None,
            "cmh_chi2": round(s * s / v, 3), "p": float(chi2.sf(s * s / v, 1))}


def analyze(rows):
    look = [x for x in rows if x["kind"] != "산술"]
    ar = [x for x in rows if x["kind"] == "산술"]
    d = {"열_전부_부름_비율": {k: {"n": len(xs), "col_named": sum(x["col_named"] for x in xs),
                                  "row_named": sum(x["row_named"] for x in xs)}
                           for k, xs in (("조회", look), ("조회_답이_머리글", [x for x in look if x["kind"] == "조회_답이_머리글"]),
                                         ("조회_나머지", [x for x in look if x["kind"] == "조회_나머지"]), ("산술", ar))},
         "전체": {"조회": rate(look), "산술": rate(ar), "fisher_p": None}, "층": [], "층_통합": None}
    t = [[d["전체"]["산술"]["rank1_hit"], len(ar) - d["전체"]["산술"]["rank1_hit"]],
         [d["전체"]["조회"]["rank1_hit"], len(look) - d["전체"]["조회"]["rank1_hit"]]]
    d["전체"]["fisher_p"] = float(fisher_exact(t)[1])
    tabs = []
    for cn in (True, False):
        for rn in (True, False):
            L = [x for x in look if x["col_named"] == cn and x["row_named"] == rn]
            A = [x for x in ar if x["col_named"] == cn and x["row_named"] == rn]
            tab = [[sum(x["rank1"] for x in A), sum(1 - x["rank1"] for x in A)],
                   [sum(x["rank1"] for x in L), sum(1 - x["rank1"] for x in L)]]
            tabs.append(tab)
            d["층"].append({"열_전부_부름": cn, "행_전부_부름": rn, "조회": rate(L), "산술": rate(A),
                           "조회_답이_머리글": rate([x for x in L if x["kind"] == "조회_답이_머리글"]),
                           "조회_나머지": rate([x for x in L if x["kind"] == "조회_나머지"]),
                           "fisher_p_산술_대_조회": float(fisher_exact(tab)[1]) if L and A else None})
    d["층_통합"] = cmh(tabs)
    vt = []
    d["값조건_층"] = []
    for vc in (True, False):
        L = [x for x in look if x["value_condition"] == vc]
        A = [x for x in ar if x["value_condition"] == vc]
        tab = [[sum(x["rank1"] for x in A), sum(1 - x["rank1"] for x in A)],
               [sum(x["rank1"] for x in L), sum(1 - x["rank1"] for x in L)]]
        vt.append(tab)
        d["값조건_층"].append({"값_조건_표현": vc, "조회": rate(L), "산술": rate(A),
                             "조회_답이_머리글": rate([x for x in L if x["kind"] == "조회_답이_머리글"]),
                             "조회_나머지": rate([x for x in L if x["kind"] == "조회_나머지"]),
                             "fisher_p_산술_대_조회": float(fisher_exact(tab)[1])})
    d["값조건_층_통합"] = cmh(vt)
    d["층_통합_없이_OR"] = round((t[0][0] * t[1][1]) / (t[0][1] * t[1][0]), 3)
    return d


if __name__ == "__main__":
    queries, docs, _ = ma.load_population("train")
    res = {}
    for name, ver in (("처음_버전_v1", "v1"), ("최종_버전_v3.3u", "v3.3u")):
        rows = rows_for(ver, docs, queries)
        res[name] = analyze(rows)
        with open(OUT / f"why_rank1_rows_{ver}.jsonl", "w") as f:
            f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
    (OUT / "why_rank1.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))
