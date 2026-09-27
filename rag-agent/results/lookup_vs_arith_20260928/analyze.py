"""2026-09-28 조회 대 산술 검색 정확도: 채점 기준 효과와 질문 표현 효과. 새 검색·임베딩 없음.

입력 = 원고 주 결과(표 5-2, 5-5)의 records:
  HiTab  results/rerun_20260926/hitab/hitab_test_gold_s3c_records.jsonl (질문의 표 안, 예산 20)
  MH     results/rerun_20260926/mh/mh_train_s3c_records.jsonl          (문서 안 'doc', v3.3u, 예산 20)
그룹 = 조회1/조회2+/산술1/산술2+. HiTab 조회 = aggregation none, 산술 = 그 밖(retrieval_accuracy.query_type);
mode 'any'(답이 머리글) 336건은 원고처럼 뺀다. MH 조회/산술 = records 의 kind, 셀 수 = m.
전부 포함 = G ⊆ R (원고 주 지표), 1개 이상 = G ∩ R ≠ ∅, 셀 재현율 = |G ∩ R| / |G|
  (질의 평균과 셀 합산 둘 다).
HiTab 은 records 의 gold_cells·context_cells 로 바로 센다. MH records 에는 셀 좌표가 없어서,
mh_arms 의 색인 코드로 v3.3u 셀 문장을 다시 만들고(결정적, 임베딩 없음) 정답 셀 문장이
records 의 doc 문맥 문자열에 있는지로 센다 — 문서 안 셀 문장은 코드가 서로 다름을 보장한다.
재구성이 records 의 correct·any_DIAGNOSTIC·m·layer 를 전부 재현하는지 assert 한다.
질문 표현 = hitab_path_overlap.py 와 같은 정의: _tokenize(소문자) − sklearn 불용어, 서로 다른 단어,
비율 = |질문 ∩ 경로| / |질문|. 경로 = 정답 셀들의 행 경로 + 열 경로 단어의 합집합(표 제목·값 제외).
MH 경로는 v3.3u 표 객체의 경로에서 식별자 원소('Table k', 'row N', 'column N')를 뺀 것
(원고 3.3절: 라벨이 아니라 같은 문장을 가르는 마지막 수단). 식별자를 넣은 값도 json 에 둔다.
실행: .venv/bin/python results/lookup_vs_arith_20260928/analyze.py
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from rag_agent.bench import hitab_grid as hg                          # noqa: E402
from rag_agent.retrieve.encoders import _tokenize                     # noqa: E402

G = ("조회1", "조회2+", "산술1", "산술2+")
words = lambda s: set(_tokenize(s)) - ENGLISH_STOP_WORDS
IDENT = re.compile(r"^(Table \d+|row \d+|column \d+)$")


def row(qid, grp, m, n_in, correct, question, path_words, path_words_ident=None):
    q = words(question)
    ratio = lambda p: len(q & p) / len(q) if q else None
    r = {"query_id": qid, "group": grp, "m": m, "n_gold_in_context": n_in, "all": int(n_in == m),
         "any": int(n_in > 0), "recall": n_in / m, "ratio": ratio(path_words)}
    assert r["all"] == correct, qid
    if path_words_ident is not None:
        r["ratio_with_identifiers"] = ratio(path_words_ident)
    return r


def hitab():
    tables, rows = {}, []
    for r in map(json.loads, open(ROOT / "results/rerun_20260926/hitab/hitab_test_gold_s3c_records.jsonl")):
        if "excluded" in r or r["mode"] != "all":
            continue
        gold, ctx = {tuple(c) for c in r["gold_cells"]}, {tuple(c) for c in r["context_cells"]}
        assert len(gold) == r["m"]
        arith = (r.get("aggregation") or "none") != "none"
        grp = G[2 * arith + (r["m"] > 1)]
        p = set()
        for tid, i, j in gold:
            t = tables.setdefault(tid, hg.load_table(tid, "data/hitab").table)
            p |= words(" ".join([*t.row_path(i), *t.col_path(j)]))
        rows.append(row(r["query_id"], grp, r["m"], len(gold & ctx), r["correct"], r["question"], p))
    n = {g: sum(x["group"] == g for x in rows) for g in G}
    assert n == {"조회1": 991, "조회2+": 38, "산술1": 60, "산술2+": 156}, n
    # 원고 표 5-2 본 방법 행
    for g, acc in (("조회1", .9637), ("조회2+", .9211)):
        assert round(np.mean([x["all"] for x in rows if x["group"] == g]), 4) == acc
    assert round(np.mean([x["all"] for x in rows if x["group"].startswith("산술")]), 4) == .8611
    return rows


def mh():
    import mh_arms as ma
    from retrieval_accuracy import build_corpus
    queries, docs, _ = ma.load_population("train")
    tables, hdr = ma.build_tables(docs, "v3.3u", "none")
    texts, covers, _, _, _ = build_corpus("", sorted(tables), "s3c", "cell", {}, 1000, "leaf", "values", 200, None,
                                          load=lambda tid, _d: tables.get(tid))
    text_of = {}
    for t, cv in zip(texts, covers):
        assert len(cv) == 1
        text_of[next(iter(cv))] = t
    live = {(tid, i, j) for tid, tab in tables.items()
            for i, rr in enumerate(tab.table.data) for j, v in enumerate(rr) if str(v).strip()}
    qs = {q["uid"]: q for q in ma.resolve_gold(queries, tables, hdr, live)}
    rows = []
    for r in map(json.loads, open(ROOT / "results/rerun_20260926/mh/mh_train_s3c_records.jsonl")):
        if "excluded" in r:
            continue
        q, d = qs[r["query_id"]], r["doc"]
        assert not q["excluded"] and len(q["gold"]) == r["m"] and ma.layer(q) == r["layer"]
        ctx = set(d["context"])
        n_in = sum(text_of[c] in ctx for c in q["gold"])
        assert int(n_in > 0) == d["any_DIAGNOSTIC"], r["query_id"]
        grp = G[2 * (r["kind"] == "arith") + (r["m"] > 1)]
        p, pi = set(), set()
        for tid, i, j in q["gold"]:
            t = tables[tid].table
            path = [*t.row_path(i), *t.col_path(j)]
            p |= words(" ".join(x for x in path if not IDENT.match(x)))
            pi |= words(" ".join(path))
        rows.append(row(r["query_id"], grp, r["m"], n_in, d["correct"], r["question"], p, pi))
    n = {g: sum(x["group"] == g for x in rows) for g in G}
    assert n == {"조회1": 212, "조회2+": 367, "산술1": 71, "산술2+": 2235}, n
    for g, acc in zip(G, (.9575, .8774, .9577, .8492)):         # 원고 표 5-5 본 방법 행
        assert round(np.mean([x["all"] for x in rows if x["group"] == g]), 4) == acc
    return rows


def ratio_stats(xs, key="ratio"):
    v = np.array([x[key] for x in xs if x[key] is not None])
    return {"n": len(xs), "n_no_content_words": len(xs) - len(v),
            "ratio_median": float(np.median(v)) if len(v) else None,
            "ratio_mean": float(v.mean()) if len(v) else None,
            "ratio_zero_share": float((v == 0).mean()) if len(v) else None}


def summary(xs):
    return {"n": len(xs), "m_median": float(np.median([x["m"] for x in xs])),
            "any": float(np.mean([x["any"] for x in xs])), "all": float(np.mean([x["all"] for x in xs])),
            "recall_query_mean": float(np.mean([x["recall"] for x in xs])),
            "recall_cell_pooled": sum(x["n_gold_in_context"] for x in xs) / sum(x["m"] for x in xs),
            "n_all": sum(x["all"] for x in xs), "n_any": sum(x["any"] for x in xs)}


def matched(rows):
    """조회2+ 대 산술2+ 를 정답 셀 수 m 이 같은 층끼리. 두 그룹이 다 있는 층만 합계·표준화에 쓴다."""
    by = defaultdict(lambda: defaultdict(list))
    for x in rows:
        if x["group"] in ("조회2+", "산술2+"):
            by[x["m"]][x["group"]].append(x)
    strata = []
    for m in sorted(by):
        L, A = by[m]["조회2+"], by[m]["산술2+"]
        s = {"m": m, "조회2+": summary(L) if L else {"n": 0}, "산술2+": summary(A) if A else {"n": 0}}
        if L and A:
            s["fisher_p_all"] = float(fisher_exact([[s["조회2+"]["n_all"], len(L) - s["조회2+"]["n_all"]],
                                                    [s["산술2+"]["n_all"], len(A) - s["산술2+"]["n_all"]]])[1])
        strata.append(s)
    common = [s for s in strata if s["조회2+"]["n"] and s["산술2+"]["n"]]
    nl = sum(s["조회2+"]["n"] for s in common)
    pooled = {g: summary([x for s in common for x in by[s["m"]][g]]) for g in ("조회2+", "산술2+")}
    # 산술2+ 층별 비율을 조회2+ 의 층 분포(공통 층)로 가중한 값
    std = {k: sum(s["산술2+"][k] * s["조회2+"]["n"] / nl for s in common)
           for k in ("all", "any", "recall_query_mean")}
    return {"strata": strata, "common_m": [s["m"] for s in common],
            "common_pooled": pooled, "산술2+_standardized_to_조회2+_m": std}


def main():
    out = {}
    for name, rows in (("hitab", hitab()), ("multihiertt", mh())):
        with open(OUT / f"{name}_rows.jsonl", "w") as f:
            f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
        d = {"groups": {}, "matched_m": matched(rows), "fisher_p_all": {}}
        for a, b in (("조회1", "산술1"), ("조회2+", "산술2+"), ("조회", "산술")):   # "조회" = 조회1 + 조회2+
            k = [[sum(x["all"] == v for x in rows if x["group"] == g or x["group"][:2] == g) for v in (1, 0)]
                 for g in (a, b)]
            d["fisher_p_all"][f"{a} 대 {b}"] = {"table_all_notall": k, "p": float(fisher_exact(k)[1])}
        for g in G:
            xs = [x for x in rows if x["group"] == g]
            d["groups"][g] = {**summary(xs), "path_overlap": ratio_stats(xs),
                              "path_overlap_success": ratio_stats([x for x in xs if x["all"]]),
                              "path_overlap_fail": ratio_stats([x for x in xs if not x["all"]])}
            if name == "multihiertt":
                d["groups"][g]["path_overlap_with_identifiers"] = ratio_stats(xs, "ratio_with_identifiers")
        out[name] = d
    (OUT / "analyze.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
