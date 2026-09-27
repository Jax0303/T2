"""2026-09-28 원고 5.7절(1위 적중률) 출처·검정, 조회1 을 "정답 = 정답 셀 경로의 머리글" 여부로 나눈 값. 새 검색 없음.

MultiHiertt train, 문서 안, s3c 셀 문장, α=.7:
  처음 버전(v1) = results/mh_arms/mh_train_cell_hv1_none_doc_kladder_records.jsonl 의 correct_at.doc['1'·'20']
    (같은 순위 목록을 예산별로 재채점한 기록; 예산 1 실행 _b1_records.jsonl 과 문항별로 같은지 assert).
    원고처럼 2,885건 기준: v1 이 채점하지 못한 14건은 최종 버전의 그룹에 넣고 실패로 센다.
  최종 버전(v3.3u) = results/rerun_20260926/mh/mh_train_s3c_records.jsonl (표 5-5). 1위 정보가 기록에 없어서
    doc.context[0](budget_select 가 순위대로 넣은 첫 셀 문장)이 정답 셀 문장과 같은지로 센다. 최종 버전은 문서 안
    셀 문장이 서로 다름을 코드가 보장한다. 정답 셀 문장은 색인 코드로 다시 만든다(결정적, 임베딩 없음).
HiTab test 단일 셀 조회 991, 질문의 표 안, s3c = results/rerun_20260926/hitab/hitab_test_gold_s3c_records.jsonl.
  1위 = gold_rank == 1 (context_units[0] 의 셀과 같은지 assert), 예산 20 = correct.
(가) = 정답 문자열 하나라도 정답 셀의 행·열 경로 머리글 원소 하나와 정규화 후 같음. 정규화 = 소문자, [a-z0-9.] 밖 문자 삭제,
  양끝 '.' 삭제, 숫자면 Decimal 정규형(12.0 → 12). 경로는 그 버전의 표 객체(t.row_path(i) + t.col_path(j)).
실행: HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=. .venv/bin/python results/lookup_vs_arith_20260928/rank1.py
"""
import json
import re
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import mh_arms as ma                                                  # noqa: E402
from retrieval_accuracy import build_corpus                           # noqa: E402
from rag_agent.bench import hitab_grid as hg                          # noqa: E402

jl = lambda p: [json.loads(x) for x in open(ROOT / p)]


def norm(s):
    t = re.sub(r"[^a-z0-9.]", "", str(s).lower()).strip(".")
    try:
        return format(Decimal(t).normalize(), "f") if re.fullmatch(r"\d+(\.\d+)?", t) else t
    except InvalidOperation:
        return t


def is_header(answers, path):
    heads = {norm(h) for h in path} - {""}
    return any(norm(a) in heads for a in answers)


def rate(xs, k):
    return {"n": len(xs), "hit": sum(x[k] for x in xs), "rate": sum(x[k] for x in xs) / len(xs) if xs else None}


def fisher(a, b):
    t = [[a["hit"], a["n"] - a["hit"]], [b["hit"], b["n"] - b["hit"]]]
    return {"table_hit_miss": t, "p": float(fisher_exact(t)[1])}


def split(rows):
    """조회1 을 (가)/(나) 로. rows: {query_id, header, rank1, b20}."""
    out = {}
    for name, keep in (("가_정답이_머리글", True), ("나_나머지", False)):
        xs = [x for x in rows if x["header"] == keep]
        out[name] = {"n": len(xs), "rank1": rate(xs, "rank1"), "budget20": rate(xs, "b20"),
                     "query_ids": sorted(x["query_id"] for x in xs) if keep else None}
    return out


def gold_paths(queries, tables, hdr):
    live = {(tid, i, j) for tid, tab in tables.items()
            for i, rr in enumerate(tab.table.data) for j, v in enumerate(rr) if str(v).strip()}
    qs = {q["uid"]: dict(q) for q in ma.resolve_gold([dict(q) for q in queries], tables, hdr, live)}
    return qs


def mh():
    queries, docs, _ = ma.load_population("train")
    out = {}
    # 최종 버전: 표, 셀 문장, 정답
    t3, h3 = ma.build_tables(docs, "v3.3u", "none")
    texts, covers, *_ = build_corpus("", sorted(t3), "s3c", "cell", {}, 1000, "leaf", "values", 200, None,
                                     load=lambda tid, _d: t3.get(tid))
    text_of = {next(iter(c)): t for t, c in zip(texts, covers)}
    q3 = gold_paths(queries, t3, h3)
    path = lambda tabs, cell: [*tabs[cell[0]].table.row_path(cell[1]), *tabs[cell[0]].table.col_path(cell[2])]
    fin = {}
    for r in jl("results/rerun_20260926/mh/mh_train_s3c_records.jsonl"):
        if "excluded" in r or r["layer"] not in ("lookup_m1", "arith_m1"):
            continue
        (g,) = q3[r["query_id"]]["gold"]
        fin[r["query_id"]] = {"query_id": r["query_id"], "layer": r["layer"], "answer": r["answer"],
                              "rank1": int(r["doc"]["context"][0] == text_of[g]), "b20": r["doc"]["correct"],
                              "header": is_header([r["answer"]], path(t3, g))}
        assert fin[r["query_id"]]["rank1"] <= fin[r["query_id"]]["b20"]
    # 처음 버전
    t1, h1 = ma.build_tables(docs, "v1", "none")
    q1 = gold_paths(queries, t1, h1)
    b1 = {r["query_id"]: r["doc"]["correct"] for r in jl("results/mh_arms/mh_train_cell_hv1_none_doc_b1_records.jsonl")
          if "excluded" not in r}
    # 방법 점검: 첫 문맥 문장 == 정답 셀 문장 으로 센 1위가 v1 의 correct_at['1'] 과 같은가.
    # v1 은 문서 안 같은 문장이 있어서 그런 문항은 따로 센다.
    tx1, cv1, *_ = build_corpus("", sorted(t1), "s3c", "cell", {}, 1000, "leaf", "values", 200, None,
                                load=lambda tid, _d: t1.get(tid))
    text1 = {next(iter(c)): t for t, c in zip(tx1, cv1)}
    n_same = Counter((c[0].split("::")[0], t) for c, t in text1.items())
    check = {"n": 0, "agree": 0, "disagree_gold_text_duplicated_in_doc": 0, "disagree_other": 0}
    v1 = {}
    for r in jl("results/mh_arms/mh_train_cell_hv1_none_doc_kladder_records.jsonl"):
        if "excluded" in r or r["layer"] not in ("lookup_m1", "arith_m1"):
            continue
        ca = r["correct_at"]["doc"]
        assert ca["20"] == r["doc"]["correct"] and ca["1"] == b1[r["query_id"]]
        (g,) = q1[r["query_id"]]["gold"]
        same = n_same[(g[0].split("::")[0], text1[g])]
        check["n"] += 1
        if int(r["doc"]["context"][0] == text1[g]) == ca["1"]:
            check["agree"] += 1
        else:
            check["disagree_gold_text_duplicated_in_doc" if same > 1 else "disagree_other"] += 1
        v1[r["query_id"]] = {"query_id": r["query_id"], "layer": r["layer"], "rank1": ca["1"], "b20": ca["20"],
                             "header": is_header([r["answer"]], path(t1, g)), "v1_scored": True}
    for qid, x in fin.items():                     # v1 이 채점 못 한 문항: 최종 버전 그룹·경로, 실패로 셈
        if qid not in v1:
            assert q1[qid]["excluded"], qid
            v1[qid] = {**x, "rank1": 0, "b20": 0, "v1_scored": False, "v1_excluded": q1[qid]["excluded"]}
    assert set(v1) == set(fin)
    out["방법_점검_v1_첫문맥문장_대_correct_at1"] = check
    for name, rows in (("처음_버전_v1", v1), ("최종_버전_v3.3u", fin)):
        L = [x for x in rows.values() if x["layer"] == "lookup_m1"]
        A = [x for x in rows.values() if x["layer"] == "arith_m1"]
        d = {"records": "results/mh_arms/mh_train_cell_hv1_none_doc_kladder_records.jsonl" if name.startswith("처음")
             else "results/rerun_20260926/mh/mh_train_s3c_records.jsonl",
             "조회1": {"rank1": rate(L, "rank1"), "budget20": rate(L, "b20")},
             "산술1": {"rank1": rate(A, "rank1"), "budget20": rate(A, "b20")}}
        d["fisher_rank1_산술1_대_조회1"] = fisher(d["산술1"]["rank1"], d["조회1"]["rank1"])
        d["fisher_budget20_산술1_대_조회1"] = fisher(d["산술1"]["budget20"], d["조회1"]["budget20"])
        d["조회1_split"] = split(L)
        if name.startswith("처음"):
            Ls = [x for x in L if x["v1_scored"]]
            d["v1_채점_문항만"] = {"조회1_rank1": rate(Ls, "rank1"), "조회1_budget20": rate(Ls, "b20"),
                              "fisher_rank1_산술1_대_조회1": fisher(d["산술1"]["rank1"], rate(Ls, "rank1")),
                              "조회1_split": split(Ls)}
            d["v1_미채점_문항"] = [{k: x[k] for k in ("query_id", "layer", "header", "v1_excluded")}
                              for x in rows.values() if not x["v1_scored"]]
        out[name] = d
    # 원고 5.7절 값
    v = out["처음_버전_v1"]
    assert (v["조회1"]["rank1"]["hit"], v["조회1"]["rank1"]["n"], v["산술1"]["rank1"]["hit"], v["산술1"]["rank1"]["n"]) == (75, 212, 38, 71)
    assert (v["조회1"]["budget20"]["hit"], v["산술1"]["budget20"]["hit"]) == (200, 68)
    assert round(out["최종_버전_v3.3u"]["조회1"]["budget20"]["rate"], 4) == .9575
    return out


def hitab():
    rows, tables = [], {}
    for r in jl("results/rerun_20260926/hitab/hitab_test_gold_s3c_records.jsonl"):
        if "excluded" in r or r["mode"] != "all" or r["m"] != 1 or (r.get("aggregation") or "none") != "none":
            continue
        (g,) = [tuple(c) for c in r["gold_cells"]]
        assert r["correct"] == int(r["gold_rank"] is not None and r["gold_rank"] <= 20)
        rank1 = int(r["gold_rank"] == 1)
        assert rank1 == int([tuple(c) for c in r["context_units"][0]["cells"]] == [g])
        t = tables.setdefault(g[0], hg.load_table(g[0], "data/hitab").table)
        rows.append({"query_id": r["query_id"], "rank1": rank1, "b20": r["correct"],
                     "header": is_header(r["answer"], [*t.row_path(g[1]), *t.col_path(g[2])])})
    assert len(rows) == 991 and sum(x["b20"] for x in rows) == 955
    return {"records": "results/rerun_20260926/hitab/hitab_test_gold_s3c_records.jsonl",
            "조회1": {"rank1": rate(rows, "rank1"), "budget20": rate(rows, "b20")}, "조회1_split": split(rows)}


if __name__ == "__main__":
    res = {"multihiertt": mh(), "hitab": hitab()}
    (OUT / "rank1.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))
