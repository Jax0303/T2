"""논문 4단계(서술 수정, 2026-09-27)에 쓰는 값 중 기존 결과 파일에 없는 것 — 기존 결과에서 다시 센다. 새 검색·생성 없음.

실행:  cd rag-agent && .venv/bin/python results/thesis_fix_20260927/step4_values.py  -> step4_values.json
1. 전달 셀 평균: HiTab 단일 셀 조회 991건(질문의 표 안)·MultiHiertt 2,885건(문서 안), 재실행 레코드의 cells_in_context.
2. HiTab 표 단위(질문의 표 안) 유형별 정확도.
3. MultiHiertt 882건 답변: 셀 문장(최종) 대 고정 청크(최종) — 가중 정확도·층화 부트스트랩 CI(mh_cap300_report 와 같은 방식, 가중 2,885).
4. 답변 차이가 나온 문항 묶음: 둘 다 검색 성공 / 셀 문장만 / 비교군만 / 둘 다 실패 (HiTab 300 s3c 대 고정 청크, MultiHiertt 882 최종 대 최종).
5. 고정 청크 머리글 규칙 효과: 처음 규칙 문맥 366 대 최종 규칙 문맥 368 (McNemar).
"""
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("step4_values.json")
GROUPS = ["lookup_m1", "lookup_m2+", "arith_m1", "arith_m2+"]
POP = {"lookup_m1": 212, "lookup_m2+": 367, "arith_m1": 71, "arith_m2+": 2235}
W = {g: n / sum(POP.values()) for g, n in POP.items()}


def jl(p):
    return [json.loads(l) for l in open(ROOT / p, encoding="utf-8") if l.strip()]


def p_exact(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def mc(a, b, qs):
    x = sum(1 for q in qs if a[q] and not b[q])
    y = sum(1 for q in qs if b[q] and not a[q])
    return {"b": x, "c": y, "p": p_exact(x, y)}


out = {}
# 1-2. HiTab 전달 셀 평균(단일 셀 조회 991), 표 단위 유형별
RH = "results/rerun_20260926/hitab"
single = [x["query_id"] for x in jl(f"{RH}/hitab_test_gold_s3c_type_accuracy.jsonl") if x["query_type"] == "single_cell"]
out["hitab_cells_delivered_single991"] = {}
for arm in ["s3c", "sleaf", "table", "chunk", "trag_hetero", "rowcol", "randrow", "tablerag_path", "tablerag_leaf"]:
    rec = {x["query_id"]: x for x in jl(f"{RH}/hitab_test_gold_{arm}_records.jsonl")}
    out["hitab_cells_delivered_single991"][arm] = round(sum(rec[q]["cells_in_context"] for q in single) / len(single), 2)
tbl = json.loads((ROOT / f"{RH}/hitab_test_gold_table.json").read_text())
out["hitab_table_unit_gold"] = {"type_accuracy": {t: tbl["type_accuracy"][t]["accuracy"] for t in ("single_cell", "multi_cell", "arithmetic")},
                                "cells_delivered_mean_all_scored": round(tbl["cells_delivered_mean"], 1)}
# 1. MultiHiertt 전달 셀 평균(2,885, 문서 안)
RM = "results/rerun_20260926/mh"
out["mh_cells_delivered_2885"] = {}
for arm in ["s3c", "sleaf", "chunk", "trag_hetero", "rowcol", "tablerag_path", "tablerag_leaf", "randrow"]:
    j = json.loads((ROOT / f"{RM}/mh_train_{arm}.json").read_text())
    out["mh_cells_delivered_2885"][arm] = j["by_layer"]["ALL"]["doc"]["cells_delivered_mean"]

# 3. MultiHiertt 882 가중 정확도
CAP = "results/mh_arms/cap300_20260924"
rows = {"cell": {x["query_id"]: x for x in jl(f"{CAP}/cell_uniq.jsonl")},
        "chunk_final": {x["query_id"]: x for x in jl("results/reader_format_20260927/test/mh_chunk_final.jsonl")},
        "chunk_v1": {x["query_id"]: x for x in jl(f"{CAP}/chunk.jsonl")}}
ids = sorted(rows["cell"])
assert all(sorted(v) == ids for v in rows.values())
grp = {g: [q for q in ids if rows["cell"][q]["layer"] == g] for g in GROUPS}
ok = {k: {q: v[q]["answer_correct"] for q in ids} for k, v in rows.items()}
rng = np.random.default_rng(0)


def weighted(d, sel=None):
    return sum(W[g] * np.mean([d[q] for q in (sel[g] if sel else grp[g])]) for g in GROUPS)


def boot(f):
    vals = np.empty(10_000)
    idx = {g: np.array(grp[g]) for g in GROUPS}
    for b in range(10_000):
        vals[b] = f({g: rng.choice(idx[g], len(idx[g])) for g in GROUPS})
    return [round(float(x), 4) for x in np.percentile(vals, [2.5, 97.5])]


out["mh882_weighted"] = {
    "note": "가중 = 최종 머리글 규칙 그룹 크기 212/367/71/2,235, 층화 부트스트랩 10,000회 seed 0(이 파일에서 새로 뽑음)",
    "chunk_final": {"weighted_em": round(float(weighted(ok["chunk_final"])), 4),
                    "ci95": boot(lambda s: weighted(ok["chunk_final"], s))},
    "cell_minus_chunk_final": {"weighted_diff": round(float(weighted(ok["cell"]) - weighted(ok["chunk_final"])), 4),
                               "ci95": boot(lambda s: weighted(ok["cell"], s) - weighted(ok["chunk_final"], s))},
    "groups_em_chunk_final": {g: round(float(np.mean([ok["chunk_final"][q] for q in grp[g]])), 4) for g in GROUPS}}

# 4. 답변 차이가 나온 문항 묶음
def split(a_ok, a_ret, b_ok, b_ret, qs):
    parts = {"both_retrieved": [q for q in qs if a_ret[q] and b_ret[q]],
             "only_ours_retrieved": [q for q in qs if a_ret[q] and not b_ret[q]],
             "only_other_retrieved": [q for q in qs if b_ret[q] and not a_ret[q]],
             "neither_retrieved": [q for q in qs if not a_ret[q] and not b_ret[q]]}
    res = {k: {"n": len(v), "ours_correct": sum(a_ok[q] for q in v), "other_correct": sum(b_ok[q] for q in v),
               "diff": sum(a_ok[q] for q in v) - sum(b_ok[q] for q in v)} for k, v in parts.items()}
    res["total_diff"] = sum(a_ok[q] for q in qs) - sum(b_ok[q] for q in qs)
    res["diff_outside_both"] = res["total_diff"] - res["both_retrieved"]["diff"]
    return res


s3 = {x["query_id"]: x for x in jl("results/s3c_answer_hitab300_20260926/rows.jsonl")}
ch = {x["query_id"]: x for x in jl("results/fair_filter_20260921/rows.jsonl") if x["arm"] == "chunk"}
h_ids = sorted(s3)
out["diff_split"] = {
    "hitab300_s3c_vs_chunk": split({q: s3[q]["correct_base"] for q in h_ids}, {q: s3[q]["retrieval_correct"] for q in h_ids},
                                   {q: ch[q]["correct_base"] for q in h_ids}, {q: ch[q]["retrieval_correct"] for q in h_ids}, h_ids),
    "mh882_final_vs_chunk_final": split(ok["cell"], {q: rows["cell"][q]["retrieval_correct"] for q in ids},
                                        ok["chunk_final"], {q: rows["chunk_final"][q]["retrieval_correct"] for q in ids}, ids)}
# 5. 고정 청크 머리글 규칙 효과
out["mh882_chunk_v1_vs_final"] = {"chunk_v1": sum(ok["chunk_v1"].values()), "chunk_final": sum(ok["chunk_final"].values()),
                                  "mcnemar_final_vs_v1": mc(ok["chunk_final"], ok["chunk_v1"], ids)}
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
