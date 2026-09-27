"""논문 5단계(원고 점검 반영, 2026-09-28)에 쓰는 값 중 기존 결과 파일에 없는 것 — 기존 기록에서 다시 센다. 새 검색·생성 없음.

실행:  cd rag-agent && .venv/bin/python results/thesis_fix_20260928/values.py  -> values.json
1. HiTab 538개 표 한 색인(단일 셀 조회 991건): 방법별 전달 셀 평균(재실행 레코드의 cells_in_context).
2. MultiHiertt 882건 답변:
   a. 산술·셀 2개 이상에서 두 방법(최종 대 최종) 모두 검색 성공한 문항의 답변 정확도.
   b. 고정 청크 처음 규칙 대 최종 규칙: 문맥(context_sha256)·검색 성공 여부가 다른 문항 수.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("values.json")


def jl(p):
    return [json.loads(l) for l in open(ROOT / p, encoding="utf-8") if l.strip()]


out = {}
# 1. HiTab 538개 표 한 색인 전달 셀 평균
RH = "results/rerun_20260926/hitab"
single = [x["query_id"] for x in jl(f"{RH}/hitab_test_split_s3c_type_accuracy.jsonl") if x["query_type"] == "single_cell"]
assert len(single) == 991
out["hitab_split_cells_delivered_991"] = {}
for arm in ["s3c", "sleaf", "table", "row", "chunk", "trag_hetero", "tablerag_leaf", "tablerag_path", "rowcol"]:
    rec = {x["query_id"]: x for x in jl(f"{RH}/hitab_test_split_{arm}_records.jsonl")}
    out["hitab_split_cells_delivered_991"][arm] = round(sum(rec[q]["cells_in_context"] for q in single) / len(single), 4)

# 2. MultiHiertt 882
CAP = "results/mh_arms/cap300_20260924"
rows = {"cell_uniq": {x["query_id"]: x for x in jl(f"{CAP}/cell_uniq.jsonl")},
        "chunk_v1": {x["query_id"]: x for x in jl(f"{CAP}/chunk.jsonl")},
        "chunk_final": {x["query_id"]: x for x in jl("results/reader_format_20260927/test/mh_chunk_final.jsonl")}}
ids = sorted(rows["cell_uniq"])
assert all(sorted(v) == ids for v in rows.values())
ok = {k: {q: v[q]["answer_correct"] for q in ids} for k, v in rows.items()}
ret = {k: {q: v[q]["retrieval_correct"] for q in ids} for k, v in rows.items()}
lay = {q: rows["cell_uniq"][q]["layer"] for q in ids}

# a
both = [q for q in ids if lay[q] == "arith_m2+" and ret["cell_uniq"][q] and ret["chunk_final"][q]]
out["arith_m2plus_both_retrieved_final"] = {
    "n": len(both), "cell_correct": sum(ok["cell_uniq"][q] for q in both), "chunk_correct": sum(ok["chunk_final"][q] for q in both),
    "cell_acc": round(sum(ok["cell_uniq"][q] for q in both) / len(both), 4),
    "chunk_acc": round(sum(ok["chunk_final"][q] for q in both) / len(both), 4)}
# b
cv, cf = rows["chunk_v1"], rows["chunk_final"]
rd = [q for q in ids if ret["chunk_v1"][q] != ret["chunk_final"][q]]
out["chunk_v1_vs_final_context"] = {
    "context_differs": sum(cv[q]["context_sha256"] != cf[q]["context_sha256"] for q in ids),
    "retrieval_differs": len(rd), "retrieval_final_only": sum(ret["chunk_final"][q] for q in rd),
    "retrieval_v1_only": sum(ret["chunk_v1"][q] for q in rd),
    "retrieval_success_v1": sum(ret["chunk_v1"].values()), "retrieval_success_final": sum(ret["chunk_final"].values())}

OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
