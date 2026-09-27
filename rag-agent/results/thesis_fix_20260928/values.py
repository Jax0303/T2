"""논문 5단계(원고 점검 반영, 2026-09-28)에 쓰는 값 중 기존 결과 파일에 없는 것 — 기존 기록에서 다시 센다. 새 검색·생성 없음.

실행:  cd rag-agent && HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python results/thesis_fix_20260928/values.py  -> values.json
1. HiTab 538개 표 한 색인(단일 셀 조회 991건): 방법별 전달 셀 평균(재실행 레코드의 cells_in_context).
2. MultiHiertt 882건 답변:
   a. 조회 두 그룹, 셀 문장(최종) 대 고정 청크(최종) McNemar.
   b. 산술·셀 2개 이상에서 두 방법(최종 대 최종) 모두 검색 성공한 문항의 답변 정확도.
   c. 고정 청크 처음 규칙 대 최종 규칙: 문맥(context_sha256)·검색 성공 여부가 다른 문항 수.
   d. 처음 규칙(v1) 셀 문장이 같은 90건 재현: 조회 검색 성공 459건 중 정답 셀 문장(값 제외)이 문맥 속 다른 문장(값 제외)과
      같은 문항. 처음 규칙 검색 레코드의 문맥 + 처음 규칙으로 다시 만든 정답 셀 문장. 그 문항들에서 조건별 정확도.
      ※ 재현되지 않음: 기록(PREREG-2026-09-24-mh-answer-cap300.md:116)은 90건·5:17 이나, 이 정의로는 110건·6:28 이다
        (값까지 같은 문장 제외·정답 셀 전부 조건 등 변형도 101~110건, 5~6:28). 원 분석 코드와 90건 목록은 저장되지 않았다.
        논문은 기록된 값을 쓰고, 이 항목은 논문에 쓰지 않는다.
   e. 처음 규칙 조회 두 그룹의 순손실(셀 문장 대 고정 청크, 둘 다 처음 규칙) 분해.
"""
import json
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
OUT = Path(__file__).with_name("values.json")


def jl(p):
    return [json.loads(l) for l in open(ROOT / p, encoding="utf-8") if l.strip()]


def p_exact(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def mc(a, b, qs):
    x = sum(1 for q in qs if a[q] and not b[q])
    y = sum(1 for q in qs if b[q] and not a[q])
    return {"n": len(qs), "b": x, "c": y, "p": p_exact(x, y)}


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
        "cell_hv33": {x["query_id"]: x for x in jl(f"{CAP}/cell_hv33r.jsonl")},
        "cell_v1": {x["query_id"]: x for x in jl(f"{CAP}/cell.jsonl")},
        "chunk_v1": {x["query_id"]: x for x in jl(f"{CAP}/chunk.jsonl")},
        "chunk_final": {x["query_id"]: x for x in jl("results/reader_format_20260927/test/mh_chunk_final.jsonl")}}
ids = sorted(rows["cell_uniq"])
assert all(sorted(v) == ids for v in rows.values())
ok = {k: {q: v[q]["answer_correct"] for q in ids} for k, v in rows.items()}
ret = {k: {q: v[q]["retrieval_correct"] for q in ids} for k, v in rows.items()}
lay = {q: rows["cell_uniq"][q]["layer"] for q in ids}
look = [q for q in ids if lay[q].startswith("lookup")]

# a
out["lookup_final_vs_chunk_final"] = mc(ok["cell_uniq"], ok["chunk_final"], look)
# b
both = [q for q in ids if lay[q] == "arith_m2+" and ret["cell_uniq"][q] and ret["chunk_final"][q]]
out["arith_m2plus_both_retrieved_final"] = {
    "n": len(both), "cell_correct": sum(ok["cell_uniq"][q] for q in both), "chunk_correct": sum(ok["chunk_final"][q] for q in both),
    "cell_acc": round(sum(ok["cell_uniq"][q] for q in both) / len(both), 4),
    "chunk_acc": round(sum(ok["chunk_final"][q] for q in both) / len(both), 4)}
# c
cv, cf = rows["chunk_v1"], rows["chunk_final"]
rd = [q for q in ids if ret["chunk_v1"][q] != ret["chunk_final"][q]]
out["chunk_v1_vs_final_context"] = {
    "context_differs": sum(cv[q]["context_sha256"] != cf[q]["context_sha256"] for q in ids),
    "retrieval_differs": len(rd), "retrieval_final_only": sum(ret["chunk_final"][q] for q in rd),
    "retrieval_v1_only": sum(ret["chunk_v1"][q] for q in rd),
    "retrieval_success_v1": sum(ret["chunk_v1"].values()), "retrieval_success_final": sum(ret["chunk_final"].values())}

# d. 처음 규칙 셀 문장이 같은 90건
from mh_arms import build_tables, load_population, resolve_gold           # noqa: E402
from retrieval_accuracy import build_corpus                               # noqa: E402

hit = [q for q in look if ret["cell_v1"][q]]
queries, docs, _ = load_population("train")
queries = [q for q in queries if q["uid"] in set(hit)]
docs = {u: docs[u] for u in {q["uid"] for q in queries}}
tables, hdr = build_tables(docs, "v1", "none")
texts, covers, _, _, _ = build_corpus("", sorted(tables), "s3c", "cell", {}, 1000, "leaf", "sentence", 200, None,
                                      load=lambda tid, _d: tables.get(tid), trag_dtype="infer")
text_of = {c: t for t, cs in zip(texts, covers) for c in cs}
live = {(tid, i, j) for tid, tab in tables.items() for i, row in enumerate(tab.table.data) for j, v in enumerate(row) if str(v).strip()}
queries = {q["uid"]: q for q in resolve_gold(queries, tables, hdr, live)}
ctx = {x["query_id"]: x["doc"]["context"] for x in jl("results/mh_arms/mh_train_cell_hv1_none_doc_records.jsonl") if x["query_id"] in set(hit)}
strip = lambda s: s.rsplit(": ", 1)[0]


def same_sentence(q):
    c = ctx[q]
    for g in queries[q]["gold"]:
        t = text_of[g]
        assert t in c, q
        if sum(strip(s) == strip(t) for s in c) > 1:
            return True
    return False


dup = [q for q in hit if same_sentence(q)]
rest = [q for q in hit if q not in set(dup)]
acc = lambda k, qs: round(sum(ok[k][q] for q in qs) / len(qs), 4)
out["v1_same_sentence"] = {
    "lookup_hit_v1": len(hit), "n": len(dup), "n_rest": len(rest),
    "acc_on_dup": {k: acc(k, dup) for k in ("cell_v1", "chunk_v1", "cell_hv33", "cell_uniq", "chunk_final")},
    "acc_on_rest": {k: acc(k, rest) for k in ("cell_v1", "chunk_v1")},
    "cell_v1_vs_chunk_v1_dup": mc(ok["cell_v1"], ok["chunk_v1"], dup),
    "cell_v1_vs_chunk_v1_rest": mc(ok["cell_v1"], ok["chunk_v1"], rest),
    "cell_uniq_vs_chunk_final_dup": mc(ok["cell_uniq"], ok["chunk_final"], dup)}
# e. 처음 규칙 조회 두 그룹 순손실 분해
miss = [q for q in look if not ret["cell_v1"][q]]
parts = {"all_lookup": look, "same_sentence": dup, "rest_hit": rest, "v1_retrieval_miss": miss}
out["v1_lookup_net_loss"] = {k: {**mc(ok["cell_v1"], ok["chunk_v1"], qs), "net_loss": mc(ok["cell_v1"], ok["chunk_v1"], qs)["c"] - mc(ok["cell_v1"], ok["chunk_v1"], qs)["b"]}
                             for k, qs in parts.items()}
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
