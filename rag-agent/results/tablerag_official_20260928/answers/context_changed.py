"""2026-09-28 답변 표본에서 TableRAG(Chen) 재구현 문맥이 공식 규칙으로 바뀐 문항 수. 모델·검색 실행 없음(기존 records 비교).
HiTab 300(results/ksweep_population_300.json): 답변에 쓴 문맥 = results/retrieval_accuracy/t_tablerag_{arm}_v2_gold_records.jsonl
  (PREREG-2026-09-21-fair-llm-filtering-hitab.md), 공식 규칙 = results/tablerag_official_20260928/hitab/hitab_test_gold_tablerag_{arm}_official_records.jsonl,
  참고로 원고 검색 표의 원고 규칙 재실행 = results/rerun_20260926/hitab/hitab_test_gold_tablerag_{arm}_records.jsonl.
MultiHiertt 882(results/mh_arms/sample_cap300_seed20260913.json): 답변에 쓴 문맥 = 처음 규칙 results/mh_arms/mh_train_tablerag_{arm}_hv1_none_doc_records.jsonl,
  머리글 고친 규칙 ..._hv3.3_none_doc_records.jsonl, 공식 규칙(최종 머리글) = results/tablerag_official_20260928/mh/mh_train_tablerag_{arm}_official_records.jsonl,
  원고 규칙(최종 머리글) = results/rerun_20260926/mh/mh_train_tablerag_{arm}_records.jsonl.
문맥 비교 = 리더에 들어가는 문자열 목록(HiTab: context, MultiHiertt: doc.context)이 같은가.
실행: .venv/bin/python results/tablerag_official_20260928/answers/context_changed.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
jl = lambda p: {r["query_id"]: r for r in map(json.loads, open(ROOT / p)) if "excluded" not in r}


def ids(p):
    d = json.load(open(ROOT / p))
    return set(d["query_ids"]) if "query_ids" in d else {q for v in d["by_layer"].values() for q in v}


out = {"hitab_300": {}, "mh_882": {}}
H = ids("results/ksweep_population_300.json")
for arm in ("leaf", "path"):
    used = jl(f"results/retrieval_accuracy/t_tablerag_{arm}_v2_gold_records.jsonl")
    off = jl(f"results/tablerag_official_20260928/hitab/hitab_test_gold_tablerag_{arm}_official_records.jsonl")
    inf = jl(f"results/rerun_20260926/hitab/hitab_test_gold_tablerag_{arm}_records.jsonl")
    assert H <= set(used) and H <= set(off) and H <= set(inf)
    out["hitab_300"][arm] = {"n": len(H),
                             "answer_context_vs_official_changed": sum(used[q]["context"] != off[q]["context"] for q in H),
                             "answer_context_vs_rerun_infer_changed": sum(used[q]["context"] != inf[q]["context"] for q in H),
                             "rerun_infer_vs_official_changed": sum(inf[q]["context"] != off[q]["context"] for q in H)}
M = ids("results/mh_arms/sample_cap300_seed20260913.json")
for arm in ("leaf", "path"):
    off = jl(f"results/tablerag_official_20260928/mh/mh_train_tablerag_{arm}_official_records.jsonl")
    inf = jl(f"results/rerun_20260926/mh/mh_train_tablerag_{arm}_records.jsonl")
    v1 = jl(f"results/mh_arms/mh_train_tablerag_{arm}_hv1_none_doc_records.jsonl")
    v33 = jl(f"results/mh_arms/mh_train_tablerag_{arm}_hv3.3_none_doc_records.jsonl")
    c = lambda d, q: d[q]["doc"]["context"] if q in d else None
    out["mh_882"][arm] = {"n": len(M), "missing": {k: len(M - set(d)) for k, d in (("official", off), ("infer_final", inf), ("v1", v1), ("v33", v33))},
                          "v1_answer_context_vs_official_changed": sum(c(v1, q) != c(off, q) for q in M),
                          "v33_answer_context_vs_official_changed": sum(c(v33, q) != c(off, q) for q in M),
                          "infer_final_vs_official_changed": sum(c(inf, q) != c(off, q) for q in M)}
(Path(__file__).parent / "context_changed.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
