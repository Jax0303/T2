"""표 단위 검색(538표 한 색인, 절단 허용) 대 s3c_v2(셀 단위, 같은 색인 범위): 단일 셀 조회 991문항의
'문맥에 정답 표 포함'(gold_table_in_context) 짝 비교. 결과 파일 읽기 + 정확 McNemar 만."""
import json
from pathlib import Path
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
def load(p):
    rs = [json.loads(l) for l in open(ROOT / p, encoding="utf-8")]
    return {r["query_id"]: r for r in rs if "correct" in r and r.get("mode") == "all" and r.get("m") == 1
            and (r.get("aggregation") or "none") == "none"}
A = load("results/problem_def_audit_20260925/table_split_truncate_records.jsonl")
B = load("results/evaluation_v2/s3c_v2_records.jsonl")
assert set(A) == set(B) and len(A) == 991
s = json.load(open(ROOT / "results/problem_def_audit_20260925/table_split_truncate.json"))
b = sum(A[q]["gold_table_in_context"] and not B[q]["gold_table_in_context"] for q in A)
c = sum(B[q]["gold_table_in_context"] and not A[q]["gold_table_in_context"] for q in A)
tabs = [len({x[0] for x in A[q]["context_cells"]}) for q in A] if "context_cells" in next(iter(A.values())) else None
print(json.dumps({
    "encoder_overflow": {k: s["embedding_input_audit"]["documents"][k] for k in ("n", "n_overflow", "overflow_ratio", "max_tokens", "max_seq_length", "policy")},
    "table_unit": {"file": "results/problem_def_audit_20260925/table_split_truncate_records.jsonl",
                   "gold_table_in_context": sum(A[q]["gold_table_in_context"] for q in A),
                   "cell_retrieval_correct": sum(A[q]["correct"] for q in A),
                   "tables_in_context_hist": None if tabs is None else {k: tabs.count(k) for k in sorted(set(tabs))}},
    "s3c_v2": {"file": "results/evaluation_v2/s3c_v2_records.jsonl",
               "gold_table_in_context": sum(B[q]["gold_table_in_context"] for q in B),
               "cell_retrieval_correct": sum(B[q]["correct"] for q in B)},
    "query_count": len(A),
    "table_unit_only": b, "s3c_v2_only": c,
    "mcnemar_exact_p": binomtest(b, b + c).pvalue if b + c else 1.0,
}, indent=1, ensure_ascii=False))
