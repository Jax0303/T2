"""교집합 비교(2026-09-27): 두 조건이 모두 검색에 성공한 문항만 모아 답변 정답률 비교. 새 실행 없음.

실행:  cd rag-agent && .venv/bin/python results/answer_decomp_20260927/intersection.py  -> intersection.json
- 입력은 answer_decomp.py 와 같은 답변 행(retrieval_correct, 답변 정답).
- McNemar = 정확 이항(scipy binomtest), b = 본 방법만 맞힘, c = 비교군만 맞힘 (교집합 안에서).
- MultiHiertt 그룹별 값은 탐색적 결과(사전등록 없음).
"""
import json
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
GROUPS = ["lookup_m1", "lookup_m2+", "arith_m1", "arith_m2+"]


def jl(p):
    return {x["query_id"]: x for x in (json.loads(l) for l in open(p, encoding="utf-8") if l.strip())}


def pair(a, b, qs):
    """a, b: qid -> (retrieval_correct, answer_correct)."""
    both = [q for q in qs if a[q][0] and b[q][0]]
    x = sum(1 for q in both if a[q][1] and not b[q][1])
    y = sum(1 for q in both if b[q][1] and not a[q][1])
    only_a = [q for q in qs if a[q][0] and not b[q][0]]
    only_b = [q for q in qs if b[q][0] and not a[q][0]]
    acc = lambda d, s: {"k": sum(d[q][1] for q in s), "n": len(s), "rate": round(sum(d[q][1] for q in s) / len(s), 4) if s else None}
    return {"n_questions": len(qs), "both_retrieved": len(both),
            "ours_answer_on_both": acc(a, both), "other_answer_on_both": acc(b, both),
            "mcnemar_on_both": {"b_ours_only": x, "c_other_only": y, "p": float(binomtest(x, x + y).pvalue) if x + y else 1.0},
            "only_ours_retrieved": len(only_a), "only_other_retrieved": len(only_b),
            "ours_answer_on_only_ours_retrieved": acc(a, only_a)}


out = {}
# HiTab 300: s3c 대 고정 청크
s3 = {q: (r["retrieval_correct"], r["correct_base"]) for q, r in jl(ROOT / "results/s3c_answer_hitab300_20260926/rows.jsonl").items()}
ff = {}
for q, r in ((x["query_id"], x) for x in map(json.loads, open(ROOT / "results/fair_filter_20260921/rows.jsonl", encoding="utf-8"))):
    if r["arm"] == "chunk":
        ff[q] = (r["retrieval_correct"], r["correct_base"])
assert set(s3) == set(ff) and len(s3) == 300
out["hitab_300"] = {"s3c_vs_chunk": pair(s3, ff, sorted(s3))}

# MultiHiertt 882
CAP = ROOT / "results/mh_arms/cap300_20260924"
load = lambda n: {q: (r["retrieval_correct"], r["answer_correct"]) for q, r in jl(CAP / f"{n}.jsonl").items()}
layer = {q: r["layer"] for q, r in jl(CAP / "cell.jsonl").items()}
ids = sorted(layer)
PAIRS = [("cell", "chunk"), ("cell", "trag_hetero"), ("cell_uniq", "chunk")]
m = {"note": "그룹별 값은 탐색적 결과(사전등록 없음)", "context_header_rule": {}}
for a, b in PAIRS:
    A, B = load(a), load(b)
    assert set(A) == set(B) == set(ids) and len(ids) == 882
    m[f"{a}_vs_{b}"] = {"ALL": pair(A, B, ids)} | {g: pair(A, B, [q for q in ids if layer[q] == g]) for g in GROUPS}
    for n in (a, b):
        m["context_header_rule"][n] = json.loads((CAP / f"{n}.run.json").read_text())["header_rule"]
out["mh_882"] = m
Path(__file__).with_name("intersection.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
