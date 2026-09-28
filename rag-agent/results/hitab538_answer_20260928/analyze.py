"""HiTab 538표 한 색인 조건 답변 300건 집계 (PREREG-2026-09-28-hitab538-answer.md, 실행 전 커밋).

실행:  PYTHONPATH=. .venv/bin/python results/hitab538_answer_20260928/analyze.py   (rag-agent/ 에서)
입력:  이 폴더의 s3c_rows.jsonl·chunk_rows.jsonl, 표 안 s3c 답변 results/s3c_answer_hitab300_20260926/rows.jsonl
출력:  이 폴더의 analyze.json
"""
import json
from pathlib import Path

from scripts.fair_filter_eval import mcnemar

HERE = Path(__file__).parent


def rows(path, arm):
    rs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return {r["query_id"]: r for r in rs if r["arm"] == arm}


s3c, chunk = rows(HERE / "s3c_rows.jsonl", "s3c"), rows(HERE / "chunk_rows.jsonl", "chunk")
intable = rows("results/s3c_answer_hitab300_20260926/rows.jsonl", "s3c")
pop = json.load(open("results/ksweep_population_300.json"))["query_ids"]
assert set(s3c) == set(chunk) == set(intable) == set(pop) and len(pop) == 300

ans = lambda d: {q: d[q]["correct_base"] for q in pop}
out = {"n": len(pop)}
for name, d in (("s3c_538", s3c), ("chunk_538", chunk), ("s3c_intable", intable)):
    out[name] = {"answer_correct": sum(ans(d).values()),
                 "answer": round(sum(ans(d).values()) / len(pop), 4),
                 "retrieval_hits": sum(d[q]["retrieval_correct"] for q in pop),
                 "reader_input_tokens_mean": round(sum(d[q]["reader_input_tokens"] for q in pop) / len(pop), 1)}
# 주 검정 (비교 1개): a = s3c 538, b = 청크 538
out["mcnemar_s3c_vs_chunk_538"] = mcnemar(ans(s3c), ans(chunk))
# 기술 통계 (검정 없음): 표 안 s3c 237 과의 차이와 불일치 수
out["s3c_538_minus_intable"] = out["s3c_538"]["answer_correct"] - out["s3c_intable"]["answer_correct"]
pair = mcnemar(ans(s3c), ans(intable))
out["s3c_538_vs_intable_counts"] = {"only_538": pair["a_only"], "only_intable": pair["b_only"]}

(HERE / "analyze.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
