"""GATE 0-1: README .7074(s3c_v2) 의 검색 범위를 records 로 확인. 추론 없음, 파일 읽기만."""
import collections, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REC = ROOT / "results/evaluation_v2/s3c_v2_records.jsonl"
ANS = ROOT / "results/evaluation_v2/s3c_v2_answer_retrieved.jsonl"

recs = [json.loads(l) for l in REC.open()]
scored = [r for r in recs if "correct" in r]
primary = [r for r in scored if r.get("mode") == "all" and r.get("m") == 1
           and (r.get("aggregation") or "none") == "none"]
ans = {json.loads(l)["query_id"]: json.loads(l) for l in ANS.open()}

def foreign(r):  # 문맥 셀 중 질의의 표가 아닌 표의 셀 개수
    return sum(1 for c in r["context_cells"] if str(c[0]) != str(r["table_id"]))

out = {"records": str(REC.relative_to(ROOT)),
       "records_sha256": hashlib.sha256(REC.read_bytes()).hexdigest(),
       "answers": str(ANS.relative_to(ROOT))}
for name, pop in (("scored_1581", scored), ("primary_991", primary)):
    f = [foreign(r) for r in pop]
    tabs = [len({str(c[0]) for c in r["context_cells"]}) for r in pop]
    ok = [ans[r["query_id"]] for r in pop]
    out[name] = {
        "query_count": len(pop),
        "queries_with_foreign_table_cells": sum(x > 0 for x in f),
        "ratio": round(sum(x > 0 for x in f) / len(pop), 4),
        "foreign_cells_total": sum(f),
        "context_cells_total": sum(len(r["context_cells"]) for r in pop),
        "queries_all_20_foreign": sum(x == len(r["context_cells"]) for x, r in zip(f, pop)),
        "distinct_tables_in_context_hist": dict(sorted(collections.Counter(tabs).items())),
        "gold_table_in_context_0": sum(r["gold_table_in_context"] == 0 for r in pop),
        "retrieval_correct": sum(r["correct"] for r in pop),
        "answer_correct": sum(a["answer_correct"] for a in ok),
        "answer_context_sha_matches_records": sum(a.get("context_sha256") == r["context_sha256"]
                                                 for a, r in zip(ok, pop)),
    }
print(json.dumps(out, indent=1, ensure_ascii=False))
