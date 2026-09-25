"""GATE 0-4: CLAUDE.md §1 '중앙 1,561토큰, 86%가 512 초과' 재계산 (커밋 97b9bbc 시점 records 파일).
+ 0-1 보조: HiTab test 538표 raw texts 에 들여쓰기(앞 공백) 셀이 있는지. 추론 없음."""
import json, statistics as st, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
blob = subprocess.run(["git", "-C", str(ROOT), "show",
                       "97b9bbc:rag-agent/results/rhb_dense_512_s3_casefix_records.jsonl"],
                      capture_output=True, text=True, check=True).stdout
rows = [json.loads(l) for l in blob.splitlines()]
q = [r["gold_table_tokens"] for r in rows]
per_table = {r["gold_table"]: r["gold_table_tokens"] for r in rows}
t = list(per_table.values())
ids = {json.loads(l)["table_id"] for l in open(ROOT / "data/hitab/data/test_samples.jsonl")}
cells = lead = 0
for tid in ids:
    for row in json.load(open(ROOT / f"data/hitab/data/tables/raw/{tid}.json"))["texts"]:
        for v in row:
            cells += 1
            lead += bool(str(v)) and str(v)[0].isspace()
print(json.dumps({
    "source": "97b9bbc:rag-agent/results/rhb_dense_512_s3_casefix_records.jsonl (field gold_table_tokens)",
    "per_query": {"query_count": len(q), "median": st.median(q), "max": max(q),
                  "n_over_512": sum(x > 512 for x in q), "ratio_over_512": round(sum(x > 512 for x in q) / len(q), 4)},
    "per_distinct_table": {"n_tables": len(t), "median": st.median(t), "max": max(t),
                           "n_over_512": sum(x > 512 for x in t), "ratio_over_512": round(sum(x > 512 for x in t) / len(t), 4)},
    "hitab_test_raw_texts": {"tables": len(ids), "cells": cells, "cells_with_leading_whitespace": lead},
}, indent=1))
