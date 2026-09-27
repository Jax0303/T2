#!/usr/bin/env python3
"""정답 셀의 table_description 유무와 문서 안 같은 문장 비율 (실행 없음 — 임베딩·모델을 부르지 않는다). 2026-09-28.

사전등록(PREREG-2026-09-28-table-description.md) 머리말의 수치를 세션 임시 스크립트에서 이 파일로 옮겨 다시 센다.
모집단·색인은 inspect_desc.py 와 같다(train 표 근거 질의, 머리글 최종 규칙 v3.3u, s3c 셀 문장).

  PYTHONPATH=. .venv/bin/python results/table_description_20260928/gold_dup.py
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from mh_arms import build_tables, load_population, resolve_gold      # noqa: E402

_DESC = re.compile(r"^(Table \d+ shows .*) is (.*?)\s*\.?\s*$", re.S)
_NUM = re.compile(r"^[\s$€£(\-–—−]*[\d.,]+\s*%?\s*\)?$")

queries, docs, _ = load_population("train")
tables, hdr = build_tables(docs, "v3.3u")
live = {(tid, i, j) for tid, tab in tables.items()
        for i, row in enumerate(tab.table.data) for j, v in enumerate(row) if str(v).strip()}
queries = resolve_gold(queries, tables, hdr, live)
desc = {u: json.loads(d[1]) if isinstance(d[1], str) else d[1] for u, d in docs.items()}
key = lambda tid, i, j: f"{tid.split('::')[1]}-{i + hdr[tid][0]}-{j + hdr[tid][1]}"

scored = [q for q in queries if not q["excluded"] and q["gold"]]
gold = [(q["uid"], c) for q in scored for c in q["gold"]]
missing = Counter()
per_doc = {}
for tid, i, j in live:
    uid = tid.split("::")[0]
    s = desc[uid].get(key(tid, i, j))
    if s is None:
        v = tables[tid].table.data[i][j].strip()
        missing["numeric" if _NUM.match(v) else "dash" if re.fullmatch(r"[—–\-−]+", v) else "text"] += 1
        continue
    m = _DESC.match(s)
    per_doc.setdefault(uid, Counter())[m.group(1) if m else s] += 1
n_desc = sum(sum(c.values()) for c in per_doc.values())
ours = {}
for tid, i, j in live:
    t = tables[tid].table
    ours.setdefault(tid.split("::")[0], Counter())[tuple(t.row_path(i) + t.col_path(j))] += 1
ours_dup = sum(v for c in ours.values() for v in c.values() if v > 1)
dup = sum(v for c in per_doc.values() for v in c.values() if v > 1)
out = {
    "scored_queries": len(scored),
    "gold_cells": len(gold),
    "gold_cells_without_table_description": sum(1 for u, c in gold if key(*c) not in desc[u]),
    "index_cells": len(live),
    "index_cells_without_table_description": sum(missing.values()),
    "without_table_description_by_value": dict(missing),
    "index_cells_with_table_description": n_desc,
    "same_sentence_without_value_in_doc": {
        "table_description_cells": dup,
        "table_description_rate": round(dup / n_desc, 4),
        "definition": "table_description 이 있는 색인 셀 중, 끝의 ' is <값> .' 를 뗀 문장이 같은 문서의 다른 셀과 같은 셀",
        "ours_cells": ours_dup, "ours_rate": round(ours_dup / len(live), 4),
        "ours_definition": "본 방법 색인 셀 중 행 경로 + 열 경로(값을 뺀 s3c 문장)가 같은 문서의 다른 셀과 같은 셀"},
}
(Path(__file__).resolve().parent / "gold_dup.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps(out, indent=1, ensure_ascii=False))
