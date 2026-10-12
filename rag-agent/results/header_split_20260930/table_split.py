"""표 단위 머리글 분리 비율(2026-10-06, 사용자 질문 "전체 표 중 몇 %") — header_split.py 와 같은 대리 기준
(표의 첫 데이터 행을 담지 않은 청크 = 열 머리글 줄 없음). LangChain 청크 기록 2,885문항의 문서 2,146개(중복 제거)·표 8,901개."""
import json, statistics
from collections import defaultdict
from pathlib import Path
REC = Path(__file__).resolve().parents[2] / "results/mh_only_20260928/baselines/langchain_chunk/langchain_chunk_records.jsonl"
parse = lambda c: tuple(int(v) for v in c.split("-"))
seen = set(); n_tab = n_split = n_cell = n_bad = 0; sizes = []
for l in open(REC, encoding="utf-8"):
    r = json.loads(l)
    key = tuple(tuple(sorted(u["cells"])) for u in r["ranked_units"])
    if key in seen: continue
    seen.add(key)
    chunks = [set(u["cells"]) for u in r["ranked_units"]]
    allc = set().union(*chunks); rows = defaultdict(set)
    for c in allc: rows[parse(c)[0]].add(parse(c)[1])
    for t, v in rows.items():
        first = {c for c in allc if parse(c)[0] == t and parse(c)[1] == min(v)}
        tcells = {c for c in allc if parse(c)[0] == t}
        bad = {c for c in tcells if not any(ch & first for ch in chunks if c in ch)}
        n_tab += 1; n_split += bool(bad); n_cell += len(tcells); n_bad += len(bad); sizes.append((len(tcells), bool(bad)))
out = {"n_docs": len(seen), "n_tables": n_tab, "tables_split": n_split, "tables_split_pct": round(100 * n_split / n_tab, 1),
       "n_cells": n_cell, "cells_only_in_headerless_chunks": n_bad, "cells_pct": round(100 * n_bad / n_cell, 1),
       "cells_per_table_median": statistics.median(s for s, _ in sizes),
       "by_table_size": {f"{lo}-{hi}": {"tables": len(g), "split": sum(g), "pct": round(100 * sum(g) / max(len(g), 1), 1)}
                         for lo, hi in ((1, 20), (21, 50), (51, 100), (101, 10 ** 9)) for g in [[b for s, b in sizes if lo <= s <= hi]]}}
Path(__file__).with_suffix(".json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False))
