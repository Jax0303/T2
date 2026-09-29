"""원고 3.1절(서론·결론) 청크 머리글 분리 비율 재계산 — 2026-09-29 첫 계산의 코드가 저장되지 않아 같은 규칙으로 다시 씀.

입력: LangChain 튜토리얼 청크(1,000자, 겹침 200자) 검색 기록 2,885문항. 문서의 모든 청크와 각 청크가 통째로 담은 셀 좌표가 들어 있다.
대리 기준: 표의 첫 데이터 행(그 표에서 데이터 셀이 있는 가장 위 행)을 담지 않은 청크에는 열 머리글 줄이 없다고 본다
(마크다운으로 펼친 표에서 열 머리글 줄은 첫 데이터 행 바로 앞까지이므로). 정답 셀을 담은 청크가 모두 이런 청크이면 그 셀은 '머리글 없는 청크에만' 있다.
'첫 데이터 행을 담는다'를 그 행 셀 하나 이상/전부로 두 가지로 셌다 — 결과가 같다. 새 실행 없음.
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REC = ROOT / "results/mh_only_20260928/baselines/langchain_chunk/langchain_chunk_records.jsonl"
parse = lambda c: tuple(int(v) for v in c.split("-"))          # "표-행-열"
R = [json.loads(l) for l in open(REC, encoding="utf-8")]
out = {"n_queries": len(R), "n_gold_cells": sum(len(r["gold_ids"]) for r in R)}
for variant in ("any", "all"):
    n_cells = n_q = 0
    for r in R:
        chunks = [set(u["cells"]) for u in r["ranked_units"]]
        assert len(chunks) == r["n_chunks_in_doc"]                  # 문서의 청크 전부가 기록돼 있다
        allc = set().union(*chunks)
        rows = defaultdict(set)
        for c in allc:
            rows[parse(c)[0]].add(parse(c)[1])
        first = {t: {c for c in allc if parse(c)[0] == t and parse(c)[1] == min(v)} for t, v in rows.items()}
        has_first = lambda ch, t: bool(ch & first[t]) if variant == "any" else first[t] <= ch
        bad = sum(not any(has_first(ch, parse(g)[0]) for ch in chunks if g in ch) for g in r["gold_ids"])
        n_cells += bad; n_q += bad > 0
    out[variant] = {"cells": n_cells, "cells_pct": round(100 * n_cells / out["n_gold_cells"], 1),
                    "queries": n_q, "queries_pct": round(100 * n_q / len(R), 1)}
(Path(__file__).parent / "header_split.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False))
