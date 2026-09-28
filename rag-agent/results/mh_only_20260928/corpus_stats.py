"""GATE 2 점검용 코퍼스 통계 (리더·임베딩 없음, CPU): 문서당 청크 수 분포, 청크 색인에 본문 단락 포함 여부, 청크 경계에서 잘린 셀 비율,
문서당 행+열 수 분포, 문서당 표 수·셀 수. 대상 = 2,885 채점 문항의 문서.

실행:  PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python results/mh_only_20260928/corpus_stats.py
"""
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts")
import mh_arms as ma  # noqa: E402
from retrieval_accuracy import build_corpus  # noqa: E402

HERE = Path(__file__).parent
scored = {r["query_id"] for r in map(json.loads, open("results/fair_mh_20260928/retrieval/mh_train_s3c_records.jsonl")) if "excluded" not in r}
queries, docs, _ = ma.load_population("train")
tables, hdr = ma.build_tables(docs, "v3.3u", "none")
uids = sorted(scored)
by_doc = {u: sorted(t for t in tables if t.startswith(u + "::")) for u in uids}
live = {tid: {(tid, i, j) for i, row in enumerate(tables[tid].table.data) for j, v in enumerate(row) if str(v).strip()} for tid in tables}


def dist(xs):
    return {"mean": round(statistics.mean(xs), 2), "median": statistics.median(xs), "max": max(xs), "min": min(xs),
            "<=20 비율": round(sum(x <= 20 for x in xs) / len(xs), 4), "<=5 비율": round(sum(x <= 5 for x in xs) / len(xs), 4)}


out = {"n_docs": len(uids), "n_tables_per_doc": dist([len(by_doc[u]) for u in uids]),
       "n_cells_per_doc": dist([sum(len(live[t]) for t in by_doc[u]) for u in uids])}
for unit, tag in (("chunk", "1,000자 청크"), ("trag_hetero", "TableRAG(Yu) 청킹"), ("row", "행 단위"), ("table", "표 단위")):
    texts, covers, *_ = build_corpus("", sorted(tables), "s3c", unit, {}, 1000, "leaf", "values", 200, None, load=lambda tid, _d: tables.get(tid))
    per_doc = Counter(); covered = {}
    for t, cv in zip(texts, covers):
        tid = next(iter(cv))[0] if cv else None
        if tid is None:
            continue
        per_doc[tid.split("::")[0]] += 1
        covered.setdefault(tid, set()).update(cv)
    n_units = [per_doc[u] for u in uids]
    cells_all = sum(len(live[t]) for u in uids for t in by_doc[u])
    cells_cov = sum(len(live[t] & covered.get(t, set())) for u in uids for t in by_doc[u])
    out[unit] = {"name": tag, "문서당 단위 수": dist(n_units), "색인 단위 총수": len(texts),
                 "본문 단락 포함": "아니오 — 색인은 표 markdown 만(build_corpus 는 tables 만 받음)",
                 "셀 커버율(어느 단위에도 온전히 담기지 않은 셀 = 경계에서 잘림)": {"cells_total": cells_all, "cells_covered": cells_cov, "uncovered_ratio": round(1 - cells_cov / cells_all, 4)},
                 "단위 예시(첫 단위 앞 200자)": texts[0][:200]}
rows_cols = []
for u in uids:
    n = 0
    for t in by_doc[u]:
        tb = tables[t].table; n += tb.n_rows + tb.n_cols
    rows_cols.append(n)
out["rowcol"] = {"name": "행·열 단위", "문서당 행+열 수": dist(rows_cols)}
(HERE / "corpus_stats.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
