"""TableRAG(Yu) 전체 파이프라인이 LLM 에 실제로 넣은 셀의 합집합 채점 (200 답변 문항). 새 실행 없음 — answer trace 재사용.
합집합 = 초기 프롬프트의 상위 1 표 전체(markdown) ∪ 모든 회차·하위 질의의 재정렬 상위 3 청크.
지표: 검색 성공률(정답 셀 전부 포함), 셀 정밀도 |G∩R|/|R|, 전달 셀 수. SQL 경로가 꺼낸 값은 셀 대응이 없어 제외.
실행: PYTHONPATH=. .venv/bin/python results/mh_only_20260928/final/yu_union.py → yu_union.md"""
import importlib.util, json, statistics
from pathlib import Path
spec = importlib.util.spec_from_file_location("trag_yu_baseline", "scripts/baselines/tablerag_yu.py")
Y = importlib.util.module_from_spec(spec); spec.loader.exec_module(Y)
B = Path("results/mh_only_20260928/baselines/tablerag_yu")
ans = [json.loads(l) for l in open(B / "tablerag_yu_answer.jsonl", encoding="utf-8")]
uids = [a["query_id"] for a in ans]
_q, docs_raw, _ = Y.load_population("train")
tables, hdr = Y.build_tables({u: docs_raw[u] for u in uids}, "v3.3u", "none")
rows = []
for a in ans:
    d = Y.Doc(a["query_id"], tables, hdr, 42)
    t = a["raw"]; gold = set(a["gold_ids"])
    top1_tid = d.chunk_tid[t["init_retrieval"]["reranked"][0]]
    cells = set().union(*(set(d.chunk_cells[i]) for i, tid in enumerate(d.chunk_tid) if tid == top1_tid))
    n_chunks = 0
    for r in t["rounds"]:
        for run in r.get("tool_runs", []):
            for i in run["reranked_top3"]:
                cells |= set(d.chunk_cells[i]); n_chunks += 1
    first = set().union(*(set(d.chunk_cells[i]) for i in t["init_retrieval"]["reranked"][:1]))
    rows.append({"q": a["query_id"], "layer": a["layer"], "success": gold <= cells, "recall": len(gold & cells) / len(gold), "cellP": len(gold & cells) / len(cells),
                 "n_cells": len(cells), "n_chunks": n_chunks, "first_success": gold <= first, "doc_cells": sum(len(set(c)) for c in [set().union(*map(set, d.chunk_cells))])})
def agg(rs):
    return {"n": len(rs), "success": sum(r["success"] for r in rs), "success_pct": round(100 * statistics.mean(r["success"] for r in rs), 1),
            "recall_pct": round(100 * statistics.mean(r["recall"] for r in rs), 1), "cellP_pct": round(100 * statistics.mean(r["cellP"] for r in rs), 1), "cells_mean": round(statistics.mean(r["n_cells"] for r in rs), 1),
            "chunks_mean": round(statistics.mean(r["n_chunks"] for r in rs), 1), "doc_cells_mean": round(statistics.mean(r["doc_cells"] for r in rs), 1),
            "first_chunk_success_pct": round(100 * statistics.mean(r["first_success"] for r in rs), 1)}
out = {"all": agg(rows), **{L: agg([r for r in rows if r["layer"] == L]) for L in sorted({r["layer"] for r in rows})}}
Path("results/mh_only_20260928/final/yu_union.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
L = ["| 범위 | 문항 | 검색 성공(정답 셀 전부) | 셀 정밀도 | 전달 셀 수 평균 | 문서 셀 수 평균 | 하위 질의 청크 수 평균 |", "|---|---:|---:|---:|---:|---:|---:|"]
for k, v in out.items():
    L.append(f"| {k} | {v['n']} | {v['success']} ({v['success_pct']}%) | {v['cellP_pct']}% | {v['cells_mean']} | {v['doc_cells_mean']} | {v['chunks_mean']} |")
Path("results/mh_only_20260928/final/yu_union.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
