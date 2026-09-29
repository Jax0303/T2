"""위층 행 머리글(절 제목 행)이 청크에서 떨어지는 비율 — 2026-09-30, 새 실행 없음(CPU).

LangChain 청크 비교군과 같은 문서 텍스트·분할(1,000자, 겹침 200자)을 다시 만든다. 절 제목 행 = 데이터 칸이 모두 빈 행
(원고 3.3.1절 규칙). 정답 셀 바로 위의 가장 가까운 절 제목 행을 그 셀의 위층 행 머리글로 보고, 정답 셀을 담은 청크 가운데
어느 것도 그 줄을 통째로 담지 않으면 '위층 행 머리글이 떨어짐'으로 센다. 절 제목 행이 없는 셀은 떨어질 것이 없다.
열 머리글 기준(header_split.py 의 첫 데이터 행 규칙)을 같은 청크로 다시 세어 730/339 가 나오는지 함께 확인한다.
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mh_arms import NO_TITLE, build_tables, load_population, resolve_gold
from rag_agent.serialization.chunks import markdown_source

queries, docs, _ = load_population("train")
tables, hdr = build_tables(docs, "v3.3u", "none")
live = {(tid, i, j) for tid, tab in tables.items() for i, row in enumerate(tab.table.data) for j, v in enumerate(row) if str(v).strip()}
queries = [q for q in resolve_gold(queries, tables, hdr, live) if q["gold"] and not q["excluded"]]
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
blank = lambda row: not any(str(v).strip() for v in row)
S = {"n_queries": len(queries), "n_gold_cells": 0, "col_cells": 0, "col_queries": 0, "has_section_parent": 0,
     "row_cells": 0, "row_queries": 0, "either_cells": 0, "either_queries": 0}
for q in queries:
    uid = q["uid"]
    text, off, cellpos, rowline, secrow = "", 0, {}, {}, {}
    tids = sorted((t for t in tables if t.split("::")[0] == uid), key=lambda t: int(t.split("::")[1]))
    for k, tid in enumerate(tids):
        tab = tables[tid]
        md, spans = markdown_source(tab, tab.table, NO_TITLE)
        md = md[len("Table name: \n"):]; shift = len("Table name: \n")
        pos = 0
        for r, row in enumerate(tab.raw.get("texts") or []):
            if blank(row): continue
            end = md.index("\n", pos) if "\n" in md[pos:] else len(md)
            i = tab.row_map.get(r)
            if i is not None:
                rowline[(tid, i)] = (off + pos, off + end)
            pos = end + 1
        for s, e, (i, j) in spans:
            cellpos[(tid, i, j)] = (off + s - shift, off + e - shift)
        data = tab.table.data
        secrow[tid] = {i for i in range(len(data)) if blank(data[i]) and (tid, i) in rowline}
        text += md + ("\n\n" if k < len(tids) - 1 else "")
        off = len(text)
    chunks = [(d.metadata["start_index"], d.metadata["start_index"] + len(d.page_content)) for d in splitter.create_documents([text])]
    inside = lambda span, ch: ch[0] <= span[0] and span[1] <= ch[1]
    first = {}
    for (tid, i, j) in cellpos:
        first[tid] = min(first.get(tid, i), i)
    cq = rq = eq = False
    for (tid, i, j) in q["gold"]:
        S["n_gold_cells"] += 1
        has = [ch for ch in chunks if inside(cellpos[(tid, i, j)], ch)]
        col_bad = not any(any(inside(p, ch) for (t2, i2, _), p in cellpos.items() if t2 == tid and i2 == first[tid]) for ch in has)
        parents = [s for s in secrow[tid] if s < i]
        row_bad = False
        if parents:
            S["has_section_parent"] += 1
            row_bad = not any(inside(rowline[(tid, max(parents))], ch) for ch in has)
        S["col_cells"] += col_bad; S["row_cells"] += row_bad; S["either_cells"] += col_bad or row_bad
        cq |= col_bad; rq |= row_bad; eq |= col_bad or row_bad
    S["col_queries"] += cq; S["row_queries"] += rq; S["either_queries"] += eq
for k in ("col", "row", "either"):
    S[k + "_cells_pct"] = round(100 * S[k + "_cells"] / S["n_gold_cells"], 1)
    S[k + "_queries_pct"] = round(100 * S[k + "_queries"] / S["n_queries"], 1)
(Path(__file__).parent / "row_header_split.json").write_text(json.dumps(S, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(S, ensure_ascii=False))
