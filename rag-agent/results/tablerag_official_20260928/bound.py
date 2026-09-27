#!/usr/bin/env python3
"""TableRAG 셀 검색 재구현의 전달 가능 상한 — 숫자 열 판정 규칙별 (실행 없음 — 임베딩·모델을 부르지 않는다). 2026-09-28.

전달 가능 = 질문의 정답 셀 전부가 TableRAG 문서의 셀 매핑(tablerag_units 의 covers)에 들어 있다. 예산·검색과 무관한 상한.
규칙: infer(원고 결과에 쓴 규칙), official(원본 utils.infer_dtype, trag.official_frame), all_object.
HiTab = test 단일 셀 조회 991건, MultiHiertt = train 채점 2,885건(머리글 최종 규칙 v3.3u).
official 에서 날짜 dtype 으로 바뀐 열(원본처럼 요약 문서 하나로 접힘)의 수를 따로 적는다.
infer 규칙의 색인 글이 코드 변경 전 결과(rerun_20260926)와 같은지 corpus_text_sha256 으로 확인한다.

  PYTHONPATH=. .venv/bin/python results/tablerag_official_20260928/bound.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from retrieval_accuracy import build_corpus, load_queries, query_type, tablerag_units   # noqa: E402
from rag_agent.eval.artifacts import digest                                # noqa: E402
from rag_agent.serialization import tablerag_unit as trag               # noqa: E402
import mh_arms                                                         # noqa: E402

RULES, MODES = ("infer", "official", "all_object"), ("leaf", "path")
_cov, _dt = {}, set()


def covered(tab, m, r):
    """표 하나의 전달 가능 셀 {(i, j)}."""
    k = (tab.table.table_id, m, r)
    if k not in _cov:
        t = tab.table
        if r == "official":
            df = trag.official_frame(t, m)
            _dt.update((t.table_id, m, c) for c in range(t.n_cols) if str(df.iloc[:, c].dtype).startswith("datetime"))
        _cov[k] = set().union(*(set(cs) for _, cs in tablerag_units(tab, t, m, r)))
    return _cov[k]


def count(per_q):
    """per_q: 질의마다 [(tab, 그 표의 정답 셀 {(i, j)})]."""
    out = {r: {m: 0 for m in MODES} for r in RULES}
    for items in per_q:
        for r in RULES:
            for m in MODES:
                out[r][m] += all(g <= covered(tab, m, r) for tab, g in items)
    return out


def group(gold, tab_of):
    by = {}
    for tid, i, j in gold:
        by.setdefault(tid, set()).add((i, j))
    return [(tab_of(tid), g) for tid, g in by.items()]


tabs = {}
hq = [q for q in load_queries("data/hitab", "test", tabs) if not q.get("excluded") and query_type(q) == "single_cell"]
hitab = count([group(q["gold"], tabs.get) for q in hq])
hdt = len(_dt)
# infer 규칙 색인 글이 코드 변경 전과 같은가 (HiTab 질문의 표 안 leaf, 538표 path)
sha = {}
for m, f in (("leaf", "hitab_test_gold_tablerag_leaf"), ("path", "hitab_test_split_tablerag_path")):
    texts = build_corpus("data/hitab", sorted({q["table_id"] for q in load_queries("data/hitab", "test", {})}),
                         "s3c", "tablerag", json.loads((ROOT / "results/tableconf/totto_page_titles.json").read_text()),
                         1000, m, "values", 200, None, trag_dtype="infer")[0]
    sha[f] = digest(texts) == json.loads((ROOT / f"results/rerun_20260926/hitab/{f}.json").read_text())["corpus_text_sha256"]

queries, docs, _ = mh_arms.load_population("train")
tables, hdr = mh_arms.build_tables(docs, "v3.3u")
live = {(tid, i, j) for tid, tab in tables.items() for i, row in enumerate(tab.table.data) for j, v in enumerate(row) if str(v).strip()}
mq = [q for q in mh_arms.resolve_gold(queries, tables, hdr, live) if not q["excluded"] and q["gold"]]
_dt.clear()
mh = count([group(q["gold"], tables.get) for q in mq])
mdt = len(_dt)
texts = mh_arms.build_corpus("", sorted(tables), "s3c", "tablerag", {}, 1000, "leaf", "values", 200, None,
                             load=lambda tid, _d: tables.get(tid), trag_dtype="infer")[0]
sha["mh_train_tablerag_leaf"] = digest(texts) == json.loads((ROOT / "results/rerun_20260926/mh/mh_train_tablerag_leaf.json").read_text())["corpus_text_sha256"]

rate = lambda d, n: {r: {m: round(v / n, 4) for m, v in x.items()} for r, x in d.items()}
res = {"hitab_test_single_cell": {"n": len(hq), "deliverable": hitab, "rate": rate(hitab, len(hq)),
                                  "official_datetime_columns_table_mode": hdt},
       "multihiertt_train": {"n": len(mq), "deliverable": mh, "rate": rate(mh, len(mq)),
                             "official_datetime_columns_table_mode": mdt},
       "infer_corpus_unchanged_vs_rerun_20260926": sha}
(Path(__file__).resolve().parent / "bound.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
