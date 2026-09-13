#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""MultiHiertt 검색 정확도 — HiTab 과 **같은 계측기**로, arm 을 갈아 끼우며 잰다.

`scripts/retrieval_accuracy.py` 의 색인 단위 빌더(`build_corpus`)를 그대로 호출한다.
표 객체만 MultiHiertt 용 어댑터로 바꿔 끼우므로, "arm" 이 두 데이터셋에서 같은 것을
뜻한다 — 단위 코드가 한 벌뿐이라 갈라질 자리가 없다.

지표는 `CLAUDE.md` §0.1 그대로: 운영점 하나(셀 20), 질의 하나를 맞았다/틀렸다.

    검색 정확도 = (문맥이 정답 근거 셀 **전부**를 담은 질의 수) / (질의 수)

층은 데이터셋 필드가 정한다: `program` 이 비면 조회, 있으면 산술. `m` 은
`table_evidence` 의 셀 수. 다중 조회 = `lookup_m2+`, 다중 산술 = `arith_m2+`.

범위 둘 다 싣는다. `doc` 이 MultiHiertt 의 과제 정의이자 MT2Net 의 범위(문서 내
재정렬)이고, `corpus` 는 문서 전부를 한 색인에 넣은 스트레스 조건이다.

  PYTHONPATH=. .venv/bin/python scripts/mh_arms.py --unit cell --tag mh_s3c
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                                                    # noqa: E402

from retrieval_accuracy import (budget_select, build_corpus,          # noqa: E402
                                rowcol_select)
from rag_agent.eval.artifacts import (Selection, digest, evidence_fields,  # noqa: E402
                                     provenance, write_pair)
from rag_agent.reconstruct import (guess_n_header_cols,               # noqa: E402
                                   guess_n_header_rows, parse_html_table,
                                   reconstruct_col_paths, reconstruct_row_paths)
from rag_agent.retrieve.encoders import _tokenize, default_encoder    # noqa: E402
from rag_agent.retrieve.hybrid_index import _minmax                   # noqa: E402
from rag_agent.retrieve.sparse_bm25 import SparseBM25                 # noqa: E402

# MultiHiertt 표는 캡션·제목을 싣지 않는다. s3c 는 제목이 없으면 S2 경로 문자열로
# 떨어지도록 이미 정의돼 있다. 없는 제목을 지어내지 않는다.
NO_TITLE = ""


class MHTable:
    """HiTab 의 표 객체와 같은 읽기 인터페이스. 헤더 경로는 **자가복원**이다."""

    __slots__ = ("table_id", "grid", "nhr", "nhc", "data", "_rows", "_cols")

    def __init__(self, table_id: str, grid, nhr: int, nhc: int):
        self.table_id, self.grid, self.nhr, self.nhc = table_id, grid, nhr, nhc
        self.data = [[(grid[r][c] or "").strip() for c in range(nhc, len(grid[0]))]
                     for r in range(nhr, len(grid))]
        self._rows = reconstruct_row_paths(grid, nhr, nhc)
        self._cols = reconstruct_col_paths(grid, nhr, nhc)

    @property
    def n_rows(self) -> int:
        return len(self.data)

    @property
    def n_cols(self) -> int:
        return len(self.data[0]) if self.data else 0

    def cell(self, r: int, c: int):
        return self.data[r][c]

    def row_path(self, r: int):
        return self._rows[r] if r < len(self._rows) else []

    def col_path(self, c: int):
        return self._cols[c] if c < len(self._cols) else []


class MHDoc:
    """`tab` 자리. 청킹 arm 이 읽는 raw 격자와 raw->data 좌표 맵을 함께 준다."""

    __slots__ = ("table", "title", "raw", "row_map", "col_map")

    def __init__(self, table: MHTable, title: str = NO_TITLE):
        self.table, self.title = table, title
        self.raw = {"texts": table.grid}
        self.row_map = {r: r - table.nhr for r in range(table.nhr, len(table.grid))}
        self.col_map = {c: c - table.nhc for c in range(table.nhc, len(table.grid[0]))}


def load_population(split: str):
    """(queries, docs) — 표 근거만 있는 질의 전부와 그 문서. 필드 정의는 데이터셋 것."""
    from datasets import load_dataset
    rows = load_dataset("bevaya/MultiHiertt", split=split)
    queries, docs, skipped = [], {}, Counter()
    for row in rows:
        ev = row.get("table_evidence") or []
        if row.get("text_evidence"):
            skipped["needs_text_evidence"] += 1       # 셀 색인이 원리적으로 못 담는다
            continue
        if not ev:
            skipped["no_table_evidence"] += 1
            continue
        coords = []
        for e in ev:
            parts = e.split("-")
            if len(parts) != 3:
                coords = None
                break
            coords.append(tuple(int(x) for x in parts))
        if coords is None:
            skipped["bad_evidence_coord"] += 1
            continue
        queries.append({"uid": row["uid"], "question": row["question"],
                        "answer": row.get("answer"), "coords": coords,
                        "kind": "arith" if (row.get("program") or "").strip() else "lookup",
                        "n_gold_tables": len({c[0] for c in coords})})
        docs[row["uid"]] = (row["tables"], row["table_description"], row["paragraphs"])
    return queries, docs, skipped


_MARKER = __import__("re").compile(r"^## Table (\d+)")


def table_labels(paragraphs, rule: str = "none") -> dict:
    """{표 번호: 라벨} — 문서가 표 자리에 둔 `## Table N` 표시 직전 문단에서만 가져온다.

    생성하지 않는다. 규칙은 PREREG-2026-09-13-table-label.md:
      none  라벨 없음
      L1    표시 직전의 비어 있지 않은 문단 그대로
      L2    L1 이 ':' 로 끝나거나 25단어 이하일 때만
    """
    if rule == "none":
        return {}
    out = {}
    for i, s in enumerate(paragraphs or []):
        m = _MARKER.match(s)
        if not m:
            continue
        prev = next((p.strip() for p in reversed(paragraphs[:i])
                     if p.strip() and not _MARKER.match(p)), "")
        if rule == "L2" and not (prev.endswith(":") or len(prev.split()) <= 25):
            prev = ""
        out[int(m.group(1))] = " ".join(prev.split())
    return out


def build_tables(docs, header_rule: str = "v1", label_rule: str = "none"):
    """{table_id: MHDoc} — 표 하나가 색인의 한 '표'다. id 는 ``{uid}::{표 번호}``."""
    tables, hdr = {}, {}
    for uid in sorted(docs):
        labels = table_labels(docs[uid][2] if len(docs[uid]) > 2 else [], label_rule)
        for t_idx, html in enumerate(docs[uid][0]):
            grid = parse_html_table(html)
            if len(grid) < 3 or len(grid[0]) < 2:
                continue
            # 행 경계를 먼저(열 기본값 1), 그 경계로 열 경계를 잡는다 — gold nhc 가
            # 있는 tree_reconstruct_hitab_raw.py 에서 검증된 순서.
            nhr = max(1, min(guess_n_header_rows(grid, n_header_cols=1, rule=header_rule),
                             len(grid) - 1))
            nhc = max(1, min(guess_n_header_cols(grid, n_header_rows=nhr),
                             len(grid[0]) - 1))
            tid = f"{uid}::{t_idx}"
            tables[tid] = MHDoc(MHTable(tid, grid, nhr, nhc), labels.get(t_idx, NO_TITLE))
            hdr[tid] = (nhr, nhc)
    return tables, hdr


_DESC = __import__("re").compile(r"^Table (\d+) shows (.*) is (.*?)\s*\.?\s*$", __import__("re").S)


def mt2net_units(tables, docs, form: str = "desc"):
    """MT2Net 의 색인 단위 — **데이터셋이 실어 준 그 문장 그대로**.

    `psunlpgroup/MultiHiertt` 의 `utils/retriever_utils.py` 는 `table_description`
    의 문자열을 그대로 후보로 쓴다(생성하지 않는다). 그래서 이 arm 은 재현이 아니라
    **원문 그대로**다 — HiTab 쪽 `--template mt2net` 이 논문 예시 한 줄에서 역설계한
    추정인 것과 다르다. 바뀌는 것은 검색기뿐이고, 그건 모든 arm 이 공유한다.

    ``form="s3c"`` 는 같은 문장에서 헤더 구절과 값만 꺼내 본 방법의 제목 없는
    형태(``"{경로}: {값}"``)로 다시 쓴다. 헤더 **내용**은 MT2Net 과 같고(데이터셋의
    구조 주석에서 나온 것) **형태**만 본 방법 것이다. 그래서
      cell(자가복원 헤더) 대 이것  = 헤더 출처의 몫
      이것 대 mt2net_desc          = 문장 형태의 몫
    으로 MultiHiertt 에서 MT2Net 이 이긴 차이를 가른다. train 22,498문장 표본에서
    ``Table N shows … is … .`` 형태가 100% 맞았다.
    """
    texts, covers, unit_tids = [], [], []
    for uid in sorted(docs):
        desc = json.loads(docs[uid][1]) if isinstance(docs[uid][1], str) else docs[uid][1]
        for key in sorted(desc):
            parts = key.split("-")
            if len(parts) != 3:
                continue
            t_idx, r, c = (int(x) for x in parts)
            tid = f"{uid}::{t_idx}"
            tab = tables.get(tid)
            if tab is None:
                continue
            t = tab.table
            i, j = r - t.nhr, c - t.nhc
            m = _DESC.match(desc[key]) if form == "s3c" else None
            text = f"{m.group(2)}: {m.group(3)}" if m else desc[key]
            if form == "label" and tab.title:
                # 본 방법이 제목을 앉히는 자리와 같은 틀. 헤더 렌더링만 다르다.
                text = f"In the table '{tab.title}', {desc[key]}"
            texts.append(text)
            # 헤더 영역의 문장은 데이터 셀을 배달하지 않는다 — TableRAG 의 스키마
            # 문서와 같은 취급이고, 예산을 쓰지 않으므로 이 arm 에 불리하지 않다.
            cells = ([(tid, i, j)] if 0 <= i < t.n_rows and 0 <= j < t.n_cols
                     and str(t.data[i][j]).strip() else [])
            covers.append(frozenset(cells))
            unit_tids.append(tid)
    return texts, covers, [], unit_tids, {}


def resolve_gold(queries, tables, hdr, live):
    """질의마다 gold 셀 집합과 제외 사유. 제외는 이름을 붙여 세고 숨기지 않는다."""
    for q in queries:
        gold, why = set(), None
        for t_idx, r, c in q["coords"]:
            tid = f"{q['uid']}::{t_idx}"
            h = hdr.get(tid)
            if h is None:
                why = why or "gold_table_unparsed"
                continue
            key = (tid, r - h[0], c - h[1])
            if key in live:
                gold.add(key)
            elif r < h[0] or c < h[1]:
                why = why or "gold_in_header"       # 데이터셀 색인이 담을 수 없는 칸
            else:
                why = why or "gold_cell_missing"    # 빈 칸으로 파싱됨
        # 좌표 하나라도 데이터 셀로 못 내려오면 그 질의는 제외한다 — 부분 gold 로
        # 채점하면 요구 셀이 조용히 줄어 그 arm 이 쉬워진다.
        q["gold"], q["excluded"] = gold, (why or (None if gold else "gold_empty"))
    return queries


def layer(q) -> str:
    return f"{q['kind']}_m{'1' if len(q['gold']) == 1 else '2+'}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train",
                    help="MultiHiertt test 는 gold 를 공개하지 않는다.")
    ap.add_argument("--unit", default="cell",
                    choices=["cell", "row", "chunk", "trag_hetero", "tablerag",
                             "rowcol", "mt2net_desc", "mt2net_header_s3c",
                             "mt2net_desc_label"])
    ap.add_argument("--template", default="s3c")
    ap.add_argument("--row-text", default="sentence", choices=["sentence", "values"])
    ap.add_argument("--tablerag-colmode", default="leaf", choices=["leaf", "path"])
    ap.add_argument("--tablerag-dtype", default="infer", choices=["infer", "all_object"])
    ap.add_argument("--header-rule", default="v1", choices=["v1", "v2"],
                    help="헤더 행 추정 규칙. v2 = PREREG-2026-09-13-header-units-note.md")
    ap.add_argument("--label-rule", default="none", choices=["none", "L1", "L2"],
                    help="표 고유 라벨 규칙. PREREG-2026-09-13-table-label.md")
    ap.add_argument("--chunk-chars", type=int, default=1000)
    ap.add_argument("--chunk-overlap", type=int, default=200)
    ap.add_argument("--embed-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--alpha", type=float, default=0.7,
                    help="HiTab arm 에 고정된 값을 그대로 쓴다 — 재선택하지 않는다")
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--rowcol-max-pairs", type=int, default=200)
    ap.add_argument("--max-docs", type=int, default=0, help="배관 점검용. 보고용은 0")
    ap.add_argument("--shard", type=int, default=50000)
    ap.add_argument("--dump-context", type=int, default=1)
    ap.add_argument("--cache-dir", default=".cache/mh_arms")
    ap.add_argument("--out-dir", default="results/mh_arms")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    if a.budget <= 0 or not 0 <= a.alpha <= 1:
        ap.error("invalid budget or alpha")
    out = Path(a.out_dir)
    if any((out / f"{a.tag}{s}").exists() for s in (".json", "_records.jsonl")):
        ap.error("output exists; use a new tag")

    t0 = time.time()
    queries, docs, skipped = load_population(a.split)
    if a.max_docs:
        keep = set(sorted(docs)[:a.max_docs])
        queries = [q for q in queries if q["uid"] in keep]
        docs = {u: docs[u] for u in keep}
    print(f"[pop] {len(queries)} queries / {len(docs)} docs (제외: {dict(skipped)}) "
          f"{time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    tables, hdr = build_tables(docs, a.header_rule, a.label_rule)
    if a.unit in ("mt2net_desc", "mt2net_header_s3c", "mt2net_desc_label"):
        texts, covers, is_row, unit_tids, grid = mt2net_units(
            tables, docs, form={"mt2net_desc": "desc", "mt2net_header_s3c": "s3c",
                                "mt2net_desc_label": "label"}[a.unit])
    else:
        texts, covers, is_row, unit_tids, grid = build_corpus(
            "", sorted(tables), a.template, a.unit, {}, a.chunk_chars,
            a.tablerag_colmode, a.row_text, a.chunk_overlap, None,
            load=lambda tid, _d: tables.get(tid), trag_dtype=a.tablerag_dtype)
    # gold 해석은 arm 과 무관해야 한다. 이 arm 이 배달할 수 있는 셀(covers)로
    # 가르면, 숫자 열을 min/max 로 접는 TableRAG 처럼 셀을 못 담는 arm 은 그 질의가
    # '오답'이 아니라 '제외'가 되어 분모가 줄고 정확도가 부푼다 (2026-09-13 실측:
    # TableRAG 만 188건이 더 빠져 2,685 대 2,871 이었다). 표의 비어 있지 않은
    # 데이터 셀 전부로 가른다.
    live = {(tid, i, j) for tid, tab in tables.items()
            for i, row in enumerate(tab.table.data)
            for j, v in enumerate(row) if str(v).strip()}
    queries = resolve_gold(queries, tables, hdr, live)
    uid_arr = np.array([t.split("::")[0] for t in unit_tids])
    assert len(uid_arr) == len(texts), "unit->doc map lost a unit"
    print(f"[corpus] {len(tables)} tables / {len(texts)} {a.unit} units "
          f"({time.time() - t0:.0f}s)", flush=True)

    t0 = time.time()
    bm = SparseBM25(_tokenize(t) for t in texts)
    print(f"[bm25] {len(bm.vocab)} terms in {time.time() - t0:.0f}s", flush=True)

    enc = default_encoder(model_name=a.embed_model)
    cache = Path(a.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    key = digest(texts)[:16]
    emb = np.empty((len(texts), 768), dtype=np.float32)
    for s in range(0, len(texts), a.shard):
        f = cache / f"{a.unit}_{len(texts)}_{key}_{s}.npy"
        if f.exists():
            v = np.load(f)
        else:
            t1 = time.time()
            v = enc.encode(texts[s:s + a.shard])
            np.save(f, v)
            print(f"[dense] {s}..{s + len(v)} in {time.time() - t1:.0f}s", flush=True)
        emb[s:s + len(v)] = v
        del v

    TOP = 2048            # 진단·선택에 충분한 상한. 전체 argsort 를 피한다.
    recs, t0 = [], time.time()
    for n, q in enumerate(queries, 1):
        if q["excluded"] or not q["gold"]:
            recs.append({"query_id": q["uid"], "kind": q["kind"],
                         "m_annotated": len(q["coords"]),
                         "excluded": q["excluded"] or "gold_empty"})
            continue
        sp = bm.get_scores(_tokenize(q["question"]))
        dn = emb @ enc.encode_query([q["question"]])[0].astype(np.float32)
        sc = a.alpha * _minmax(dn) + (1 - a.alpha) * _minmax(sp)
        r = {"query_id": q["uid"], "kind": q["kind"], "m": len(q["gold"]),
             "m_annotated": len(q["coords"]), "n_gold_tables": q["n_gold_tables"],
             "layer": layer(q), "question": q["question"], "answer": q["answer"],
             "mode": "all"}
        for scope in ("corpus", "doc"):
            if scope == "doc":
                sel = np.flatnonzero(uid_arr == q["uid"])
                k = min(TOP, len(sel) - 1)
                top = sel[np.argpartition(-sc[sel], k)[:k + 1]]
            else:
                k = min(TOP, len(sc) - 1)       # 작은 코퍼스(--max-docs)에서 kth 초과 방지
                top = np.argpartition(-sc, k)[:k + 1]
            order = top[np.argsort(-sc[top], kind="stable")]
            if a.unit == "rowcol":
                selected = rowcol_select(order, covers, texts, is_row, a.budget,
                                         a.dump_context, grid, a.template, a.row_text,
                                         cap=a.rowcol_max_pairs)
            else:
                selected = budget_select(order, covers, texts, a.budget, a.dump_context)
            got, n_cells, ctx = selected
            r[scope] = {"correct": int(q["gold"] <= got),
                        "any_DIAGNOSTIC": int(bool(q["gold"] & got)),
                        "cells_in_context": n_cells,
                        "gold_doc_in_context": int(any(c[0].split("::")[0] == q["uid"]
                                                       for c in got))}
            if a.dump_context:
                r[scope]["context"] = ctx
                r[scope]["context_sha256"] = digest(ctx)
                # 채점한 셀 집합과 리더에게 배달한 단위가 같은지 — HiTab 레그와
                # 같은 불변식. 어긋나면 그 수치는 성립하지 않는다.
                unit_cells = set().union(*(set(map(tuple, u["cells"]))
                                           for u in selected.units)) if selected.units else set()
                if unit_cells != got:
                    raise ValueError("selected cells differ from delivered units")
        recs.append(r)
        if n % 200 == 0:
            print(f"  {n}/{len(queries)}  {time.time() - t0:.0f}s", flush=True)

    scored = [r for r in recs if "corpus" in r]
    excl = Counter(r["excluded"] for r in recs if "excluded" in r)
    by = defaultdict(list)
    for r in scored:
        by[r["layer"]].append(r)
        by["ALL"].append(r)

    def rate(v):
        return round(sum(v) / len(v), 4) if v else None

    def block(rows):
        o = {"n": len(rows)}
        for scope in ("corpus", "doc"):
            o[scope] = {
                "accuracy_all": rate([x[scope]["correct"] for x in rows]),
                "accuracy_any_DIAGNOSTIC": rate([x[scope]["any_DIAGNOSTIC"] for x in rows]),
                "gold_doc_in_context": rate([x[scope]["gold_doc_in_context"] for x in rows]),
                "cells_delivered_mean": (round(sum(x[scope]["cells_in_context"]
                                                   for x in rows) / len(rows), 1)
                                         if rows else None)}
        o["mean_m"] = round(sum(x["m"] for x in rows) / len(rows), 3) if rows else None
        o["m_over_budget"] = sum(1 for x in rows if x["m"] > a.budget)
        return o

    summary = {
        "dataset": "MultiHiertt (Zhao et al., ACL 2022) via bevaya/MultiHiertt",
        "split": a.split, "unit": a.unit, "template": a.template,
        "row_text": a.row_text, "tablerag_colmode": a.tablerag_colmode,
        "chunk_chars": a.chunk_chars, "chunk_overlap": a.chunk_overlap,
        "metric": "질의 단위 맞았다/틀렸다, 운영점 하나 (CLAUDE.md §0.1). 주지표 = all.",
        "hierarchy_source": ("dataset table_description (원문 그대로)"
                             if a.unit.startswith("mt2net")
                             else "self-reconstructed"),
        "provenance": provenance(ROOT), "arguments": vars(a),
        "encoder": enc.name, "query_prefix": enc.query_prefix, "alpha": a.alpha,
        "budget_cells": a.budget, "header_rule": a.header_rule, "label_rule": a.label_rule,
        "n_tables_labelled": sum(1 for t in tables.values() if t.title),
        "context_version": 2 if a.dump_context else None,
        "n_docs": len(docs), "n_tables": len(tables), "n_units": len(texts),
        "corpus_text_sha256": digest(texts),
        "population_skipped_upstream": dict(skipped),
        "n_queries": len(queries), "n_scored": len(scored),
        "n_excluded": sum(excl.values()), "excluded_by_reason": dict(excl),
        "query_ids_sha256": digest(sorted(r["query_id"] for r in scored)),
        "by_layer": {k: block(v) for k, v in sorted(by.items())},
    }
    write_pair(out / f"{a.tag}_records.jsonl", recs, summary,
               summary_path=out / f"{a.tag}.json")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in {"provenance", "arguments"}},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
