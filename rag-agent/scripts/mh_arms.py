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

집합 채점 (`rag_agent/eval/strict_recall.py`, doc 범위, hybrid 질문도 표 근거로 포함):
`--metric-mode strict_fixed_budget` 은 budget_select 가 리더에게 주는 --budget 셀 문맥을 채점하는
기준선이고, `strict_no_k` 는 예산 없이 문서 안 관련성 점수 >= 임계값인 단위를 고른다 — 임계값은
한 분할(validation)에서 macro F1 로 고르고 다른 분할에 그대로 적용한다. budget_select 는 부르지 않는다.

  PYTHONPATH=. .venv/bin/python scripts/mh_arms.py --unit cell --tag mh_s3c
  PYTHONPATH=. .venv/bin/python scripts/mh_arms.py --metric-mode strict_no_k --split validation
  PYTHONPATH=. .venv/bin/python scripts/mh_arms.py --metric-mode strict_no_k --split train \\
      --threshold-from results/strict_no_k/mh_validation_cell_hv3.3_none_doc_strict_no_k_summary.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                                                    # noqa: E402

from retrieval_accuracy import (budget_select, build_corpus,          # noqa: E402
                                rowcol_select, threshold_select)
from rag_agent.eval.artifacts import (Selection, digest, evidence_fields,  # noqa: E402
                                     provenance, write_pair)
from rag_agent.eval import strict_recall as sr                        # noqa: E402
from rag_agent.reconstruct import (guess_n_header_cols,               # noqa: E402
                                   guess_n_header_rows, parse_html_table,
                                   reconstruct_col_paths, reconstruct_row_paths)
from rag_agent.reconstruct.header_grid import (corner_scope,           # noqa: E402
                                              parse_html_table_layout)
from rag_agent.retrieve.encoders import _tokenize, default_encoder    # noqa: E402
from rag_agent.retrieve.hybrid_index import _minmax                   # noqa: E402
from rag_agent.retrieve.sparse_bm25 import SparseBM25                 # noqa: E402

# MultiHiertt 표는 캡션·제목을 싣지 않는다. s3c 는 제목이 없으면 S2 경로 문자열로
# 떨어지도록 이미 정의돼 있다. 없는 제목을 지어내지 않는다.
NO_TITLE = ""


class MHTable:
    """HiTab 의 표 객체와 같은 읽기 인터페이스. 헤더 경로는 **자가복원**이다."""

    __slots__ = ("table_id", "grid", "nhr", "nhc", "data", "_rows", "_cols")

    def __init__(self, table_id: str, grid, nhr: int, nhc: int, cover=None, rules=()):
        self.table_id, self.grid, self.nhr, self.nhc = table_id, grid, nhr, nhc
        self.data = [[(grid[r][c] or "").strip() for c in range(nhc, len(grid[0]))]
                     for r in range(nhr, len(grid))]
        rules = frozenset(rules)
        self._rows = reconstruct_row_paths(grid, nhr, nhc,
                                           cover=cover if rules & {"S2", "S2n"} else None,
                                           isolated_blank_only="S2n" in rules,
                                           close_on_total=("own" if "S4n2" in rules else
                                                           "named" if "S4n" in rules else "S4" in rules),
                                           disambiguate=("no_numbers_guard" if "S5n3" in rules else
                                                         "no_numbers" if "S5n2" in rules else
                                                         "guarded" if "S5n" in rules else "S5" in rules))
        cols = reconstruct_col_paths(grid, nhr, nhc, cover=cover if rules & {"S1", "S1n"} else None,
                                     narrow="S1n" in rules)
        if "S6n" in rules:
            scope = corner_scope(grid, nhr, nhc, cols, row_paths=self._rows, max_len=60)
        else:
            scope = corner_scope(grid, nhr, nhc, cols) if "S6" in rules else ""
        self._cols = [[scope, *p] for p in cols] if scope else cols

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


def load_population(split: str, keep_hybrid: bool = False):
    """(queries, docs) — 표 근거만 있는 질의 전부와 그 문서. 필드 정의는 데이터셋 것.

    ``keep_hybrid`` 는 text_evidence 도 있는 질의를 표 근거로 남긴다(strict_no_k). 그때
    표 근거가 없는 text-only 질의는 ``skipped["text_only"]`` 로 센다."""
    from datasets import load_dataset
    rows = load_dataset("bevaya/MultiHiertt", split=split)
    queries, docs, skipped = [], {}, Counter()
    for row in rows:
        ev = row.get("table_evidence") or []
        text = bool(row.get("text_evidence"))
        if text and not keep_hybrid:
            skipped["needs_text_evidence"] += 1       # 셀 색인이 원리적으로 못 담는다
            continue
        if not ev:
            skipped["text_only" if text else "no_table_evidence"] += 1
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
                        "n_gold_tables": len({c[0] for c in coords}),
                        "has_text_evidence": text,
                        "program_ops": re.findall(r"([a-z_]+)\(", row.get("program") or "")})
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


V3_RULES = ("S1", "S2", "S3", "S4", "S5", "S6")
V31_RULES = ("S1n", "S2n", "S3n", "S4n", "S5n", "S6n")
V32_RULES = ("S1n", "S2n", "S3n", "S4n2", "S5n2", "S6n")
V33_RULES = ("S1n", "S2n", "S3n", "S4n2", "S5n3", "S6n")


def build_tables(docs, header_rule: str = "v1", label_rule: str = "none", rules=None):
    """{table_id: MHDoc} — 표 하나가 색인의 한 '표'다. id 는 ``{uid}::{표 번호}``.

    ``header_rule="v3"`` 은 v2 + S1~S6, ``"v3.1"`` 은 v2 + 좁힌 S1n~S6n(정정 1), ``"v3.2"`` 는 v3.1 에서
    S4n·S5n 을 S4n2·S5n2 로 바꾼 것(정정 3), ``"v3.3"`` 은 v3.2 에서 S5n2 를 S5n3 로 바꾼 것이다(정정 4,
    PREREG-2026-09-14-header-v3.md). ``rules`` 는 영향 분석이 규칙을 하나씩 v2 위에 얹어 보려고 두는 인자다.
    None 이면 v1/v2 는 규칙 없음, v3·v3.1·v3.2·v3.3 은 각자의 전부.
    """
    if rules is None:
        rules = {"v3": V3_RULES, "v3.1": V31_RULES, "v3.2": V32_RULES, "v3.3": V33_RULES}.get(header_rule, ())
    rules = frozenset(rules)
    row_rule = ("v3.1" if "S3n" in rules else "v3" if "S3" in rules
                else "v2" if header_rule in ("v3", "v3.1", "v3.2", "v3.3") else header_rule)
    tables, hdr = {}, {}
    for uid in sorted(docs):
        labels = table_labels(docs[uid][2] if len(docs[uid]) > 2 else [], label_rule)
        for t_idx, html in enumerate(docs[uid][0]):
            grid, cover = parse_html_table_layout(html) if rules else (parse_html_table(html), None)
            if len(grid) < 3 or len(grid[0]) < 2:
                continue
            # 행 경계를 먼저(열 기본값 1), 그 경계로 열 경계를 잡는다 — gold nhc 가
            # 있는 tree_reconstruct_hitab_raw.py 에서 검증된 순서.
            nhr = max(1, min(guess_n_header_rows(grid, n_header_cols=1, rule=row_rule),
                             len(grid) - 1))
            nhc = max(1, min(guess_n_header_cols(grid, n_header_rows=nhr),
                             len(grid[0]) - 1))
            tid = f"{uid}::{t_idx}"
            tables[tid] = MHDoc(MHTable(tid, grid, nhr, nhc, cover, rules),
                                labels.get(t_idx, NO_TITLE))
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


def cell_id(cell, hdr) -> str:
    """``(표 id, i, j)`` -> ``"{uid}::{표}-{행}-{열}"`` — ``table_evidence`` 와
    ``table_description`` 이 쓰는 펼친 격자 좌표. ``MHTable.data`` 는 ``grid[nhr:][nhc:]``."""
    tid, i, j = cell
    return f"{tid}-{i + hdr[tid][0]}-{j + hdr[tid][1]}"


def evidence_anomalies(q, docs, tables) -> list:
    """gold 좌표가 ``table_description`` 의 같은 키·같은 값을 가리키는지. 어긋나면 기록만 한다."""
    desc = docs[q["uid"]][1]
    desc = json.loads(desc) if isinstance(desc, str) else desc
    norm = lambda s: re.sub(r"[,\s]", "", s or "")
    out = []
    for t, r, c in q["coords"]:
        key, tab = f"{t}-{r}-{c}", tables.get(f"{q['uid']}::{t}")
        m = _DESC.match(desc.get(key, ""))
        if m is None:
            out.append({"cell": key, "problem": "no_table_description_sentence"})
            continue
        grid = tab.table.grid if tab is not None else []
        got = (grid[r][c] or "") if r < len(grid) and c < len(grid[r]) else None
        if got is None or norm(m.group(3)) != norm(got):
            out.append({"cell": key, "problem": "value_differs",
                        "description_value": m.group(3), "grid_value": got})
    return out


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
    ap.add_argument("--header-rule", default=None, choices=["v1", "v2", "v3", "v3.1", "v3.2", "v3.3"],
                    help="헤더 행 추정 규칙. 기본 v1, strict_no_k 는 채택 규칙 v3.3 "
                         "(PREREG-2026-09-14-header-v3.md 정정 4). v2 = PREREG-2026-09-13-header-units-note.md, "
                         "v3·v3.1·v3.2·v3.3 = PREREG-2026-09-14-header-v3.md")
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
    ap.add_argument("--out-dir", default=None,
                    help="기본 results/mh_arms, results/strict_fixed_budget, results/strict_no_k")
    ap.add_argument("--tag", default="", help="필수 (strict_* 는 자동)")
    ap.add_argument("--metric-mode", default="accuracy",
                    choices=["accuracy", "strict_fixed_budget", "strict_no_k"],
                    help="strict_fixed_budget: doc 범위, --budget 셀 문맥을 Strict Recall 로 채점. "
                         "strict_no_k: 예산 없음, 문서 안 관련성 점수 >= 임계값 (budget_select 안 부름)")
    ap.add_argument("--threshold-from", default="",
                    help="strict_no_k: 임계값을 고른 실행의 summary JSON. 없으면 이 분할에서 고른다 (test 금지)")
    ap.add_argument("--context-tokenizer", default="Qwen/Qwen3-8B",
                    help="strict_*: 문맥 토큰 수를 셀 리더 토크나이저 ('' = 세지 않음)")
    a = ap.parse_args()
    strict, no_k = a.metric_mode != "accuracy", a.metric_mode == "strict_no_k"
    if a.budget <= 0 or not 0 <= a.alpha <= 1:
        ap.error("invalid budget or alpha")
    if not (a.tag or strict):
        ap.error("--tag is required")
    if a.threshold_from and not no_k:
        ap.error("--threshold-from is only for strict_no_k")
    if no_k and (a.unit == "rowcol" or (a.split == "test" and not a.threshold_from)):
        ap.error("strict_no_k: not rowcol, and the threshold is never selected on test")
    a.header_rule = a.header_rule or ("v3.3" if strict else "v1")
    a.out_dir = a.out_dir or {"accuracy": "results/mh_arms",
                              "strict_fixed_budget": "results/strict_fixed_budget",
                              "strict_no_k": "results/strict_no_k"}[a.metric_mode]
    mark = {"strict_no_k": "strict_no_k",
            "strict_fixed_budget": f"fixed_budget{a.budget}"}.get(a.metric_mode)
    a.tag = a.tag or f"mh_{a.split}_{a.unit}_h{a.header_rule}_{a.label_rule}_doc_{mark}"
    out = Path(a.out_dir)
    if strict and (mark not in f"{out}/{a.tag}" or (not no_k and "strict_no_k" in f"{out}/{a.tag}")):
        ap.error(f"{a.metric_mode} outputs need '{mark}' in --out-dir or --tag, "
                 "and fixed-budget outputs never carry 'strict_no_k'")
    targets = (sr.outputs(out, a.tag) if strict else
               [out / f"{a.tag}{s}" for s in (".json", "_records.jsonl")])
    if any(p.exists() for p in targets):
        ap.error("output exists; use a new tag")

    t0 = time.time()
    queries, docs, skipped = load_population(a.split, keep_hybrid=strict)
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
    # strict_no_k 는 질문의 문서 행만 읽으므로 조각을 메모리 매핑한다 — train 전체 행렬(약 3GB)을
    # 이 상자 RAM 에 올리지 않는다. 다른 모드는 예전 그대로 한 행렬에 채운다.
    emb = None if no_k else np.empty((len(texts), 768), dtype=np.float32)
    shards = []
    for s in range(0, len(texts), a.shard):
        f = cache / f"{a.unit}_{len(texts)}_{key}_{s}.npy"
        if f.exists():
            v = np.load(f, mmap_mode="r" if no_k else None)
        else:
            t1 = time.time()
            v = enc.encode(texts[s:s + a.shard])
            np.save(f, v)
            print(f"[dense] {s}..{s + len(v)} in {time.time() - t1:.0f}s", flush=True)
            if no_k:
                v = np.load(f, mmap_mode="r")
        if no_k:
            shards.append(v)
        else:
            emb[s:s + len(v)] = v
        del v

    TOP = 2048            # 진단·선택에 충분한 상한. 전체 argsort 를 피한다.
    count, tok_info = sr.token_counter(a.context_tokenizer) if strict else (None, None)
    strict_rows, pending = [], []
    ids = lambda cells: {cell_id(c, hdr) for c in cells}

    def strict_row(q, selected):
        # 채점 대상은 선택기가 리더에게 넘긴 셀 전부다. 여기서 다시 자르지 않는다.
        unit_texts = [u["text"] for u in selected.units]
        return sr.row(
            "multihiertt", a.split, q["uid"],
            sr.question_type(q["kind"] == "arith", len(q["gold"])),
            ids(q["gold"]), ids(selected.cells),
            table_of=lambda s: s.rsplit("-", 2)[0], table_metrics=True,
            population="hybrid_questions" if q["has_text_evidence"] else "table_only_questions",
            layer=layer(q), program_ops=q["program_ops"], context_units=len(unit_texts),
            context_tokens=count(unit_texts) if count else None,
            context_sha256=digest(unit_texts))

    recs, t0 = [], time.time()
    for n, q in enumerate(queries, 1):
        if q["excluded"] or not q["gold"]:
            recs.append({"query_id": q["uid"], "kind": q["kind"],
                         "m_annotated": len(q["coords"]),
                         "excluded": q["excluded"] or "gold_empty"})
            continue
        sp = bm.get_scores(_tokenize(q["question"]))
        if no_k:
            # 관련성 점수는 질문의 문서 안에서 min-max 한다 — 다른 분할에서 고른 임계값이 같은
            # 척도를 보도록. (고정 예산 모드는 코퍼스 전체 min-max 뒤 문서로 마스크한다.)
            sel = np.flatnonzero(uid_arr == q["uid"])
            qv = enc.encode_query([q["question"]])[0].astype(np.float32)
            dn = np.concatenate([shards[k][sel[sel // a.shard == k] - k * a.shard]
                                 for k in np.unique(sel // a.shard)]) @ qv
            pending.append((q, sel, a.alpha * _minmax(dn) + (1 - a.alpha) * _minmax(sp[sel])))
            continue
        dn = emb @ enc.encode_query([q["question"]])[0].astype(np.float32)
        sc = a.alpha * _minmax(dn) + (1 - a.alpha) * _minmax(sp)
        r = {"query_id": q["uid"], "kind": q["kind"], "m": len(q["gold"]),
             "m_annotated": len(q["coords"]), "n_gold_tables": q["n_gold_tables"],
             "layer": layer(q), "question": q["question"], "answer": q["answer"],
             "mode": "all"}
        for scope in (("doc",) if strict else ("corpus", "doc")):
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
            if strict:
                strict_rows.append(strict_row(q, selected))
                continue
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

    if strict:
        threshold = None
        if no_k:
            config = {"unit": a.unit, "template": a.template, "alpha": a.alpha,
                      "header_rule": a.header_rule, "label_rule": a.label_rule}
            if a.threshold_from:
                threshold = sr.load_threshold(a.threshold_from, "multihiertt", a.split, config)
            else:
                threshold = sr.selected_threshold(
                    sr.sweep([(ids(q["gold"]), None,
                               [(float(v), ids(covers[p])) for v, p in zip(s, sel)])
                              for q, sel, s in pending]),
                    a.split, f"{a.alpha}*minmax(BGE cosine) + {round(1 - a.alpha, 4)}*minmax(BM25, "
                             "corpus IDF), min-max over the units of the question's own document",
                    config)
            for q, sel, s in pending:
                strict_rows.append(strict_row(q, threshold_select(
                    s, sel, covers, texts, threshold["applied_threshold"], a.dump_context)))
            if budget_select.calls:
                raise RuntimeError("budget_select was called on the strict_no_k path")
        pop = lambda q: "hybrid_questions" if q["has_text_evidence"] else "table_only_questions"
        excluded = [{"query_id": q["uid"], "reason": q["excluded"] or "gold_empty",
                     "population": pop(q)} for q in queries if q["excluded"] or not q["gold"]]
        anomalies = [{"query_id": q["uid"], "cells": bad} for q in queries
                     if (bad := evidence_anomalies(q, docs, tables))]
        groups = {"overall": lambda r: True,
                  **{t: (lambda r, t=t: r["question_type"] == t) for t in sr.TYPES},
                  "all_table_questions": lambda r: True,
                  "table_only_questions": lambda r: r["population"] == "table_only_questions",
                  "hybrid_questions": lambda r: r["population"] == "hybrid_questions"}
        summary = sr.write(out, a.tag, strict_rows, groups, {
            "dataset": "multihiertt", "split": a.split,
            "metric": ("Strict Recall, budget-free: R_q = units with relevance score >= threshold"
                       if no_k else
                       f"Strict Recall under a {a.budget}-cell context (fixed-budget baseline)"),
            "selector": ({"function": "threshold_select",
                          "threshold": threshold["applied_threshold"],
                          "budget_select_calls": budget_select.calls,
                          "search_scope": "doc: tables of the question's own document"}
                         if no_k else
                         {"function": "rowcol_select" if a.unit == "rowcol" else "budget_select",
                          "stop_after_distinct_cells": a.budget, "last_unit_kept_whole": True,
                          "search_scope": "doc: tables of the question's own document"}),
            "threshold": threshold,
            "gold_rule": "table_evidence '{table}-{row}-{col}' on the expanded grid, the same key "
                         "as table_description; cell id '{uid}::{table}-{row}-{col}' (cell_id)",
            "question_type_rule": "arithmetic = non-empty program (repo rule, load_population "
                                  "kind); otherwise single/multi by gold cell count",
            "arithmetic_without_program_operator": sum(
                1 for r in strict_rows if r["question_type"] == "arithmetic" and not r["program_ops"]),
            "non_arithmetic_with_program_operator": sum(
                1 for r in strict_rows if r["question_type"] != "arithmetic" and r["program_ops"]),
            "table_metric": "Derived Table Coverage: tables recovered from the selected cells; "
                            "no separate table retrieval stage exists",
            "population_counts": {
                "all_table_questions": len(queries),
                "table_only_questions": sum(not q["has_text_evidence"] for q in queries),
                "hybrid_questions": sum(q["has_text_evidence"] for q in queries),
                "text_only_questions_excluded": skipped.get("text_only", 0),
                "no_evidence_excluded": skipped.get("no_table_evidence", 0),
                "bad_evidence_coord_excluded": skipped.get("bad_evidence_coord", 0)},
            "excluded_by_reason": dict(Counter(x["reason"] for x in excluded)),
            "header_rule": a.header_rule, "label_rule": a.label_rule, "unit": a.unit,
            "template": a.template, "alpha": a.alpha, "encoder": enc.name,
            "query_prefix": enc.query_prefix, "context_tokenizer": tok_info,
            "n_docs": len(docs), "n_tables": len(tables), "n_units": len(texts),
            "corpus_text_sha256": digest(texts), "provenance": provenance(ROOT),
            "arguments": vars(a)}, anomalies, excluded)
        sr.print_groups(summary["groups"])
        if threshold:
            print(f"threshold {threshold['applied_threshold']} ({threshold['role']}, selected on "
                  f"{threshold['selected_on_split']}); budget_select calls {budget_select.calls}")
        print(f"population {summary['population_counts']}; excluded {len(excluded)} "
              f"{summary['excluded_by_reason']}; gold anomalies {len(anomalies)}; "
              f"wrote -> {out / a.tag}_*")
        return 0

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
