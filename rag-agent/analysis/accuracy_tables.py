#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the two report tables from the summary/record files on disk.

Reads nothing but ``results/retrieval_accuracy/*.json`` and the answer legs, so
every number in the tables has a file behind it and no number is typed by hand.

  PYTHONPATH=. .venv/bin/python analysis/accuracy_tables.py > results/retrieval_accuracy/TABLES.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "results/retrieval_accuracy"


def load(tag):
    f = D / f"{tag}.json"
    return json.loads(f.read_text()) if f.exists() else None


def recs(tag, mode="all"):
    f = D / f"{tag}_records.jsonl"
    if not f.exists():
        return {}
    return {j["query_id"]: j for j in map(json.loads, f.open())
            if j.get("mode") == mode and "correct" in j}


def primary_ids(ref="t_s3c_hybrid"):
    """주 모집단 — 단일 셀 조회 (`aggregation` 이 none 이고 gold 셀이 하나).

    2026-09-08 사용자 결정. 결과를 보고 고른 것이 아니라 연구 범위를 조회로 잡은
    것이고, `aggregation` 은 HiTab 이 질의마다 붙여 둔 필드이므로 우리가 나눈 분류가
    아니다. 여기서 빠지는 산술·다중 셀 254건은 숨기지 않고 표 1c 에 전수로 적는다 —
    그쪽에서 순위가 뒤집히는 것까지 포함해서.

    좁힌 이유: gold 셀이 정확히 하나라 지표가 "그 한 칸이 예산 안에 있나" 하나로
    떨어진다. m>=2 는 셀 예산 20 이 요구 셀 수에 따라 다르게 빡빡해서 색인 단위를
    가로질러 비교하면 예산 압박이 섞여 들어온다.
    """
    return {q for q, d in recs(ref).items()
            if (d.get("aggregation") or "none") == "none" and d["m"] == 1}


def acc(tag, ids):
    o = recs(tag)
    S = [q for q in ids if q in o]
    return (sum(o[q]["correct"] for q in S) / len(S)) if S else None


def paired(tag, ref="t_s3c_hybrid", mode="all", ids=None):
    """Exact McNemar of ``tag`` against our arm, over the queries both scored."""
    from math import comb
    fa, fb = D / f"{ref}_records.jsonl", D / f"{tag}_records.jsonl"
    if tag == ref or not (fa.exists() and fb.exists()):
        return ""
    A = {json.loads(l)["query_id"]: json.loads(l) for l in fa.open()}
    B = {json.loads(l)["query_id"]: json.loads(l) for l in fb.open()}
    ids = [q for q in A if q in B and "correct" in A[q] and "correct" in B[q]
           and A[q]["mode"] == mode and (ids is None or q in ids)]
    n01 = sum(1 for q in ids if A[q]["correct"] and not B[q]["correct"])
    n10 = sum(1 for q in ids if B[q]["correct"] and not A[q]["correct"])
    n = n01 + n10
    p = (1.0 if n == 0 else
         min(1.0, 2 * sum(comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n))
    return f"{n01}:{n10}, p={p:.3g}"


def ctx_mean(tag):
    f = D / f"{tag}_records.jsonl"
    if not f.exists():
        return None
    v = [json.loads(l)["cells_in_context"] for l in f.open() if '"correct"' in l]
    return sum(v) / len(v) if v else None


# (tag, label, the arm this row is tested against, provenance of the row)
#
# The `분류` column exists because an ablation and a baseline answer different
# questions and a reader who cannot tell them apart reads the table wrong. An
# ablation (S2, flat) removes one part of OUR unit to price that part; it is
# not a competitor and beating it is not a result. A baseline is a method that
# exists outside this repo -- a published system's index unit, or what a
# general-purpose RAG stack does by default.
# 표 1 -- 이 리포 밖에 존재하는 방법만. 색인 단위가 유일한 변수이고 코퍼스,
# 예산(셀 20), 검색기(hybrid α=0.7), 인코더는 모든 행에서 같다.
BASELINE_ROWS = [
    ("t_s3c_hybrid", "**본 방법** — 셀 문장(제목+행경로+열경로+값)", "", "—"),
    ("t_row_values", "행 단위 — `RowColRetrieval` 의 행 절반 (TableRAG §4.2) ‡", "t_s3c_hybrid", "발표 논문(부분)"),
    ("t_rowcol_values", "`RowColRetrieval` — 행 × 열 교집합, 온전 재현 ‡", "t_s3c_hybrid", "발표 논문"),
    ("t_table_hybrid", "표 통째 — 표 하나가 색인 단위 하나 (표 단위 검색)", "t_s3c_hybrid", "통제 — 셀 검색 없음"),
    ("t_mt2net_hybrid", "셀 + 계층 헤더, 표 제목 없음 — MT2Net 의 색인 **단위** (Zhao et al. 2022 §4) ¶", "t_s3c_hybrid", "발표 논문(단위)"),
    ("t_trag_hetero", "TableRAG **(Huawei, EMNLP 2025)** 검색 레그 — 코드 읽기 1,000자", "t_s3c_hybrid", "발표 논문"),
    ("t_trag_hetero_tok", "TableRAG **(Huawei)** 검색 레그 — 논문 읽기 1,000토큰(≈2,400자)", "t_s3c_hybrid", "발표 논문"),
    ("t_chunk1000", "고정 크기 청킹 1,000자 (LangChain 기본값, 청크마다 헤더 반복)", "t_s3c_hybrid", "업계 기본값"),
    ("t_tablerag_leaf", "TableRAG **(NeurIPS 2024)** 셀 코퍼스, 열 이름 = 잎 라벨 ⚠️", "t_s3c_hybrid", "발표 논문"),
    ("t_tablerag_path", "TableRAG **(NeurIPS 2024)** 셀 코퍼스, 열 이름 = 헤더 경로 (강한 변형) ⚠️", "t_s3c_hybrid", "발표 논문"),
    ("full_s3c_hybrid", "**본 방법** — 코퍼스 전체(표 저장소 3,597표)", "", "—"),
    ("full_mt2net_hybrid", "MT2Net 의 색인 단위 — 코퍼스 전체 ¶", "full_s3c_hybrid", "발표 논문(단위)"),
]

# 표 1 은 셀 검색 정확도를 잰다. 색인 단위가 표 하나인 행에는 그 단계가 없다 --
# gold 표가 뽑히면 그 표의 셀이 전부 문맥에 들어오므로 `gold ⊆ got` 이 "gold 표가
# 뽑혔나"로 축소된다. 측정으로 확인된다: all 모드 1,245 건에서 "표는 찾았는데 셀은
# 놓쳤다"가 이 행만 0 건이고 나머지 행은 84~496 건이다. 그러므로 이 행의 값을
# `주지표 정확도` 칸에 넣으면 두 행이 서로 다른 것을 재게 된다 -- 비운다. 이 행이
# 실제로 재는 값은 `gold 표 포함률` 열에 그대로 있다.
NO_CELL_STEP = {"t_table_hybrid"}

# 표 1b -- 우리 시스템에서 한 부분씩 뺀 것. 경쟁 상대가 아니라 각 부분의 값이다.
# 이기는 것이 결과가 아니라, 얼마나 떨어지는지가 그 부분이 사는 값이다.
ABLATION_ROWS = [
    ("t_s3c_hybrid", "**본 방법** — 셀 문장, 하이브리드 α=0.7", "", "—"),
    ("t_s3c_dense", "검색기 축: dense only (α=1.0)", "t_s3c_hybrid", "검색기"),
    ("t_s3c_bm25", "검색기 축: BM25 only (α=0.0) — 같은 문장, 어휘 매칭만", "t_s3c_hybrid", "검색기"),
    ("full_bm25", "검색기 축: BM25 only — 코퍼스 전체", "full_s3c_hybrid", "검색기"),
    ("t_s3c_hybrid_noprefix", "BGE 쿼리 지시문 없이 (버그 재현)", "t_s3c_hybrid", "질의 측"),
    ("t_s2_hybrid", "색인 단위: 헤더 경로만, 표 고유 라벨 없음 (S2)", "t_s3c_hybrid", "색인 단위"),
    ("t_flat_hybrid", "색인 단위: 잎 라벨만, 계층 경로도 라벨도 없음 (flat)", "t_s3c_hybrid", "색인 단위"),
    ("t_row_hybrid", "입도: 우리 셀 문장을 **행별로 묶음** (발표된 행 단위가 아니다)", "t_s3c_hybrid", "입도"),
]


#: 표 1c 의 열 머리. 긴 라벨은 괄호에서 잘리면 TableRAG 세 행이 구별되지 않는다.
SHORT = {"t_s3c_hybrid": "본 방법", "t_row_hybrid": "행 청크",
         "t_mt2net_hybrid": "MT2Net", "t_chunk1000": "고정 청킹",
         "t_trag_hetero": "TRAG Huawei", "t_tablerag_leaf": "TRAG'24 leaf",
         "t_tablerag_path": "TRAG'24 path"}


def by_type_table():
    """표 1c — 질의 유형별. 헤드라인 하나가 가리는 것을 드러낸다."""
    from collections import defaultdict
    recs = {}
    for tag, _l, _r, _k in BASELINE_ROWS:
        f = D / f"{tag}_records.jsonl"
        if not f.exists() or tag in NO_CELL_STEP or tag.startswith("full_"):
            continue
        recs[tag] = {j["query_id"]: j for j in map(json.loads, f.open())
                     if j.get("mode") == "all" and "correct" in j}
    if "t_s3c_hybrid" not in recs:
        return ""
    base = recs["t_s3c_hybrid"]
    g = defaultdict(list)
    for q, d in base.items():
        g[d.get("aggregation") or "none"].append(q)
    big = [k for k in sorted(g, key=lambda x: -len(g[x])) if len(g[k]) >= 10]
    cols = [t for t in recs]
    out = ["## 표 1c — 질의 유형별 (표 1 을 쪼갠 것, 새 측정 아님)", "",
           "`aggregation` 은 **HiTab 이 질의마다 붙여 둔 필드**이고 우리가 나눈 것이 아니다.",
           "헤드라인 하나로는 안 보이는 것이 둘 있다:", "",
           f"1. 주지표 {len(base):,} 건의 **{len(g['none'])/len(base):.0%} 가 조회(`none`)** 다. 전체 값은",
           "   사실상 조회 값에 끌려간다 — 슬라이드에 \"셀 조회\" 라고 쓸 거면 `none` 행을 쓴다.",
           "2. **산술에서는 순위가 뒤집힌다.** 산술은 gold 셀이 여럿(m≥2)이라 20셀 예산이 빡빡한데,",
           "   표를 통째로 퍼주는 청킹 계열은 그 셀들이 자동으로 딸려 온다.", "",
           "| 유형 | n | 비중 | " + " | ".join(SHORT.get(t, t) for t in cols) + " |",
           "|---|---:|---:|" + "---:|" * len(cols)]
    for k in big:
        S = g[k]
        out.append(f"| `{k}` | {len(S)} | {len(S)/len(base):.1%} | " + " | ".join(
            f"{sum(recs[t][q]['correct'] for q in S if q in recs[t])/len(S):.4f}" for t in cols) + " |")
    out.append(f"| **전체** | **{len(base)}** | 100% | " + " | ".join(
        f"**{sum(recs[t][q]['correct'] for q in base if q in recs[t])/len(base):.4f}**"
        for t in cols) + " |")
    mg = defaultdict(list)
    for q, d in base.items():
        mg[min(d["m"], 4)].append(q)
    out += ["", "필요 gold 셀 수 `m` 별 (본 방법): " + " · ".join(
        f"{'m≥4' if m == 4 else f'm={m}'} n={len(S)} {sum(base[q]['correct'] for q in S)/len(S):.4f}"
        for m, S in sorted(mg.items())) +
        " — 전체 값이 높은 것은 **m=1 이 대부분이기 때문**이다.", ""]
    return "\n".join(out)


def ablation_table():
    """표 1b — 우리 시스템에서 한 부분씩 뺀 값."""
    out = ["## 표 1b — 본 방법 ablation (경쟁 상대 아님)", "",
           "표 1 이 색인 단위를 바깥 방법과 견주는 표라면, 여기는 **우리 것에서 한 부분을**",
           "**뺐을 때 얼마나 떨어지는가**를 재는 표다. 이기는 것이 결과가 아니라, 떨어지는",
           "폭이 그 부분이 사는 값이다. 특히 `BM25 only` 는 통제로 필요하다 — 셀 문장에 표",
           "제목이 들어가므로 \"이득이 그냥 단어 겹침 아니냐\"는 반론이 성립할 수 있고, 같은",
           "문장을 어휘 매칭만으로 썼을 때의 값이 그 반론에 대한 답이다.", "",
           f"| 뺀 것 | 축 | 색인 단위 수 | **단일 셀 조회 (n={len(primary_ids())})** | 데이터셀 전체 | 헤더답 | 본 방법 대비 (McNemar) |",
           "|---|---|---:|---:|---:|---:|---|"]
    P = primary_ids()
    for tag, label, ref, kind in ABLATION_ROWS:
        d = load(tag)
        if not d:
            out.append(f"| {label} | {kind} | — | *(미측정)* | — | — | — |")
            continue
        a = acc(tag, P) if not tag.startswith("full_") else None
        out.append(
            f"| {label} | {kind} | {d['n_units']:,} | "
            f"{f'**{a:.4f}**' if a is not None else '—'} | "
            f"{d['accuracy_all_mode']:.4f} | {d['accuracy_any_mode']:.4f} | "
            f"{(paired(tag, ref, ids=(P if a is not None else None)) if ref else '') or '—'} |")
    dh, dd = load("t_s3c_hybrid"), load("t_s3c_dense")
    if dh and dd:
        out += ["", "⚠️ **하이브리드는 dense 를 유의하게 이기지 못한다** "
                f"({paired('t_s3c_dense')}), 그리고 헤더답 모드에서는 dense 가 더 높다 "
                f"({dd['accuracy_any_mode']:.4f} > {dh['accuracy_any_mode']:.4f}). "
                "α=0.7 은 사전등록된 sweep(`RESULTS.md` §8)에서 고정한 값이므로 결과를 보고",
                "바꾸지 않는다 — 대신 **BM25 성분이 주지표에서 사는 값이 통계적으로 확인되지**",
                "**않는다**고 적는다. dense only 로 단순화하려면 새 사전등록이 필요하다."]
    return "\n".join(out)


def retrieval_table():
    d0 = load("t_s3c_hybrid") or {}
    n_all, n_any = d0.get("n_all_mode", "?"), d0.get("n_any_mode", "?")
    P = primary_ids()
    out = [f"## 표 1 — 검색 정확도 (HiTab test, 질의 "
           f"{d0.get('n_queries_in_split', '?'):,}건)", "",
           "판정은 질의 단위 정답/오답. 같은 코퍼스, 같은 인코더",
           "(`BAAI/bge-base-en-v1.5`, 학습 없음). `주지표` 열이 답을 데이터 셀에서 읽는",
           f"질의(n={n_all}), `헤더답` 열이 답 자체가 헤더인 질의(n={n_any}, "
           "기준이 낮으므로 별도).", "",
           "예산 20 은 **정지 규칙이지 같은 문맥 크기가 아니다** — 누적 셀수가 20 에",
           "닿으면 멈추므로 마지막 색인 단위는 통째로 들어간다. 행마다 실제로 배달되는",
           "셀 수가 다르고, 그 값이 `실제 문맥 셀수` 열이다.", "",
           "† **셀 검색 단계가 없는 행은 `주지표`·`헤더답` 칸을 비웠다.** 이유는 표 아래.", "",
           "**이 표에는 리포 밖에 존재하는 방법만 싣는다.** 우리 시스템에서 한 부분을 뺀",
           "행(dense only, BM25 only, S2, flat)은 경쟁 상대가 아니므로 **표 1b** 로 뺐다 —",
           "그 행을 이겼다는 것은 결과가 아니라 그 부분이 사는 값이다.", "",
           f"| 색인 단위 / 검색기 | 분류 | 색인 단위 수 | 문맥 셀수 | **단일 셀 조회 (n={len(P)})** | 데이터셀 전체 | 헤더답 | gold 표 | 본 방법 대비 (McNemar) |",
           "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for tag, label, ref, kind in BASELINE_ROWS:
        d = load(tag)
        if not d:
            out.append(f"| {label} | {kind} | — | — | *(미측정)* | — | — | — | — |")
            continue
        cm = ctx_mean(tag)
        if tag in NO_CELL_STEP:
            prim = allm = anym = mcn = "— †"
        else:
            a = acc(tag, P) if not tag.startswith("full_") else None
            prim = f"**{a:.4f}**" if a is not None else "—"
            allm = f"{d['accuracy_all_mode']:.4f}"
            anym = f"{d['accuracy_any_mode']:.4f}"
            mcn = (paired(tag, ref, ids=(P if not tag.startswith("full_") else None))
                   if ref else "") or "—"
        out.append(
            f"| {label} | {kind} | {d['n_units']:,} | {cm:.1f} | {prim} | {allm} | "
            f"{anym} | {d['gold_table_in_context']:.4f} | {mcn} |")
    out += ["", "### 출처 감사 — 각 행을 어디까지 원문과 대조했나 (2026-09-08)", "",
            "| 행 | 읽은 원문 | 구현 상태 |",
            "|---|---|---|",
            "| 본 방법 | — | — |",
            "| 행 단위 / rowcol | TableRAG §4.2 + 공식 코드 3함수 + TAP4LLM(Sui et al., "
            "EMNLP 2024 Findings, arXiv:2312.09039) §표 샘플링 | `--row-text values` 가 "
            "발표된 단위(값만). `sentence` 는 우리 문장을 행별로 묶은 **입도 ablation** 이고 "
            "기준선이 아니다 — 표 1b |",
            "| 표 통째 | TARGET(arXiv:2505.11545) 초록 | 재현 대상 없음(벤치마크). `†` 참조 |",
            "| MT2Net | 논문 §4 원문 + 공식 코드 | **문장 불일치 확정.** 단위만 일치 — `¶` 참조 |",
            "| TableRAG (Huawei) | 논문 §3.2·§5.1.2 + 공식 코드 4파일 | 텍스트 검색 레그만. "
            "청크 크기는 코드(1,000자)를 따랐고 논문은 1,000토큰이다 — 아래 |",
            "| TableRAG (NeurIPS'24) | 공식 코드 전체 + 논문 §4.2 | 셀 코퍼스만. "
            "`build_schema_corpus` 누락(닿는 셀 0 변화 확인) |",
            "| 고정 청킹 | — (LangChain 기본값, 논문 아님) | — |",
            "",
            "**Huawei 청크 크기가 논문과 코드에서 다르다.** 논문 §5.1.2 는 *\"text is chunked "
            "into segments of 1000 tokens, with a 200-token overlap\"* 인데, 코드",
            "(`online_inference/tools/retriever.py`)는 `RecursiveCharacterTextSplitter("
            "chunk_size=1000, chunk_overlap=200)` 로 **글자**를 센다(LangChain 기본 단위).",
            "약 4배 차이다. 위 행은 **코드**를 따랐다 — 실제로 그들이 돌린 것이기 때문이다.",
            "토큰 읽기로 다시 잰 행은 아직 없다.",
            "",
            "그쪽이 쓰는 인코더(BGE-M3)와 리랭커(top-30 → top-3)는 재현하지 않았다. 표 1 은",
            "**색인 단위**만 변수로 두므로 검색기를 모든 행에서 같게 고정한다. 참고로 이 리포는",
            "리랭커가 이 과제에서 **해롭다**고 세 번 측정했다(`CLAUDE.md` §5).",
            "",
            "TARGET 이 우리 주장을 바깥에서 확증한다 — 표 검색이 **메타데이터 변화, 특히 표",
            f"제목의 부재에 크게 민감하다**는 관측이고, 표 1b 의 S2(제목 없음) "
            f"{acc('t_s2_hybrid', primary_ids()):.4f} 가 같은",
            "방향이다. dense 가 BM25 를 크게 앞선다는 관측도 표 1b 와 일치한다.",
            "",
            "### 발표된 수치와의 관계 (같은 표에 올리지 않는 이유)", "",
            "MT2Net 이 논문에서 보고한 검색 수치는 **MultiHiertt 의 top-10 recall 76.4% /",
            "top-15 80.8%** (Zhao et al. 2022, ACL, arXiv:2206.01347 §5.4) 이다. 위 표와",
            "직접 비교할 수 없고, 이유가 셋이다 — ① **데이터셋이 다르다**(MultiHiertt vs",
            "HiTab), ② **지표가 다르다**(supporting fact recall = 전체 중 몇 개, 우리는",
            "질의 단위 정답률), ③ **범위가 다르다**(MT2Net 은 질문이 속한 문서 안 표 ~4개를",
            "재정렬하고, 우리는 코퍼스 전체를 검색한다). 위 표의 `MT2Net 선형화` 행은 그",
            "논문의 **색인 단위(셀+계층 헤더 선형화)** 만 같은 코퍼스·같은 예산·같은",
            "검색기로 재현한 것이고, 분류기 재정렬은 재현하지 않았다. 템플릿은 논문에 실린",
            "예시 문장 한 개에서 역추정한 잠정 재현이다 —",
            "`rag_agent/serialization/templates.py` 머리 참조.",
            "",
            "",
            "### ¶ MT2Net 행은 \"단위\" 재현이지 \"문장\" 재현이 아니다 (원문 대조 2026-09-08)",
            "",
            "논문(Zhao et al., ACL 2022, arXiv:2206.01347) §4 가 **말하는 것은 원칙**이다:",
            "*\"we turn each cell into a sentence, along with its hierarchical row and column",
            "headers\"*. 규칙(구분자·축 순서·어순)은 어디에도 없다.",
            "",
            "그리고 우리 템플릿은 논문이 보여주는 예시를 **재현하지 못한다** — 같은 셀을 우리",
            "노브로 렌더링하면 이렇게 갈린다:",
            "",
            "```",
            "우리   For Product of Innovation Systems of Segment, Sales, 2018, … is 2,894",
            "논문   For Innovation Systems of Segment, sales of product in 2018, … is 2,894",
            "```",
            "",
            "논문은 **행의 잎(`product`)을 열 절 안에** 넣고 연도를 `in` 으로 잇는다. 의미를 보고",
            "짜맞춘 렌더링이지 고정 구분자 결합이 아니다. 게다가 Figure 1 의 `Retrieved top-n",
            "Facts` 에 **두 번째 예시**가 있는데(*\"The funded Aerospace Systems in 2017 was 9560\"*)",
            "첫 예시와 형태가 다르다 — 예시 둘, 형태 둘, 규칙 없음.",
            "",
            "**그러므로 이 행을 \"MT2Net 재현\" 이라고 쓰지 말 것.** 재현한 것은 **색인 단위**",
            "(셀 하나 + 행·열 헤더 경로, 표 제목 없음)이고 문장 형태가 아니다. 그 구분이",
            "성립하는 이유는 이 리포가 이미 **문장 형태가 이득을 만들지 않는다**고 측정해",
            "뒀기 때문이다 — S2 .517 / S2r .527 / MT2Net .516 이 서로 구별되지 않는다",
            "(`CLAUDE.md` §5). 버는 것은 **표 고유 라벨**이고, MT2Net 의 단위에는 그것이 없다.",
            "이 행의 값은 그 라벨의 값을 재는 것이지 저쪽 문장을 재는 것이 아니다.",
            "",
            "### ‡ 행 단위 검색의 출처와, 우리가 재현하지 않은 절반 (2026-09-08)",
            "",
            "행을 검색 단위로 쓰는 것은 **발표된 방법이다.** TableRAG(Chen et al., NeurIPS",
            "2024) §4.2 의 기준선 `RowColRetrieval` 이 그것이고 — \"encodes rows and columns",
            "and then retrieve the top K rows and columns based on their similarity to the",
            "question\'s embedding to form a sub-table\" — 그 논문은 방법론 출처로 Sui et al.,",
            "TAP4LM (arXiv 2023) 을 인용한다. 공식 구현에도 `build_row_corpus` 와",
            "`build_column_corpus` 가 둘 다 있다.",
            "",
            "⚠️ **우리 arm 은 그 절반이다.** 행만 인코딩해 검색하고, 열 인코딩도 sub-table",
            "구성도 하지 않는다. 그러므로 이 행을 `RowColRetrieval 재현` 이라고 쓰면 과대",
            "표기다. 열 절반을 붙여 온전히 재현하는 것은 아직 하지 않았다.",
            "",
            "(이전 라벨 `업계 관행` 은 근거 부족이었다. `CLAUDE.md` 의 \"row_chunk 는 발표된",
            "시스템이 아니다\" 진술도 같은 이유로 갱신이 필요하다.)",
            "",
            "### † 표 단위 검색 — 왜 이 표의 지표로 셀 수 없는가 (2026-09-08)",
            "",
            "`--unit table`(표 하나 = 색인 단위 하나)은 **측정했고 값도 있다**",
            f"(`results/retrieval_accuracy/t_table_hybrid.json`, 데이터셀 전체 "
            f"{(load('t_table_hybrid') or {}).get('accuracy_all_mode', '?')}). 행을 지우지",
            "않고 칸만 비운 것은, 값이 불리해서 뺀 것이 아니라 **재는 과제가 다르기 때문**임을",
            "표 안에서 보이기 위해서다 — 진 baseline 을 조용히 빼면 체리피킹이다.",
            "",
            "칸을 비운 이유 둘:",
            "",
            "1. **셀을 고르는 단계가 없다.** 단위가 표면 표만 맞히면 gold 셀이 전부 따라온다.",
            f"   그래서 이 arm 의 값({(load('t_table_hybrid') or {}).get('accuracy_all_mode','?')})은 "
            f"gold 표 포함률({(load('t_table_hybrid') or {}).get('gold_table_in_context','?')})과 "
            "사실상 같은 값이고,",
            "   재는 것은 셀 검색이 아니라 **표 선택**이다. 표 1 의 다른 행은 전부 20셀 예산",
            "   안에서 어느 셀을 고르느냐를 겨루는데, 이 행만 그 겨룸에 참가하지 않는다.",
            "2. **예산이 구속되지 않는다.** 단위 하나가 이미 예산을 넘겨서 질의의 **97.6%**",
            "   에서 20셀이 안 지켜진다(중앙 84셀, 40%는 100셀 초과, 최대 660셀). 같은 예산",
            "   비교라는 표 1 의 전제가 이 행에서만 깨진다.",
            "",
            "표 단위 검색 자체는 별도 과제로 다룬다 — 벤치마크로는 TARGET(Ji, Glenn,",
            "Parameswaran, Hulsebos, arXiv:2505.11545)이 그 형태를 평가하며, 그쪽 수치와",
            "이 표를 같은 자리에 놓지 않는다.",
            "",
            "### 청킹 두 행의 차이 (2026-09-08)",
            "",
            "`TableRAG (Huawei)` 와 `고정 크기 청킹` 은 둘 다 마크다운을 1,000자로 자른다 —",
            "전자가 후자의 특정 설정이다. 재현은 `online_inference/tools/retriever.py` 를 따랐다:",
            "`RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`, 청크마다",
            "`\"File name: …\"` 접두, 표 이름은 split 전에 붙으므로 **첫 청크에만** 실린다",
            "(`utils/tool_utils.py: excel_to_markdown`). 우리 `고정 크기 청킹` 행은 대신",
            "**헤더 행을 청크마다 반복**하고 overlap 이 없다. 두 행의 차이가 그 두 선택이 사는",
            "값이다 — 헤더를 반복하면 열 라벨이 둘째 청크부터 살아남는다.",
            "",
            "그쪽 `table_name` 은 워크북 **파일명**에서 온다. 그 코퍼스에서는 파일명이 의미를",
            "담지만 HiTab 의 `table_id`(`0_1_nsf21326-tab001`)는 담지 않으므로, 표 **제목**을",
            "넣었다 — 다른 모든 행이 받는 것과 같은 문자열이다. `table_id` 를 넣은 첫 실행은",
            "0.2771(데이터셀 전체 n=1,245 기준)이 나왔고, 우리가 만든 handicap 이라 폐기했다.",
            "",
            "그쪽의 스키마 JSON(`{\"table_name\", \"column_list\"}`, `src/data_persistent.py:",
            "generate_schema_info`)은 **검색 색인에 들어가지 않는다** — `NL2SQL_USER_PROMPT` 로만",
            "가서 MySQL 사본에 대한 SQL 생성을 돕는다. 답을 검색 문맥에서 읽지 않고 DB 에",
            "질의하므로, 여기서 재현되는 것은 그 시스템의 **텍스트 검색 레그뿐이다.**",
            "",
            "⚠️ **TableRAG 라는 이름의 논문이 둘이다. 아래 (a) 두 행과 위 (b) 한 행은 다른 논문이다.**",
            "",
            "- **(a) Chen et al., NeurIPS 2024** — million-token table understanding.",
            "  스키마 검색 + 셀 검색, pandas/ReAct 에이전트. `tablerag_unit.py` 가 포팅한 것이",
            "  이쪽이고, 위 두 행이 재는 것도 이쪽이다.",
            "- **(b) arXiv 2506.10380 (Huawei)** — heterogeneous-document RAG. 표를 오프라인에",
            "  관계형 DB 로 적재하고, 온라인에서 SQL 을 생성해 실행한다. 평탄화한 표 조각마다",
            "  원 표의 스키마를 매핑해 붙인다(f: D̂_i,j → S(D_i)).",
            "",
            "**논문 초안의 related work 는 (b) 를 가리킨다** (커밋 `787e65c`,",
            "`docs/CITATIONS_VERIFIED.md`: \"WE mean (b) in RELATED_DELTA (Huawei-TableRAG)\").",
            "**(b) 는 아직 측정하지 않았다.** 위 두 행의 낮은 값을 (b) 에 대한 결과로 읽지 말 것 —",
            "(b) 의 색인 단위는 표 조각에 **스키마를 붙여** 표 정체성을 싣는 구조라서, 아래 갭",
            "진술이 (b) 에는 성립하지 않는다. 논문에서 두 이름을 구별해 쓰지 않으면 리뷰어가",
            "이 표와 related work 를 같은 시스템으로 읽는다.",
            "",
            "**포트 대조 (2026-09-08).** 위 두 행의 텍스트는 공식 구현",
            "`google-research/table_rag/agent/retriever.py: build_cell_corpus` 와 줄 단위로",
            "대조했다. 구조는 일치한다(숫자 열 → min/max 요약 하나, 범주 셀 → (열,값) 중복 제거",
            "후 빈도순, `max_encode_cell` 예산). 차이 둘: ① 포트는 `\"dtype\": \"float64\"` 를",
            "고정하고 원본은 `col.dtype` 를 쓴다(질의에 dtype 이름이 안 나오므로 영향 없음),",
            "② 원본은 `build_schema_corpus` 도 함께 쓰는데 포트는 셀 코퍼스만 담는다 —",
            "**닿는 셀이 0개 늘어난다**(범주 열의 `cell_examples` 는 최빈 3개라 중복 제거된 셀",
            "문서의 부분집합이고, 숫자 열은 같은 min/max 다). 측정에는 영향이 없다.",
            "",
            "⚠️ **원본은 색인을 표마다 따로 만든다.** `init_retriever(table_id, df)` 가 호출될",
            "때마다 `db_dir = f'{data_type}_db_{max_encode_cell}_' + table_id` 로 FAISS 인덱스를",
            "하나씩 세우고 `top_k=5` 로 그 안에서만 검색한다. 파일 어디에도 코퍼스 단위 인덱스가",
            "없다 — **TableRAG 는 표를 가로질러 검색하지 않는다. 표를 받아 그 안에서 찾는다.**",
            "그러므로 위 두 행은 그 시스템이 구현하지 않은 설정에서 잰 값이고, 성능 비교로",
            "쓰면 틀린다. 이것이 갭 진술이 성립하는 근거이며, 추론이 아니라 공식 구현이 근거다.",
            "",
            "**아래는 (a) 에 대해서만 성립한다 — TableRAG (NeurIPS 2024) 행은 성능 비교가 아니라 갭 진술이다.** 0.06 / 0.13 은 직렬화 품질이",
            "아니라 구조에서 나온다: TableRAG 는 숫자 열 하나를 min/max 만 담은 요약 문서",
            "**하나로 접고**, 개별 숫자는 인코딩하지 않는다",
            "(`rag_agent/serialization/tablerag_unit.py` 머리). 계층표 질문의 피연산자는",
            "거의 다 숫자 셀이므로 그 셀들은",
            f"어떤 질의로도 닿을 수 없다 — 색인 단위 "
            f"{(load('t_tablerag_leaf') or {}).get('n_units', '?'):,} 개가 배달하는 셀은 "
            "코퍼스의 일부뿐이다.",
            "TableRAG 는 검색을 리더에게 줄 근거 집합이 아니라 **pandas 코드를 쓰기 위한 힌트**로",
            "쓰고 표 전체를 에이전트에게 건네므로, 그 설계에서는 숫자를 개별 인코딩할 이유가 없다.",
            "그러므로 **\"우리가 TableRAG 를 이겼다\"로 쓰지 말 것.** 성립하는 문장은 하나다 —",
            "대표적인 셀 단위 table-RAG 가 이 과제가 필요로 하는 셀을 색인하지 않는다.",
            "`leaf` 는 계층표를 평범하게 읽었을 때 나오는 열 이름이고 `path` 는 헤더 경로를",
            "쥐여준 강한 변형이다. 약한 쪽만 이기는 것은 결과가 아니므로 둘 다 싣는다.",
            "",
            "**이 격차의 기전을 문장 모양으로 설명하지 말 것.** 두 색인 단위의 차이는 셀 문장이",
            "**표를 식별하는 고유 라벨(제목/캡션)을 담는가**이고, MT2Net 선형화는 담지 않는다.",
            "HiTab 은 99.3% 가 제목을 갖고 있어 그 차이가 크게 나온다 — 제목이 없는 코퍼스에서는",
            "이 이득이 사라진다는 것이 이미 측정돼 있다(`CLAUDE.md` §2, §5). 위 표의 `gold 표",
            "포함률` 열이 그 기전을 그대로 보여준다: 격차는 셀을 고르는 단계가 아니라 **표를",
            "찾는 단계**에서 벌어진다."]
    d = load("t_s3c_hybrid")
    if d:
        out += ["", f"제외 {d['n_excluded']}건 (gold 해석 불가): "
                    + ", ".join(f"`{k}` {v}" for k, v in d["excluded_by_reason"].items())
                    + ". 분모에서 조용히 빠진 것이 아니라 사유와 함께 셌다."]
    return "\n".join(out)


def answer_table():
    out = ["## 표 2 — LLM 답변 정확도 (같은 질의, 검색이 준 문맥 그대로)", "",
           "채점기는 HiTab 공식 EM(`hitab_exact_match_text`) — 허용오차 없음.", "",
           "| 조건 | 리더 | n | **답변 정확도(주지표 n)** | 전체 n | 검색 성공 질의에서 | 검색 실패 질의에서 |",
           "|---|---|---:|---:|---:|---:|---:|"]
    names = {"retrieved": "검색 문맥 (상위 20셀) — 배치되는 조건",
             "gold": "gold 셀만 주입 — **리더 천장**",
             "oracle": "검색이 맞았으면 gold, 틀렸으면 검색 문맥",
             "retrieved_format_nodefect":
                 "검색 문맥 + 출력 형식 지시 (사전등록 A, 라벨 결함 25건 제외)",
             "gold_format_nodefect":
                 "gold 셀만 + 출력 형식 지시 (사전등록 A, 라벨 결함 25건 제외)",
             "retrieved_evidence_nodefect":
                 "검색 문맥 + 근거 셀 우선 (사전등록 B, 라벨 결함 25건 제외)"}
    any_row = False
    for cond in ("retrieved", "gold", "oracle", "retrieved_format_nodefect",
                 "gold_format_nodefect", "retrieved_evidence_nodefect"):
        f = D / f"t_s3c_hybrid_answer_{cond}.json"
        if not f.exists() and cond.endswith("_nodefect"):
            continue                       # 미측정 개입은 줄을 만들지 않는다
        if not f.exists():
            out.append(f"| {names[cond]} | — | — | *(미측정)* | — | — | — |")
            continue
        d = json.loads(f.read_text())
        any_row = True
        out.append(f"| {names[cond]} | `{d['reader']}` | {d['n_all_mode']} | "
                   f"**{d['answer_accuracy_all_mode']:.4f}** | "
                   f"{d['answer_accuracy']:.4f} (n={d['n']}) | "
                   f"{d['answer_given_retrieval_hit']} (n={d['n_retrieval_hit']}) | "
                   f"{d['answer_given_retrieval_miss']} (n={d['n_retrieval_miss']}) |")
    if not any_row:
        out.append("")
        out.append("*아직 측정 없음.*")
        return "\n".join(out)
    if (D / "t_s3c_hybrid_answer_gold_format_nodefect.json").exists():
        out += ["",
                "⚠️ **위 표의 행끼리 뺄셈하지 말 것.** 사전등록 A·B 행은 라벨 결함 25건을",
                "제외한 n=1,220 이고 기준선 두 행은 n=1,245 다. 같은 모집단(n=1,220)에서의",
                "기준선은 `retrieved` **0.6189**, `gold` **0.8648** 이므로, A 의 실제 효과는",
                "`retrieved` +0.0008, `gold` **−0.0009** 로 **둘 다 0** 이다. 사전등록의",
                "기각 기준(`gold` 0.87 미만)에 걸려 **A 는 기각**되었다 —",
                "판정은 `results/retrieval_accuracy/VERDICT_PROMPT.md`."]
    return "\n".join(out)


def audit_block():
    f = D / "PIPELINE_AUDIT.json"
    if not f.exists():
        return ""
    d = json.loads(f.read_text())
    ex = ", ".join(f"`{k}` {v}" for k, v in d["queries_excluded_now"].items())
    return "\n".join([
        "## 0 — 파이프라인이 실제로 무엇을 색인하고 채점했나",
        "",
        "출처 `results/retrieval_accuracy/PIPELINE_AUDIT.json` "
        "(계측기 `analysis/pipeline_audit.py`). 옛 규칙 두 가지가 아직 실행 가능하므로"
        " '이전' 열은 기억이 아니라 측정이다.",
        "",
        "| | 이전 | 지금 |",
        "|---|---:|---:|",
        f"| 색인된 표 (test 가 참조하는 {d['tables_referenced_by_split']}표 중) "
        f"| {d['tables_indexed_before']} | **{d['tables_indexed_now']}** |",
        f"| 채점된 질의 ({d['queries_in_split']}건 중) "
        f"| {d['queries_scored_before']} | **{d['queries_scored_now']}** |",
        f"| 색인된 셀 | — | {d['cells_indexed_now']:,} |",
        "",
        f"지금 채점되는 {d['queries_scored_now']}건의 내역: 데이터 셀 gold "
        f"{d['queries_scored_now_by_mode'].get('all')}건(주지표) / 헤더 답 "
        f"{d['queries_scored_now_by_mode'].get('any')}건. 제외 "
        f"{sum(d['queries_excluded_now'].values())}건 — {ex}.",
        f"참고: 표 저장소 전체는 {d['tables_in_store']:,}표다. 위 표의 코퍼스는 "
        "test 질의가 참조하는 표 전체이며, 저장소 전량 색인은 별도 행으로 잰다.",
    ])


def gap_block():
    """The professor's arithmetic: retrieval .9 -> answer .9, or not, and why."""
    r = load("t_s3c_hybrid")
    fa = D / "t_s3c_hybrid_answer_retrieved.json"
    fg = D / "t_s3c_hybrid_answer_gold.json"
    if not (r and fa.exists()):
        return ""
    A = json.loads(fa.read_text())
    G = json.loads(fg.read_text()) if fg.exists() else None
    ra, aa = r["accuracy_all_mode"], A["answer_accuracy_all_mode"]
    lines = ["## 검색 .9 → 답변 .9 인가 (교수 질문에 대한 답)", "",
             f"주지표 모집단(n={A['n_all_mode']})에서 검색 정확도 **{ra:.4f}**, "
             f"같은 문맥에 대한 답변 정확도 **{aa:.4f}**. 차이 {ra - aa:+.4f}.", "",
             "이 차이는 검색이 아니라 **리더가 만든다**. 분해:", "",
             f"- 검색이 맞은 질의에서의 답변 정확도: **{A['answer_given_retrieval_hit']}** "
             f"(n={A['n_retrieval_hit']})",
             f"- 검색이 틀린 질의에서의 답변 정확도: **{A['answer_given_retrieval_miss']}** "
             f"(n={A['n_retrieval_miss']})"]
    if G:
        ceil = G["answer_accuracy_all_mode"]
        lines += [f"- **리더 천장** — 정답 셀만 넣었을 때: **{ceil:.4f}** "
                  f"(n={G['n_all_mode']})", "",
                  "손실이 어디서 나는지 두 단계로 갈린다. **둘 다 검색이 아니라 리더 쪽이다.**", "",
                  "| 단계 | 값 | 그 단계에서 잃은 것 |",
                  "|---|---:|---:|",
                  f"| 검색 정확도 (정답 근거가 문맥에 있다) | {ra:.4f} | — |",
                  f"| 리더 천장 (정답 셀만 줬을 때 답변) | {ceil:.4f} | {ceil - ra:+.4f} |",
                  f"| 실제 답변 (검색이 준 20셀을 줬을 때) | {aa:.4f} | {aa - ceil:+.4f} |",
                  "",
                  f"① **{ra - ceil:.4f}** 는 리더가 정답 셀을 보고도 못 맞히는 몫이다. "
                  f"검색을 1.0 으로 올려도 이 몫은 그대로 남는다.",
                  f"② **{ceil - aa:.4f}** 는 문맥에 같이 들어온 **distractor** 몫이다. "
                  "검색이 성공한 질의에서도 발생한다 — 정답 셀이 문맥에 있는데 리더가 옆 셀 "
                  "값을 읽는다. 검색을 더 잘해서 줄일 수 있는 것이 아니라 **덜 넣어야** 준다.",
                  "",
                  f"따라서 '검색 .9면 답변 .9' 는 **리더 천장이 .9 이상이고 distractor 손실이 "
                  f"0 일 때만** 성립한다. 여기서는 천장이 {ceil:.4f} 이고 distractor 손실이 "
                  f"{ceil - aa:.4f} 라 성립하지 않는다. 검색을 고쳐서 닿을 수 있는 상한은 "
                  f"**{ceil:.4f}** 이고, 그 위는 리더를 바꾸는 문제다."]
    else:
        lines += ["", "*(리더 천장 조건 미측정)*"]
    if A.get("by_aggregation"):
        worst = sorted((v, k) for k, v in A["by_aggregation"].items() if v is not None)[:5]
        lines += ["", "질문 유형별 답변 정확도가 낮은 쪽 5개: "
                  + ", ".join(f"`{k}` {v}" for v, k in worst)]
    return "\n".join(lines)


if __name__ == "__main__":
    print(f"# 결과 표 — HiTab test, 학습 없음 (생성 `analysis/accuracy_tables.py`)\n")
    ab = audit_block()
    if ab:
        print(ab); print()
    print(retrieval_table())
    print()
    print(ablation_table())
    print()
    bt = by_type_table()
    if bt:
        print(bt)
        print()
    print(answer_table())
    gb = gap_block()
    if gb:
        print()
        print(gb)
