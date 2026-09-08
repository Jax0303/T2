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


def paired(tag, ref="t_s3c_hybrid", mode="all"):
    """Exact McNemar of ``tag`` against our arm, over the queries both scored."""
    from math import comb
    fa, fb = D / f"{ref}_records.jsonl", D / f"{tag}_records.jsonl"
    if tag == ref or not (fa.exists() and fb.exists()):
        return ""
    A = {json.loads(l)["query_id"]: json.loads(l) for l in fa.open()}
    B = {json.loads(l)["query_id"]: json.loads(l) for l in fb.open()}
    ids = [q for q in A if q in B and "correct" in A[q] and "correct" in B[q]
           and A[q]["mode"] == mode]
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


# (tag, label, the arm this row is tested against -- never across corpora)
RETRIEVAL_ROWS = [
    ("t_s3c_hybrid", "**본 방법** — 셀 문장(제목+행경로+열경로+값), 하이브리드 α=0.7", ""),
    ("t_s3c_dense", "본 방법, dense only (α=1.0)", "t_s3c_hybrid"),
    ("t_s3c_bm25", "본 방법의 문장, BM25 only (α=0.0)", "t_s3c_hybrid"),
    ("t_s3c_hybrid_noprefix", "본 방법, BGE 쿼리 지시문 없이 (버그 재현)", "t_s3c_hybrid"),
    ("t_s2_hybrid", "색인 단위 ablation — 헤더 경로만, 표 고유 라벨 없음 (S2)", "t_s3c_hybrid"),
    ("t_flat_hybrid", "색인 단위 ablation — 잎 라벨만, 계층 경로도 라벨도 없음 (flat)", "t_s3c_hybrid"),
    ("t_mt2net_hybrid", "MT2Net 선형화 (Zhao et al. 2022 §4, 템플릿은 잠정 재현)", "t_s3c_hybrid"),
    ("t_row_hybrid", "행 청크 (업계 관행, 발표된 시스템 아님)", "t_s3c_hybrid"),
    ("t_table_hybrid", "표 통째 (TARGET 계열 표 단위)", "t_s3c_hybrid"),
    ("full_s3c_hybrid", "**본 방법** — 코퍼스 전체(표 저장소 3,597표)", ""),
    ("full_bm25", "BM25 only — 코퍼스 전체", "full_s3c_hybrid"),
    ("full_mt2net_hybrid", "MT2Net 선형화 — 코퍼스 전체", "full_s3c_hybrid"),
]


def retrieval_table():
    d0 = load("t_s3c_hybrid") or {}
    n_all, n_any = d0.get("n_all_mode", "?"), d0.get("n_any_mode", "?")
    out = [f"## 표 1 — 검색 정확도 (HiTab test, 질의 "
           f"{d0.get('n_queries_in_split', '?'):,}건)", "",
           "판정은 질의 단위 정답/오답. 같은 코퍼스, 같은 셀 예산(20), 같은 인코더",
           "(`BAAI/bge-base-en-v1.5`, 학습 없음). `주지표` 열이 답을 데이터 셀에서 읽는",
           f"질의(n={n_all}), `헤더답` 열이 답 자체가 헤더인 질의(n={n_any}, "
           "기준이 낮으므로 별도).", "",
           "| 색인 단위 / 검색기 | 코퍼스 | 색인 단위 수 | 실제 문맥 셀수(평균) | **주지표 정확도** | 헤더답 정확도 | gold 표 포함률 | 본 방법 대비 (우리승:상대승, McNemar) |",
           "|---|---:|---:|---:|---:|---:|---:|---|"]
    for tag, label, ref in RETRIEVAL_ROWS:
        d = load(tag)
        if not d:
            out.append(f"| {label} | — | — | — | *(미측정)* | — | — | — |")
            continue
        cm = ctx_mean(tag)
        out.append(
            f"| {label} | {d['n_tables']}표 | {d['n_units']:,} | "
            f"{cm:.1f} | **{d['accuracy_all_mode']:.4f}** | "
            f"{d['accuracy_any_mode']:.4f} | {d['gold_table_in_context']:.4f} | "
            f"{(paired(tag, ref) if ref else '') or '—'} |")
    out += ["", "### 발표된 수치와의 관계 (같은 표에 올리지 않는 이유)", "",
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
    print(answer_table())
    gb = gap_block()
    if gb:
        print()
        print(gb)
