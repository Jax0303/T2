# SPDX-License-Identifier: MIT
"""2026-09-13 작업 기록 — 레코드에서 표를 다시 세어 `REPORT-2026-09-13.md` 를 쓴다.

요약 JSON 이 아니라 레코드에서 센다. 파일이 없으면 그 칸은 '—' 로 두고 실행 로그에 남긴다.
검정은 짝지음 McNemar(정확 이항), 한 표 안의 비교 묶음마다 Holm 보정.

  PYTHONPATH=. .venv/bin/python analysis/mh_final_report.py
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import sys
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
MH = ROOT / "results/mh_arms"
OUT = ROOT / "REPORT-2026-09-13.md"
LAYERS = ["lookup_m1", "lookup_m2+", "arith_m1", "arith_m2+", "ALL"]
LNAME = {"lookup_m1": "단일 조회", "lookup_m2+": "다중 조회", "arith_m1": "단일 산술",
         "arith_m2+": "다중 산술", "ALL": "전체"}

# 조건별 arm -> 태그. MT2Net 원문 문장은 표 파서·라벨과 무관한 텍스트라 v1 레그 하나를 기준으로 쓴다.
RET = {
    "v1 (헤더 수정 전, 라벨 없음)": [
        ("본 방법", "mh_cell"), ("MT2Net 원문", "mh_mt2net"),
        ("MT2Net 헤더+본 방법 형태", "mh_mt2net_header_s3c"), ("chunk", "mh_chunk"),
        ("Huawei TableRAG", "mh_huawei"), ("RowCol", "mh_rowcol"),
        ("TableRAG (유리한 설정)", "mh_tablerag_allobj_path")],
    "v2 (헤더 수정, 라벨 없음)": [
        ("본 방법", "mh_cell_hv2"), ("MT2Net 원문", "mh_mt2net"),
        ("chunk", "mh_chunk_hv2"), ("Huawei TableRAG", "mh_huawei_hv2"),
        ("RowCol", "mh_rowcol_hv2"), ("TableRAG (유리한 설정)", "mh_tablerag_allobj_path_hv2")],
    "v2+L1 (헤더 수정 + 표 고유 라벨)": [
        ("본 방법", "mh_cell_hv2_L1"), ("MT2Net 원문", "mh_mt2net"),
        ("MT2Net 원문+라벨", "mh_mt2net_desc_label_hv2_L1"), ("chunk", "mh_chunk_hv2_L1"),
        ("Huawei TableRAG", "mh_huawei_hv2_L1"), ("RowCol", "mh_rowcol_hv2_L1"),
        ("TableRAG (유리한 설정)", "mh_tablerag_allobj_path_hv2_L1")],
}
GOLD = {"v1 (헤더 수정 전, 라벨 없음)": "mh_GOLD_doc", "v2 (헤더 수정, 라벨 없음)": "mh_GOLD_hv2_doc",
        "v2+L1 (헤더 수정 + 표 고유 라벨)": "mh_GOLD_hv2_L1_doc"}
missing: list = []


def load(path):
    p = MH / path
    if not p.exists():
        missing.append(str(p.relative_to(ROOT)))
        return None
    return {r["query_id"]: r for r in map(json.loads, p.open(encoding="utf-8"))}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [1.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def mcnemar(a, b, ids, key):
    w = sum(1 for i in ids if key(a[i]) and not key(b[i]))
    l = sum(1 for i in ids if not key(a[i]) and key(b[i]))
    return w, l, (binomtest(w, w + l, .5).pvalue if w + l else 1.0)


def fmt_p(p):
    return f"{p:.2g}" if p >= 1e-4 else f"{p:.1e}"


def retrieval_section(md):
    md.append("## 2. MultiHiertt 검색 정확도 (train, 셀 20개, 주지표 all)\n")
    md.append("같은 조건 표 안에서는 **모든 arm 이 가진 질의의 교집합**으로 센다(분모가 같다).\n")
    for cond, arms in RET.items():
        recs = [(name, load(f"{tag}_records.jsonl")) for name, tag in arms]
        recs = [(n, {k: v for k, v in r.items() if "doc" in v}) for n, r in recs if r]
        if len(recs) < 2 or recs[0][0] != "본 방법":
            md.append(f"### {cond}\n\n레코드 부족 — 생략.\n")
            continue
        ids = sorted(set.intersection(*[set(r) for _n, r in recs]))
        base = recs[0][1]
        for scope in ("doc", "corpus"):
            md.append(f"### {cond} — {scope} 범위 (query count={len(ids)})\n")
            md.append("| arm | " + " | ".join(LNAME[k] for k in LAYERS) + " |")
            md.append("|---" * (len(LAYERS) + 1) + "|")
            for name, r in recs:
                cells = []
                for k in LAYERS:
                    q = [i for i in ids if k == "ALL" or base[i]["layer"] == k]
                    cells.append(f"{sum(r[i][scope]['correct'] for i in q) / len(q):.4f}" if q else "—")
                md.append(f"| {name} | " + " | ".join(cells) + " |")
            tests = []
            for name, r in recs[1:]:
                for k in ("ALL", "lookup_m2+", "arith_m2+"):
                    q = [i for i in ids if k == "ALL" or base[i]["layer"] == k]
                    w, l, p = mcnemar(base, r, q, lambda x, s=scope: x[s]["correct"])
                    tests.append((name, k, w, l, p))
            adj = holm([t[4] for t in tests])
            md.append("\n본 방법 대 각 arm (본 방법 승 : 상대 승, Holm 보정 p):\n")
            md.append("| 상대 arm | 전체 | 다중 조회 | 다중 산술 |")
            md.append("|---|---|---|---|")
            for name, _r in recs[1:]:
                row = [f"{w}:{l}, p={fmt_p(p2)}" for (n2, k, w, l, _p), p2 in zip(tests, adj) if n2 == name]
                md.append(f"| {name} | " + " | ".join(row) + " |")
            md.append("")


def answer_section(md):
    md.append("## 3. MultiHiertt 답변 EM (doc 범위, 같은 1,047개 질의, Qwen2.5-7B-Instruct 4bit, 답만 출력)\n")
    md.append("채점: MultiHiertt 공식 채점기 포팅. 산술 EM 은 리더 프로토콜(풀이 금지) 때문에 바닥이다 — §5.\n")
    for cond, arms in RET.items():
        recs = [(name, load(f"{tag}_answer_doc.jsonl")) for name, tag in arms]
        recs = [(n, r) for n, r in recs if r]
        if not recs or recs[0][0] != "본 방법":
            md.append(f"### {cond}\n\n답변 레코드 부족 — 생략.\n")
            continue
        g = load(f"{GOLD[cond]}.jsonl")
        ids = sorted(set.intersection(*[set(r) for _n, r in recs]))
        base = recs[0][1]
        md.append(f"### {cond} (query count={len(ids)})\n")
        md.append("| arm | " + " | ".join(f"{LNAME[k]} 검색 / EM" for k in LAYERS) + " |")
        md.append("|---" * (len(LAYERS) + 1) + "|")
        rows = recs + ([("**gold 셀만 (리더 천장)**", g)] if g else [])
        for name, r in rows:
            cells = []
            for k in LAYERS:
                q = [i for i in ids if (k == "ALL" or base[i]["layer"] == k) and i in r]
                if not q:
                    cells.append("—"); continue
                em = sum(r[i]["answer_correct"] for i in q) / len(q)
                rt = ("—" if r is g else f"{sum(r[i]['retrieval_correct'] for i in q) / len(q):.3f}")
                cells.append(f"{rt} / {em:.3f}")
            md.append(f"| {name} | " + " | ".join(cells) + " |")
        tests = []
        for name, r in recs[1:]:
            for k in ("ALL", "lookup_m2+", "arith_m2+"):
                q = [i for i in ids if k == "ALL" or base[i]["layer"] == k]
                w, l, p = mcnemar(base, r, q, lambda x: x["answer_correct"])
                tests.append((name, k, w, l, p))
        if tests:
            adj = holm([t[4] for t in tests])
            md.append("\n본 방법 대 각 arm, 답변 EM (본 방법 승 : 상대 승, Holm 보정 p):\n")
            md.append("| 상대 arm | 전체 | 다중 조회 | 다중 산술 |")
            md.append("|---|---|---|---|")
            for name, _r in recs[1:]:
                row = [f"{w}:{l}, p={fmt_p(p2)}" for (n2, k, w, l, _p), p2 in zip(tests, adj) if n2 == name]
                md.append(f"| {name} | " + " | ".join(row) + " |")
        md.append("")


def ablation_section(md):
    md.append("## 4. 본 방법 조건 사다리 (헤더 수정, 라벨)\n")
    md.append("| 조건 | 검색 doc 전체 | 검색 corpus 전체 | 답변 EM 전체 (1,047) |")
    md.append("|---|---:|---:|---:|")
    prev = None
    for cond, tag in (("v1", "mh_cell"), ("v2", "mh_cell_hv2"), ("v2+L1", "mh_cell_hv2_L1")):
        p = MH / f"{tag}.json"
        a = load(f"{tag}_answer_doc.jsonl")
        if not p.exists():
            md.append(f"| {cond} | — | — | — |"); continue
        b = json.loads(p.read_text())["by_layer"]["ALL"]
        em = f"{sum(x['answer_correct'] for x in a.values()) / len(a):.3f}" if a else "—"
        md.append(f"| {cond} | {b['doc']['accuracy_all']:.4f} | {b['corpus']['accuracy_all']:.4f} | {em} |")
    md.append("\n각 단계 수치는 레코드 요약(모집단 v1 2,871 / v2 2,878)이다. 짝지은 비교는 §2 표를 본다.\n")


def overflow_section(md):
    md.append("## 6. 인코더 입력 절단 감사 (bge-base 512토큰, v2+L1)\n")
    try:
        from transformers import AutoTokenizer
        import mh_arms as M
        from retrieval_accuracy import build_corpus
        tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
        _q, docs, _s = M.load_population("train")
        tables, _h = M.build_tables(docs, "v2", "L1")
        md.append("| arm | 단위 수 | 512토큰 초과 비율 |")
        md.append("|---|---:|---:|")
        for name, unit, kw in (("chunk", "chunk", {}), ("Huawei TableRAG", "trag_hetero", {}),
                               ("RowCol", "rowcol", {"row_text": "values"})):
            texts, *_ = build_corpus("", sorted(tables), "s3c", unit, {}, 1000, "leaf",
                                     kw.get("row_text", "sentence"), 200, None,
                                     load=lambda tid, _d: tables.get(tid))
            lens = tok(texts, add_special_tokens=True)["input_ids"]
            md.append(f"| {name} | {len(texts)} | {sum(len(x) > 512 for x in lens) / len(lens):.2%} |")
        md.append("\n초과분은 인코더가 잘라서 임베딩한다(모든 청킹 arm 공통 조건).\n")
    except Exception as e:                                   # 기록은 남기고 멈추지 않는다
        md.append(f"감사 실패: `{type(e).__name__}: {e}`\n")


def pilot_section(md):
    md.append("## 5. 산술 리더 파일럿 (탐색용, validation, gold 셀만)\n")
    p = MH / "reader_cot_pilot_validation.json"
    if not p.exists():
        md.append("파일럿 결과 없음.\n"); missing.append(str(p.relative_to(ROOT))); return
    s = json.loads(p.read_text())
    md.append(f"query count={s['n']} (seed {s['seed']}), 리더 {s['reader']}. **보고용 수치가 아니다** — 다음 사전등록의 근거.\n")
    md.append("| 프롬프트 | EM | 'Final answer' 표시 누락 |")
    md.append("|---|---:|---:|")
    md.append(f"| 답만 (현행 neutral, 64토큰) | {s['direct']['em']:.3f} | — |")
    md.append(f"| 풀이 허용 CoT (384토큰) | {s['cot']['em']:.3f} | {s['cot']['marker_missing']} |")
    md.append("")


def hitab_section(md):
    md.append("## 1. HiTab (test) — 오늘 확인한 것\n")
    from analysis import audit_2026_09_13 as A
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        A.gap_table()
        A.scope_ladder()
    text = buf.getvalue().replace("## 표 1", "### 표 1-1").replace("## 표 2", "### 표 1-2")
    md.append(text)


def runlog_section(md):
    md.append("## 7. 실행 기록\n")
    for log in ("run_ans.log", "run_hv2_cell.log", "run_label.log", "run_v2base.log"):
        p = MH / log
        if not p.exists():
            continue
        lines = p.read_text(errors="replace").splitlines()
        bad = [l for l in lines if l.startswith("exit=") and not l.startswith("exit=0")]
        md.append(f"- `{log}`: 단계 {sum(l.startswith('#####') for l in lines)}개, "
                  f"비정상 종료 {len(bad)}개" + (f" ({'; '.join(bad)})" if bad else ""))
    if missing:
        md.append("\n없어서 비워 둔 파일:\n")
        md.extend(f"- `{m}`" for m in sorted(set(missing)))
    md.append("")


HEADER = """# 2026-09-13 작업 기록 — 비교군 실사, 검색-EM 간극, MultiHiertt 다중 조회·산술

> 이 문서는 `analysis/mh_final_report.py` 가 레코드에서 표를 다시 세어 만든다. 서술 부분은
> 실행 전에 적은 것이고, 수치는 전부 아래 표에서 온다.

## 0. 오늘 한 일과 결정 (요약)

**비교군 구현 실사** (원본 저장소를 받아 함수를 실행해 대조, `VERIFICATION-2026-09-13.md`)
- HiTab 공식 채점기 포팅: 원본과 일치, gold 를 예측으로 넣으면 EM 1.0.
- 채점한 셀 = 배달한 단위 = 리더가 본 문자열: 모든 레코드에서 강제됨.
- **고친 것 ①** 비교군을 자기 논문의 검색 범위 밖(538표 전역)에서만 재고 있었다 → 표 1개 범위 추가.
- **고친 것 ②** TableRAG 숫자 열 처리가 원본(pandas dtype + 중복 열 이름) 동작과 달랐다 → 유리한 읽기 추가.
- **고친 것 ③** MultiHiertt 스크립트의 gold 해석이 arm 에 의존해 TableRAG 만 188건이 제외됐다 → arm 무관으로.
- **고친 것 ④** MultiHiertt 헤더 행 추정이 괄호 없는 단위 표기·연도 행을 놓쳤다(경로 충돌 21.9%)
  → 규칙 v2 (`PREREG-2026-09-13-header-units-note.md`, 검증 지표 통과 후 채택).

**검색-EM 간극** (교수님 지적): 코드 결함이 아니다. HiTab 조회 질의에 gold 셀만 주면 EM .95.
간극 = 함께 배달된 distractor 셀 + 리더의 산술 한계 (§1 표 1-1).

**방법 정의** (사용자 확정, `CLAUDE.md` §1): 셀을 구별되게 만드는 데이터(고유 라벨 + 행/열 경로 + 값)로
문장을 생성해 임베딩한다. MultiHiertt 라벨은 표 직전 문단(L1), validation 분할의 구별력으로 선택
(`PREREG-2026-09-13-table-label.md`).

**표 라벨(L1) 결과와 사전등록 이탈**: 표 직전 문단을 라벨로 붙이자 본 방법 검색이 **유의하게 떨어졌다**
(doc .8495 → .8304, p=.0014; corpus .2654 → .2264, p=7.9e-8). 같은 라벨의 Huawei 도 .696 → .647.
사전등록 예측 1(corpus 상승)은 틀렸고 예측 3(잡음 문장이면 하락)이 맞았다. 선택 지표(U_doc, 문장 구별력)는
검색 정확도를 예측하지 못했다. 이 결과로 **공정 비교 기준 조건을 v2(헤더 수정, 라벨 없음)로 두고**, 사용자
결정에 따라 라벨 레그의 답변 EM 6개·gold, TableRAG·MT2Net 라벨 검색을 **실행하지 않았다**(이탈). 라벨 규칙을
L2 로 바꿔 재실행하지 않았다 — 결과를 보고 규칙을 고르는 것이 되기 때문이다.

**MT2Net 위치** (원문 arXiv:2206.01347 확인): RAG 가 아니라 문서 내 지도학습 근거 선택기. 시스템이 아니라
**라벨 없는 선행 색인 단위**로 비교한다. MT2Net 문장은 데이터셋 구조 주석으로 만든 것이다.

**리더 산술**: "7B 가 산술을 못 한다"는 과한 서술이었다. Qwen2.5-7B-Instruct 는 GSM8K 91.6 / MATH 75.5
(arXiv:2412.15115), 가중치 4bit 는 거의 무손실(arXiv:2504.04823). 현행 프로토콜이 풀이를 금지한다(§5).
표 산술에서 7~8B 가 GPT-4o 의 절반 수준이라는 선행 결과는 TableBench(arXiv:2408.09174),
DocMath-Eval(arXiv:2311.09805).

**표본 크기**: 5pt EM 차이를 잡으려면 짝지은 질의 약 1,000건(query count=100 이면 검정력 .10~.14).
"""

NEXT = """## 8. 다음 단계 (2026-09-13 사용자와 확정)

**A. 리더 교체 검증** — 리더가 병목인지 확인한다. 모든 방법에 같은 리더·프롬프트·greedy·질의.
1. 파일럿: validation 산술 60건, gold 셀만, `local:Qwen/Qwen3-8B?quantization=4bit` (생각 모드 끔)으로
   답만 / 풀이 허용 두 프롬프트. §5 의 Qwen2.5-7B 파일럿과 나란히 보고 질의당 생성 속도를 잰다.
   근거: Qwen3 기술 보고서 Table 18 비생각 모드 MATH-500 Qwen3-8B 87.4 대 Qwen2.5-7B-Instruct 77.6.
   생각 모드는 greedy 금지·최대 32,768토큰이라 제외. Qwen2.5-Math(수학 외 비권장), R1-Distill(긴 추론),
   Table-R1(HiTab train 학습)은 제외 사유를 기록.
2. 사전등록: 리더·프롬프트 선택 규칙(gold 문맥 산술 EM 최고, 시간 허용), 표본(조회 1,047 + 산술 최대 1,000).
3. 본 실행: v2(헤더 수정·라벨 없음) 조건 7개 방법 답변 EM 재측정. 생성 속도에 따라 배치 생성 구현.
   현재 리더 결과도 함께 싣는다.

**B. 보강 (선택)**
4. HiTab 답변 EM: 표 1개 범위, TableRAG 수정판.
5. MT2Net 검색 모듈(학습된 분류기)과 dev 분할 비교 — 공개 체크포인트. 학습 없는 검색이 이길 가능성은 낮다.
6. RAG 비교군 추가 후보 T-RAG(arXiv:2504.01346, 코드 공개). FT-RAG(arXiv:2605.01495)는 코드 미확인.

**C. 논문 서술**
7. MT2Net 은 "선행 표현(비RAG)" 블록. 주장은 "학습 없이 코퍼스 전체에서 검색, RAG 비교군 대비 검색 우위,
   데이터셋 제목이 있으면(HiTab) MT2Net 문장 대비 우위, 없으면(MultiHiertt) 동등".
8. 문단 캡션 라벨(L1)은 검색을 떨어뜨린 부정적 결과로 싣는다. 라벨 규칙을 바꿔 재실행하지 않는다.
"""


def main():
    md = [HEADER]
    hitab_section(md)
    retrieval_section(md)
    answer_section(md)
    ablation_section(md)
    pilot_section(md)
    overflow_section(md)
    runlog_section(md)
    md.append(NEXT)
    OUT.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {OUT}  (없는 파일 {len(set(missing))}개)")


if __name__ == "__main__":
    main()
