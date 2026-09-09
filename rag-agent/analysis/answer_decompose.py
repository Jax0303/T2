#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""검색 실패와 리더 실패를 가른다 — arm × 모집단 표.

`accuracy_tables.py` 가 `build()` 를 불러 `TABLES.md` 안에 그대로 싣는다 —
표 1 과 표 2 가 한 파일에 있어야 "검색 .9 면 답변 .9 인가"를 나란히 읽는다
(`CLAUDE.md` §0.2).

`End-to-end EM` 하나로는 어느 쪽이 깎였는지 알 수 없다. 검색이 성공한 질의만
따로 보면(`Conditional EM`) 남는 차이는 문맥의 성질이지 검색이 아니다.

    End-to-end EM        전체 질의
    Conditional EM       검색 성공(correct=1) 질의만          <- 핵심
    Retrieval-limited    검색 실패(correct=0) 질의 비율

채점은 `rag_agent/eval/answer_em.py` 로 **다시 한다** — 디스크의 `answer_correct`
와 대조해 어긋나면 그 수를 보고한다. 두 벌이 어긋난 채 표가 나가는 것을 막는
유일한 방법이 매번 다시 세는 것이다.

  PYTHONPATH=. .venv/bin/python analysis/answer_decompose.py   # 단독 확인용
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.answer_stats import (bootstrap_ci, holm,            # noqa: E402
                                   mcnemar_exact, paired_bootstrap)
from rag_agent.eval.answer_em import score                        # noqa: E402

D = ROOT / "results/retrieval_accuracy"
REF = "본 방법 (셀+헤더경로)"

ARMS = [("본 방법 (셀+헤더경로)", "t_s3c_hybrid"),
        ("행 단위 (RowColRetrieval 행 절반)", "t_row_values"),
        ("RowColRetrieval (행×열 온전)", "t_rowcol_values"),
        ("MT2Net 색인 단위", "t_mt2net_hybrid"),
        ("TableRAG Huawei (코드 1,000자)", "t_trag_hetero"),
        ("TableRAG Huawei (논문 1,000토큰)", "t_trag_hetero_tok"),
        ("TableRAG NeurIPS'24 (leaf)", "t_tablerag_leaf"),
        ("TableRAG NeurIPS'24 (path)", "t_tablerag_path"),
        ("고정 청킹 1,000자 (업계 기본값)", "t_chunk1000")]
#: `표 통째` 는 답변 표에서 뺀다 (2026-09-09 사용자 결정). 표 1 에서도 셀 검색
#: 단계가 없어 주지표 칸이 비어 있는 통제 행이고, 문맥이 평균 118.6셀 / 최대
#: 13,729토큰이라 8GB 에서 6.2% 의 질의가 VRAM 절벽에 걸려 10시간이 든다.
#: 측정하지 않았으므로 표에 행을 만들지 않는다.
#: 표 1b — 우리 ablation. 경쟁 상대가 아니므로 본표에서 뺀다.
ABLATION = [("우리 문장을 행별로 묶음", "t_row_hybrid")]
#: 모집단 — 표 1 과 같은 모양으로 자른다. `any`(답이 헤더인 질의)는 검색 판정
#: 기준이 다르고(gold 중 하나만 있으면 성공) 답도 숫자가 아니므로, 산술 층에
#: 섞으면 그 층의 절반이 헤더 문자열 질의가 된다 — 2026-09-09 에 실제로 그랬다.
POPS = [("단일 셀 조회 (주지표)", lambda r: r["mode"] == "all"
         and (r.get("aggregation") or "none") == "none" and r["m"] == 1),
        ("조회", lambda r: r["mode"] == "all" and r["type"] == "lookup"),
        ("산술", lambda r: r["mode"] == "all" and r["type"] == "arithmetic"),
        ("데이터셀 전체", lambda r: r["mode"] == "all"),
        ("헤더답 (별도)", lambda r: r["mode"] == "any")]
#: 조건쌍 검정과 공통 Conditional EM 은 이 셋에서만 — 나머지는 그 합이다.
TEST_POPS = ["단일 셀 조회 (주지표)", "조회", "산술"]


def gold_size() -> dict:
    """`{query_id: m}` — 답변 레코드에 없는 gold 셀 수를 검색 레코드에서 가져온다.
    arm 마다 같다는 것은 `tests/test_arm_fairness.py` 가 보증한다."""
    f = D / "t_s3c_hybrid_records.jsonl"
    return {j["query_id"]: j["m"] for j in map(json.loads, f.open()) if "m" in j}


_M = None


def load(tag: str, condition: str = "retrieved") -> list:
    global _M
    if _M is None:
        _M = gold_size()
    f = D / f"{tag}_answer_{condition}.jsonl"
    if not f.exists():
        return []
    rows = []
    for j in map(json.loads, f.open()):
        j.update(score(j["pred"], j["answer"], j.get("aggregation")))
        j["m"] = _M.get(j["query_id"])
        rows.append(j)
    return rows


def rescore_drift(rows: list) -> int:
    """디스크의 `answer_correct` 와 지금 채점이 어긋난 건수."""
    return sum(1 for r in rows if r["official"] != r.get("answer_correct"))


def cell(rows: list) -> dict:
    n = len(rows)
    if not n:
        return {}
    off = [r["official"] for r in rows]
    hit = [r["official"] for r in rows if r["retrieval_correct"]]
    lo, hi = bootstrap_ci(off)
    return {"n": n, "em": sum(off) / n, "ci": (lo, hi),
            "defect": sum(r["format_defect"] for r in rows),
            "cond_em": (sum(hit) / len(hit)) if hit else None, "n_hit": len(hit),
            "ret_limited": 1 - len(hit) / n}


def build() -> str:
    size = json.loads((D / "INPUT_SIZE.json").read_text()) \
        if (D / "INPUT_SIZE.json").exists() else {"conditions": {}}
    data, missing, drift = {}, [], {}
    for name, tag in ARMS + ABLATION:
        rows = load(tag)
        if not rows:
            missing.append(name)
            continue
        data[name] = rows
        drift[name] = rescore_drift(rows)

    out = ["## 표 2b — 색인 단위별 답변 정확도 (표 1 과 같은 행 집합)", "",
           "사전등록 `PREREG-2026-09-08-policy-answer-em.md`. 조작 변인은 검색 조건",
           "하나뿐이고 리더·프롬프트·디코딩·배치는 모든 행에서 동일하다.", "",
           "**행 집합은 표 1 과 같다** — 리포 밖에 존재하는 방법(발표 논문 + 업계",
           "기본값)만 싣고, 우리 ablation 은 아래 `표 2c` 로 뺀다. 2026-09-09 정정:",
           "처음에는 P1~P4 라는 이름으로 우리 ablation 둘(행 묶음·표 통째)을 본표에",
           "올렸는데, 그것은 표 1 이 금지한 비교다(`TABLES.md` — \"그 행을 이겼다는",
           "것은 결과가 아니라 그 부분이 사는 값이다\").", "",
           "`EM` 은 HiTab 공식 채점기(허용오차 없음)이고 이것만 정확도로 쓴다.",
           "`형식결함` 은 공식이 기각했는데 **숫자 집합이 정확히 같은** 건수 —",
           "표기만 다른 경우이고 정확도가 아니다. 허용오차 채점(±1%)은 2026-09-09 에",
           "**폐기**했다: 실제로는 ×100/÷100 재척도·부호 뒤집힘·예측 속 아무 숫자",
           "하나까지 받아 줘서 P4 산술에서 추가로 통과한 92건 중 76건이 오답이었다.", "",
           "**모집단은 표 1 과 같은 모양이다** (2026-09-09 통일): 주지표는 단일 셀",
           "조회 n=991, `데이터셀 전체` 가 n=1,245, `헤더답` n=336 은 별도로 싣는다.",
           "표 1·표 2 와 같은 자리의 값끼리만 비교한다.", ""]
    if missing:
        out += [f"⚠️ **미실행: {', '.join(missing)}** — 아래 표에 행이 없다.", ""]
    if any(drift.values()):
        out += ["⚠️ **재채점 불일치**: " + ", ".join(
            f"{k} {v}건" for k, v in drift.items() if v) +
            " — 디스크의 `answer_correct` 와 지금 채점이 다르다. 원인을 밝히기 전에는"
            " 이 표를 쓰지 말 것.", ""]

    out += ["| 조건 | 모집단 | n | EM | 95% CI | Cond. EM (자기 모집단) | 검색실패 비율 | "
            "형식결함 | 평균 입력 토큰 | 한계초과율 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, _tag in ARMS:
        if name not in data:
            continue
        sz = size["conditions"].get(name, {})
        for t, keep in POPS:
            rows = [r for r in data[name] if keep(r)]
            c = cell(rows)
            if not c:
                continue
            ce = f"{c['cond_em']:.4f}" if c["cond_em"] is not None else "—"
            out.append(
                f"| {name} | {t} | {c['n']} | **{c['em']:.4f}** | "
                f"[{c['ci'][0]:.4f}, {c['ci'][1]:.4f}] | {ce} (n={c['n_hit']}) | "
                f"{c['ret_limited']:.4f} | {c['defect']} | "
                f"{sz.get('tokens_mean', '—')} | "
                f"{sz.get('over_limit_ratio', '—')} |")

    # --- 조건쌍 검정: 각 조건 vs 본 방법(P4), 유형별로 나눠서 -----------------
    if REF in data:
        ref = {r["query_id"]: r for r in data[REF]}
        raw, rowbuf = {}, []
        for name, _tag in ARMS:
            if name == REF or name not in data:
                continue
            other = {r["query_id"]: r for r in data[name]}
            ids_all = [q for q in ref if q in other]
            for t, keep in POPS:
                if t not in TEST_POPS:
                    continue
                ids = [q for q in ids_all if keep(ref[q])]
                if not ids:
                    continue
                A = [ref[q]["official"] for q in ids]
                B = [other[q]["official"] for q in ids]
                n01, n10, p = mcnemar_exact(A, B)
                bs = paired_bootstrap(A, B)
                key = f"{name}/{t}"
                raw[key] = p
                rowbuf.append((key, len(ids), n01, n10, p, bs))
        adj = holm({k: v for k, v in raw.items()})
        out += ["", f"## 조건쌍 검정 — 각 arm vs `{REF}` (짝지음, 전체 모집단)", "",
                "McNemar 정확검정. `본방법만:다른조건만` 은 불일치쌍이다. Holm 보정은",
                f"검정 {len(raw)} 개에 건다(`데이터셀 전체`·`헤더답` 행은 그 합이거나",
                "다른 판정 기준이라 검정에 넣지 않는다). 차이 CI 는 paired bootstrap",
                "B=10,000, seed=42.", "",
                "| 비교 | 모집단 | n | 본방법만:다른조건만 | p | Holm p | 차이(다른−본) | 95% CI |",
                "|---|---|---:|---:|---:|---:|---:|---|"]
        for key, n, n01, n10, p, bs in rowbuf:
            name, t = key.split("/")
            hp = f"{adj[key]:.3g}" if key in adj else "—"
            out.append(f"| {name} | {t} | {n} | {n01}:{n10} | {p:.3g} | {hp} | "
                       f"{bs['diff']:+.4f} | [{bs['ci_lo']:+.4f}, {bs['ci_hi']:+.4f}] |")

    # --- Conditional EM 은 **쌍별** 공통 성공 집합에서 비교한다 -----------------
    # arm 전체의 교집합을 쓰면 가장 약한 arm(TableRAG'24 leaf 는 68건만 성공)이
    # 집합을 정하고, 남는 것은 "정확도 0.07 짜리도 찾는 쉬운 질의"뿐이라 n 이
    # 50 으로 무너지고 표본도 편향된다 — 2026-09-09 에 실제로 그랬다. 비교는
    # arm 하나 대 본 방법이므로 조건부 집합도 그 둘이 함께 성공한 질의로 잡는다.
    if REF in data:
        byq = {n: {r["query_id"]: r for r in v} for n, v in data.items()}
        out += ["", "## Conditional EM — **둘 다** 검색에 성공한 질의만 (쌍별)", "",
                "위 표의 `Cond. EM` 은 arm 마다 모집단이 달라 서로 뺄 수 없다. 여기서는",
                "각 arm 을 본 방법과 **그 둘이 함께 검색에 성공한 질의**에서만 비교한다.",
                "검색이 이미 성공한 자리이므로 남는 차이는 문맥의 성질이지 검색이 아니다.", "",
                "| 비교 arm | 모집단 | 둘 다 성공 n | 본 방법 | 그 arm | 차이 | p | Holm p |",
                "|---|---|---:|---:|---:|---:|---:|---:|"]
        raw2, buf2 = {}, []
        for name, _tg in ARMS + ABLATION:
            if name == REF or name not in data:
                continue
            for ty, keep in POPS:
                if ty not in TEST_POPS:
                    continue
                S = [q for q in byq[REF]
                     if q in byq[name] and keep(byq[REF][q])
                     and byq[REF][q]["retrieval_correct"]
                     and byq[name][q]["retrieval_correct"]]
                if len(S) < 30:            # 이보다 작으면 비교가 성립하지 않는다
                    buf2.append((name, ty, len(S), None, None, None, None))
                    continue
                A = [byq[REF][q]["official"] for q in S]
                B = [byq[name][q]["official"] for q in S]
                _n01, _n10, p = mcnemar_exact(A, B)
                bs = paired_bootstrap(A, B)
                raw2[f"{name}/{ty}"] = p
                buf2.append((name, ty, len(S), sum(A) / len(S), sum(B) / len(S), p, bs))
        adj2 = holm(raw2)
        for name, ty, n, a, b, p, bs in buf2:
            if a is None:
                out.append(f"| {name} | {ty} | {n} | — | — | *(n<30, 비교 불가)* | — | — |")
                continue
            out.append(f"| {name} | {ty} | {n} | {a:.4f} | {b:.4f} | "
                       f"{bs['diff']:+.4f} [{bs['ci_lo']:+.4f},{bs['ci_hi']:+.4f}] | "
                       f"{p:.3g} | {adj2[f'{name}/{ty}']:.3g} |")

    abl = [(n, t) for n, t in ABLATION if n in data]
    if abl:
        out += ["", "## 표 2c — 우리 ablation (경쟁 상대 아님)", "",
                "| 뺀 것 | 모집단 | n | EM | Cond. EM (자기 모집단) | 검색실패 비율 |",
                "|---|---|---:|---:|---:|---:|"]
        for name, _tg in abl:
            for ty, keep in POPS:
                c = cell([r for r in data[name] if keep(r)])
                if not c:
                    continue
                ce = f"{c['cond_em']:.4f}" if c["cond_em"] is not None else "—"
                out.append(f"| {name} | {ty} | {c['n']} | **{c['em']:.4f}** | "
                           f"{ce} (n={c['n_hit']}) | {c['ret_limited']:.4f} |")

    out += ["", "## 읽는 법", "",
            "- **`Cond. EM (자기 모집단)` 열끼리 빼지 말 것.** arm 마다 검색 성공",
            "  질의가 다르므로 그 열은 서로 다른 모집단의 값이다. 비교는 위의",
            "  `둘 다 검색에 성공한 질의` 절에서만 성립하고, 사전등록 §6 의 주 예측도",
            "  거기서 판정된다.",
            "- `검색실패 비율` 이 클수록 그 조건의 End-to-end EM 은 검색에 눌린 것이다.",
            "- 입력 토큰은 `analysis/answer_input_size.py` 가 잰 값이고 생성 없이",
            "  토크나이저로만 센 것이라 실행 전에 이미 알 수 있었다.", ""]
    return "\n".join(out)


if __name__ == "__main__":
    print(build())
