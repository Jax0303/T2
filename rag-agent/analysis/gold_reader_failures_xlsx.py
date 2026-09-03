#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""gold 주입률 100%인데 왜 틀리나 — 리더 실패 전수 원장.

`cond=gold` 는 정답 셀 문장을 **강제로** 프롬프트에 넣는다. 그러므로 여기 남은
오답은 검색 탓이 아니라 **전부 리더(또는 채점기, 또는 gold 라벨) 탓**이다.
dev 830 에서 .9024 이므로 81 건이 남는다. 이 스크립트는 그 81 건을 분류하고,
맞은 749 건까지 포함한 830 건 전부를 셀 문장과 함께 워크북으로 떨군다.

분류는 **자동**이며 규칙 기반이다. 사람이 다시 봐야 하는 열(`판정_사람`)을
비워 둔다 -- `results/audit/manual_check_*.xlsx` 와 같은 관례다.

시트:
  요약            분류별 집계와 각 분류의 정의
  오답_81         gold 조건 오답 전수, 셀 문장·경로·분류 포함
  전체_830        맞은 것까지 포함한 830 건

모델을 부르지 않는다. 출력 `results/audit/gold_reader_failures.xlsx`.

  PYTHONPATH=. .venv/bin/python analysis/gold_reader_failures_xlsx.py
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "analysis")]

import pandas as pd                                                   # noqa: E402
from header_path_coverage import load_corpus                          # noqa: E402
from cell_rank_dump import cell_texts                                 # noqa: E402
from phase4_summary import em, em_lenient                             # noqa: E402

NUM = re.compile(r"-?\d[\d,]*\.?\d*")

# 분류 정의. 순서가 곧 우선순위다 -- 위에서 먼저 걸리면 아래는 안 본다.
KINDS = [
    ("A 부호 반대",        "절댓값이 같고 부호만 다르다. 표가 감소를 양수로 적는데 질문이 '얼마나 줄었나'를 묻는 식."),
    ("B 스케일 차이",      "10의 거듭제곱만큼 다르다. 표 제목의 '천 달러 단위' 같은 배율을 리더가 적용했거나 무시했다."),
    ("C 여집합",           "예측 + 정답 ≈ 100. 퍼센트 질문에서 반대편을 읽었다."),
    ("D 100 더함",         "예측 − 정답 ≈ 100. 증감률에 기준선 100을 얹었다."),
    ("E 표기만 다름",      "위에 안 걸리고 관대 채점(em_lenient)으로는 통과. 값은 맞고 형식(%, 쉼표, 통화기호)만 다르다. **채점기 문제이지 리더 문제가 아니다.**"),
    ("F 셀 값 그대로",     "예측이 주입된 셀의 값과 일치하는데 gold 답과는 다르다. **gold 라벨을 의심해야 하는 칸이다.**"),
    ("G 문맥에 없는 값",   "예측한 숫자가 주입된 셀 문장 어디에도 없다. 셀 하나만 넣었으므로 **리더가 지어냈거나 계산한 것**이다."),
    ("H 문자열 불일치",    "답이 숫자가 아니다. 서술형이라 표기 차이로 갈린다."),
    ("I 그 밖",            "위 어디에도 안 걸린다."),
]


def nums(s):
    return [float(x.replace(",", "")) for x in NUM.findall(str(s))]


def gold_str(raw):
    """gold_answer 는 리스트를 문자열로 박은 것이다: \"[55269.5]\" / \"['a, b']\"."""
    try:
        v = ast.literal_eval(str(raw))
        if isinstance(v, (list, tuple)):
            return ", ".join(str(x) for x in v)
        return str(v)
    except (ValueError, SyntaxError):
        return str(raw)


def classify(pred, gold, cell_value, lenient, sentence):
    """규칙 기반 자동 분류. 순서가 우선순위다.

    부호·스케일·여집합을 관대 채점보다 **먼저** 본다 -- em_lenient 가 부호를
    용서하므로, 나중에 보면 진짜 읽기 오류가 '표기 문제'로 숨는다.
    """
    p, g = nums(pred), nums(gold)
    cv = nums(cell_value)
    ctx = set(nums(sentence))
    if p and g:
        a, b = p[0], g[0]
        if a == -b and a != 0:
            return KINDS[0][0]
        if b != 0 and a != 0:
            r = abs(a) / abs(b)
            for k in (10, 100, 1000, 10000, .1, .01, .001, .0001):
                if abs(r - k) < 1e-6 * max(1, k):
                    return KINDS[1][0]
        if abs((abs(a) + abs(b)) - 100) < 0.5:
            return KINDS[2][0]
        if abs((a - b) - 100) < 0.5:
            return KINDS[3][0]
    if lenient:
        return KINDS[4][0]
    if p and cv and abs(p[0] - cv[0]) < 1e-9:
        return KINDS[5][0]
    if p and not any(abs(p[0] - c) < 1e-9 for c in ctx):
        return KINDS[6][0]
    if not p or not g:
        return KINDS[7][0]
    return KINDS[8][0]


def qform(q):
    q = q.lower().strip()
    for pre in ("how many", "how much", "what percentage", "what proportion",
                "what was", "what is", "which", "who", "when", "where", "why", "how"):
        if q.startswith(pre):
            return pre
    return "그 밖"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="results/answer_ret/p0_dev_all.jsonl")
    ap.add_argument("--population", default="hitab_dev_lookup_all")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--title-mode", default="page")
    ap.add_argument("--out", default="results/audit/gold_reader_failures.xlsx")
    a = ap.parse_args()
    a.dataset, a.cell_scheme, a.data_dir = "hitab", "S3c", "data/hitab"
    a.seed, a.max_queries, a.mh_queries = 42, 0, 400
    a.rhb_question_types, a.rhb_em_only = [], False

    C = load_corpus(a)
    txt = cell_texts(C, a.cell_scheme, a.title_mode)
    pos = {c: n for n, c in enumerate(C.cell_owner)}
    qmeta = {q["query_id"]: q for q in C.queries}

    recs = [json.loads(l) for l in open(a.records)]
    gold = [r for r in recs if r["cond"] == "gold"]
    print(f"[gold] {len(gold)} 건")

    rows = []
    for r in gold:
        q = qmeta.get(r["query_id"], {})
        cells = [pos[c] for c in sorted(q.get("gold_cells", [])) if c in pos]
        sent = "\n".join(txt[p] for p in cells)
        tid, ri, ci = (C.cell_owner[cells[0]] if cells else ("", "", ""))
        rp, cp, val = (C.cell_paths[cells[0]] if cells else ((), (), ""))
        gs = gold_str(r["gold_answer"])
        ok = int(em(r["pred_parsed"], r["gold_answer"]))
        lok = int(em_lenient(r["pred_parsed"], r["gold_answer"]))
        rows.append({
            "query_id": r["query_id"], "정답여부": ok, "관대채점": lok,
            "분류": "" if ok else classify(r["pred_parsed"], gs, val, lok, sent),
            "질문형태": qform(r["question"]), "질문": r["question"],
            "정답(gold)": gs, "리더 예측": r["pred_parsed"],
            "주입된 셀의 값": val,
            "행 경로": " > ".join(map(str, rp)), "열 경로": " > ".join(map(str, cp)),
            "표 id": tid, "행": ri, "열": ci, "주입 셀 수": len(cells),
            "주입된 셀 문장": sent,
            "프롬프트 토큰": r.get("prompt_tokens"),
            "생성 캡 도달": int(bool(r.get("hit_token_cap"))),
            "판정_사람": "", "메모_사람": "",
        })

    df = pd.DataFrame(rows)
    bad = df[df["정답여부"] == 0].copy()
    print(f"[오답] {len(bad)} 건")

    cnt = Counter(bad["분류"])
    summary = pd.DataFrame(
        [{"분류": k, "건수": cnt.get(k, 0),
          "오답 중 비율": round(cnt.get(k, 0) / max(1, len(bad)), 4),
          "830 중 비율": round(cnt.get(k, 0) / len(df), 4), "정의": d}
         for k, d in KINDS])
    qs = (bad.groupby("질문형태").size().sort_values(ascending=False)
          .rename("오답 수").reset_index())
    qall = df.groupby("질문형태").size().rename("전체 수").reset_index()
    qs = qs.merge(qall, on="질문형태", how="right").fillna(0)
    qs["오답 수"] = qs["오답 수"].astype(int)
    qs["오답률"] = (qs["오답 수"] / qs["전체 수"]).round(4)
    qs = qs.sort_values("오답 수", ascending=False)

    head = pd.DataFrame([
        {"항목": "조건", "값": "cond=gold — 정답 셀 문장을 강제 주입"},
        {"항목": "모집단", "값": f"{a.population} n={len(df)} (전부 m=1)"},
        {"항목": "인코더", "값": "models/bge-base-cell-ft-p0 · α=0.8 · title-mode page"},
        {"항목": "리더", "값": "Qwen2.5-7B-Instruct 4-bit NF4 · temp=0 · seed=42 · max_new_tokens=32"},
        {"항목": "EM", "값": round(df["정답여부"].mean(), 4)},
        {"항목": "관대채점 EM", "값": round(df["관대채점"].mean(), 4)},
        {"항목": "gold 주입률", "값": 1.0},
        {"항목": "오답", "값": len(bad)},
        {"항목": "읽는 법", "값": "gold 주입률이 1.000이므로 검색 탓이 아니다. 남은 오답은 리더·채점기·gold 라벨 탓이다."},
        {"항목": "주의", "값": "분류는 규칙 기반 자동이다. 판정_사람 열은 비워 뒀다."},
    ])

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        head.to_excel(w, sheet_name="요약", index=False, startrow=0)
        summary.to_excel(w, sheet_name="요약", index=False, startrow=len(head) + 2)
        qs.to_excel(w, sheet_name="요약", index=False,
                    startrow=len(head) + len(summary) + 4)
        bad.sort_values(["분류", "질문형태"]).to_excel(w, sheet_name="오답_81", index=False)
        df.to_excel(w, sheet_name="전체_830", index=False)
        for sh, widths in (("요약", [22, 96]), ("오답_81", None), ("전체_830", None)):
            ws = w.sheets[sh]
            if widths:
                for i, wd in enumerate(widths, 1):
                    ws.column_dimensions[chr(64 + i)].width = wd
            else:
                for col, wd in zip("ABCDEFGHIJKLMNOPQRS",
                                   [34, 9, 9, 15, 14, 60, 22, 22, 16, 26, 26,
                                    9, 6, 6, 9, 70, 12, 11, 12, 24]):
                    ws.column_dimensions[col].width = wd
                ws.freeze_panes = "B2"
    print(f"-> {out}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
