#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""표 안 실패 166건 전수 원장 — 원본 표·셀 문장·질의·정답·리더 출력.

`results/colsig/VERDICT.md` §E 가 이 166 건을 셋으로 갈랐다: 질문이 오답 쪽
105 / 정답 쪽 56 / 경로 완전 동일 5. 그 분류는 **표면 코사인 비교**이지
gold 라벨이 맞는지를 잰 것이 **아니다**. 이 스크립트는 사람이 그것을 가를 수
있도록 한 건마다 필요한 것을 전부 한 줄에 모은다.

자동으로 하는 점검은 둘뿐이고, 나머지 판정 열은 비워 둔다
(`results/audit/manual_check_*.xlsx` 와 같은 관례):

  문장점검_값   셀 문장의 값이 원본 표의 그 좌표 텍스트와 같은가 (문장 생성 점검)
  문장점검_답   gold 답 숫자가 gold 셀 문장 안에 있는가 (없으면 라벨 의심)

시트:
  요약        출처·집계·판정 갈래 정의
  케이스_166  한 건 = 한 줄
  원본표      등장하는 표마다 그리드 원문 (gold 【】, 1등 «»)

모델을 부르지 않는다. GPU 불필요.

  python3 analysis/intable_failures_xlsx.py
"""
from __future__ import annotations

import argparse
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
from gold_reader_failures_xlsx import gold_str, qform                 # noqa: E402

NUM = re.compile(r"-?\d[\d,]*\.?\d*")
WORD = re.compile(r"[a-z0-9']+")

KINDS = [
    ("미명세",     "질문이 두 칸을 가르는 말을 아예 안 한다. 연도·단위·카테고리 미지정. 여러 칸이 답이 될 수 있는데 gold 가 하나만 지정된 것 — 라벨 오류가 아니라 질문이 덜 쓰인 것이다."),
    ("동의어",     "질문은 정답 칸을 가리키는데 표현이 달라 어휘로 안 잡힌다 (other countries = total, excluding syria). **재정렬기가 고칠 수 있는 자리다.**"),
    ("판정기오류", "colsig 가 오답 쪽이라 했지만 사람이 보면 질문이 정답 칸을 가리킨다. 기성품 bge-base 구절 코사인의 한계. |margin| 이 작은 건에 몰려 있을 것이다."),
    ("질문결함",   "gold 는 원본 문장과 맞는데 질문이 잘못 쓰였다. 특히 질문의 말이 다른 열 이름과 우연히 겹칠 때 (martina: 문장은 '10.35초 완주', 질문은 'final result', 표에는 `final` 열이 따로 있음). **라벨 오류가 아니다** -- 원본 문장 열을 보고 가를 것."),
    ("라벨오류",   "원본 문장을 봐도 gold 칸이 질문과 안 맞는다. 진짜 데이터셋 결함."),
    ("추론필요",   "질문을 풀려면 표 구조나 문맥 추론이 필요하다 ('in contrast', 여집합). 어휘로는 못 하지만 원리적으로 불가능하지도 않다."),
    ("기타",       "위 어디에도 안 걸린다."),
]


def nums(s):
    return [float(x.replace(",", "")) for x in NUM.findall(str(s))]


def same_val(a, b):
    """쉼표·소수점 표기를 넘어 같은 값인가. 숫자면 숫자로, 아니면 문자열로 본다."""
    if b is None:
        return None
    na, nb = nums(a), nums(b)
    if na and nb:
        return abs(na[0] - nb[0]) < 1e-9
    return str(a).strip().lower() == str(b).strip().lower()


def toks(xs):
    return set(WORD.findall(" ".join(str(x) for x in xs).lower()))


def hint(question, gold_only, win_only):
    """어휘 신호만으로 사람의 분류를 좁힌다. 판정이 아니라 힌트다."""
    q = set(WORD.findall(question.lower()))
    g, w = toks(gold_only), toks(win_only)
    if not gold_only:
        return "gold쪽 차집합 없음(강제분류)"
    if not win_only:
        return "1등쪽 차집합 없음"
    hits = []
    if g & q:
        hits.append("질문⊃gold")
    if w & q:
        hits.append("질문⊃1등")
    return " ".join(hits) if hits else "미명세(양쪽 다 질문에 없음)"


def md_parts(line):
    s = line.strip().strip("|")
    return [x.strip() for x in s.split("|")]


def layout(C, tid):
    """md 표에서 구분선 위치와 왼쪽 헤더 열 수를 잡는다.

    `C.md_lines` 는 헤더 행 + `|---|` 구분선 + 데이터 행이고, `C.shape` 의 열 수는
    **데이터 열만** 센다. 둘의 차가 곧 왼쪽 헤더 열 수다. 셀 좌표 (r,c) 는 데이터
    영역 기준이므로 이 오프셋 없이는 원본 표와 못 맞춘다.
    """
    lines = C.md_lines[tid]
    # 빈 헤더 행(`|  |  |  |`)도 조건을 만족하므로 `-` 가 실제로 있어야 한다.
    sep = next(i for i, l in enumerate(lines)
               if "-" in l and set(l.replace("|", "").strip()) <= set("- "))
    return lines, sep, len(md_parts(lines[sep])) - C.shape[tid][1]


def cell_at(C, tid, r, c):
    lines, sep, left = layout(C, tid)
    i = sep + 1 + r
    if i >= len(lines):
        return None
    row = md_parts(lines[i])
    return row[left + c] if left + c < len(row) else None


def grid(C, tid, gold_rc, win_rc):
    """표 원문. gold 는 【】, 1등은 «» 로 표시한다."""
    lines, sep, left = layout(C, tid)
    out = []
    for i, l in enumerate(lines):
        r = i - sep - 1
        parts = md_parts(l)
        for rc, mark in ((gold_rc, "【%s】"), (win_rc, "«%s»")):
            if r == rc[0] and left + rc[1] < len(parts):
                parts[left + rc[1]] = mark % parts[left + rc[1]]
        tag = f"r{r}" if r >= 0 else ("---" if i == sep else "hdr")
        out.append(f"{tag}\t" + "\t".join(parts))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--above", default="results/colsig/"
                    "hitab_dev_lookup_all_S3c_page_hybrid0.8_above.jsonl")
    ap.add_argument("--phrase", default="results/colsig/col_signal_phrase.json")
    ap.add_argument("--records", default="results/answer_ret/p0_dev_all.jsonl")
    ap.add_argument("--out", default="results/audit/intable_failures_166.xlsx")
    a = ap.parse_args()
    a.dataset, a.cell_scheme, a.data_dir = "hitab", "S3c", "data/hitab"
    a.population, a.split, a.title_mode = "hitab_dev_lookup_all", "dev", "page"
    a.seed, a.max_queries, a.mh_queries = 42, 0, 400
    a.rhb_question_types, a.rhb_em_only = [], False

    C = load_corpus(a)
    txt = cell_texts(C, a.cell_scheme, a.title_mode)
    pos = {c: n for n, c in enumerate(C.cell_owner)}

    phrase = {r["query_id"]: r
              for r in json.load(open(a.phrase))["rows"]}
    above = {r["query_id"]: r for r in
             (json.loads(l) for l in open(a.above))
             if r["query_id"] in phrase}
    print(f"[표 안 실패] {len(above)} 건")

    # HiTab 질문은 원본 위키 문장에서 역으로 만들어진다. 그 문장을 봐야
    # "gold 가 틀렸다" 와 "질문이 나쁘다" 를 가를 수 있다 -- 예: martina 건은
    # 문장이 "10.35초에 완주" 이고 질문이 "final result" 인데, 표에 `final` 열이
    # 따로 있어서 검색기가 그리 간다. 라벨 오류가 아니라 질문 결함이다.
    src = {}
    for l in open(Path(a.data_dir) / "data" / f"{a.split}_samples.jsonl"):
        r = json.loads(l)
        src[r["id"]] = r.get("sub_sentence") or ""

    reads = {}
    for l in open(a.records):
        r = json.loads(l)
        if r["cond"] in ("gold", "top1", "top10"):
            reads.setdefault(r["query_id"], {})[r["cond"]] = r

    rows, used = [], {}
    for qid, A in above.items():
        P, R = phrase[qid], reads.get(qid, {})
        w = A["top_above"][0]
        tid = A["gold_table"]
        gr, gc = A["gold_row"], A["gold_col"]
        gsent = txt[pos[(tid, gr, gc)]] if (tid, gr, gc) in pos else ""
        wsent = txt[pos[(tid, w["row"], w["col"])]] \
            if (tid, w["row"], w["col"]) in pos else ""

        raw = cell_at(C, tid, gr, gc)
        used.setdefault(tid, (gr, gc, w["row"], w["col"]))

        gs = gold_str(A["gold_answer"])
        gnum = nums(gs)
        row = {
            "query_id": qid,
            "복구가능성": "",
            "colsig판정": P["kind"], "margin": P["margin"],
            "관계": w["relation"], "자동힌트": hint(A["question"],
                                                P["gold_only"], P["win_only"]),
            "판정_사람": "", "메모_사람": "",
            "질문": A["question"], "질문형태": qform(A["question"]),
            "원본 문장(질문의 출처)": src.get(qid, ""),
            "질문결함": "⚠️ 1등쪽 단어가 원본 문장에 없음"
                        if (toks(P["win_only"]) & set(WORD.findall(
                            A["question"].lower()))
                            and not (toks(P["win_only"]) & set(WORD.findall(
                                src.get(qid, "").lower()))))
                        else ("⚠️ gold쪽 단어를 질문이 빠뜨림"
                              if (toks(P["gold_only"]) & set(WORD.findall(
                                  src.get(qid, "").lower()))
                                  and not (toks(P["gold_only"]) & set(WORD.findall(
                                      A["question"].lower()))))
                              else ""),
            "정답(gold)": gs,
            "gold만 있는 구절": " | ".join(P["gold_only"]),
            "1등만 있는 구절": " | ".join(P["win_only"]),
            "표 id": tid,
            "gold 행": gr, "gold 열": gc,
            "gold 행경로": " > ".join(map(str, A["gold_row_path"])),
            "gold 열경로": " > ".join(map(str, A["gold_col_path"])),
            "gold 값": A["gold_value"], "원본표의 그 칸": raw,
            "표 제목": C.title.get(tid, ""),
            "표 크기": "%d행 x %d열" % C.shape[tid],
            "gold 셀 문장": gsent,
            "1등 행": w["row"], "1등 열": w["col"],
            "1등 행경로": " > ".join(map(str, w["row_path"])),
            "1등 열경로": " > ".join(map(str, w["col_path"])),
            "1등 값": w["value"], "1등 셀 문장": wsent,
            "gold 순위": A["gold_rank"], "위에 있는 셀 수": A["n_above"],
            "문장점검_값": {True: "OK", False: "⚠️불일치",
                            None: "⚠️좌표 없음"}[same_val(A["gold_value"], raw)],
            # 부호는 봐주고(표는 감소를 음수로 적고 질문은 "얼마나 줄었나"를 묻는다)
            # 문자열 답은 문장 안 등장으로 본다. 그러고도 없으면 라벨을 의심할 칸이다.
            "문장점검_답": "OK" if (
                (gnum and any(abs(abs(gnum[0]) - abs(c)) < 1e-9
                              for c in nums(gsent)))
                or (not gnum and gs.strip().lower() in gsent.lower()))
                else "⚠️답이 문장에 없음",
        }
        # 현 파이프라인(top10)이 이 건을 이미 푸는지, 못 푼다면 무엇이 막는지.
        # gold 가 문맥에 들어갔는데 틀렸다면 고르기 문제(재정렬 표적)이고,
        # gold 조건에서도 틀리면 리더 문제이며, 애초에 안 들어갔으면 검색 문제다.
        t10, gcond = R.get("top10"), R.get("gold")
        if t10 is None:
            row["복구가능성"] = ""
        elif em(t10["pred_parsed"], t10["gold_answer"]):
            row["복구가능성"] = "① 이미 맞음"
        elif not int(t10.get("gold_in_ctx", 0)):
            row["복구가능성"] = "③ 검색 부족 (gold 가 top10 밖)"
        elif gcond and em(gcond["pred_parsed"], gcond["gold_answer"]):
            row["복구가능성"] = "② 재정렬 표적 (10개 중 고르기)"
        else:
            row["복구가능성"] = "④ 리더 한계 (gold 만 줘도 틀림)"

        for cond in ("gold", "top1", "top10"):
            r = R.get(cond)
            row[f"리더 {cond} 예측"] = r["pred_parsed"] if r else ""
            row[f"리더 {cond} 정답"] = int(em(r["pred_parsed"], r["gold_answer"])) \
                if r else ""
            row[f"리더 {cond} 관대"] = int(em_lenient(r["pred_parsed"],
                                                   r["gold_answer"])) if r else ""
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["복구가능성", "colsig판정", "margin"])

    tb = pd.DataFrame([
        {"표 id": t, "제목": C.title.get(t, ""),
         "크기": "%d행 x %d열" % C.shape[t],
         "등장 건수": int((df["표 id"] == t).sum()),
         "그리드 (gold 【】, 1등 «»)":
             grid(C, t, (g[0], g[1]), (g[2], g[3]))[:32000]}
        for t, g in sorted(used.items())])

    head = pd.DataFrame([
        {"항목": "무엇", "값": "표 안 실패 166건 — 1등 셀이 정답 표 안인데 칸을 틀린 것"},
        {"항목": "출처", "값": f"{a.above} + {a.phrase} + {a.records}"},
        {"항목": "설정", "값": "models/bge-base-cell-ft-p0 · α=0.8 · title-mode page · "
                              "S3c · hitab_dev_lookup_all(830)"},
        {"항목": "리더", "값": "Qwen2.5-7B-Instruct 4-bit NF4 · temp=0 · seed=42 · "
                              "max_new_tokens=32"},
        {"항목": "깔때기", "값": "dev 830 → R@1 597 · 실패 233 → 표 밖 67 / 표 안 166"},
        {"항목": "표 수", "값": f"{len(tb)}개"},
        {"항목": "읽는 법", "값": "colsig판정은 **표면 코사인 비교**다. gold 라벨이 맞는지 "
                                "잰 것이 아니다. 그것을 가르는 것이 판정_사람 열이다."},
        {"항목": "주의", "값": "자동 점검은 문장점검_값/문장점검_답 둘뿐이다. "
                              "자동힌트는 어휘 신호이지 판정이 아니다."},
    ])
    agg = [pd.DataFrame([{"항목": k, "건수": v,
                          "166 중": round(v / len(df), 4),
                          "830 중": round(v / 830, 4)}
                         for k, v in Counter(df[col]).most_common()]).assign(축=col)
           for col in ("colsig판정", "관계", "자동힌트",
                       "문장점검_값", "문장점검_답")]
    kinds = pd.DataFrame([{"판정_사람 갈래": k, "정의": d} for k, d in KINDS])

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as W:
        r0 = 0
        head.to_excel(W, sheet_name="요약", index=False, startrow=r0); r0 += len(head) + 2
        kinds.to_excel(W, sheet_name="요약", index=False, startrow=r0); r0 += len(kinds) + 2
        for g in agg:
            g.to_excel(W, sheet_name="요약", index=False, startrow=r0); r0 += len(g) + 2
        df.to_excel(W, sheet_name="케이스_166", index=False)
        tb.to_excel(W, sheet_name="원본표", index=False)
        from openpyxl.styles import Alignment
        for sh, widths in (("요약", [24, 110, 12, 12, 14]),
                           ("케이스_166", None), ("원본표", None)):
            ws = W.sheets[sh]
            ws.freeze_panes = "A2"
            if widths:
                for i, wd in enumerate(widths, 1):
                    ws.column_dimensions[ws.cell(1, i).column_letter].width = wd
            else:
                for c in ws[1]:
                    ws.column_dimensions[c.column_letter].width = min(
                        46, max(10, len(str(c.value)) + 4))
                for r in ws.iter_rows(min_row=2):
                    for c in r:
                        c.alignment = Alignment(wrap_text=True, vertical="top")
    print(f"[출력] {out}  케이스 {len(df)} · 표 {len(tb)}")
    for col in ("복구가능성", "colsig판정", "질문결함"):
        print(f"  {col}: {dict(Counter(df[col]).most_common())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
