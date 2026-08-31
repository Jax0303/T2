#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Phase 4 Task C2 -- reader_records.xlsx + summary.md from the jsonl."""
from __future__ import annotations

import json
from pathlib import Path

import ast
import re

import numpy as np
import pandas as pd

# --- EM, re-scored here. The reader wrote is_correct against the RAW
# gold_answer string, which for HiTab is the list repr "[47.0]" -- so every
# row came out 0. The model outputs were fine; only the gold parse was wrong.
_CUR = re.compile(r"[$€£¥%]")
_THOU = re.compile(r"(?<=\d),(?=\d)")


def norm_em(s):
    s = _CUR.sub("", str(s))
    s = _THOU.sub("", s)
    s = " ".join(s.split()).strip().lower()
    if re.fullmatch(r"-?\d+\.\d+", s):
        s = s.rstrip("0").rstrip(".")
    return s


def gold_parts(g):
    """-> list of gold strings. A length-1 list is unwrapped to its element; a
    length>=2 list stays a list and is judged as a set (see em()).

    Only a bracketed literal is parsed as a container: literal_eval reads a
    bare '1,179' as the tuple (1, 179), which sent thousands-separated golds
    down the multi-gold path and past norm_em entirely (see BUGFIX_LOG.md)."""
    if str(g).strip()[:1] in "[(":
        try:
            v = ast.literal_eval(g)
            if isinstance(v, (list, tuple)):
                return [str(x) for x in v]
        except (ValueError, SyntaxError):
            pass
    return [g]


_NUM = re.compile(r"-?\d+(?:\.\d+)?")
REL_TOL = 0.01                     # PREREGISTER rev5, rule R1


def _one_num(s):
    m = _NUM.findall(str(s))
    return float(m[0]) if len(m) == 1 else None


def _same(p, g, rel):
    """One predicted value against one gold value."""
    if norm_em(p) == norm_em(g):
        return True
    if not rel:
        return False
    pv, gv = _one_num(p), _one_num(g)
    # R1: numeric answers only, relative tolerance, never an absolute-value or
    # percent-rescaling escape (rev5 rejects R2/R4).
    return (pv is not None and gv is not None and gv != 0
            and abs(pv - gv) / abs(gv) < REL_TOL)


def em(pred, gold, rel=True):
    """Single gold: match after normalisation, plus R1 relative tolerance when
    rel=True. Multi gold (list length >= 2): ALL elements must match, none
    extra -- the prediction is split on commas and paired greedily,
    order-insensitive. No partial credit. No sign normalisation."""
    gs = gold_parts(gold)
    if len(gs) == 1:
        return int(_same(pred, gs[0], rel))
    ps = [x for x in str(pred).split(",")]
    if len(ps) != len(gs):
        return 0
    left = list(ps)
    for g in gs:
        hit = next((x for x in left if _same(x, g, rel)), None)
        if hit is None:
            return 0
        left.remove(hit)
    return int(not left)

SRC = Path("results/phase4/reader_records.jsonl")
XLS = Path("results/phase4/reader_records.xlsx")
MD = Path("results/phase4/summary.md")
COLS = ["query_id", "query_type", "dataset", "pool", "policy", "n_chunks_used",
        "hit_chunk_cap", "query", "gold_answer", "gold_table_id", "gold_cell",
        "retrieved_topk", "gold_in_topk", "gold_all_in_topk", "gold_rank",
        "pred_answer_raw", "pred_parsed", "is_correct", "prompt_tokens",
        "latency_sec"]
POLS = ["P1_fixed_512", "P4_path_cell", "gold_cell"]
POOLS = ["hitab_lookup", "hitab_arith", "aitqa", "rhb_fact", "rhb_num"]
XL_MAX = 32767


def main() -> int:
    df = pd.DataFrame([json.loads(l) for l in open(SRC)])[COLS]
    df["is_correct"] = [em(p, g) for p, g in
                        zip(df["pred_parsed"], df["gold_answer"])]
    df["n_gold_answers"] = df["gold_answer"].map(lambda g: len(gold_parts(g)))
    # Excel refuses a cell over 32767 chars; drop the chunk TEXT (it stays in
    # results/phase4/retrieval_294/) and keep ranks+coords so the row still says
    # what was retrieved. Counted and reported, never silently truncated.
    n_trim = 0
    def fit(s):
        nonlocal n_trim
        if len(s) <= XL_MAX:
            return s
        n_trim += 1
        v = json.loads(s)
        for c in v:
            c.pop("text", None)
        return json.dumps({"text_dropped_see": "results/phase4/retrieval_294/",
                           "chunks": v})[:XL_MAX]
    df["retrieved_topk"] = df["retrieved_topk"].map(fit)
    df.to_excel(XLS, index=False)

    def block(sel, label):
        n = len(sel)
        return f"{sel['is_correct'].mean():.4f} (n={n})" if n else "n=0"

    L = [f"# Phase 4 Task C -- 리더 EM (sample_294)", "",
         f"행 {len(df)} = 검색 2조건 x 294 쿼리 + gold_cell {int((df.policy=='gold_cell').sum())}. "
         f"`Qwen/Qwen2.5-7B-Instruct` 4-bit, temperature=0, seed=42, "
         f"max_new_tokens=32, B_reader=4096.", "",
         "EM 정규화: 통화기호/퍼센트 제거 -> 천단위 콤마(`숫자,숫자`)만 제거 -> "
         "공백 축약 + 소문자화 -> 순수 소수(`-?\\d+\\.\\d+`)일 때만 후행 0 제거. "
         "그 외 정규화 없음. 부호 정규화 없음.", "",
         "`is_correct`는 이 스크립트에서 **재채점**한 값이다. 리더 실행 시점의 "
         "`is_correct`는 gold를 리스트 repr(`\"[47.0]\"`) 그대로 비교해 전건 0이었다. "
         "모델 출력은 정상이었고 gold 파싱만 틀렸다.", ""]

    L += ["## gold_answer 원소 수 분포 (쿼리 단위, pool별)", "",
          "| pool | 1개 | 2개 이상 | 최대 |", "|---|---:|---:|---:|"]
    u = df.drop_duplicates(["pool", "query_id"])
    for p in POOLS:
        d0 = u[u["pool"] == p]
        L.append(f"| {p} | {int((d0.n_gold_answers == 1).sum())} | "
                 f"{int((d0.n_gold_answers >= 2).sum())} | "
                 f"{int(d0.n_gold_answers.max())} |")
    L += [f"| **합계** | {int((u.n_gold_answers == 1).sum())} | "
          f"{int((u.n_gold_answers >= 2).sum())} | {int(u.n_gold_answers.max())} |", "",
          "다중 정답 판정: 원소 **전부 일치**를 요구한다. 예측을 콤마로 나눠 정규화한 "
          "뒤 gold 원소 집합과 다중집합으로 비교하며, 순서는 무시하고 부분 일치는 "
          "인정하지 않는다.", "",
          "`gold_cell` 조건은 PREREGISTER 개정 4에 따라 계산 자원 제약으로 pool당 "
          "30건으로 축소했다(`seed=42` 부분집합). `hitab_lookup`은 축소 결정 전에 60건 "
          "전량이 이미 실행돼 60을 그대로 쓴다. `hitab_arith`의 31은 재배치 이전 실행에서 "
          "1건이 먼저 기록돼 있었기 때문이다. pool별 실제 n은 아래 표에 병기했다.", "",
          "## pool x 조건별 EM", "",
          "| pool | dataset | " + " | ".join(POLS) + " |",
          "|---|---|" + "---|" * len(POLS)]
    for p in POOLS:
        d0 = df[df["pool"] == p]
        L.append(f"| {p} | {d0['dataset'].iloc[0]} | " + " | ".join(
            block(d0[d0["policy"] == pol], pol) for pol in POLS) + " |")
    L += ["", "## 유형별 EM (조회 180 / 산술 114)", "",
          "| query_type | " + " | ".join(POLS) + " |", "|---|" + "---|" * len(POLS)]
    for t in ("lookup", "arith"):
        d0 = df[df["query_type"] == t]
        L.append(f"| {t} | " + " | ".join(
            block(d0[d0["policy"] == pol], pol) for pol in POLS) + " |")

    L += ["", "## HiTab 내부 lookup(60) vs arith(60)", "",
          "| pool | " + " | ".join(POLS) + " |", "|---|" + "---|" * len(POLS)]
    for p in ("hitab_lookup", "hitab_arith"):
        d0 = df[df["pool"] == p]
        L.append(f"| {p} | " + " | ".join(
            block(d0[d0["policy"] == pol], pol) for pol in POLS) + " |")

    L += ["", "## gold_in_topk 군별 EM (검색 조건만)", "",
          "| policy | pool | gold_in_topk=True | gold_in_topk=False |",
          "|---|---|---|---|"]
    for pol in POLS[:2]:
        for p in POOLS:
            d0 = df[(df["policy"] == pol) & (df["pool"] == p)]
            L.append(f"| {pol} | {p} | {block(d0[d0['gold_in_topk']], '')} | "
                     f"{block(d0[~d0['gold_in_topk']], '')} |")

    L += ["", "## gold_cell 조건 = 리더 상한", "",
          "| pool | EM |", "|---|---|"]
    for p in POOLS:
        L.append(f"| {p} | {block(df[(df['pool'] == p) & (df['policy'] == 'gold_cell')], '')} |")
    L.append(f"| **전체** | {block(df[df['policy'] == 'gold_cell'], '')} |")

    cap = df[df["policy"] == "P4_path_cell"].groupby("pool")["hit_chunk_cap"].sum()
    L += ["", "## hit_chunk_cap (P4 상위 200 상한 도달)", "",
          "| pool | 건수 |", "|---|---|"]
    for p in POOLS:
        L.append(f"| {p} | {int(cap.get(p, 0))} |")
    L.append(f"| **합계** | {int(cap.sum())} |")

    L += ["", "## prompt_tokens / latency / n_chunks_used 분포", "",
          "| policy | ptok mean | median | max | lat mean | median | max | "
          "n_chunks mean | median | max |", "|---|" + "---|" * 9]
    for pol in POLS:
        d0 = df[df["policy"] == pol]
        L.append(f"| {pol} | {d0.prompt_tokens.mean():.0f} | "
                 f"{d0.prompt_tokens.median():.0f} | {d0.prompt_tokens.max()} | "
                 f"{d0.latency_sec.mean():.2f} | {d0.latency_sec.median():.2f} | "
                 f"{d0.latency_sec.max():.2f} | {d0.n_chunks_used.mean():.1f} | "
                 f"{d0.n_chunks_used.median():.0f} | {d0.n_chunks_used.max()} |")

    fail = df[df["pred_parsed"].astype(str).str.strip() == ""]
    L += ["", "## 파싱 실패 (pred_parsed 비어 있음)", "",
          "| policy | 건수 |", "|---|---|"]
    for pol in POLS:
        L.append(f"| {pol} | {int((fail['policy'] == pol).sum())} |")
    L += [f"| **합계** | {len(fail)} |", "",
          f"`retrieved_topk`이 Excel 셀 상한(32767자)을 넘어 청크 본문을 뺀 행: "
          f"**{n_trim}** (본문은 `results/phase4/retrieval_294/`에 그대로 있다).",
          f"총 소요 {df.latency_sec.sum()/3600:.2f}시간.", ""]

    MD.write_text("\n".join(L))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
