#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""주지표(조회 m=1, n=991)에서 답이 틀린 질의 전수 — 한 파일로.

이 리포의 실패 분석이 세 군데(§검색 실패 / 리더 천장 / distractor)에 흩어져 있어서
같은 질의를 두 번 세거나 한 층만 보고 결론을 내는 사고가 났다. 여기서는 **한 질의가
정확히 한 bucket** 에 들어가고, 세 bucket 의 합이 오답 수와 맞는지 실행할 때마다
검사한다(아래 assert).

  PYTHONPATH=. .venv/bin/python analysis/failure_dump.py
  -> results/retrieval_accuracy/FAILURES.json
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.accuracy_tables import primary_ids                     # noqa: E402

D = ROOT / "results/retrieval_accuracy"
#: 셀 문장 = "In the table 'T', among <행경로>, the value of <열경로> is <값>."
SENT = re.compile(r"^In the table '(?P<table>.*?)', among (?P<row>.*?), "
                  r"the value of (?P<col>.*?) is (?P<value>.*?)\.?$")


def leg(name):
    return {j["query_id"]: j for j in
            map(json.loads, (D / f"t_s3c_hybrid_answer_{name}.jsonl").open())}


def parse(line):
    m = SENT.match(line.strip())
    return m.groupdict() if m else None


def misread(rec, pred, gold_pred):
    """리더가 읽은 줄이 gold 줄과 어떤 관계인가. 못 찾으면 문맥 밖의 값이다."""
    ctx = [parse(c) for c in rec.get("context") or []]
    if any(c is None for c in ctx):
        return {"where": "문장 파싱 실패"}
    p, g = pred.strip().rstrip("."), str(gold_pred).strip().rstrip(".")
    hit = [c for c in ctx if c["value"] == p]
    ref = [c for c in ctx if c["value"] == g]
    if not hit:
        return {"where": "문맥에 없는 값"}
    if not ref:
        return {"where": "gold 줄 특정 실패"}
    h, r = hit[0], ref[0]
    where = ("다른 표의 셀" if h["table"] != r["table"] else
             "같은 행, 다른 열" if h["row"] == r["row"] else
             "같은 열, 다른 행" if h["col"] == r["col"] else
             "같은 표, 행·열 모두 다름")
    return {"where": where, "read_row_path": h["row"], "read_col_path": h["col"],
            "gold_row_path": r["row"], "gold_col_path": r["col"]}


def main() -> int:
    P = primary_ids()
    recs = {j["query_id"]: j for j in
            map(json.loads, (D / "t_s3c_hybrid_records.jsonl").open())
            if "correct" in j}
    ret, gold = leg("retrieved"), leg("gold")
    k10 = {j["query_id"]: j for j in map(
        json.loads, (D / "t_s3c_k10_answer_retrieved.jsonl").open())}
    ceil = {c["query_id"]: c for c in
            json.loads((D / "CEILING_CASES.json").read_text())}

    rows = []
    for q in sorted(P):
        r, a, g = recs[q], ret[q], gold[q]
        if a["answer_correct"] and g["answer_correct"]:
            continue                       # 두 조건 다 맞은 질의는 실패가 아니다
        # 순서가 규칙이다: 배치 조건(retrieved)이 맞은 질의는 배치 실패가 아니다.
        # 이 순서를 뒤집으면 gold 만 틀린 17건이 리더 천장으로 새어 든다.
        if a["answer_correct"]:
            bucket = "gold_only_fail"      # 20셀은 맞고 gold 한 줄은 틀림
        elif not r["correct"]:
            bucket = "retrieval_miss"      # 근거가 문맥에 없다
        elif not g["answer_correct"]:
            bucket = "reader_ceiling"      # 정답 셀만 줘도 틀린다
        else:
            bucket = "distractor"          # 근거는 있는데 옆 셀을 읽었다
        row = {"query_id": q, "bucket": bucket, "table_id": r["table_id"],
               "question": r["question"], "gold_answer": r["answer"],
               "aggregation": r.get("aggregation"), "m": r["m"],
               "retrieval_correct": r["correct"],
               "pred_retrieved_k20": a["pred"], "correct_retrieved_k20": a["answer_correct"],
               "pred_retrieved_k10": k10.get(q, {}).get("pred"),
               "correct_retrieved_k10": k10.get(q, {}).get("answer_correct"),
               "pred_gold_only": g["pred"], "correct_gold_only": g["answer_correct"],
               "n_context_cells": len(r.get("context") or []),
               "gold_cells": r.get("gold_cells")}
        if bucket == "distractor":
            row["misread"] = misread(r, a["pred"], g["pred"])
        if q in ceil:
            row["ceiling_class"] = ceil[q]["class"]
            row["ceiling_route"] = ceil[q]["route"]
            row["dataset_defect"] = ceil[q]["dataset_defect"]
        rows.append(row)

    by = Counter(x["bucket"] for x in rows)
    wrong = sum(1 for q in P if not ret[q]["answer_correct"])
    assert by["retrieval_miss"] + by["reader_ceiling"] + by["distractor"] == wrong, \
        f"세 bucket 합 {by} 이 오답 {wrong} 과 다르다"
    assert sum(1 for q in P if not gold[q]["answer_correct"]) == \
        sum(1 for x in rows if not x["correct_gold_only"]), \
        "gold 조건 실패가 전수로 실려 있지 않다"

    out = {
        "population": "단일 셀 조회 (주지표) n=991 — aggregation none & gold 셀 1개",
        "scorer": "hitab_exact_match_text (HiTab 공식, 허용오차 없음)",
        "reader": "local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit, temperature 0",
        "source": {"records": "t_s3c_hybrid_records.jsonl",
                   "legs": ["t_s3c_hybrid_answer_retrieved.jsonl",
                            "t_s3c_hybrid_answer_gold.jsonl",
                            "t_s3c_k10_answer_retrieved.jsonl"],
                   "classes": "CEILING_CASES.json"},
        "caveat": ("gold 레그는 2026-09-09 제목 버그 수정 이전 실행이다 "
                   "(BUGFIX_LOG.md). bucket 판정 중 gold 조건에 기대는 부분은 "
                   "재실행 전까지 잠정이다."),
        "buckets": {
            "retrieval_miss": "정답 셀이 20칸 안에 없다 — 검색의 몫",
            "reader_ceiling": "정답 셀만 줘도 틀린다 — 리더의 몫",
            "distractor": "정답 셀은 문맥에 있는데 다른 셀을 읽었다 — 문맥의 몫",
            "gold_only_fail": "20셀에서는 맞고 gold 한 줄에서 틀렸다"},
        "counts": dict(by), "n_failures": len(rows),
        "answer_em_k20": round(sum(ret[q]["answer_correct"] for q in P) / len(P), 4),
        "cases": rows,
    }
    f = D / "FAILURES.json"
    f.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps({"file": str(f), "n_failures": len(rows), "counts": dict(by)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
