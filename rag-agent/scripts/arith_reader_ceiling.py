#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""산술 질의의 **리더 천장**: 정답 셀만 주입했을 때의 EM (`CLAUDE.md §7` 요구).

`--pilot` 은 "리더가 산술을 못하는가, 프롬프트가 못 하게 막았는가"를 가르는 2×2 파일럿이다.
현행 `neutral` 프롬프트는 "답만 쓰라 + 64토큰"이라 **풀이(chain-of-thought)를 구조적으로
금지**한다. Qwen2.5-7B-Instruct 의 공식 GSM8K 는 95.2 이므로, 우리 산술 EM .17 은 모델
무능보다 측정 설정을 가리킨다 — 그래서 모델(리더)과 프롬프트(direct/cot)를 분리해 잰다.

`answer_accuracy.py --condition gold` 를 그대로 쓰지 못하는 이유: 그 경로의 무결성 가드가
`tables_sha256` 을 **전체 실행의 표 집합**(538개) 기준으로 비교하는데, per-type 분할 파일
(`*_arithmetic_records.jsonl`)의 표는 부분집합(90개)이라 항상 불일치한다. 가드의 내용 검사
두 개(split sha256, page_titles sha256)는 통과함을 확인했다. 렌더러(`gold_context`)·
프롬프트·채점(`hitab_exact_match_text`)은 `answer_accuracy.py` 것을 그대로 import 해 쓴다.

**Qwen3 thinking 모드는 꺼진다**(`rag_agent/llm/local_qwen.py` 가 `enable_thinking=False`
로 고정). 켜면 `<think>` 블록이 토큰 예산을 먹고 EM 채점 문자열에 섞인다.

  PYTHONPATH=. .venv/bin/python scripts/arith_reader_ceiling.py --pilot --n 100
  PYTHONPATH=. .venv/bin/python scripts/arith_reader_ceiling.py --reader local:Qwen/Qwen3-8B?quantization=4bit --prompt cot
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rag_agent.eval.metrics import hitab_exact_match_text              # noqa: E402
from rag_agent.llm.factory import build_llm                            # noqa: E402
from scripts.answer_accuracy import PROMPTS, gold_context, load_evidence  # noqa: E402
from answer_accuracy_mh import COT, extract                            # noqa: E402

RECS = ROOT / "results" / "retrieval_accuracy" / "t_sleaf_gold_arithmetic_records.jsonl"
POP = ROOT / "results" / "fair_filter_arith_population_216.json"
OUT_DIR = ROOT / "results" / "fair_filter_arith_20260921"
DATA = "data/hitab"
QWEN25 = "local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit"
QWEN3 = "local:Qwen/Qwen3-8B?quantization=4bit"


def run_cell(llm, recs, ids, tabs, prompt: str, max_tokens: int) -> list:
    """한 칸(리더 × 프롬프트). gold 셀만 준 문맥에서 EM 을 잰다."""
    sys_prompt = COT if prompt == "cot" else PROMPTS["neutral"]
    rows = []
    for q in ids:
        r = recs[q]
        ctx = gold_context(r, tabs, DATA)          # 정답 셀만, 그 셀의 헤더 경로와 함께
        user = "Context:\n" + "\n".join(ctx) + f"\n\nQuestion: {r['question']}\nAnswer:"
        raw = llm.complete(sys_prompt, user, max_tokens=max_tokens, temperature=0.0)
        pred, marked = extract(raw) if prompt == "cot" else (raw, False)
        rows.append({"query_id": q, "m": r.get("m"), "aggregation": r.get("aggregation"),
                     "n_gold_lines": len(ctx), "question": r["question"],
                     "answer": r["answer"], "raw": raw, "pred": pred,
                     "final_answer_marked": int(marked),
                     "correct": int(hitab_exact_match_text(pred, r["answer"]))})
    return rows


def summarize(rows, label) -> dict:
    by_m: dict = {}
    for x in rows:
        b = by_m.setdefault(str(x["m"]), [0, 0])
        b[0] += x["correct"]
        b[1] += 1
    return {"cell": label, "n": len(rows),
            "em": round(sum(x["correct"] for x in rows) / len(rows), 4),
            "final_answer_marked": round(
                sum(x["final_answer_marked"] for x in rows) / len(rows), 4),
            "em_by_m": {k: round(v[0] / v[1], 4) for k, v in sorted(by_m.items())},
            "n_by_m": {k: v[1] for k, v in sorted(by_m.items())}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", action="store_true",
                    help="2x2 파일럿: Qwen3{direct,cot} + Qwen2.5{direct}(재현 확인)")
    ap.add_argument("--n", type=int, default=0, help="앞 n 건만 (0=전수 216)")
    ap.add_argument("--reader", default=QWEN3)
    ap.add_argument("--prompt", default="cot", choices=["neutral", "cot"])
    ap.add_argument("--max-tokens", type=int, default=0, help="0=direct 64 / cot 384")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    recs, _meta = load_evidence(RECS)
    ids = json.load(open(POP))["query_ids"]
    if a.n:
        ids = ids[:a.n]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tabs: dict = {}
    t0 = time.time()

    cells = ([(QWEN3, "neutral", 64), (QWEN3, "cot", 384), (QWEN25, "neutral", 64)]
             if a.pilot else
             [(a.reader, a.prompt, a.max_tokens or (384 if a.prompt == "cot" else 64))])
    out = []
    for reader, prompt, mt in cells:
        label = f"{reader.split('/')[-1].split('?')[0]} × {prompt}({mt}tok)"
        print(f"[cell] {label}  n={len(ids)}", flush=True)
        llm = build_llm(reader)
        rows = run_cell(llm, recs, ids, tabs, prompt, mt)
        name = (f"{'pilot_' if a.pilot else 'ceiling_'}"
                f"{reader.split('/')[-1].split('?')[0].replace('.', '')}_{prompt}"
                + (f"_{a.tag}" if a.tag else ""))
        (OUT_DIR / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        s = summarize(rows, label)
        s["reader"], s["prompt"], s["max_tokens"] = reader, prompt, mt
        s["enable_thinking"] = False           # local_qwen.py 가 고정
        s["file"] = f"{name}.jsonl"
        out.append(s)
        print(json.dumps(s, ensure_ascii=False), flush=True)
        del llm
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()

    res = {"elapsed_s": round(time.time() - t0, 1), "population": POP.name,
           "n": len(ids), "context": "gold(정답 셀만 주입)", "cells": out}
    name = "pilot_reader_2x2.json" if a.pilot else f"ceiling_{a.tag or 'main'}.json"
    (OUT_DIR / name).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
