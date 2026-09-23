#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""탐색용 파일럿 — 같은 7B 리더에게 풀이를 허용하면 산술 EM 이 오르는가.

**결과 보고용이 아니다.** validation 분할, gold 셀만 준 문맥에서 두 프롬프트를 같은 질의에
돌려 리더 천장과 질의당 시간을 잰다. 다음 사전등록(산술 답변 EM 레그)의 프롬프트·표본 크기·
소요 시간을 정하는 근거로만 쓴다.

  direct  현행 `neutral` (답만, 64토큰)
  cot     풀이를 쓰고 마지막 줄에 'Final answer:' (384토큰). 표시가 없으면 출력 전체로 채점
          — 추출을 관대하게 하지 않는다.
  think_boxed  Qwen3 모델 카드의 수학 권장 그대로: 생각 모드, 사용자 메시지 끝에 "Please reason step by
          step, and put your final answer within \\boxed{}.", 샘플링 temp .6 / top_p .95 / top_k 20 /
          min_p 0 (생각 모드에서 greedy 금지). 마지막 \\boxed{} 안만 채점하고 없으면 답 부분 전체로
          채점한다. PREREG-2026-09-23-reader-thinking-pilot.md
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from answer_accuracy_mh import COT, PROMPTS, extract, gold_context    # noqa: E402
import mh_arms as M                                                   # noqa: E402
from rag_agent.eval.artifacts import write_pair                       # noqa: E402
from rag_agent.eval.multihiertt_em import docmath_match, mh_exact_match  # noqa: E402
from rag_agent.llm.factory import build_llm                           # noqa: E402

BOXED_SYSTEM = ("Answer the question using only the provided table context. Read the row and "
                "column labels to identify the relevant values.")
BOXED_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."
_BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
THINK_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}


def extract_boxed(text: str):
    """마지막 \\boxed{} 안. LaTeX 표기(\\text{}, \\%, \\$, \\,)만 벗긴다 — 채점기가 숫자로 읽게."""
    hits = _BOXED.findall(text or "")
    if not hits:
        return (text or "").strip(), False
    pred = re.sub(r"\\text\{([^{}]*)\}", r"\1", hits[-1])
    return pred.replace("\\%", "%").replace("\\$", "$").replace("\\,", ",").strip(), True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--reader", default="local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit")
    ap.add_argument("--label-rule", default="L1", choices=["none", "L1", "L2"],
                    help="gold 문장 라벨. 기본 L1 은 2026-09-13 첫 파일럿 재현용")
    ap.add_argument("--conditions", default="direct,cot",
                    help="direct,cot,think_boxed 중 쉼표 목록. 기본은 2026-09-13 파일럿 재현")
    ap.add_argument("--think-max-tokens", type=int, default=8192)
    ap.add_argument("--gen-seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    queries, docs, _ = M.load_population(a.split)
    tables, hdr = M.build_tables(docs, "v2", a.label_rule)
    live = {(tid, i, j) for tid, tab in tables.items() for i, row in enumerate(tab.table.data)
            for j, v in enumerate(row) if str(v).strip()}
    queries = M.resolve_gold(copy.deepcopy(queries), tables, hdr, live)
    arith = sorted((q for q in queries if q["kind"] == "arith" and q["gold"] and not q["excluded"]),
                   key=lambda q: q["uid"])
    pick = sorted(random.Random(a.seed).sample(arith, min(a.n, len(arith))), key=lambda q: q["uid"])
    print(f"[pilot] {a.split} 산술 {len(arith)} 중 {len(pick)}", flush=True)

    llm = build_llm(a.reader)
    rows, sec = [], {}
    spec = {"direct": (PROMPTS["neutral"], 64), "cot": (COT, 384),
            "think_boxed": (BOXED_SYSTEM, a.think_max_tokens)}
    conds = a.conditions.split(",")
    for cond in conds:
        system, max_tok = spec[cond]
        import torch
        torch.manual_seed(a.gen_seed)
        t0 = time.time()
        for q in pick:
            ctx = gold_context(sorted(q["gold"]), tables)
            user = "Context:\n" + "\n".join(ctx) + f"\n\nQuestion: {q['question']}\nAnswer:"
            t = time.time()
            if cond == "think_boxed":
                out = llm.complete(system, user.removesuffix("Answer:") + BOXED_SUFFIX,
                                   max_tokens=max_tok, thinking=True, **THINK_SAMPLING)
                pred, marked = extract_boxed(out)
            else:
                out = llm.complete(system, user, max_tokens=max_tok, temperature=0.0)
                pred, marked = extract(out) if cond == "cot" else (out, True)
            prog = q["kind"] == "arith"
            rows.append({"query_id": q["uid"], "condition": cond, "m": len(q["gold"]),
                         "answer": q["answer"], "raw": out, "pred": pred, "marker_found": marked,
                         "answer_correct": int(mh_exact_match(pred, q["answer"], prog)),
                         "answer_correct_docmath": int(docmath_match(pred, q["answer"], prog)),
                         "seconds": round(time.time() - t, 2),
                         **getattr(llm, "last_generation", {})})
        dt = time.time() - t0
        sec[cond] = round(dt / len(pick), 2)
        rs = [r for r in rows if r["condition"] == cond]
        print(f"[{cond}] EM {sum(r['answer_correct'] for r in rs)/len(rs):.3f}  "
              f"질의당 {dt/len(rs):.1f}s  표시 누락 {sum(not r['marker_found'] for r in rs)}", flush=True)
    summary = {"split": a.split, "n": len(pick), "seed": a.seed, "reader": llm.name,
               "header_rule": "v2", "label_rule": a.label_rule, "context": "gold cells only",
               "exploratory": True, "cot_prompt": COT, "direct_prompt": PROMPTS["neutral"],
               "think_boxed": {"system": BOXED_SYSTEM, "suffix": BOXED_SUFFIX, **THINK_SAMPLING,
                               "max_new_tokens": a.think_max_tokens},
               "gen_seed": a.gen_seed, "query_ids": [q["uid"] for q in pick]}
    for cond in conds:
        rs = [r for r in rows if r["condition"] == cond]
        summary[cond] = {"em": sum(r["answer_correct"] for r in rs) / len(rs),
                         "em_docmath": sum(r["answer_correct_docmath"] for r in rs) / len(rs),
                         "marker_missing": sum(not r["marker_found"] for r in rs),
                         "think_unclosed": sum(not r.get("think_closed", True) for r in rs),
                         "new_tokens_mean": (round(sum(r["new_tokens"] for r in rs) / len(rs), 1)
                                             if "new_tokens" in rs[0] else None),
                         "sec_per_query": sec[cond]}
    write_pair(Path(a.out), rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
