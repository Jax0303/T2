#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""HiTab "표 전체" 비교군 — 검색 없이 질문의 표 전체를 리더에 넣는다.
PREREG-2026-09-24-mh-answer-cap300.md §표 전체.

HiTab 7개 방법 표(`fair_filter_eval.py` 의 무필터 리더, `results/fair_filter_20260921/rows.jsonl`)와
같은 300건·같은 리더·같은 프롬프트·같은 출력 한도(64)·배치 1. 바뀌는 것은 문맥 하나:
표 전체를 청킹 arm 과 같은 markdown(`chunks.markdown_source`)으로, 이름은 색인과 같은 라벨(섹션 제목 + 페이지 제목).

  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python scripts/hitab_fulltable_answer.py

--top1-records (PREREG-2026-09-28-hitab538-answer.md): 질문의 표 대신 표 단위 색인 검색 1위 표를 같은 markdown 으로 넣는다.
질문·정답·정답 셀도 그 기록에서 읽는다(sleaf 기록을 쓰지 않는다).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_agent.bench import hitab_grid as hg                            # noqa: E402
from rag_agent.eval.metrics import hitab_exact_match_text               # noqa: E402
from rag_agent.llm.factory import build_llm                             # noqa: E402
from rag_agent.serialization.caption import with_page_title             # noqa: E402
from rag_agent.serialization.chunks import markdown_source              # noqa: E402
from scripts.answer_accuracy import (_PAGE_TITLES, PROMPTS,             # noqa: E402
                                     check_context_limit, load_evidence)

POP = ROOT / "results" / "ksweep_population_300.json"
RECORDS = ROOT / "results" / "retrieval_accuracy" / "t_sleaf_gold_records.jsonl"
OUT = ROOT / "results" / "fulltable_20260924" / "hitab_rows.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top1-records", default="", help="표 단위 색인 검색 기록 (검색 1위 표를 넣는다)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    pop = json.load(open(POP))["query_ids"]
    recs, meta = load_evidence(a.top1_records or RECORDS)
    if a.top1_records and meta["arguments"]["unit"] != "table":
        raise SystemExit("--top1-records 는 표 단위 색인 기록이어야 한다")
    # context_units 는 검색 순위 순서(budget_select) — 첫 단위가 1위 표
    top1 = {q: recs[q]["context_units"][0]["cells"][0][0] for q in pop} if a.top1_records else {}
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} 있음 — 덮어쓰지 않는다")
    out.parent.mkdir(parents=True, exist_ok=True)
    llm = build_llm("local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit")
    limit = getattr(llm, "context_limit", 0)
    n_ok, t0 = 0, time.time()
    with out.open("x", encoding="utf-8", newline="\n") as stream:
        for k, q in enumerate(pop, 1):
            r = recs[q]
            tid = top1.get(q, r["table_id"])
            tab = hg.load_table(tid)
            title = with_page_title(tab.title, _PAGE_TITLES.get(tid))
            user = ("Context:\n" + markdown_source(tab, tab.table, title)[0]
                    + f"\n\nQuestion: {r['question']}\nAnswer:")
            n_tok = llm.n_prompt_tokens(PROMPTS["neutral"], user)
            check_context_limit(n_tok, 64, limit)
            pred = llm.complete(PROMPTS["neutral"], user, max_tokens=64, temperature=0.0)
            ok = int(hitab_exact_match_text(pred, r["answer"]))
            n_ok += ok
            row = {"arm": "table_top1" if top1 else "fulltable", "query_id": q, "question": r["question"],
                   "answer": r["answer"], "reader_input_tokens": n_tok, "pred_base": pred, "correct_base": ok}
            if top1:   # 단일 셀 조회라 정답 셀 전부가 1위 표에 있으면 검색 성공
                row |= {"table_id_used": tid, "retrieval_correct": int(all(c[0] == tid for c in r["gold_cells"]))}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            if k % 50 == 0:
                print(f"  {k}/{len(pop)} {time.time() - t0:.0f}s acc={n_ok / k:.4f}", flush=True)
    print(f"{'table_top1' if top1 else 'fulltable'} {n_ok}/{len(pop)} = {n_ok / len(pop):.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
