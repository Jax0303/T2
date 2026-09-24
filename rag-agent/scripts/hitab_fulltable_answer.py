#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""HiTab "표 전체" 비교군 — 검색 없이 질문의 표 전체를 리더에 넣는다.
PREREG-2026-09-24-mh-answer-cap300.md §표 전체.

HiTab 7개 방법 표(`fair_filter_eval.py` 의 무필터 리더, `results/fair_filter_20260921/rows.jsonl`)와
같은 300건·같은 리더·같은 프롬프트·같은 출력 한도(64)·배치 1. 바뀌는 것은 문맥 하나:
표 전체를 청킹 arm 과 같은 markdown(`chunks.markdown_source`)으로, 이름은 색인과 같은 라벨(섹션 제목 + 페이지 제목).

  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python scripts/hitab_fulltable_answer.py
"""
from __future__ import annotations

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
    pop = json.load(open(POP))["query_ids"]
    recs, _ = load_evidence(RECORDS)
    if OUT.exists():
        raise SystemExit(f"{OUT} 있음 — 덮어쓰지 않는다")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    llm = build_llm("local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit")
    limit = getattr(llm, "context_limit", 0)
    n_ok, t0 = 0, time.time()
    with OUT.open("x", encoding="utf-8", newline="\n") as stream:
        for k, q in enumerate(pop, 1):
            r = recs[q]
            tab = hg.load_table(r["table_id"])
            title = with_page_title(tab.title, _PAGE_TITLES.get(r["table_id"]))
            user = ("Context:\n" + markdown_source(tab, tab.table, title)[0]
                    + f"\n\nQuestion: {r['question']}\nAnswer:")
            n_tok = llm.n_prompt_tokens(PROMPTS["neutral"], user)
            check_context_limit(n_tok, 64, limit)
            pred = llm.complete(PROMPTS["neutral"], user, max_tokens=64, temperature=0.0)
            ok = int(hitab_exact_match_text(pred, r["answer"]))
            n_ok += ok
            stream.write(json.dumps({"arm": "fulltable", "query_id": q, "question": r["question"],
                                     "answer": r["answer"], "reader_input_tokens": n_tok,
                                     "pred_base": pred, "correct_base": ok},
                                    ensure_ascii=False) + "\n")
            stream.flush()
            if k % 50 == 0:
                print(f"  {k}/{len(pop)} {time.time() - t0:.0f}s acc={n_ok / k:.4f}", flush=True)
    print(f"fulltable {n_ok}/{len(pop)} = {n_ok / len(pop):.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
