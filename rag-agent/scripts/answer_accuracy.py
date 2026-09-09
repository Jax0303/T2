#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Answer accuracy on exactly the context retrieval delivered.

Table 2 to ``retrieval_accuracy.py``'s Table 1. Same split, same queries, same
cells — the reader is handed the retrieved context verbatim, so the two numbers
sit at one operating point and the question "retrieval is .9, why is the answer
not .9?" has an arithmetic answer instead of a guess.

Three conditions, because the gap has two possible owners:

  retrieved  the top-K cells the retriever chose. The deployed number.
  gold       only the annotated gold cells. The READER CEILING — whatever this
             misses, no retriever can fix.
  oracle     gold cells when retrieval found them all, the retrieved context
             when it did not. Isolates what perfect reranking inside K buys.

Scored with ``hitab_exact_match_text`` — HiTab's own scorer, no tolerance and no
rescaling, so the number stays comparable with published HiTab accuracies.

  PYTHONPATH=. .venv/bin/python scripts/answer_accuracy.py \
      --records results/retrieval_accuracy/t_s3c_hybrid_records.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_agent.bench import hitab_grid as hg                          # noqa: E402
from rag_agent.eval.metrics import hitab_exact_match_text             # noqa: E402
from rag_agent.llm.factory import build_llm                           # noqa: E402
from rag_agent.serialization.caption import (caption_sentence,        # noqa: E402
                                             with_page_title)
from rag_agent.serialization.templates import STRUCTURAL_COMPACT      # noqa: E402
from scripts.retrieval_accuracy import PAGE_TITLES                    # noqa: E402

BASE = ("You answer questions about a table. The context lines are cells of the "
        "table, each written as its headers and its value. Use only the context. "
        "Answer with the value alone — no sentence, no units, no explanation. "
        "If the question asks for several values, separate them with commas.")

# The output-format rules exist because the reader loses points on answers a
# person would mark correct: it rounds a value the context spells out in full,
# it converts a ratio the dataset stores as a fraction into a percentage, and it
# reorders a multi-value answer. None of that is the scorer being wrong -- the
# scorer is HiTab's own -- so the fix belongs in the instruction, not the
# grading. PREREG-2026-09-08-reader-prompt.md
FORMAT = BASE + (
    " Copy the value exactly as the context writes it: do not round it, do not "
    "drop or add decimal places, and do not add a percent sign or any other "
    "unit. If the question asks for a percentage or proportion and the context "
    "gives a fraction, keep the fraction's own notation. When several values "
    "are asked for, give them in the order the question names them. When the "
    "answer requires a calculation, give only the result of that calculation, "
    "never the numbers it was computed from.")

# The single largest error class under a retrieved context is reading the wrong
# cell: 432 of 490 wrong answers name a value that IS in the context, just not
# the one the question addresses. Naming the evidence line first makes that
# choice explicit instead of implicit.
EVIDENCE = FORMAT + (
    " First find the one context line whose headers match every part of the "
    "question. Then answer using that line's value. Output only the answer.")

PROMPTS = {"base": BASE, "format": FORMAT, "evidence": EVIDENCE}


#: The index prefixes ToTTo's page title (scripts/retrieval_accuracy.py
#: build_corpus). Until 2026-09-09 gold_context did not, so the gold/oracle
#: conditions handed the reader a title the corpus never held -- 188 of the 906
#: main-population retrieval hits. BUGFIX_LOG.md 2026-09-09.
_PAGE_TITLES = json.loads(PAGE_TITLES.read_text()) if PAGE_TITLES.exists() else {}


def gold_context(rec, tabs, data_dir):
    tab = tabs.get(rec["table_id"])
    if tab is None:
        tab = tabs[rec["table_id"]] = hg.load_table(rec["table_id"], data_dir)
    t = tab.table
    title = with_page_title(tab.title, _PAGE_TITLES.get(rec["table_id"]))
    return [caption_sentence(title, t.row_path(i), t.col_path(j),
                            value=t.data[i][j], template=STRUCTURAL_COMPACT)
            for _tid, i, j in rec.get("gold_cells", [])]


def summarize(rows, limit=0):
    """Leg metrics. A pure function of the rows so that a leg composed from
    other legs (analysis/compose_oracle.py) is summarised by this same code
    instead of a second copy of the definitions."""
    def acc(v):
        return round(sum(v) / len(v), 4) if v else None

    by_mode = defaultdict(list)
    for x in rows:
        by_mode[x["mode"]].append(x["answer_correct"])
    hit = [x["answer_correct"] for x in rows if x["retrieval_correct"]]
    miss = [x["answer_correct"] for x in rows if not x["retrieval_correct"]]
    def stat(key):
        v = sorted(x[key] for x in rows if x.get(key) is not None)
        if not v:
            return None
        return {"mean": round(sum(v) / len(v), 1), "median": v[len(v) // 2],
                "max": v[-1], "n": len(v)}

    over = [x for x in rows if x.get("n_tok") and limit and x["n_tok"] > limit]
    return {
        # 입력은 어디서도 잘리지 않는다(truncation 미설정). 그래서 한계를 넘는
        # 프롬프트는 짧아지는 것이 아니라 실패한다 — 아래는 절단율이 아니라
        # "한계 초과율"이고, 0 이 아니면 그 조건의 수치는 성립하지 않는다.
        "input_tokens": stat("n_tok"), "context_limit": limit,
        "n_over_context_limit": len(over),
        "over_context_limit_ratio": round(len(over) / len(rows), 4) if rows else None,
        "cells_delivered": stat("cells_in_context"),
        "context_lines": stat("n_ctx"),
        "n": len(rows), "answer_accuracy": acc([x["answer_correct"] for x in rows]),
        "answer_accuracy_all_mode": acc(by_mode["all"]), "n_all_mode": len(by_mode["all"]),
        "answer_accuracy_any_mode": acc(by_mode["any"]), "n_any_mode": len(by_mode["any"]),
        "retrieval_accuracy_here": acc([x["retrieval_correct"] for x in rows]),
        "answer_given_retrieval_hit": acc(hit), "n_retrieval_hit": len(hit),
        "answer_given_retrieval_miss": acc(miss), "n_retrieval_miss": len(miss),
        "by_aggregation": {k: acc(v) for k, v in sorted(
            ((k, [x["answer_correct"] for x in rows if (x["aggregation"] or "none") == k])
             for k in {(x["aggregation"] or "none") for x in rows}))},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True,
                    help="a *_records.jsonl written with --dump-context")
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--reader", default="local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit")
    ap.add_argument("--condition", default="retrieved",
                    choices=["retrieved", "gold", "oracle"])
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt", default="base", choices=list(PROMPTS),
                    help="base = the original instruction; format / evidence are "
                         "the two preregistered interventions")
    ap.add_argument("--exclude-unit-defect", action="store_true",
                    help="drop the queries whose gold is a fraction while the "
                         "question asks for a percentage (analysis/unit_defect.py). "
                         "Decided from the dataset alone, so queries we answer "
                         "CORRECTLY are dropped too.")
    ap.add_argument("--seed", type=int, default=42,
                    help="greedy 디코딩이라 무해하지만, 조건 넷이 같은 상태에서 "
                         "돌았다는 것을 기록으로 남긴다")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.records)]
    scored = [r for r in recs if "correct" in r]
    if a.exclude_unit_defect:
        from analysis.unit_defect import defect_ids
        bad = defect_ids(a.data_dir, "test")
        n0 = len(scored)
        scored = [r for r in scored if r["query_id"] not in bad]
        print(f"[exclude] 단위 불일치 라벨 결함 {n0 - len(scored)}건 제외 -> {len(scored)}건",
              flush=True)
    if a.limit:
        scored = scored[:a.limit]
    out = Path(a.out or Path(a.records).with_name(
        Path(a.records).stem.replace("_records", "")
        + f"_answer_{a.condition}"
        + ("" if a.prompt == "base" else f"_{a.prompt}")
        + ("_nodefect" if a.exclude_unit_defect else "") + ".jsonl"))
    # The per-query rows and the summary are written to `out` and to
    # `out.replace(".jsonl", ".json")`. An --out that does not end in .jsonl makes
    # those the same path and the summary silently destroys the predictions.
    if out.suffix != ".jsonl":
        raise SystemExit(f"--out must end in .jsonl (got {out.name}); the summary "
                         f"is written alongside it as .json")
    if out.exists():
        raise SystemExit(f"{out} exists — answer legs never overwrite; pass a new --out")

    llm = build_llm(a.reader)
    import torch                                                   # noqa: E402
    torch.manual_seed(a.seed)
    limit = getattr(llm, "context_limit", 0)
    tabs: dict = {}
    rows, t0 = [], time.time()
    for k, r in enumerate(scored, 1):
        if a.condition == "gold" or (a.condition == "oracle" and r["correct"]):
            ctx = gold_context(r, tabs, a.data_dir)
        else:
            ctx = r.get("context") or []
        user = "Context:\n" + "\n".join(ctx) + f"\n\nQuestion: {r['question']}\nAnswer:"
        n_tok = (llm.n_prompt_tokens(PROMPTS[a.prompt], user)
                 if hasattr(llm, "n_prompt_tokens") else None)
        pred = llm.complete(PROMPTS[a.prompt], user, max_tokens=a.max_tokens,
                            temperature=0.0)
        ok = hitab_exact_match_text(pred, r["answer"])
        rows.append({"query_id": r["query_id"], "mode": r["mode"],
                     "retrieval_correct": r["correct"], "answer_correct": int(ok),
                     "aggregation": r.get("aggregation"), "n_ctx": len(ctx),
                     "n_tok": n_tok, "cells_in_context": r.get("cells_in_context"),
                     "pred": pred, "answer": r["answer"]})
        if k % 50 == 0:
            print(f"  {k}/{len(scored)}  {time.time() - t0:.0f}s  "
                  f"acc={sum(x['answer_correct'] for x in rows) / k:.4f}", flush=True)

    with open(out, "w") as fh:
        for x in rows:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")

    summary = {"records": a.records, "condition": a.condition,
               "reader": llm.name, "prompt": a.prompt, "seed": a.seed,
               "max_new_tokens": a.max_tokens,
               "batch_size": 1,      # 질의당 1건 생성 — 조건 무관 고정
               "excluded_unit_defect": bool(a.exclude_unit_defect),
               **summarize(rows, limit)}
    Path(str(out).replace(".jsonl", ".json")).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
