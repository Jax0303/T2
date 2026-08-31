#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit the answer-matched gold cells of AIT-QA / RealHiTBench, query by query.

AIT-QA and RealHiTBench annotate no gold cells. Both corpora recover them by
matching the ANSWER STRING against cell values (``corpus_dump_vs_cell.py:476``),
keeping a question only when the match count equals the answer count. That filter
catches "the value appears in several cells"; it does NOT catch "the value appears
in exactly one cell, and that cell has nothing to do with the question" -- which
is the failure mode when the answer is a COMPUTED number that happens to collide
with some unrelated cell.

This script emits one row per query so the claim can be checked rather than taken
on trust: the question, the gold cell's own sentence, and how many content words
they share. Zero shared words is the flag -- not proof that the label is wrong,
but the set a human has to look at.

  PYTHONPATH=.:scripts python3 scripts/gold_label_audit.py \
      --dataset realhitbench --records results/hyb_realhitbench_s3c_4096_a7.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_WORD = re.compile(r"[a-z0-9]+")
# closed-class words carry no topic signal, so counting them as "shared" would
# hide exactly the mismatches this audit is looking for
_STOP = {"the", "and", "for", "are", "was", "were", "what", "which", "how", "many",
         "much", "total", "number", "value", "with", "from", "that", "this", "have",
         "has", "between", "difference", "average", "sum", "all", "per", "not"}


def words(s) -> set:
    return {w for w in _WORD.findall(str(s).lower()) if len(w) > 2 and w not in _STOP}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["realhitbench", "aitqa"], required=True)
    ap.add_argument("--records", help="a run's *.json; its _records.jsonl is read "
                                      "to join per-query OSC. Optional.")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from corpus_dump_vs_cell import realhitbench_corpus, aitqa_corpus
    C = realhitbench_corpus() if args.dataset == "realhitbench" else aitqa_corpus()

    # QuestionType only exists upstream of the query dicts, so join it back by id
    meta = {}
    if args.dataset == "realhitbench":
        for q in json.load(open("data/realhitbench/QA_final.json"))["queries"]:
            meta[str(q["id"])] = {"QuestionType": q.get("QuestionType"),
                                  "SubQType": q.get("SubQType")}

    osc = {}
    if args.records:
        rf = args.records.replace(".json", "_records.jsonl")
        for line in open(rf):
            r = json.loads(line)
            osc[str(r["query_id"])] = r["cell"]["osc"]

    pos = {o: i for i, o in enumerate(C.cell_owner)}
    out_path = args.out or f"results/gold_label_audit_{args.dataset}.jsonl"
    rows = []
    for q in C.queries:
        qw = words(q["question"])
        cells = []
        best = -1
        for g in sorted(q["gold_cells"]):
            i = pos.get(g)
            text = C.cell_text[i] if i is not None else None
            # the value itself is what the gold was matched ON, so it would score
            # a spurious overlap; only the header path is evidence of aboutness
            path = text.rsplit(":", 1)[0] if text else ""
            ov = len(qw & words(path))
            best = max(best, ov)
            cells.append({"coord": list(g), "sentence": text, "shared_words": ov,
                          "shared": sorted(qw & words(path))})
        rows.append({"query_id": q["query_id"], "question": q["question"],
                     "answer": q["answer"], "gold_table": q["gold_table"],
                     "m": len(q["gold_cells"]), "max_shared_words": best,
                     "flag": "NO_SHARED_WORD" if best == 0 else "ok",
                     "osc": osc.get(str(q["query_id"])),
                     **meta.get(str(q["query_id"]), {}),
                     "gold_cells": cells})
    Path(out_path).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))

    n = len(rows)
    flagged = [r for r in rows if r["flag"] == "NO_SHARED_WORD"]
    print(f"\n{args.dataset}: 전체 {n}건 / 플래그(공통 단어 0개) {len(flagged)}건 = {len(flagged)/n:.1%}")
    if osc:
        for lab, sel in (("검색 성공", [r for r in rows if r["osc"] == 1]),
                         ("검색 실패", [r for r in rows if r["osc"] == 0])):
            if not sel:
                continue
            f = sum(1 for r in sel if r["flag"] == "NO_SHARED_WORD")
            print(f"  {lab}: {len(sel)}건 중 플래그 {f}건 = {f/len(sel):.1%}")
    if meta:
        print("\n플래그된 건의 QuestionType 분포:")
        for k, v in Counter(r.get("QuestionType") for r in flagged).most_common():
            tot = sum(1 for r in rows if r.get("QuestionType") == k)
            print(f"  {str(k):24} {v:>4}/{tot:<4} = {v/tot:.1%}")
    print(f"\n-> {out_path}  ({n}줄, 한 줄에 질의 하나)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
