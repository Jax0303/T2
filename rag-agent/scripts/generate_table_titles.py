#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Write a title for every table in a corpus that ships none. Index-time, once.

`PREREG-2026-08-27-generated-title.md`. The unique-tag experiment showed a
meaningless table tag buys the label share of what a title buys and none of the
table-count share, because only an address a query can MATCH buys the second.
A generated name has no such limit -- so the question is whether the title's
gain came from the name existing or from the name the dataset happened to ship.

The generator never sees a question or an answer: input is the header labels and
the first rows, so nothing about the evaluation can leak into the index.

Titles are cached to JSON and frozen. Regenerating would silently change the
contrast, so an existing file is reused unless --overwrite is passed.

  PYTHONPATH=. python3 scripts/generate_table_titles.py --dataset aitqa
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus_dump_vs_cell import aitqa_corpus, multihiertt_corpus
from rag_agent.llm.factory import build_llm

SYS = ("You name tables. Given the header labels and first rows of one table, "
       "reply with a short noun phrase naming what the table reports -- the "
       "subject and the unit of measurement if present. Six words at most. "
       "No quotes, no explanation, no 'Table of'. Reply with the name only.")


def prompt_for(md_lines: list[str], n_rows: int) -> str:
    body = "\n".join(md_lines[:n_rows])
    return f"{body}\n\nName this table."


def clean(txt: str) -> str:
    """One line, unquoted -- the title is pasted into a sentence verbatim."""
    if not txt or not txt.strip():
        return ""
    return txt.strip().splitlines()[0].strip().strip('"').strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="aitqa", choices=["aitqa", "multihiertt"])
    ap.add_argument("--mh-queries", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rows", type=int, default=6, help="header + data rows shown")
    ap.add_argument("--model",
                    default="local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit&dtype=float16")
    ap.add_argument("--out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = Path(args.out or f"data/{args.dataset}/generated_titles.json")
    if out.exists() and not args.overwrite:
        have = json.loads(out.read_text())
        print(f"{out} already holds {len(have)} titles; --overwrite to regenerate")
        return 0

    C = (aitqa_corpus() if args.dataset == "aitqa"
         else multihiertt_corpus(args.mh_queries, args.seed))
    shipped = sum(1 for t in C.tids if C.title.get(t))
    if shipped:
        print(f"WARNING: {shipped}/{len(C.tids)} tables already ship a title; "
              f"this script is for corpora that ship none", flush=True)

    llm = build_llm(args.model)
    titles = {}
    for n, tid in enumerate(C.tids, 1):
        titles[tid] = clean(llm.complete(system=SYS,
                                         user=prompt_for(C.md_lines[tid], args.rows),
                                         max_tokens=32))
        if n % 10 == 0 or n == len(C.tids):
            print(f"[{n}/{len(C.tids)}] {tid}: {titles[tid]!r}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(titles, ensure_ascii=False, indent=1))
    empty = sum(1 for v in titles.values() if not v)
    uniq = len({v for v in titles.values() if v})
    print(f"wrote {len(titles)} titles -> {out}  (empty {empty}, distinct {uniq})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
