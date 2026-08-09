#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Score a STOPPED four-arm baseline run from its records, no solver calls.

`baseline_comparison_llm.py` / `_multihiertt.py` only write their result file
when the loop finishes, so a run stopped on purpose (budget, time) leaves
paid-for answers in the records jsonl and nothing else. This scores those.

Only queries answered under EVERY arm are counted: an arm that got further
than the others would otherwise be scored on an easier or simply different
set of questions.

Run: PYTHONPATH=. .venv/bin/python scripts/aggregate_partial_arms.py \
        results/baseline_comparison_llm_sonnet5_records.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_comparison_llm import ARMS, mcnemar  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("--solver", default="", help="recorded in the output")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    recs = [json.loads(l) for l in Path(args.records).open()]
    key = "query_id" if "query_id" in recs[0] else "uid"
    by_q: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in recs:
        by_q[r[key]][r["arm"]] = r
    complete = [q for q, d in by_q.items() if all(a in d for a in ARMS)]
    if not complete:
        print("no query is complete across all four arms")
        return 1

    acc = {a: sum(by_q[q][a]["correct"] for q in complete) / len(complete)
           for a in ARMS}
    kinds = sorted({by_q[q][ARMS[0]].get("kind", "?") for q in complete})
    out = {
        "source_records": args.records,
        "solver": args.solver,
        "status": "PARTIAL — run stopped early on purpose; this is every query "
                  "answered under all four arms, not a planned sample size",
        "n_complete_all_arms": len(complete),
        "n_queries_touched": len(by_q),
        "accuracy": {a: round(acc[a], 4) for a in ARMS},
        "by_kind": {
            k: {"n": sum(1 for q in complete
                         if by_q[q][ARMS[0]].get("kind") == k),
                **{a: round(
                    sum(by_q[q][a]["correct"] for q in complete
                        if by_q[q][ARMS[0]].get("kind") == k)
                    / max(1, sum(1 for q in complete
                                 if by_q[q][ARMS[0]].get("kind") == k)), 4)
                   for a in ARMS}}
            for k in kinds},
        "cell_sent_vs": {},
    }
    for a in ARMS:
        if a == "cell_sent":
            continue
        b = sum(1 for q in complete
                if by_q[q]["cell_sent"]["correct"] and not by_q[q][a]["correct"])
        c = sum(1 for q in complete
                if by_q[q][a]["correct"] and not by_q[q]["cell_sent"]["correct"])
        out["cell_sent_vs"][a] = {
            "delta": round(acc["cell_sent"] - acc[a], 4),
            "ours_only": b, "other_only": c,
            "mcnemar_p": round(float(mcnemar(b, c)), 5)}

    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"wrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
