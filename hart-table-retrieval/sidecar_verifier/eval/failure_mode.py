"""Classify oracle-failure rows into reader-side error categories.

Rule-based, no LLM. Run on an answer_accuracy.json that has per-row predictions
in ``rows[*].oracle_answer`` and ``rows[*].gold_answer``.

Categories
----------
abstain                : LLM produced no numeric answer (e.g. "N/A", "I don't know").
wrong_cell             : Numeric returned, gold also numeric, but magnitudes differ
                         by >10x → likely picked an unrelated cell (different column).
off_by_small_factor    : Numeric mismatch with similar magnitude (≤10x ratio).
                         Suggests near-miss, unit/scale confusion.
aggregation_failure    : Gold is multiple numbers (list len > 1) — single-value
                         answer cannot match. Reader skipped the aggregation.
text_mismatch          : Both sides are textual, strings differ.
correct                : Either exact or contains/relaxed match passes.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sidecar_verifier.eval.answer_accuracy import _match, _to_nums


def classify(pred, gold) -> str:
    pred = pred if pred is not None else ""
    pred_nums = _to_nums(pred)
    gold_nums = _to_nums(gold)

    # Correct overrides all categories.
    ok, _mtype = _match(pred, gold, tolerance=0.01)
    if ok:
        return "correct"

    # No numeric pred and gold has numbers -> abstention.
    if gold_nums and not pred_nums:
        return "abstain"

    # Aggregation failure: gold has multiple distinct numbers, pred has only one
    # or fewer than gold.
    if len(gold_nums) > 1 and len(pred_nums) < len(gold_nums):
        return "aggregation_failure"

    if gold_nums and pred_nums:
        # Find closest pred to first gold; assess magnitude ratio.
        g = gold_nums[0]
        if g == 0:
            return "off_by_small_factor"
        p = min(pred_nums, key=lambda x: abs(x - g))
        denom = max(abs(g), abs(p), 1e-9)
        ratio = max(abs(g), abs(p)) / min(abs(g) if abs(g) > 1e-9 else 1e-9,
                                          abs(p) if abs(p) > 1e-9 else 1e-9)
        if ratio > 10:
            return "wrong_cell"
        return "off_by_small_factor"

    # Both sides textual.
    return "text_mismatch"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="results/answer_accuracy.json")
    p.add_argument("--bucket", default="oracle",
                   choices=["oracle", "vector", "verified"],
                   help="Which prediction column to classify.")
    p.add_argument("--out", default="results/failure_modes.json")
    args = p.parse_args()

    data = json.loads(Path(args.inp).read_text())
    rows = data.get("rows", [])
    if not rows:
        raise SystemExit(f"No rows in {args.inp}")

    counts = Counter()
    examples = {}
    classified = []
    for r in rows:
        pred = r.get(f"{args.bucket}_answer")
        gold = r.get("gold_answer")
        if pred is None:
            continue
        cat = classify(pred, gold)
        counts[cat] += 1
        examples.setdefault(cat, []).append({
            "query": r.get("query", "")[:120],
            "gold_answer": gold,
            f"{args.bucket}_answer": pred,
        })
        classified.append({
            "query": r.get("query", ""),
            "gold_table": r.get("gold_table"),
            "gold_answer": gold,
            "pred": pred,
            "category": cat,
        })

    n = sum(counts.values())
    print(f"\nFailure-mode distribution for bucket='{args.bucket}'  (n={n})")
    print(f"  {'category':<22} {'count':>5}  {'pct':>6}")
    for cat in ["correct", "abstain", "wrong_cell", "off_by_small_factor",
                "aggregation_failure", "text_mismatch"]:
        c = counts.get(cat, 0)
        print(f"  {cat:<22} {c:>5}  {c/n*100:>5.1f}%")

    print(f"\nExamples per category (first 2 each):")
    for cat in counts:
        print(f"\n  [{cat}]")
        for ex in examples[cat][:2]:
            print(f"    q={ex['query']!r}")
            print(f"      gold={ex['gold_answer']}  pred={ex[f'{args.bucket}_answer']!r}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "bucket": args.bucket,
        "n": n,
        "counts": dict(counts),
        "rows": classified,
    }, indent=2))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
