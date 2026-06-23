"""Re-score an existing answer_accuracy.json with new tolerance modes.

No LLM re-runs. Reads the per-row predictions already in the file and applies
the HiTab-style _match function under exact / relaxed-0.5pct / relaxed-1pct.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sidecar_verifier.eval.answer_accuracy import _TOLERANCES, _match_all_modes


def rescore(rows):
    n = len(rows)
    correct_by_mode = {m: {"vector": 0, "verified": 0, "oracle": 0} for m in _TOLERANCES}
    match_type = {b: Counter() for b in ("vector", "verified", "oracle")}
    retrieved_gold_vector = 0
    retrieved_gold_verified = 0

    for r in rows:
        gold_ans = r.get("gold_answer")
        gold = r.get("gold_table")
        retrieved_gold_vector += int(r.get("vector_top") == gold)
        retrieved_gold_verified += int(r.get("verified_top") == gold)

        for bucket, ans_key in (
            ("vector", "vector_answer"),
            ("verified", "verified_answer"),
            ("oracle", "oracle_answer"),
        ):
            pred = r.get(ans_key)
            if pred is None:
                continue
            results = _match_all_modes(pred, gold_ans)
            for mode, (ok, _) in results.items():
                correct_by_mode[mode][bucket] += int(ok)
            match_type[bucket][results["exact"][1]] += 1

    return {
        "n": n,
        "retrieval_R@1_vector": retrieved_gold_vector / n if n else 0.0,
        "retrieval_R@1_verified": retrieved_gold_verified / n if n else 0.0,
        "answer_acc": {
            mode: {b: c / n if n else 0.0 for b, c in buckets.items()}
            for mode, buckets in correct_by_mode.items()
        },
        "match_type_distribution_exact": {b: dict(c) for b, c in match_type.items()},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="results/answer_accuracy.json")
    p.add_argument("--out", default="results/answer_accuracy_rescored.json")
    args = p.parse_args()

    data = json.loads(Path(args.inp).read_text())
    rows = data.get("rows", [])
    if not rows:
        raise SystemExit(f"No rows found in {args.inp}")

    summary = rescore(rows)

    print(f"queries: {summary['n']}")
    print(f"retrieval R@1: vector={summary['retrieval_R@1_vector']:.3f}  "
          f"verified={summary['retrieval_R@1_verified']:.3f}")
    print()
    print(f"{'mode':<16} {'vector':>8} {'verified':>10} {'oracle':>8}")
    for mode in _TOLERANCES:
        acc = summary["answer_acc"][mode]
        print(f"{mode:<16} {acc['vector']:>8.3f} {acc['verified']:>10.3f} {acc['oracle']:>8.3f}")
    print()
    print("match_type distribution (exact mode):")
    for bucket, dist in summary["match_type_distribution_exact"].items():
        if dist:
            print(f"  {bucket}: {dist}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
