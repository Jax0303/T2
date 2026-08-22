#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Pair two corpus_dump_vs_cell runs by query_id and McNemar their answer EM.

For comparing one knob between two runs that share everything else -- the case
this exists for is ``--answer-mode direct`` vs ``codegen`` on the arithmetic
population, where the reader is the only thing that changed. The pipeline is
deterministic, so the query_id join is exact rather than approximate.

Only queries present in BOTH files are scored, so a run stopped early cannot
flatter itself on an easier prefix.

  PYTHONPATH=. .venv/bin/python scripts/paired_em_between_runs.py \
      results/A_records.jsonl results/B_records.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from manual_sentence_ceiling import mcnemar  # noqa: E402


def load(path: str) -> dict:
    return {r["query_id"]: r for r in
            (json.loads(l) for l in Path(path).read_text().splitlines() if l.strip())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a_records")
    ap.add_argument("b_records")
    ap.add_argument("--metric", default="answer_em")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    A, B = load(args.a_records), load(args.b_records)
    qids = sorted(set(A) & set(B))
    arms = [k for k in A[qids[0]] if isinstance(A[qids[0]][k], dict)
            and args.metric in A[qids[0]][k]]
    out = {"a": args.a_records, "b": args.b_records, "n": len(qids),
           "n_a_only": len(set(A) - set(B)), "n_b_only": len(set(B) - set(A)),
           "metric": args.metric, "arms": {}}
    print(f"n={len(qids)} paired  (a-only {len(set(A)-set(B))}, "
          f"b-only {len(set(B)-set(A))})")
    for arm in arms:
        a = [A[q][arm][args.metric] for q in qids]
        b = [B[q][arm][args.metric] for q in qids]
        st = mcnemar(a, b)          # only_first = a wins, only_second = b wins
        out["arms"][arm] = {"a": round(sum(a) / len(a), 4),
                            "b": round(sum(b) / len(b), 4), **st}
        print(f"  {arm:6s} a={sum(a)/len(a):.3f}  b={sum(b)/len(b):.3f}  "
              f"b_only={st['only_second']} a_only={st['only_first']} "
              f"p={st['exact_p']:.4g}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
