#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Diagnostic — why does the oracle arm read OSC=1.0 but accuracy_em=0.0373?

Evidence only. Writes no result file, changes no pipeline.

The oracle arm of `results/e10_cross.json` / `results/e9_tight.json` reports
OSC 1.0, numeric-match .6087 and "em" .0373 on HiTab dev arithmetic m>=2
(n=161, solver groq:llama-3.1-8b-instant, codegen). Those runs stored only
verdict flags per query, not the solver's raw text, so the raw output of THAT
run is unrecoverable. This script therefore separates what can still be
established:

  1. which function each reported number actually came from (static);
  2. how the three scorers disagree on REAL predictions from this repo, taken
     from a records file that did persist per-query predictions on the same
     population and the same query ids (`answer_accuracy_injection_*_records`);
  3. what the gold answers themselves look like after each normalisation.

Run: PYTHONPATH=. .venv/bin/python scripts/diag_oracle_em.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.bench.hitab import load_queries
from rag_agent.eval.metrics import (_hmt_process, _wtq_normalize, exact_match,
                                    hitab_exact_match, numeric_match)

ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--records",
                    default="results/answer_accuracy_injection_gpt4o_records.jsonl")
    ap.add_argument("--n-sample", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    queries, _ = load_queries(args.data_dir, "dev")
    pop = [q for q in queries
           if (q.aggregation or "none") in ARITH
           and len({(o.row, o.col) for o in q.gold_operands}) >= 2]
    by_id = {q.query_id: q for q in pop}
    print(f"(a) population = HiTab dev, arithmetic m>=2, n={len(pop)}")
    print(f"    e10_cross/e9_tight oracle: osc=1.0 nm=.6087 em=.0373 "
          f"-> {round(0.0373 * len(pop))} correct / {len(pop) - round(0.0373 * len(pop))} failed")

    recs = [json.loads(l) for l in open(args.records)]
    recs = [r for r in recs if r["qid"] in by_id]
    print(f"\n(b) predictions source: {args.records} ({len(recs)} of the same qids)")
    rng = random.Random(args.seed)
    sample = rng.sample(recs, min(args.n_sample, len(recs)))

    rows = []
    for r in sample:
        q = by_id[r["qid"]]
        gold, pred = q.answer, r.get("pred_treat")
        rows.append({
            "qid": r["qid"][:8],
            "gold_raw": repr(gold),
            "pred_raw": repr(pred),
            "gold_norm": repr(_hmt_process(gold)),
            "pred_norm": repr(_hmt_process(pred)),
            "official_hmt": hitab_exact_match(pred, gold),
            "repo_exact_match": exact_match(pred, gold),
            "numeric_match_2pct": numeric_match(pred, gold),
        })
    w = max(len(r["gold_raw"]) for r in rows)
    print(f"\n{'qid':<10}{'gold_raw':<{w+2}}{'pred_raw':<24}"
          f"{'gold_norm':<14}{'pred_norm':<24}{'hmt':>5}{'exact':>7}{'nm':>5}")
    for r in rows:
        print(f"{r['qid']:<10}{r['gold_raw']:<{w+2}}{r['pred_raw'][:23]:<24}"
              f"{r['gold_norm'][:13]:<14}{r['pred_norm'][:23]:<24}"
              f"{str(r['official_hmt']):>5}{str(r['repo_exact_match']):>7}"
              f"{str(r['numeric_match_2pct']):>5}")

    # (c) classification of the sampled 20
    def bucket(r):
        g, p = r["gold_norm"], r["pred_norm"]
        if p == "None":
            return "parse/solver produced nothing"
        if r["numeric_match_2pct"] and not r["official_hmt"]:
            return "number format / rounding (numerically right, strict-EM wrong)"
        if r["official_hmt"] and not r["repo_exact_match"]:
            return "string-equality artifact (official passes, repo exact_match fails)"
        if r["official_hmt"]:
            return "correct under every scorer"
        return "solver actually wrong"
    print("\n(c) classification of the 20:")
    for k, v in Counter(bucket(r) for r in rows).most_common():
        print(f"    {v:>3}  {k}")

    # (d) all three scorers over the whole records file
    print(f"\n(d) all {len(recs)} records, per scorer:")
    for name, fn in (("official hitab_exact_match", hitab_exact_match),
                     ("repo exact_match (what e7/e10/e9 call 'em')", exact_match),
                     ("numeric_match (+-2%, internal only)", numeric_match)):
        n = sum(1 for r in recs
                if fn(r.get("pred_treat"), by_id[r["qid"]].answer))
        print(f"    {n:>4}/{len(recs)} = {n/len(recs):.4f}  {name}")

    # (e) gold distribution after normalisation
    golds = [q.answer for q in pop]
    proc = [_hmt_process(g) for g in golds]
    print(f"\n(e) gold answers of the population (n={len(golds)}):")
    print(f"    parse to float : {sum(1 for p in proc if isinstance(p, float))}")
    print(f"    stay string    : {sum(1 for p in proc if isinstance(p, str))}")
    print(f"    list-valued    : {sum(1 for p in proc if isinstance(p, list))}")
    print(f"    empty/None     : {sum(1 for p in proc if p is None or p == '' or p == [])}")
    floats = [p for p in proc if isinstance(p, float)]
    dec = Counter(len(str(f).split('.')[1]) if '.' in str(f) else 0 for f in floats)
    print(f"    decimal places of float golds: {dict(sorted(dec.items()))}")
    print(f"    unique float golds: {len(set(floats))} of {len(floats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
