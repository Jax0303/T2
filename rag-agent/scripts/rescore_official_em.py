#!/usr/bin/env python3
"""Rescore an existing verbalize_answer_eval run under HiTab's OFFICIAL metric.

`verbalize_answer_eval.py` already writes `hmt_em` for every new run. This script
is for the *legacy* dumps written before that column existed (and as a general
"recompute the official number from stored predictions" tool): it reads the
per-arm `pred` and per-record `gold` already saved in a results file and applies
`hitab_exact_match` — no LLM calls, no HiTab data download, so a solver's daily
token budget is never spent twice just to change the scorer.

Prints official EM per arm + a paired bootstrap Δ(original_sent − each RAG), and
writes `<in>.official_em.json` next to the input.

Usage: python scripts/rescore_official_em.py results/verbalize_answer_*.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_agent.eval.metrics import hitab_exact_match  # noqa: E402

ARMS = ("oracle", "original_sent", "rag_1t1c", "rag_rowchunk")
SEED = 42
BOOT = 2000


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _paired_delta(a, b, rng):
    n = len(a)
    diffs = []
    for _ in range(BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(_mean([a[i] for i in idx]) - _mean([b[i] for i in idx]))
    diffs.sort()
    return _mean(a) - _mean(b), diffs[int(0.025 * BOOT)], diffs[int(0.975 * BOOT)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    args = ap.parse_args()
    src = Path(args.infile)
    d = json.loads(src.read_text())
    recs = d["records"]
    arms = [a for a in ARMS if any(a in r["arms"] for r in recs)]

    off = {a: [] for a in arms}
    for r in recs:
        for a in arms:
            if a in r["arms"]:
                off[a].append(1 if hitab_exact_match(r["arms"][a]["pred"], r["gold"]) else 0)

    reader = d.get("config", {}).get("reader", "?")
    print(f"n={len(recs)}  solver={reader}  (official HiTab exact_match)\n")
    summary = {}
    for a in arms:
        summary[a] = {"hmt_exact_match": round(_mean(off[a]), 4), "n": len(off[a])}
        print(f"  {a:16s} hmtEM={summary[a]['hmt_exact_match']:.3f}  (n={len(off[a])})")

    rng = random.Random(SEED)
    print()
    for base in ("rag_1t1c", "rag_rowchunk"):
        if base in off and "original_sent" in off:
            dlt, lo, hi = _paired_delta(off["original_sent"], off[base], rng)
            summary[f"delta_original_vs_{base}"] = {
                "delta": round(dlt, 4), "ci95": [round(lo, 4), round(hi, 4)]}
            print(f"  Δ original_sent − {base:13s} = {dlt:+.3f}  CI95[{lo:+.3f},{hi:+.3f}]")

    out = src.with_suffix(".official_em.json")
    out.write_text(json.dumps({
        "rescored_from": src.name,
        "reader": reader,
        "primary_metric": "hmt_exact_match (HiTab official scorer)",
        "summary": summary,
    }, indent=2, ensure_ascii=False))
    print("\nsaved →", out)


if __name__ == "__main__":
    main()
