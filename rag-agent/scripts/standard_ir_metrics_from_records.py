#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Literature-standard IR metrics (Recall@k, MRR, nDCG@k, set-EM@k) over
operand-collision rank records — the "same metrics as other papers" view
(prof's rule: report standard metrics alongside OSC, never OSC alone).

Input: *_records.jsonl from operand_collision_multihiertt.py — rows {scheme,
retriever, query, cell, rank, ...} where rank is the gold cell's 1-based
position in the full corpus ranking (never null in the full-ranking runs;
null-safe anyway: null counts as not retrieved).

Per (scheme, retriever) over the query population:
  * recall@k   — mean per-query fraction of gold operand cells ranked <= k
                 (the graceful metric FT-RAG/MT2Net-style systems report).
  * hit@k      — ANY gold cell ranked <= k (Hit-Rate-style, the most lenient
                 convention in the comparison literature; the hit@k↔set_em@k
                 gap is the "lenient metrics hide incompleteness" exhibit).
  * MRR        — mean reciprocal rank of the FIRST gold cell.
  * nDCG@k     — binary relevance (gold cells rel=1), multi-gold ideal:
                 IDCG@k = sum_{i<=min(|G|,k)} 1/log2(i+1).
  * set-EM@k   — all gold cells <= k (all-or-nothing; identical to the run
                 summary's all_covered@k, recomputed here as a cross-check).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

KS = (10, 20, 50)


def load_ranks(path: str):
    """(scheme, retriever) -> {query: [rank-or-None per gold cell]}."""
    ranks = defaultdict(lambda: defaultdict(list))
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            ranks[(r["scheme"], r["retriever"])][r["query"]].append(r["rank"])
    return ranks


def recall_at_k(gold_ranks: list, k: int) -> float:
    return sum(1 for r in gold_ranks if r is not None and r <= k) / len(gold_ranks)


def hit_at_k(gold_ranks: list, k: int) -> int:
    return int(any(r is not None and r <= k for r in gold_ranks))


def rr(gold_ranks: list) -> float:
    hit = [r for r in gold_ranks if r is not None]
    return 1.0 / min(hit) if hit else 0.0


def ndcg_at_k(gold_ranks: list, k: int) -> float:
    dcg = sum(1.0 / math.log2(r + 1)
              for r in gold_ranks if r is not None and r <= k)
    idcg = sum(1.0 / math.log2(i + 1)
               for i in range(1, min(len(gold_ranks), k) + 1))
    return dcg / idcg


def set_em_at_k(gold_ranks: list, k: int) -> int:
    return int(all(r is not None and r <= k for r in gold_ranks))


def summarize(per_query: dict) -> dict:
    qs = sorted(per_query)
    n = len(qs)
    out = {"n_queries": n,
           "mrr": round(sum(rr(per_query[q]) for q in qs) / n, 4)}
    for k in KS:
        out[f"hit@{k}"] = round(
            sum(hit_at_k(per_query[q], k) for q in qs) / n, 4)
        out[f"recall@{k}"] = round(
            sum(recall_at_k(per_query[q], k) for q in qs) / n, 4)
        out[f"ndcg@{k}"] = round(
            sum(ndcg_at_k(per_query[q], k) for q in qs) / n, 4)
        out[f"set_em@{k}"] = round(
            sum(set_em_at_k(per_query[q], k) for q in qs) / n, 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("--summary", default=None,
                    help="run summary json; cross-check set_em@k == all_covered@k")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ranks = load_ranks(args.records)
    report = {"records": args.records, "by_condition": {}}
    for (scheme, retriever) in sorted(ranks):
        report["by_condition"][f"{scheme}/{retriever}"] = summarize(
            ranks[(scheme, retriever)])

    if args.summary:
        summ = json.load(open(args.summary))
        for key, m in report["by_condition"].items():
            scheme, retriever = key.split("/")
            ref = summ["by_scheme"].get(scheme, {}).get(retriever)
            if not ref:
                continue
            for k in KS:
                want = ref.get(f"all_covered@{k}")
                if want is not None:
                    assert abs(m[f"set_em@{k}"] - want) < 5e-4, (key, k, m, want)
        report["cross_checked_against"] = args.summary

    out = args.out or str(Path(args.records).with_suffix("")).replace(
        "_records", "") + "_standard_ir_metrics.json"
    Path(out).write_text(json.dumps(report, indent=2))
    print(f"[out] {out}")
    hdr = (f"{'condition':<16} {'Hit@10':>6} {'Hit@50':>6} {'R@10':>6} "
           f"{'R@50':>6} {'MRR':>6} {'nDCG@10':>8} {'nDCG@50':>8} "
           f"{'EM@10':>6} {'EM@50':>6}")
    print(hdr)
    for key, m in report["by_condition"].items():
        print(f"{key:<16} {m['hit@10']:>6.3f} {m['hit@50']:>6.3f} "
              f"{m['recall@10']:>6.3f} {m['recall@50']:>6.3f} {m['mrr']:>6.3f} "
              f"{m['ndcg@10']:>8.3f} {m['ndcg@50']:>8.3f} "
              f"{m['set_em@10']:>6.3f} {m['set_em@50']:>6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
