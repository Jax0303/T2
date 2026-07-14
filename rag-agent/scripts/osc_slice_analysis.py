#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Slice OSC by operand-set size over operand-collision rank records.

Reviewer question this answers: is the serialization gain concentrated in
multi-cell aggregation queries, and does it GROW with operand-set size? If
yes, that is the paper's Figure-1 shape (failure (3): set completeness decays
with scope, flat decays fastest).

Input: *_records.jsonl rows {scheme, retriever, query, cell, rank, colliding,
total_like[, scope_size]} from operand_collision_multihiertt.py or
operand_collision_rerank.py. scope_size is derived as the number of gold
records per query when absent (each query's records enumerate its gold set).

Output per (retriever, scope-bin): OSC set_recall@k per scheme, the flat->S3
delta, paired sign test within the bin, and the query-count / cardinality
distribution of the population (single-cell queries, if present, form their
own "1" bin — the MultiHiertt collision population is >=2 by construction).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.eval.operand_set import (bin_scope, osc_at_k_summary,
                                        paired_set_recall_flip)

KS = (10, 20, 50)
BIN_ORDER = ["1", "2", "3-4", "5-8", "9+"]


def load_ranks(path: str):
    """(scheme, retriever) -> {query: {cell: rank}}; plus query -> scope bin."""
    ranks = defaultdict(lambda: defaultdict(dict))
    scope = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            ranks[(r["scheme"], r["retriever"])][r["query"]][r["cell"]] = r["rank"]
            if "scope_size" in r:
                scope[r["query"]] = r["scope_size"]
    if not scope:  # derive: gold-set size = records per query (any one condition)
        (_, per_q), = [next(iter(ranks.items()))]
        scope = {q: len(cells) for q, cells in per_q.items()}
    return ranks, {q: bin_scope(m) for q, m in scope.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("--baseline-scheme", default="flat")
    ap.add_argument("--treat-schemes", default="S2,S3")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ranks, qbin = load_ranks(args.records)
    schemes = sorted({s for s, _ in ranks})
    retrievers = sorted({r for _, r in ranks})
    treats = [s for s in args.treat_schemes.split(",") if s in schemes]

    dist = defaultdict(int)
    for b in qbin.values():
        dist[b] += 1
    report = {"records": args.records,
              "scope_distribution": {b: dist[b] for b in BIN_ORDER if dist[b]},
              "by_retriever": {}}

    for ret in retrievers:
        block = {}
        for b in BIN_ORDER:
            qids = sorted(q for q, bb in qbin.items() if bb == b)
            if not qids:
                continue
            entry = {"n_queries": len(qids), "set_recall": {}, "delta_vs_flat": {},
                     "flip_test": {}}
            per_scheme = {}
            for sch in schemes:
                per_q = ranks.get((sch, ret))
                if not per_q or any(q not in per_q for q in qids):
                    continue
                rs = [per_q[q] for q in qids]
                per_scheme[sch] = rs
                s = osc_at_k_summary(rs, ks=KS)
                entry["set_recall"][sch] = {f"@{k}": s[f"set_recall@{k}"] for k in KS}
            base = per_scheme.get(args.baseline_scheme)
            for sch in treats:
                if base is None or sch not in per_scheme:
                    continue
                entry["delta_vs_flat"][sch] = {
                    f"@{k}": round(entry["set_recall"][sch][f"@{k}"]
                                   - entry["set_recall"][args.baseline_scheme][f"@{k}"], 4)
                    for k in KS}
                entry["flip_test"][sch] = {
                    f"@{k}": paired_set_recall_flip(base, per_scheme[sch], k)
                    for k in KS}
            block[b] = entry
        report["by_retriever"][ret] = block

    out = args.out or str(Path(args.records).with_suffix("")).replace(
        "_records", "") + "_scope_slices.json"
    Path(out).write_text(json.dumps(report, indent=2))
    print(f"[out] {out}\nscope distribution: {report['scope_distribution']}")
    for ret, block in report["by_retriever"].items():
        for b, entry in block.items():
            deltas = {sch: d.get("@50") for sch, d in entry["delta_vs_flat"].items()}
            ps = {sch: (entry["flip_test"][sch]["@50"] or {}).get("p_two_sided")
                  for sch in entry["flip_test"]}
            print(f"  {ret:<18} scope={b:<4} n={entry['n_queries']:<4} "
                  f"delta@50={deltas} p@50={ps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
