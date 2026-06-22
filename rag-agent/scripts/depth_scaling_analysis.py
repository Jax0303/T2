#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Depth-scaling analysis: does the operand bottleneck grow with header depth?

Hypothesis: as hierarchical header depth increases, (1) decomposition gets
harder, (2) all-operands-covered drops for plain retrieval, (3) the gain of
operand-targeting grows. Stratifies existing metrics by table header depth,
within HiTab and across the flat→deep benchmark ladder (WikiSQL→FinQA→HiTab).
"""
from __future__ import annotations
import argparse, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag_agent.bench import registry
from rag_agent.serialize import serialize_table, S2
from rag_agent.retrieve.operand_retriever import HybridRetriever, retrieve, operand_recall
from rag_agent.query.operand_decomposer import header_path_match_accuracy

def table_depth(t):
    paths = list(t.top_paths) + list(t.left_paths)
    return max((len([s for s in p if s]) for p in paths), default=0)

def all_covered(chunks, ops):
    if not ops: return None
    return 1.0 if all(any(c.covers(o.row,o.col) for c in chunks) for o in ops) else 0.0

def m(xs):
    xs=[v for v in xs if v is not None]; return round(sum(xs)/len(xs),3) if xs else float('nan')

def analyze(bench, max_samples, k=5):
    q,t=registry.load(bench,max_samples=max_samples)
    qs=[x for x in q if x.gold_operands]
    rc={}
    def R(tid):
        if tid not in rc: rc[tid]=HybridRetriever(serialize_table(t[tid],S2),embedder=None)
        return rc[tid]
    buckets=defaultdict(lambda: defaultdict(list))
    for x in qs:
        d=table_depth(t[x.gold_table_id])
        b=buckets[d]
        b['ceiling'].append(header_path_match_accuracy(x.question,t[x.gold_table_id],x.gold_operands,matcher='fuzzy'))
        for mode in ('plain','operand','oracle'):
            r=retrieve(x.question,t[x.gold_table_id],x.gold_operands,mode=mode,k=k,retriever=R(x.gold_table_id))
            b[f'cov_{mode}'].append(all_covered(r.retrieved,x.gold_operands))
    return buckets

def print_bench(bench, buckets):
    print(f"\n===== {bench} — by header depth (all-operands-covered@5, BM25) =====")
    print(f"{'depth':>5} {'n':>4} {'ceiling':>8} {'plain':>7} {'operand':>8} {'oracle':>7} {'gain(op-pl)':>11} {'headroom(or-op)':>15}")
    for d in sorted(buckets):
        b=buckets[d]; n=len(b['cov_plain'])
        if n<4: continue
        pl,op,orc=m(b['cov_plain']),m(b['cov_operand']),m(b['cov_oracle'])
        print(f"{d:>5} {n:>4} {m(b['ceiling']):>8} {pl:>7} {op:>8} {orc:>7} {round(op-pl,3):>11} {round(orc-op,3):>15}")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--max-samples",type=int,default=500)
    a=ap.parse_args()
    for bench in ["wikisql","finqa","hitab"]:
        print_bench(bench, analyze(bench, a.max_samples))
