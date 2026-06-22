#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Lab-meeting analysis: where does table-RAG break, by query type?

For each query we measure, at k=5, two things per retrieval mode:
  * recall        — fraction of gold operands whose row is retrieved (partial).
  * all_covered   — 1 iff EVERY gold operand is retrieved. This is what an
    aggregation/arithmetic answer actually needs: miss one operand and the
    computation is wrong. all_covered is therefore the real answer-relevant
    ceiling on retrieval.

Stratified by (a) #operands the query needs and (b) aggregation label, so the
bottleneck (multi-operand / aggregation queries) is visible.
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag_agent.bench import registry
from rag_agent.serialize import serialize_table, S2
from rag_agent.retrieve.operand_retriever import HybridRetriever, retrieve, operand_recall
from rag_agent.retrieve.coverage import assess, apply_fallback

def all_covered(chunks, ops):
    if not ops: return None
    return 1.0 if all(any(c.covers(o.row,o.col) for c in chunks) for o in ops) else 0.0

def nbucket(n): return "1 (lookup)" if n==1 else ("2" if n==2 else "3+")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--bench",default="hitab")
    ap.add_argument("--max-samples",type=int,default=300); ap.add_argument("--k",type=int,default=5)
    a=ap.parse_args()
    q,t=registry.load(a.bench,max_samples=a.max_samples)
    qs=[x for x in q if x.gold_operands]
    rc={}
    def R(tid):
        if tid not in rc: rc[tid]=HybridRetriever(serialize_table(t[tid],S2),embedder=None)
        return rc[tid]
    # accumulators: key -> mode -> [recall...], and all_covered, and counts
    by_n=defaultdict(lambda: defaultdict(list)); by_agg=defaultdict(lambda: defaultdict(list))
    cov_by_n=defaultdict(lambda: defaultdict(list)); cov_by_agg=defaultdict(lambda: defaultdict(list))
    fb_cov=defaultdict(list)  # all_covered with fallback, by n
    for x in qs:
        tab=t[x.gold_table_id]; nb=nbucket(len(x.gold_operands)); ag=(x.aggregation or "none")
        for mode in ("plain","operand","oracle"):
            r=retrieve(x.question,tab,x.gold_operands,mode=mode,k=a.k,retriever=R(x.gold_table_id))
            rec=operand_recall(r.retrieved,x.gold_operands); ac=all_covered(r.retrieved,x.gold_operands)
            by_n[nb][mode].append(rec); by_agg[ag][mode].append(rec)
            cov_by_n[nb][mode].append(ac); cov_by_agg[ag][mode].append(ac)
            if mode=="operand":
                rep=assess(r.operands,r.retrieved,tab); ctx=apply_fallback(r.retrieved,tab,rep,S2)
                fb_cov[nb].append(all_covered(ctx,x.gold_operands))
    def m(xs):
        xs=[v for v in xs if v is not None]; return round(sum(xs)/len(xs),3) if xs else float("nan")
    print(f"\n===== {a.bench}  (n={len(qs)} operand-bearing, k={a.k}, BM25) =====")
    print("\n## A. ALL-operands-covered (answer needs this) — by #operands")
    print(f"{'#operands':14s} {'n':>4} {'plain':>7} {'operand':>8} {'+fallbk':>8} {'oracle':>7}")
    order=["1 (lookup)","2","3+"]
    for nb in order:
        if nb not in cov_by_n: continue
        n=len(cov_by_n[nb]['plain'])
        print(f"{nb:14s} {n:>4} {m(cov_by_n[nb]['plain']):>7} {m(cov_by_n[nb]['operand']):>8} {m(fb_cov[nb]):>8} {m(cov_by_n[nb]['oracle']):>7}")
    print("\n## B. Partial operand_recall@5 — by #operands")
    print(f"{'#operands':14s} {'n':>4} {'plain':>7} {'operand':>8} {'oracle':>7}")
    for nb in order:
        if nb not in by_n: continue
        n=len(by_n[nb]['plain'])
        print(f"{nb:14s} {n:>4} {m(by_n[nb]['plain']):>7} {m(by_n[nb]['operand']):>8} {m(by_n[nb]['oracle']):>7}")
    print("\n## C. ALL-covered by aggregation type")
    print(f"{'aggregation':22s} {'n':>4} {'plain':>7} {'operand':>8} {'oracle':>7}")
    for ag in sorted(cov_by_agg, key=lambda k:-len(cov_by_agg[k]['plain'])):
        n=len(cov_by_agg[ag]['plain'])
        if n<5: continue
        print(f"{ag:22s} {n:>4} {m(cov_by_agg[ag]['plain']):>7} {m(cov_by_agg[ag]['operand']):>8} {m(cov_by_agg[ag]['oracle']):>7}")
if __name__=="__main__": main()
