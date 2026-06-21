#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Diagnosis on SIMPLE hierarchical tables (header-path depth 2).

Controlled first step before depth expansion. Answers:
  1. processing by query type: lookup (1 operand) vs calc (>=2 operands)
  2. WHERE the bottleneck is — split each uncovered gold operand into:
       retrieval-fail  : not covered even in ORACLE (gold path searched, still missed)
       decomposition-fail: covered by ORACLE but missed by realistic operand mode
  3. why answers fail (calc needs ALL operands).
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag_agent.bench import registry
from rag_agent.serialize import serialize_table, S2
from rag_agent.retrieve.operand_retriever import HybridRetriever, retrieve
from rag_agent.query.operand_decomposer import header_path_match_accuracy

K=5
def covered(chunks, op): return any(c.covers(op.row,op.col) for c in chunks)
def allcov(chunks, ops): return None if not ops else (1.0 if all(covered(chunks,o) for o in ops) else 0.0)
def m(xs):
    xs=[v for v in xs if v is not None]; return round(sum(xs)/len(xs),3) if xs else float('nan')

q,t=registry.load("hitab",max_samples=800)
# simple hierarchical = max gold-operand header-path depth == 2
qs=[x for x in q if x.gold_operands and max(len(o.header_path) for o in x.gold_operands)==2]
print(f"simple-hierarchical (operand depth==2) queries: {len(qs)}")

rc={}
def R(tid):
    if tid not in rc: rc[tid]=HybridRetriever(serialize_table(t[tid],S2),embedder=None)
    return rc[tid]

groups={"lookup (1 operand)":[x for x in qs if len(x.gold_operands)==1],
        "calc (>=2 operand)":[x for x in qs if len(x.gold_operands)>=2]}

print("\n## 1+2. all-operands-covered@5 + ceiling, by query type")
print(f"{'type':22s} {'n':>4} {'ceiling':>8} {'plain':>7} {'operand':>8} {'oracle':>7}")
for name,g in groups.items():
    if not g: continue
    ceil=[header_path_match_accuracy(x.question,t[x.gold_table_id],x.gold_operands,matcher='fuzzy') for x in g]
    res={}
    for mode in ('plain','operand','oracle'):
        res[mode]=[allcov(retrieve(x.question,t[x.gold_table_id],x.gold_operands,mode=mode,k=K,retriever=R(x.gold_table_id)).retrieved, x.gold_operands) for x in g]
    print(f"{name:22s} {len(g):>4} {m(ceil):>8} {m(res['plain']):>7} {m(res['operand']):>8} {m(res['oracle']):>7}")

# failure attribution over ALL gold operands (operand mode vs oracle)
print("\n## 3. failure attribution — per gold operand (operand mode, k=5)")
n_ops=0; covered_op=0; dec_fail=0; ret_fail=0
for x in qs:
    tab=t[x.gold_table_id]
    rop=retrieve(x.question,tab,x.gold_operands,mode="operand",k=K,retriever=R(x.gold_table_id)).retrieved
    ror=retrieve(x.question,tab,x.gold_operands,mode="oracle",k=K,retriever=R(x.gold_table_id)).retrieved
    for o in x.gold_operands:
        n_ops+=1
        if covered(rop,o): covered_op+=1
        elif covered(ror,o): dec_fail+=1     # gold path would retrieve it, decomposition missed
        else: ret_fail+=1                    # even gold path can't retrieve it
print(f"total gold operands: {n_ops}")
print(f"  covered (ok):              {covered_op:>4} ({covered_op/n_ops:.1%})")
print(f"  DECOMPOSITION failure:     {dec_fail:>4} ({dec_fail/n_ops:.1%})  <- oracle gets it, our decomposition doesn't")
print(f"  RETRIEVAL failure:         {ret_fail:>4} ({ret_fail/n_ops:.1%})  <- even gold path not retrieved")
