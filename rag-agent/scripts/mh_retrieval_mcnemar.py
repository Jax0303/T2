#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""MultiHiertt 검색 정확도(문서 안, 예산 20) — 본 방법 대 각 방법, 그룹별 정확 McNemar.

  .venv/bin/python scripts/mh_retrieval_mcnemar.py > results/mh_arms/retrieval_mcnemar_doc_b20.txt
"""
import json
from scipy.stats import binomtest
ARMS = ["chunk", "rowcol", "trag_hetero", "tablerag_path", "tablerag_leaf", "randrow"]
LAYERS = ["lookup_m1", "lookup_m2+", "arith_m1", "arith_m2+"]
def load(arm):
    return {r["query_id"]: r for r in map(json.loads, open(f"results/mh_arms/mh_train_{arm}_hv1_none_doc_records.jsonl")) if "doc" in r}
ours = load("cell")
for arm in ARMS:
    other = load(arm); assert set(other) == set(ours)
    cells = []
    for L in LAYERS + ["ALL"]:
        q = [k for k, r in ours.items() if L == "ALL" or r["layer"] == L]
        b = sum(1 for k in q if ours[k]["doc"]["correct"] and not other[k]["doc"]["correct"])
        c = sum(1 for k in q if not ours[k]["doc"]["correct"] and other[k]["doc"]["correct"])
        p = binomtest(b, b + c).pvalue if b + c else 1.0
        acc_o = sum(ours[k]["doc"]["correct"] for k in q) / len(q); acc_x = sum(other[k]["doc"]["correct"] for k in q) / len(q)
        cells.append(f"{L}: {acc_o:.4f} vs {acc_x:.4f} {b}:{c} p={p:.2g} (n={len(q)})")
    print(arm, " | ".join(cells))
