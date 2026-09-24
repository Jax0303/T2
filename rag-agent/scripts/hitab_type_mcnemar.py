#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""HiTab 유형별 검색 정확도(질문의 표 안, 예산 20) — 본 방법(sleaf) 대 각 방법, 정확 McNemar.

  .venv/bin/python scripts/hitab_type_mcnemar.py > results/retrieval_accuracy/hitab_type_mcnemar_gold.txt
"""
import json
from scipy.stats import binomtest

ARMS = {"ours": "t_sleaf_gold", "chunk": "t_chunk_s3c_gold_v2", "trag_hetero": "t_trag_hetero_gold_v2",
        "rowcol": "t_rowcol_s3c_gold_v2", "randrow": "t_randrow_s3c_gold",
        "tablerag_path": "t_tablerag_path_v2_gold", "tablerag_leaf": "t_tablerag_leaf_v2_gold"}


def load(stem):
    out = {}
    for line in open(f"results/retrieval_accuracy/{stem}_type_accuracy.jsonl"):
        r = json.loads(line)
        if r["retrieval_success"] is not None:
            out[r["query_id"]] = (r["query_type"], r["retrieval_success"])
    return out


ours = load(ARMS["ours"])
for name, stem in ARMS.items():
    if name == "ours":
        continue
    other = load(stem)
    assert set(other) == set(ours)
    cells = []
    for t in ["single_cell", "multi_cell", "arithmetic"]:
        q = [k for k, v in ours.items() if v[0] == t]
        b = sum(ours[k][1] and not other[k][1] for k in q)
        c = sum(other[k][1] and not ours[k][1] for k in q)
        p = binomtest(b, b + c).pvalue if b + c else 1.0
        cells.append(f"{t}: {sum(ours[k][1] for k in q) / len(q):.4f} vs {sum(other[k][1] for k in q) / len(q):.4f} "
                     f"{b}:{c} p={p:.2g} (n={len(q)})")
    print(name, " | ".join(cells))
