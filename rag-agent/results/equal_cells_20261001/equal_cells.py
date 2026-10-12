"""원고 [표 4-7] 같은 수의 셀을 전달할 때의 재현율(200문항) + 4.3.1 방어 수치. 기존 순위 기록만 사용, 새 실행 없음.

  문항마다 비교 방법의 상위 k개 단위가 전달한 (중복 없는) 셀 수 c 를 세고, 본 방법 순위 목록의 상위 c개 셀 재현율을 구한다.
  본 방법 목록은 상위 100개 셀까지 저장(retrieval_r100) — c 가 목록보다 길면 목록 전체.
  반올림은 정확한 분수 기준 사사오입(표 4-5 의 두 칸 73.5·76.7 은 이중 반올림 오류였다: 정확히 73.448%, 76.75%).
  실행: .venv/bin/python results/equal_cells_20261001/equal_cells.py   (저장소 rag-agent 에서)
"""
import json, sys
from fractions import Fraction as F
from pathlib import Path
sys.path.insert(0, "results/mh_only_20260928/final")
from rpm import units_of

B = "results/mh_only_20260928/baselines/"
ids = [json.loads(l)["query_id"] for l in open(B + "ids200.jsonl") if l.strip()]


def load(p):
    d = {}
    for l in open(p):
        r = json.loads(l); u = units_of(r)
        if r["query_id"] not in d and u is not None and r.get("gold_ids"):
            d[r["query_id"]] = (set(r["gold_ids"]), u)
    return d


def pct(fr):
    x = fr * 1000; n = int(x)
    return (n + (1 if x - n >= F(1, 2) else 0)) / 10


def top(D, q, k):
    u = D[q][1][:k]
    return set().union(*u) if u else set()


O = load("results/mh_only_20260928/retrieval_r100/mh_train_s3c_records.jsonl")
M = {"LangChain 청크": load(B + "langchain_chunk/langchain_chunk_records.jsonl"),
     "TableRAG(Yu)": load(B + "tablerag_yu/tablerag_yu_records.jsonl"), "RAPTOR": load(B + "raptor/raptor_records.jsonl")}
out = {"rows": []}
for m, ks in (("LangChain 청크", (1, 2, 4)), ("TableRAG(Yu)", (1, 2, 5)), ("RAPTOR", (1, 5, 20))):
    for k in ks:
        c = [len(top(M[m], q, k)) for q in ids]
        r = sum((F(len(M[m][q][0] & top(M[m], q, k)), len(M[m][q][0])) for q in ids), F(0)) / len(ids)
        o = sum((F(len(O[q][0] & top(O, q, n)), len(O[q][0])) for q, n in zip(ids, c)), F(0)) / len(ids)
        out["rows"].append({"method": m, "k": k, "cells_mean": round(sum(c) / len(c), 1), "recall": pct(r), "ours_same_cells": pct(o),
                            "ours_capped_at_list": sum(n > len(O[q][1]) for q, n in zip(ids, c))})
L = M["LangChain 청크"]
c1 = [len(top(L, q, 1)) for q in ids]
out["langchain_top1_has_gold"] = pct(F(sum(bool(top(L, q, 1) & L[q][0]) for q in ids), len(ids)))
out["ours_same_cells_has_gold"] = pct(F(sum(bool(top(O, q, n) & O[q][0]) for q, n in zip(ids, c1)), len(ids)))
out["ours_R@1_max"] = pct(sum((F(1, len(O[q][0])) for q in ids), F(0)) / len(ids))
out["ours_cellP@20_max"] = pct(sum((F(min(len(O[q][0]), 20), 20) for q in ids), F(0)) / len(ids))
out["gold_per_q_200"] = float(F(sum(len(O[q][0]) for q in ids), len(ids)))
out["gold_per_q_2885"] = round(sum(len(v[0]) for v in O.values()) / len(O), 2)
out["table_4_5_recheck"] = {"ours_R@5": pct(sum((F(len(O[q][0] & top(O, q, 5)), len(O[q][0])) for q in ids), F(0)) / 200),
                            "langchain_R@1": pct(sum((F(len(L[q][0] & top(L, q, 1)), len(L[q][0])) for q in ids), F(0)) / 200)}
Path(__file__).with_suffix(".json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))
