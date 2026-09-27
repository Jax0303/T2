#!/usr/bin/env python3
"""본 방법(s3c, 최종 규칙) 대 table_description 셀 문장 — 문서 안 검색, 예산 20, 같은 2,885건. 2026-09-28.

b = 본 방법만 맞힘, c = table_description 만 맞힘. 정확 McNemar(양측 이항). 그룹 4개는 Holm 보정도 싣는다.

  PYTHONPATH=. .venv/bin/python results/table_description_20260928/compare.py
"""
import json
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
OURS = ROOT / "results/rerun_20260926/mh/mh_train_s3c_records.jsonl"
DESC = Path(__file__).resolve().parent / "mh_train_s3c_tdesc_records.jsonl"


def load(p):
    return {r["query_id"]: r for r in map(json.loads, p.open()) if "doc" in r}


ours, desc = load(OURS), load(DESC)
assert ours.keys() == desc.keys() and len(ours) == 2885, "채점 질의 집합이 다르다"
assert all(ours[q]["layer"] == desc[q]["layer"] for q in ours), "그룹이 다르다"
out = {}
for g in ("ALL", "lookup_m1", "lookup_m2+", "arith_m1", "arith_m2+"):
    qs = [q for q in ours if g == "ALL" or ours[q]["layer"] == g]
    a = [ours[q]["doc"]["correct"] for q in qs]
    d = [desc[q]["doc"]["correct"] for q in qs]
    b = sum(x and not y for x, y in zip(a, d))
    c = sum(y and not x for x, y in zip(a, d))
    out[g] = {"n": len(qs), "ours": sum(a), "ours_acc": round(sum(a) / len(qs), 4),
              "table_description": sum(d), "table_description_acc": round(sum(d) / len(qs), 4),
              "b_ours_only": b, "c_desc_only": c,
              "p_exact_mcnemar": binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0}
    print(g, out[g])
# 그룹 4개를 한 묶음으로 Holm 보정(원고 5.4.1절). 전체(ALL)는 묶음에 넣지 않는다.
G4 = sorted(("lookup_m1", "lookup_m2+", "arith_m1", "arith_m2+"), key=lambda g: out[g]["p_exact_mcnemar"])
run = 0.0
for k, g in enumerate(G4):
    run = max(run, min(1.0, (len(G4) - k) * out[g]["p_exact_mcnemar"]))
    out[g]["p_holm_4groups"] = run
    print(g, "Holm", run)
(Path(__file__).resolve().parent / "compare.json").write_text(json.dumps(
    {"ours": str(OURS.relative_to(ROOT)), "table_description": str(DESC.relative_to(ROOT)),
     "scope": "doc", "budget_cells": 20, "groups": out}, indent=1))
