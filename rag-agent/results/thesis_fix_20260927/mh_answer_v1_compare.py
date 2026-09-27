"""MultiHiertt 답변 882건: 처음 머리글 규칙의 본 방법(cell, v1 검색 문맥) 대 비교군 12개 (2026-09-27). 새 실행 없음.

실행:  cd rag-agent && .venv/bin/python results/thesis_fix_20260927/mh_answer_v1_compare.py  -> mh_answer_v1_compare.json
- 답변 행: results/mh_arms/cap300_20260924/<조건>.jsonl (answer_correct). 882건 합친 McNemar(정확 이항, b = 본 방법만 맞힘,
  c = 비교군만 맞힘) + Holm(가족 = 이 12개 비교), 그룹 4개별 b:c, p.
- 문맥의 머리글 규칙 = 각 답변 실행의 <조건>.run.json header_rule. 규칙에 따라 문맥 텍스트가 바뀌는지는 context_rule_check.json.
"""
import json
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
CAP = ROOT / "results/mh_arms/cap300_20260924"
GROUPS = ["lookup_m1", "lookup_m2+", "arith_m1", "arith_m2+"]
REF = "cell"
CMP = ["fulltable", "chunk", "trag_hetero", "rowcol", "rowcol_values", "rowcol_hv33", "randrow", "randrow_values",
       "tablerag_path", "tablerag_path_hv33", "tablerag_leaf", "tablerag_leaf_hv33"]


def rows(name):
    return {x["query_id"]: x for x in map(json.loads, open(CAP / f"{name}.jsonl", encoding="utf-8"))}


def mc(a, b, ids):
    x = sum(1 for q in ids if a[q] and not b[q])
    y = sum(1 for q in ids if b[q] and not a[q])
    return {"b": x, "c": y, "p": float(binomtest(x, x + y).pvalue) if x + y else 1.0}


ref = rows(REF)
ids = sorted(ref)
assert len(ids) == 882
ok = {n: {q: r["answer_correct"] for q, r in rows(n).items()} for n in [REF, *CMP]}
assert all(set(v) == set(ids) for v in ok.values())
grp = {g: [q for q in ids if ref[q]["layer"] == g] for g in GROUPS}
run = {n: json.loads((CAP / f"{n}.run.json").read_text()) for n in [REF, *CMP]}
out = {"reference": {"name": REF, "correct": sum(ok[REF].values()), "n": 882, "context_header_rule": run[REF]["header_rule"],
                     "groups": {g: sum(ok[REF][q] for q in v) for g, v in grp.items()}},
       "comparisons": {}}
for n in CMP:
    out["comparisons"][n] = {"correct": sum(ok[n].values()), "context_header_rule": run[n]["header_rule"],
                             "pooled": mc(ok[REF], ok[n], ids),
                             "groups": {g: mc(ok[REF], ok[n], v) | {"correct": sum(ok[n][q] for q in v), "n": len(v)}
                                        for g, v in grp.items()}}
# Holm (가족 = 12개 합친 비교)
order = sorted(CMP, key=lambda n: out["comparisons"][n]["pooled"]["p"])
run_max = 0.0
for i, n in enumerate(order):
    run_max = max(run_max, min(1.0, (len(order) - i) * out["comparisons"][n]["pooled"]["p"]))
    out["comparisons"][n]["pooled"]["p_holm"] = run_max
st = json.loads((ROOT / "results/stats_20260926/stats.json").read_text())["item3b_mh882_pooled"]   # 교차 확인(맞힌 수)
assert st[REF]["correct"] == out["reference"]["correct"]
for n, v in out["comparisons"].items():
    if n in st:
        assert st[n]["correct"] == v["correct"], n
Path(__file__).with_name("mh_answer_v1_compare.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
print(json.dumps(out, ensure_ascii=False, indent=1))
