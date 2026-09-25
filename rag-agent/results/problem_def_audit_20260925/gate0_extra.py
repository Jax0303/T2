"""GATE 0 추가 요청: 표 전체 대 본 방법의 리더 입력 토큰·EM, MultiHiertt 그룹별 McNemar(Holm).
기존 결과 파일만 읽는다. 추론 없음."""
import json, statistics as st
from pathlib import Path
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: [json.loads(l) for l in open(ROOT / p, encoding="utf-8")]
tok = lambda v: {"mean": round(st.mean(v), 1), "median": st.median(v), "max": max(v)}
out = {}

# 1. HiTab 300
full = rd("results/fulltable_20260924/hitab_rows.jsonl")
ours = [r for r in rd("results/fair_filter_20260921/rows.jsonl") if r["arm"] == "ours"]
assert {r["query_id"] for r in full} == {r["query_id"] for r in ours}
out["hitab_300"] = {
    "fulltable": {"file": "results/fulltable_20260924/hitab_rows.jsonl (reader_input_tokens, correct_base)",
                  "query_count": len(full), "reader_input_tokens": tok([r["reader_input_tokens"] for r in full]),
                  "em": round(sum(r["correct_base"] for r in full) / len(full), 4), "correct": sum(r["correct_base"] for r in full)},
    "ours_t_sleaf_gold": {"file": "results/fair_filter_20260921/rows.jsonl arm=ours (reader_input_tokens = 무필터 리더 프롬프트, scripts/fair_filter_eval.py:278-281,302; correct_base)",
                          "query_count": len(ours), "reader_input_tokens": tok([r["reader_input_tokens"] for r in ours]),
                          "em": round(sum(r["correct_base"] for r in ours) / len(ours), 4), "correct": sum(r["correct_base"] for r in ours)},
}

# 2. MultiHiertt 882
D = "results/mh_arms/cap300_20260924/"
F = {r["query_id"]: r for r in rd(D + "fulltable.jsonl")}
U = {r["query_id"]: r for r in rd(D + "cell_uniq.jsonl")}
assert set(F) == set(U)
rep = json.load(open(ROOT / D / "report.json"))
arm = lambda R, name: {"file": D + f"{name}.jsonl (n_tok, answer_correct); 가중 EM = {D}report.json",
                       "query_count": len(R), "reader_input_tokens(n_tok)": tok([r["n_tok"] for r in R.values()]),
                       "em_unweighted": round(sum(r["answer_correct"] for r in R.values()) / len(R), 4),
                       "correct": sum(r["answer_correct"] for r in R.values()),
                       "em_weighted": rep[name]["weighted_em"], "ci95": rep[name]["ci95"],
                       "em_by_group": {g: v["em"] for g, v in rep[name]["groups"].items()}}
out["mh_882"] = {"fulltable": arm(F, "fulltable"), "cell_uniq": arm(U, "cell_uniq")}
groups = ["lookup_m1", "lookup_m2+", "arith_m1", "arith_m2+"]
mc = {}
for g in groups:
    ids = [q for q in U if U[q]["layer"] == g]
    b = sum(U[q]["answer_correct"] and not F[q]["answer_correct"] for q in ids)
    c = sum(F[q]["answer_correct"] and not U[q]["answer_correct"] for q in ids)
    mc[g] = {"query_count": len(ids), "cell_uniq_only_correct": int(b), "fulltable_only_correct": int(c),
             "p_exact": binomtest(b, b + c).pvalue if b + c else 1.0,
             "report_json_says": rep["fulltable"]["vs_cell"]["mcnemar"][g]}
order = sorted(groups, key=lambda g: mc[g]["p_exact"])
run = 0.0
for k, g in enumerate(order):  # Holm step-down, 단조 보정
    run = max(run, min(1.0, (len(order) - k) * mc[g]["p_exact"]))
    mc[g]["p_holm"] = run
for g in groups:
    mc[g]["p_exact"] = round(mc[g]["p_exact"], 6); mc[g]["p_holm"] = round(mc[g]["p_holm"], 6)
out["mh_882"]["mcnemar_by_group"] = mc
out["mh_882"]["mcnemar_note"] = ("report.json 의 cell_only = 기준 arm REF=cell_uniq 만 맞힘, this_only = fulltable 만 맞힘 "
                                 "(scripts/mh_cap300_report.py:21,70-73)")
print(json.dumps(out, indent=1, ensure_ascii=False))
