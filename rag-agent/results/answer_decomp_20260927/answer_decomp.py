"""답변 정확도 분해(2026-09-27): 비교군마다 전달 셀 평균, 검색 성공률, 검색 성공·실패 시 답변 정답률. 새 실행 없음.

실행:  cd rag-agent && python3 results/answer_decomp_20260927/answer_decomp.py  -> answer_decomp.json, answer_decomp.md
- 검색 성공 = 답변 문맥(20셀 예산)에 정답 셀 전부 포함(답변 행의 retrieval_correct).
- HiTab 300: 답변 행 = s3c results/s3c_answer_hitab300_20260926/rows.jsonl, 나머지 results/fair_filter_20260921/rows.jsonl
  (무필터 correct_base), 표 전체 results/fulltable_20260924/hitab_rows.jsonl. 전달 셀 = 답변 실행이 읽은 검색 records 의
  cells_in_context(scripts/fair_filter_eval.py ARMS/EXTRA_ARMS 의 records), records correct 와 retrieval_correct 가 같은지 확인.
- MultiHiertt 882: results/mh_arms/cap300_20260924/<조건>.jsonl 의 retrieval_correct·answer_correct·cells_in_context.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
RA = ROOT / "results/retrieval_accuracy"
GROUPS = ["lookup_m1", "lookup_m2+", "arith_m1", "arith_m2+"]


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def frac(k, n):
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None}


def decomp(rows, cells=None):
    """rows: [(retrieval_correct, answer_correct)], cells: [전달 셀 수] 또는 None(검색 없음)."""
    if cells is None:
        return {"retrieval": "검색 없음", "answer": frac(sum(a for _, a in rows), len(rows))}
    hit = [a for r, a in rows if r]
    miss = [a for r, a in rows if not r]
    return {"cells_delivered_mean": round(sum(cells) / len(cells), 2),
            "retrieval_success": frac(len(hit), len(rows)),
            "answer_given_hit": frac(sum(hit), len(hit)), "answer_given_miss": frac(sum(miss), len(miss)),
            "answer": frac(sum(hit) + sum(miss), len(rows))}


out = {}
# ------------------------------------------------------------------ HiTab 300
STEM = {"s3c": "t_s3c_gold_labelabl", "sleaf": "t_sleaf_gold", "chunk": "t_chunk_s3c_gold_v2",
        "trag_hetero": "t_trag_hetero_gold_v2", "rowcol": "t_rowcol_s3c_gold_v2", "randrow": "t_randrow_s3c_gold",
        "tablerag_path": "t_tablerag_path_v2_gold", "tablerag_leaf": "t_tablerag_leaf_v2_gold"}
ff = {}
for r in jl(ROOT / "results/fair_filter_20260921/rows.jsonl"):
    ff.setdefault("sleaf" if r["arm"] == "ours" else r["arm"], {})[r["query_id"]] = r
ff["s3c"] = {r["query_id"]: r for r in jl(ROOT / "results/s3c_answer_hitab300_20260926/rows.jsonl")}
ids = sorted(ff["s3c"])
h = {}
for arm, stem in STEM.items():
    rows = ff[arm]
    assert sorted(rows) == ids, arm
    rec = {x["query_id"]: x for x in jl(RA / f"{stem}_records.jsonl")}
    assert all(int(rec[q]["correct"]) == rows[q]["retrieval_correct"] for q in ids), arm
    h[arm] = decomp([(rows[q]["retrieval_correct"], rows[q]["correct_base"]) for q in ids],
                    [rec[q]["cells_in_context"] for q in ids])
ft = {r["query_id"]: r for r in jl(ROOT / "results/fulltable_20260924/hitab_rows.jsonl")}
assert sorted(ft) == ids
h["fulltable"] = decomp([(None, ft[q]["correct_base"]) for q in ids])
out["hitab_300"] = h

# ------------------------------------------------------------------ MultiHiertt 882
CAP = ROOT / "results/mh_arms/cap300_20260924"
MH = ["cell_uniq", "cell", "fulltable", "chunk", "trag_hetero", "rowcol", "rowcol_values", "rowcol_hv33", "randrow",
      "randrow_values", "tablerag_path", "tablerag_path_hv33", "tablerag_leaf", "tablerag_leaf_hv33"]
m = {}
ref = None
for name in MH:
    rows = {x["query_id"]: x for x in jl(CAP / f"{name}.jsonl")}
    ref = ref or sorted(rows)
    assert sorted(rows) == ref and len(ref) == 882, name
    run = json.loads((CAP / f"{name}.run.json").read_text())
    none = run["condition"] == "fulltable"
    res = {"context_header_rule": run["header_rule"], "condition": run["condition"]}
    for g in ["ALL", *GROUPS]:
        qs = [q for q in ref if g == "ALL" or rows[q]["layer"] == g]
        rc = [(rows[q]["retrieval_correct"], rows[q]["answer_correct"]) for q in qs]
        res[g] = decomp(rc, None if none else [rows[q]["cells_in_context"] for q in qs])
    m[name] = res
out["mh_882"] = m
(OUT / "answer_decomp.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def f(x):
    return f"{x['rate']:.4f} ({x['k']}/{x['n']})" if x["n"] else f"— (0/0)"


def line(name, d, extra=""):
    if d.get("retrieval") == "검색 없음":
        return f"| {name}{extra} | 검색 없음 | 검색 없음 | — | — | {f(d['answer'])} |"
    return (f"| {name}{extra} | {d['cells_delivered_mean']:.2f} | {f(d['retrieval_success'])} | {f(d['answer_given_hit'])} | "
            f"{f(d['answer_given_miss'])} | {f(d['answer'])} |")


NAME = {"s3c": "본 방법(s3c)", "sleaf": "sleaf(이전 조건)", "chunk": "고정 청크", "trag_hetero": "TableRAG(Yu) 청크",
        "rowcol": "RowCol", "randrow": "RandRow", "tablerag_path": "TableRAG path", "tablerag_leaf": "TableRAG leaf",
        "fulltable": "표 전체"}
MNAME = {"cell_uniq": "본 방법 (최종 규칙)", "cell": "본 방법 (처음 규칙)", "fulltable": "표 전체", "chunk": "고정 청크 (처음)",
         "trag_hetero": "TableRAG(Yu) 청크 (처음)", "rowcol": "RowCol 셀 문장 (처음)", "rowcol_values": "RowCol 값만 (처음)",
         "rowcol_hv33": "RowCol 셀 문장 (머리글 고친)", "randrow": "RandRow 셀 문장 (처음)", "randrow_values": "RandRow 값만 (처음)",
         "tablerag_path": "TableRAG path (처음)", "tablerag_path_hv33": "TableRAG path (머리글 고친)",
         "tablerag_leaf": "TableRAG leaf (처음)", "tablerag_leaf_hv33": "TableRAG leaf (머리글 고친)"}
GNAME = {"ALL": "882건 합산(모집단 가중 아님)", "lookup_m1": "조회 셀 1개", "lookup_m2+": "조회 셀 2개+",
         "arith_m1": "산술 셀 1개", "arith_m2+": "산술 셀 2개+"}
HEAD = ["| 조건 | 전달 셀 평균 | 검색 성공률 | 성공 시 정답률 | 실패 시 정답률 | 답변 정답률 |", "|---|---:|---:|---:|---:|---:|"]
md = ["# 답변 정확도 분해 (2026-09-27)", "", "괄호 = 답변 문맥의 머리글 규칙. 검색 성공 = 답변 문맥에 정답 셀 전부 포함.", "", "## HiTab 단일 셀 조회 300건", "", *HEAD]
md += [line(NAME[a], d) for a, d in h.items()]
for g in ["ALL", *GROUPS]:
    md += ["", f"## MultiHiertt 882건 — {GNAME[g]}", "", *HEAD]
    md += [line(MNAME[n], d[g]) for n, d in m.items()]
(OUT / "answer_decomp.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
