"""2026-09-28 원고의 TableRAG(Chen) 재구현 값을 공식 숫자 열 규칙(infer_dtype) 결과로 대체할 때 쓰는 값. 새 검색·모델 실행 없음.
PREREG-2026-09-28-tablerag-official-dtype.md (사후 추가 절: 대체·답변 재생성).

검색(원고 표 5-2·5-3·5-5): results/rerun_20260926/compare.py·thesis_fix_20260927/derive.py·step4_values.py·thesis_fix_20260928/values.py
  와 같은 계산에서 tablerag_leaf/path 만 공식 규칙 records(results/tablerag_official_20260928/{hitab,mh}/*_official) 로 바꾼다.
  Holm 가족도 같은 정의(한 데이터셋·한 범위의 비교군 전부, 기준 s3c)로 다시 계산하고, 이전 가족(rerun compare.json)과 판정(.05)을 비교한다.
답변(표 5-4, 부록 A, 표 5-8, 부록 F): --answers 를 주면 answers/ 의 재생성 결과로 계산한다(실행 뒤).
  HiTab 300 가족 = stats_20260926 item3b_hitab300 과 같은 8개(sleaf·chunk·trag_hetero·rowcol·randrow·tablerag_path·tablerag_leaf·표 전체).
  MH 882 가족 = stats_20260926 item3b_mh882_pooled 에서 옛 TableRAG 4개(처음·v3.3 문맥)를 빼고 공식 규칙 2개를 넣은 12개(기준 cell_uniq).
  부록 F 가족 = thesis_fix_20260927/mh_answer_v1_compare.py 의 12개에서 옛 TableRAG 4개를 뺀 8개(기준 처음 규칙 본 방법 cell).
실행: .venv/bin/python results/tablerag_official_20260928/replace/values.py [--answers]
"""
import json
import statistics
import sys
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
R1, OF = "results/rerun_20260926", "results/tablerag_official_20260928"
GROUPS = ["lookup_m1", "lookup_m2+", "arith_m1", "arith_m2+"]
TYPES = ["single_cell", "multi_cell", "arithmetic"]
TR = ("tablerag_leaf", "tablerag_path")


def jl(rel):
    return [json.loads(x) for x in open(ROOT / rel, encoding="utf-8") if x.strip()]


def mc(ref, x, ids):
    b = sum(1 for q in ids if ref[q] and not x[q])
    c = sum(1 for q in ids if x[q] and not ref[q])
    return {"n": len(ids), "b": b, "c": c, "p": binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0}


def holm(rows):
    order = sorted(rows, key=lambda k: rows[k]["p"])
    m, run = len(rows), 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * rows[k]["p"]))
        rows[k]["p_holm"] = run
    return rows


def acc(d, ids):
    return round(sum(d[q] for q in ids) / len(ids), 4)


def verdict_changes(old, new):
    """이전·새 가족에서 같은 비교의 Holm .05 판정이 바뀐 것."""
    return [{"arm": k, "old_p_holm": old[k], "new_p_holm": new[k]["p_holm"]}
            for k in new if k in old and (old[k] < .05) != (new[k]["p_holm"] < .05)]


def stem_h(scope, arm):
    return f"{OF}/hitab/hitab_test_{scope}_{arm}_official" if arm in TR else f"{R1}/hitab/hitab_test_{scope}_{arm}"


out = {"retrieval": {}}
# ---------------------------------------------------------------- HiTab 질문의 표 안 (표 5-2)
H_ARMS = ["s3c", "sleaf", "chunk", "trag_hetero", "rowcol", "randrow", "tablerag_path", "tablerag_leaf"]
typ, succ, cells = {}, {}, {}
for arm in H_ARMS:
    rows = [r for r in jl(f"{stem_h('gold', arm)}_type_accuracy.jsonl") if r["retrieval_success"] is not None]
    succ[arm] = {r["query_id"]: r["retrieval_success"] for r in rows}
    typ.update({r["query_id"]: r["query_type"] for r in rows})
ids_t = {t: sorted(q for q, v in typ.items() if v == t) for t in TYPES}
h = {"n": {t: len(ids_t[t]) for t in TYPES}, "accuracy": {}, "success": {}, "vs_s3c": {}, "cells_delivered_single991": {}}
for arm in H_ARMS:
    h["accuracy"][arm] = {t: acc(succ[arm], ids_t[t]) for t in TYPES}
    h["success"][arm] = {t: sum(succ[arm][q] for q in ids_t[t]) for t in TYPES}
    j = json.loads((ROOT / f"{stem_h('gold', arm)}.json").read_text())["type_accuracy"]
    assert all(j[t]["success"] == h["success"][arm][t] and j[t]["n"] == h["n"][t] for t in TYPES), arm
    rec = {r["query_id"]: r for r in jl(f"{stem_h('gold', arm)}_records.jsonl")}
    h["cells_delivered_single991"][arm] = round(statistics.mean(rec[q]["cells_in_context"] for q in ids_t["single_cell"]), 4)
    if arm != "s3c":
        h["vs_s3c"][arm] = {t: mc(succ["s3c"], succ[arm], ids_t[t]) for t in TYPES}
old_d = json.loads((ROOT / "results/thesis_fix_20260927/derive.json").read_text())["hitab_retrieval_gold"]
for arm in H_ARMS:                                  # 공식 규칙으로 바꾸지 않은 방법은 원고 값과 같아야 한다
    if arm not in TR:
        assert h["accuracy"][arm] == old_d["accuracy"][arm], arm
h["max_p_randrow_path_leaf_9"] = max((h["vs_s3c"][a][t]["p"], f"{a}.{t}") for a in ("randrow",) + TR for t in TYPES)
out["retrieval"]["hitab_gold"] = h


# ---------------------------------------------------------------- Holm 가족 3개 (rerun compare.py 정의)
def hitab_sc(stem):
    return {r["query_id"]: int(r["correct"]) for r in jl(f"{stem}_records.jsonl")
            if "correct" in r and r["mode"] == "all" and r.get("m") == 1 and (r.get("aggregation") or "none") == "none"}


def mh_doc(stem):
    return {r["query_id"]: (r["layer"], int(r["doc"]["correct"])) for r in jl(f"{stem}_records.jsonl") if "doc" in r}


cmp_old = json.loads((ROOT / f"{R1}/compare.json").read_text())["item1"]
ARMS = ["s3c", "sleaf", "table", "row", "chunk", "trag_hetero", "tablerag_leaf", "tablerag_path", "rowcol", "randrow"]
for key, scope in (("hitab_gold", "gold"), ("hitab_split", "split")):
    res = {a: hitab_sc(stem_h(scope, a)) for a in ARMS if (ROOT / f"{stem_h(scope, a)}.json").exists()}
    ids = sorted(res["s3c"])
    fam = holm({a: mc(res["s3c"], v, ids) for a, v in res.items() if a != "s3c"})
    for a, v in fam.items():
        v.update(correct=sum(res[a][q] for q in ids), accuracy=acc(res[a], ids))
    old = {r["arm"]: r["vs_s3c"]["p_holm"] for r in cmp_old[key] if "vs_s3c" in r}
    fam_out = {"family": fam, "verdict_changes_vs_rerun_compare": verdict_changes(old, fam)}
    if scope == "split":
        fam_out["cells_delivered_single991"] = {}
        for a in res:
            rec = {r["query_id"]: r for r in jl(f"{stem_h(scope, a)}_records.jsonl")}
            fam_out["cells_delivered_single991"][a] = round(statistics.mean(rec[q]["cells_in_context"] for q in ids), 4)
        fam_out["s3c"] = {"correct": sum(res["s3c"].values()), "accuracy": acc(res["s3c"], ids)}
        worst = max((v["p_holm"], a) for a, v in fam.items() if a != "sleaf")
        fam_out["max_holm_excluding_sleaf"] = {"p_holm": worst[0], "arm": worst[1]}
    out["retrieval"][f"{key}_family"] = fam_out

# ---------------------------------------------------------------- MultiHiertt 문서 안 (표 5-5)
stem_m = lambda a: f"{OF}/mh/mh_train_{a}_official" if a in TR else f"{R1}/mh/mh_train_{a}"
M_ARMS = ["s3c", "sleaf", "table", "row", "chunk", "trag_hetero", "tablerag_leaf", "tablerag_path", "rowcol", "randrow"]
md = {a: mh_doc(stem_m(a)) for a in M_ARMS if (ROOT / f"{stem_m(a)}.json").exists()}
ids = sorted(md["s3c"])
assert len(ids) == 2885 and all(set(v) == set(ids) for v in md.values())
gids = {g: [q for q in ids if md["s3c"][q][0] == g] for g in GROUPS} | {"ALL": ids}
ok = {a: {q: v[q][1] for q in ids} for a, v in md.items()}
m = {"n": {g: len(v) for g, v in gids.items()}, "accuracy": {}, "vs_s3c": {}, "cells_delivered_2885": {}}
for a in md:
    m["accuracy"][a] = {g: acc(ok[a], v) for g, v in gids.items()}
    bl = json.loads((ROOT / f"{stem_m(a)}.json").read_text())["by_layer"]
    assert all(abs(bl[g]["doc"]["accuracy_all"] - m["accuracy"][a][g]) < 1e-9 for g in gids), a
    m["cells_delivered_2885"][a] = bl["ALL"]["doc"]["cells_delivered_mean"]
    if a != "s3c":
        m["vs_s3c"][a] = {g: mc(ok["s3c"], ok[a], v) for g, v in gids.items()}
fam = holm({a: dict(m["vs_s3c"][a]["ALL"]) for a in md if a != "s3c"})
old = {r["arm"]: r["vs_s3c"]["p_holm"] for r in cmp_old["mh_doc"] if "vs_s3c" in r}
m["family_ALL"] = fam
m["verdict_changes_vs_rerun_compare"] = verdict_changes(old, fam)
BASE6 = ["chunk", "trag_hetero", "rowcol", "tablerag_path", "tablerag_leaf", "randrow"]
cells24 = [(m["vs_s3c"][b][g]["p"], f"{b} {g}") for b in BASE6 for g in GROUPS]
m["six_baselines_24_cells"] = {"max_p": max(cells24), "n_p_lt_05": sum(1 for p, _ in cells24 if p < .05)}
out["retrieval"]["mh_doc"] = m

# ---------------------------------------------------------------- 전달 가능 상한 (bound.json 그대로)
b = json.loads((ROOT / f"{OF}/bound.json").read_text())
out["bound"] = {"source": f"{OF}/bound.json", "hitab": b["hitab_test_single_cell"], "multihiertt": b["multihiertt_train"]}



# ---------------------------------------------------------------- 답변 (재생성 뒤, --answers)
def answers():
    A = f"{OF}/answers"
    res = {}
    # HiTab 300 (표 5-4): 기준 s3c, 가족 8개 = stats_20260926 item3b_hitab300 과 같은 구성, TableRAG 2개만 새 답변
    s3 = {r["query_id"]: r for r in jl("results/s3c_answer_hitab300_20260926/rows.jsonl")}
    ids = sorted(s3)
    ff = {}
    for r in jl("results/fair_filter_20260921/rows.jsonl"):
        ff.setdefault(r["arm"], {})[r["query_id"]] = r
    new = {a: {r["query_id"]: r for r in jl(f"{A}/hitab_{a}_official_rows.jsonl")} for a in TR}
    filt = {a: {r["query_id"]: r for r in jl(f"{A}/hitab_{a}_official_filter_rows.jsonl")} for a in TR}
    ft = {r["query_id"]: r for r in jl("results/fulltable_20260924/hitab_rows.jsonl")}
    arms = {"sleaf": ff["ours"], "chunk": ff["chunk"], "trag_hetero": ff["trag_hetero"], "rowcol": ff["rowcol"],
            "randrow": ff["randrow"], "tablerag_path": new["tablerag_path"], "tablerag_leaf": new["tablerag_leaf"],
            "fulltable": ft}
    assert all(set(v) == set(ids) for v in arms.values())
    ok = {q: s3[q]["correct_base"] for q in ids}
    fam = holm({a: mc(ok, {q: v[q]["correct_base"] for q in ids}, ids) for a, v in arms.items()})
    old = json.loads((ROOT / "results/stats_20260926/stats.json").read_text())["item3b_hitab300"]
    h = {"family": fam, "verdict_changes_vs_stats_item3b": verdict_changes({k: v["p_holm"] for k, v in old.items()}, fam),
         "other_rows_holm_changed": {a: [old[a]["p_holm"], fam[a]["p_holm"]] for a in fam
                                     if a not in TR and abs(old[a]["p_holm"] - fam[a]["p_holm"]) > 1e-9 * old[a]["p_holm"]},
         "tablerag": {}}
    for a in TR:
        rec = {r["query_id"]: r for r in jl(f"{OF}/hitab/hitab_test_gold_{a}_official_records.jsonl")}
        rows = new[a]
        hit = [q for q in ids if rows[q]["retrieval_correct"]]
        assert all(rows[q]["retrieval_correct"] == rec[q]["correct"] for q in ids)
        h["tablerag"][a] = {"cells_delivered_300": round(statistics.mean(rec[q]["cells_in_context"] for q in ids), 2),
                            "retrieval_300": acc({q: rows[q]["retrieval_correct"] for q in ids}, ids),
                            "answer": acc({q: rows[q]["correct_base"] for q in ids}, ids),
                            "answer_correct": sum(rows[q]["correct_base"] for q in ids),
                            "answer_given_hit": acc({q: rows[q]["correct_base"] for q in hit}, hit), "hits": len(hit),
                            "reader_input_tokens_mean": round(statistics.mean(rows[q]["reader_input_tokens"] for q in ids), 1),
                            "diff_s3c_minus_this": round(acc(ok, ids) - acc({q: rows[q]["correct_base"] for q in ids}, ids), 4)}
        # 부록 A: 필터 실행. 이 실행의 무필터 답이 무필터 실행과 같은지 먼저 확인한다
        fr = filt[a]
        same = sum(fr[q]["pred_base"] == rows[q]["pred_base"] for q in ids)
        base = {q: fr[q]["correct_base"] for q in ids}
        fil = {q: fr[q]["correct_filtered"] for q in ids}
        x = mc(base, fil, ids)
        h["tablerag"][a]["appendix_A"] = {
            "base_pred_same_as_nofilter_run": same, "answer_base": acc(base, ids), "answer_filtered": acc(fil, ids),
            "delta": round(acc(fil, ids) - acc(base, ids), 4), "base_vs_filtered": {"a_only": x["b"], "b_only": x["c"],
                                                                                    "p_value": round(x["p"], 6)},
            "lines_mean": round(sum(fr[q]["n_lines"] for q in ids) / len(ids), 1),
            "lines_kept_mean": round(sum(fr[q]["n_kept"] for q in ids) / len(ids), 1),
            "filter_fallback_n": sum(fr[q]["filter_fallback"] for q in ids)}
    res["hitab300"] = h

    # MultiHiertt 882: 가족 12개(기준 cell_uniq) = stats item3b_mh882_pooled − 옛 TableRAG 4 + 공식 2
    done = [ROOT / f"{A}/mh_{a}_official.jsonl" for a in TR]
    if not all(p.exists() and sum(1 for _ in open(p)) == 882 for p in done):
        res["mh882_final"] = "pending"
        return res
    CAP = ROOT / "results/mh_arms/cap300_20260924"
    load = lambda p: {r["query_id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}
    mh = {p.stem: load(p) for p in sorted(CAP.glob("*.jsonl")) if "." not in p.stem}
    mh = {k: v for k, v in mh.items() if len(v) == 882}
    ref = mh.pop("cell_uniq")
    ids = sorted(ref)
    for k in ("tablerag_leaf", "tablerag_path", "tablerag_leaf_hv33", "tablerag_path_hv33"):
        mh.pop(k)
    for a in TR:
        mh[f"{a}_official"] = load(ROOT / f"{A}/mh_{a}_official.jsonl")
    assert all(set(v) == set(ids) for v in mh.values()) and len(mh) == 12
    em = lambda d: {q: int(d[q]["answer_correct"]) for q in ids}
    fam = holm({k: mc(em(ref), em(v), ids) | {"correct": sum(em(v).values())} for k, v in mh.items()})
    old = json.loads((ROOT / "results/stats_20260926/stats.json").read_text())["item3b_mh882_pooled"]
    grp = {g: [q for q in ids if ref[q]["layer"] == g] for g in GROUPS}
    res["mh882_final"] = {"reference": {"name": "cell_uniq", "correct": sum(em(ref).values())}, "family": fam,
                          "verdict_changes_vs_stats_item3b": verdict_changes({k: v["p_holm"] for k, v in old.items()}, fam),
                          "other_rows_holm_changed": {k: [old[k]["p_holm"], fam[k]["p_holm"]] for k in fam
                                                      if k in old and abs(old[k]["p_holm"] - fam[k]["p_holm"]) > 1e-9 * old[k]["p_holm"]},
                          "tablerag_groups": {f"{a}_official": {g: mc(em(ref), em(mh[f"{a}_official"]), v)
                                                                 | {"correct": sum(em(mh[f"{a}_official"])[q] for q in v), "n": len(v)}
                                                                 for g, v in grp.items()} for a in TR}}
    # 부록 F: 기준 처음 규칙 본 방법(cell), 옛 가족 12개에서 옛 TableRAG 4개를 뺀 8개
    v1c = json.loads((ROOT / "results/thesis_fix_20260927/mh_answer_v1_compare.json").read_text())
    ref1 = load(CAP / "cell.jsonl")
    keep = ["fulltable", "chunk", "trag_hetero", "rowcol", "rowcol_values", "rowcol_hv33", "randrow", "randrow_values"]
    ok1 = {q: int(ref1[q]["answer_correct"]) for q in ids}
    okk = {k: {q: int(r["answer_correct"]) for q, r in load(CAP / f"{k}.jsonl").items()} for k in keep}
    fam8 = holm({k: mc(ok1, okk[k], ids) for k in keep})
    for k in keep:
        assert (fam8[k]["b"], fam8[k]["c"]) == (v1c["comparisons"][k]["pooled"]["b"], v1c["comparisons"][k]["pooled"]["c"]), k
    res["appendixF_v1_family8"] = {"family": fam8, "old_p_holm": {k: v1c["comparisons"][k]["pooled"]["p_holm"] for k in keep},
                                   "verdict_changes": verdict_changes({k: v1c["comparisons"][k]["pooled"]["p_holm"] for k in keep}, fam8),
                                   "max_holm_rowcol_randrow_5": max((fam8[k]["p_holm"], k) for k in keep
                                                                    if k.startswith(("rowcol", "randrow")))}
    return res


if "--answers" in sys.argv:
    out["answers"] = answers()
(OUT / "values.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in out["retrieval"].items() if k.endswith("family")}, ensure_ascii=False, indent=1)[:4000])
