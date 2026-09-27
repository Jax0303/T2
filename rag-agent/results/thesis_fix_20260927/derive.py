"""논문 초안 3단계 수정(2026-09-27)에 쓰는 수치 — 기존 결과 파일에서 다시 센다. 새 검색·생성 실행은 없다.

실행:  cd rag-agent && .venv/bin/python results/thesis_fix_20260927/derive.py   -> derive.json (같은 폴더)

- HiTab: 본 방법 = s3c. 검색은 재실행(results/rerun_20260926/hitab, 질문의 표 안), 답변은 같은 300건
  (s3c = results/s3c_answer_hitab300_20260926, 비교군 = results/fair_filter_20260921 무필터, 표 전체 = results/fulltable_20260924).
- MultiHiertt: 모집단 2,885(최종 머리글 규칙의 채점 문항). 검색은 재실행(results/rerun_20260926/mh, 문서 안).
  처음 규칙(v1)은 2,871건만 채점했다 — 나머지 14건(정답 셀을 머리글 칸으로 파싱)은 검색 실패로 센다(2026-09-27 사용자 지시).
- McNemar = 정확 이항(scipy binomtest), b = 앞 조건만 맞힘, c = 뒤 조건만 맞힘.
"""
import json
import sys
from collections import Counter
from pathlib import Path

from scipy.stats import binomtest, fisher_exact

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).with_name("derive.json")
RH, RM = "results/rerun_20260926/hitab", "results/rerun_20260926/mh"
GROUPS = ["lookup_m1", "lookup_m2+", "arith_m1", "arith_m2+"]


def jl(rel):
    with open(ROOT / rel, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def jload(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def mc(a, b, ids):
    """a, b: qid -> 0/1. (a만, b만, p)"""
    x = sum(1 for q in ids if a[q] and not b[q])
    y = sum(1 for q in ids if b[q] and not a[q])
    return {"b": x, "c": y, "p": float(binomtest(x, x + y).pvalue) if x + y else 1.0}


def acc(d, ids):
    return round(sum(d[q] for q in ids) / len(ids), 4)


out = {}

# ------------------------------------------------------------------ HiTab 검색 (유형별, 질문의 표 안)
H_ARMS = ["s3c", "sleaf", "chunk", "trag_hetero", "rowcol", "randrow", "tablerag_path", "tablerag_leaf"]
typ, succ = {}, {}
for arm in H_ARMS:
    rows = jl(f"{RH}/hitab_test_gold_{arm}_type_accuracy.jsonl")
    rows = [r for r in rows if r["retrieval_success"] is not None]      # 정답 근거를 못 푸는 3건 제외
    succ[arm] = {r["query_id"]: r["retrieval_success"] for r in rows}
    typ.update({r["query_id"]: r["query_type"] for r in rows})
TYPES = ["single_cell", "multi_cell", "arithmetic"]
ids_t = {t: sorted(q for q, v in typ.items() if v == t) for t in TYPES}
h = {"source": [f"{RH}/hitab_test_gold_{{arm}}_type_accuracy.jsonl"], "n": {t: len(ids_t[t]) for t in TYPES},
     "accuracy": {}, "success": {}, "vs_s3c": {}}
for arm in H_ARMS:
    h["accuracy"][arm] = {t: acc(succ[arm], ids_t[t]) for t in TYPES}
    h["success"][arm] = {t: sum(succ[arm][q] for q in ids_t[t]) for t in TYPES}
    j = jload(f"{RH}/hitab_test_gold_{arm}.json")["type_accuracy"]          # 결과 JSON 과 같아야 한다
    assert all(j[t]["success"] == h["success"][arm][t] and j[t]["n"] == h["n"][t] for t in TYPES), arm
    if arm != "s3c":
        h["vs_s3c"][arm] = {t: mc(succ["s3c"], succ[arm], ids_t[t]) for t in TYPES}
        h["vs_s3c"][arm]["single_cell"]["diff_s3c_minus_this"] = round(
            h["accuracy"]["s3c"]["single_cell"] - h["accuracy"][arm]["single_cell"], 4)
        h["vs_s3c"][arm]["arithmetic"]["diff_s3c_minus_this"] = round(
            h["accuracy"]["s3c"]["arithmetic"] - h["accuracy"][arm]["arithmetic"], 4)
cmp = jload("results/rerun_20260926/compare.json")["item1"]["hitab_gold"]       # 단일 셀 b:c 교차 확인
for r in cmp:
    if r["arm"] in h["vs_s3c"]:
        v = h["vs_s3c"][r["arm"]]["single_cell"]
        assert (v["b"], v["c"]) == (r["vs_s3c"]["b"], r["vs_s3c"]["c"]), r["arm"]
h["max_p_randrow_path_leaf_9"] = max((h["vs_s3c"][a][t]["p"], f"{a}.{t}")
                                     for a in ("randrow", "tablerag_path", "tablerag_leaf") for t in TYPES)
s3c_json = jload(f"{RH}/hitab_test_gold_s3c.json")
h["s3c_max_doc_tokens"] = s3c_json["embedding_input_audit"]["documents"]["max_tokens"]
h["s3c_n_units"] = s3c_json["n_units"]
rec = {r["query_id"]: r for r in jl(f"{RH}/hitab_test_gold_s3c_records.jsonl")}
single = ids_t["single_cell"]
h["s3c_recall_at_k_single_cell"] = {
    k: round(sum(1 for q in single if rec[q]["gold_rank"] is not None and rec[q]["gold_rank"] <= k) / len(single), 4)
    for k in (1, 5, 10, 20)}
out["hitab_retrieval_gold"] = h

# ------------------------------------------------------------------ HiTab 답변 300건
S3A = "results/s3c_answer_hitab300_20260926/rows.jsonl"
FF = "results/fair_filter_20260921/rows.jsonl"
FT = "results/fulltable_20260924/hitab_rows.jsonl"
s3 = {r["query_id"]: r for r in jl(S3A)}
ids300 = sorted(s3)
ff = {}
for r in jl(FF):
    ff.setdefault(r["arm"], {})[r["query_id"]] = r
ft = {r["query_id"]: r for r in jl(FT)}
assert set(ft) == set(ids300) and all(set(v) == set(ids300) for v in ff.values())
ok = {q: r["correct_base"] for q, r in s3.items()}
hits = [q for q in ids300 if s3[q]["retrieval_correct"]]
miss = [q for q in ids300 if not s3[q]["retrieval_correct"]]
a = {"source": [S3A, FF, FT, f"{RH}/hitab_test_gold_{{arm}}_type_accuracy.jsonl"],
     "n": len(ids300),
     "s3c": {"answer": acc(ok, ids300), "answer_correct": sum(ok.values()),
             "retrieval": acc({q: s3[q]["retrieval_correct"] for q in ids300}, ids300), "retrieval_hits": len(hits),
             "retrieval_misses": len(miss), "answer_given_hit": acc(ok, hits),
             "answer_given_miss": acc(ok, miss) if miss else None, "correct_among_misses": sum(ok[q] for q in miss),
             "reader_input_tokens_mean": round(sum(s3[q]["reader_input_tokens"] for q in ids300) / 300, 1)},
     "retrieval_rerun_300": {}, "vs_s3c": {}}
FFARM = {"sleaf": "ours", "chunk": "chunk", "trag_hetero": "trag_hetero", "rowcol": "rowcol", "randrow": "randrow",
         "tablerag_path": "tablerag_path", "tablerag_leaf": "tablerag_leaf"}
for arm, fa in FFARM.items():
    rr = acc(succ[arm], ids300)
    fr = acc({q: ff[fa][q]["retrieval_correct"] for q in ids300}, ids300)
    assert rr == fr, (arm, rr, fr)            # 재실행 검색 = 답변 실행이 읽은 검색
    a["retrieval_rerun_300"][arm] = rr
    other = {q: ff[fa][q]["correct_base"] for q in ids300}
    a["vs_s3c"][arm] = mc(ok, other, ids300) | {"answer": acc(other, ids300),
                                                "diff_s3c_minus_this": round(acc(ok, ids300) - acc(other, ids300), 4)}
assert acc(succ["s3c"], ids300) == a["s3c"]["retrieval"]
fto = {q: ft[q]["correct_base"] for q in ids300}
a["vs_s3c"]["fulltable"] = mc(ok, fto, ids300) | {"answer": acc(fto, ids300), "answer_correct": sum(fto.values())}
st = jload("results/stats_20260926/stats.json")["item3b_hitab300"]           # stats_20260926 과 교차 확인
for arm, v in a["vs_s3c"].items():
    assert (v["b"], v["c"]) == (st[arm]["b"], st[arm]["c"]), arm
a["retrieval_991_minus_300"] = round(h["accuracy"]["s3c"]["single_cell"] - a["s3c"]["retrieval"], 4)
out["hitab_answer300"] = a

# ------------------------------------------------------------------ HiTab 같은 표 안에서 값만 빼고 같은 셀 문장 (s3c)
import retrieval_accuracy as ra                                              # noqa: E402

args = s3c_json["arguments"]
page_titles = json.loads(ra.PAGE_TITLES.read_text()) if ra.PAGE_TITLES.exists() else {}
tabs: dict = {}
queries = ra.load_queries(str(ROOT / args["data_dir"]), args["split"], tabs, False)
tids = sorted({q["table_id"] for q in queries})
texts, covers, *_ = ra.build_corpus(str(ROOT / args["data_dir"]), tids, "s3c", "cell", page_titles)
assert ra.digest(texts) == s3c_json["corpus_text_sha256"], "셀 문장이 재실행 색인과 다르다"
keys = []
for tid in tids:
    tab = ra.hg.load_table(tid, str(ROOT / args["data_dir"]))
    t = tab.table
    title = ra.with_page_title(tab.title, page_titles.get(tid))
    for i in range(t.n_rows):
        for j in range(t.n_cols):
            if str(t.data[i][j]).strip():
                keys.append((tid, ra.cell_unit(title, t.row_path(i), t.col_path(j), "\x00", "s3c")))
assert len(keys) == len(texts)
cnt = Counter(keys)
dup = sum(1 for k in keys if cnt[k] > 1)
out["hitab_same_sentence_s3c"] = {"source": [args["data_dir"], "scripts/retrieval_accuracy.py build_corpus(s3c, cell)"],
                                  "corpus_text_sha256": s3c_json["corpus_text_sha256"], "n_cells": len(keys),
                                  "cells_with_same_sentence_in_table": dup, "ratio": round(dup / len(keys), 4)}

# ------------------------------------------------------------------ MultiHiertt 검색 (2,885, 문서 안)
def mh_doc(rel):
    d = {}
    for x in jl(rel):
        if "excluded" not in x:
            doc = x["doc"] if isinstance(x["doc"], dict) else eval(x["doc"])   # 레코드가 dict 를 문자열로 저장한 경우
            d[x["query_id"]] = (x["layer"], int(doc["correct"]))
    return d


M_ARMS = ["s3c", "sleaf", "chunk", "trag_hetero", "rowcol", "tablerag_path", "tablerag_leaf", "randrow"]
md = {arm: mh_doc(f"{RM}/mh_train_{arm}_records.jsonl") for arm in M_ARMS}
ids = sorted(md["s3c"])
assert len(ids) == 2885 and all(set(v) == set(ids) for v in md.values())
layer = {q: md["s3c"][q][0] for q in ids}
gids = {g: [q for q in ids if layer[q] == g] for g in GROUPS} | {"ALL": ids}
m = {"source": [f"{RM}/mh_train_{{arm}}_records.jsonl"], "n": {g: len(v) for g, v in gids.items()},
     "accuracy": {}, "vs_s3c": {}}
okm = {arm: {q: md[arm][q][1] for q in ids} for arm in M_ARMS}
for arm in M_ARMS:
    m["accuracy"][arm] = {g: acc(okm[arm], v) for g, v in gids.items()}
    bl = jload(f"{RM}/mh_train_{arm}.json")["by_layer"]
    assert all(abs(bl[g]["doc"]["accuracy_all"] - m["accuracy"][arm][g]) < 1e-9 for g in gids), arm
    if arm != "s3c":
        m["vs_s3c"][arm] = {g: mc(okm["s3c"], okm[arm], v) | {"diff_s3c_minus_this": round(
            m["accuracy"]["s3c"][g] - m["accuracy"][arm][g], 4)} for g, v in gids.items()}
BASE6 = ["chunk", "trag_hetero", "rowcol", "tablerag_path", "tablerag_leaf", "randrow"]
cells24 = [(m["vs_s3c"][b][g]["p"], f"{b} {g}") for b in BASE6 for g in GROUPS]
m["six_baselines_24_cells"] = {"max_p": max(cells24), "n_p_lt_05": sum(1 for p, _ in cells24 if p < .05),
                               "s3c_first_in_every_group": all(m["accuracy"]["s3c"][g] > m["accuracy"][b][g]
                                                               for b in BASE6 for g in GROUPS)}
out["mh_retrieval_doc"] = m

# ------------------------------------------------------------------ MultiHiertt 머리글 규칙별 (본 방법)
V1 = "results/mh_arms/mh_train_cell_hv1_none_doc_records.jsonl"
V33 = "results/mh_interim200_v33/mh_cell_hv33_records.jsonl"
v1raw = mh_doc(V1)
v1_excl = {x["query_id"]: x["excluded"] for x in jl(V1) if "excluded" in x}
extra = sorted(set(ids) - set(v1raw))
assert len(extra) == 14 and all(v1_excl[q] == "gold_in_header" for q in extra)
rule = {"v1": {q: (v1raw[q][1] if q in v1raw else 0) for q in ids},            # 14건 = 검색 실패
        "v33": {q: c for q, (_, c) in mh_doc(V33).items()},
        "v33u": okm["s3c"]}
assert set(rule["v33"]) == set(ids)
hr = {"source": [V1, V33, f"{RM}/mh_train_s3c_records.jsonl"],
      "v1_gold_in_header_counted_as_failure": {"n": len(extra), "by_group": dict(Counter(layer[q] for q in extra))},
      "accuracy": {r: {g: acc(rule[r], v) for g, v in gids.items()} for r in rule},
      "v33_vs_v1": {g: mc(rule["v33"], rule["v1"], v) for g, v in gids.items()},
      "v33u_vs_v33": {g: mc(rule["v33u"], rule["v33"], v) for g, v in gids.items()},
      "v33u_vs_v1": {g: mc(rule["v33u"], rule["v1"], v) for g, v in gids.items()}}
for r1, r2 in (("v33", "v1"), ("v33u", "v33"), ("v33u", "v1")):
    hr[f"diff_{r1}_minus_{r2}"] = {g: round(hr["accuracy"][r1][g] - hr["accuracy"][r2][g], 4) for g in gids}
common = sorted(v1raw)                                                        # 두 규칙이 함께 채점하는 2,871건
hr["common_2871"] = {"n": len(common), "v1_correct": sum(rule["v1"][q] for q in common),
                     "v33u_correct": sum(rule["v33u"][q] for q in common),
                     "v33u_vs_v1": mc(rule["v33u"], rule["v1"], common)}
out["mh_header_rules"] = hr

# ------------------------------------------------------------------ 5.7: 처음 규칙의 셀 1개 두 그룹 (2,885, 14건 실패)
B1 = "results/mh_arms/mh_train_cell_hv1_none_doc_b1_records.jsonl"
b1raw = mh_doc(B1)
assert set(b1raw) == set(v1raw)
b1 = {q: (b1raw[q][1] if q in b1raw else 0) for q in ids}
g57 = {}
for name, d in (("top1", b1), ("budget20", rule["v1"])):
    lk, ar = gids["lookup_m1"], gids["arith_m1"]
    kl, ka = sum(d[q] for q in lk), sum(d[q] for q in ar)
    g57[name] = {"lookup_m1": {"correct": kl, "n": len(lk), "accuracy": round(kl / len(lk), 4)},
                 "arith_m1": {"correct": ka, "n": len(ar), "accuracy": round(ka / len(ar), 4)},
                 "fisher_exact_two_sided_p": float(fisher_exact([[kl, len(lk) - kl], [ka, len(ar) - ka]])[1])}
out["mh_v1_m1_groups"] = {"source": [B1, V1], **g57}

OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUT)
