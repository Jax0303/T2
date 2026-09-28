"""RandRow 삭제(지도교수 지시 §4) 뒤 원고 Holm 가족 재계산. 새 실행 없음, 문항별 결과 파일에서 McNemar·Holm 을 다시 센다.
가족 정의는 results/tablerag_official_20260928/replace/values.py 와 같고 RandRow(randrow, randrow_values)만 뺀다.
  검색(표 5-1): MultiHiertt 문서 안 2,885, 기준 s3c, 9개 → 8개. 비교군 5개(BASE6 − randrow) × 4그룹 = 20칸 최대 p.
  답변(표 5-6): 882, 기준 cell_uniq, 12개 → 10개.
  부록 C(표 C-1): 882, 기준 처음 규칙 cell, 8개 → 6개.
실행: .venv/bin/python results/mh_only_20260928/no_randrow_holm.py  (DATA_ROOT 로 큰 records 가 있는 저장소 루트를 줄 수 있다)
"""
import json
import os
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parents[2]))
OUT = Path(__file__).parent / "no_randrow_holm.json"
GROUPS = ["lookup_m1", "lookup_m2+", "arith_m1", "arith_m2+"]
R1, OF, CAP = "results/rerun_20260926", "results/tablerag_official_20260928", "results/mh_arms/cap300_20260924"
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


out = {}
# 검색 (values.py mh_doc 과 같음, randrow 제외)
stem = lambda a: f"{OF}/mh/mh_train_{a}_official" if a in TR else f"{R1}/mh/mh_train_{a}"
ARMS = ["s3c", "sleaf", "table", "row", "chunk", "trag_hetero", "tablerag_leaf", "tablerag_path", "rowcol"]
md = {a: {r["query_id"]: (r["layer"], int(r["doc"]["correct"])) for r in jl(f"{stem(a)}_records.jsonl") if "doc" in r} for a in ARMS}
ids = sorted(md["s3c"])
assert len(ids) == 2885 and all(set(v) == set(ids) for v in md.values())
ok = {a: {q: v[q][1] for q in ids} for a, v in md.items()}
gids = {g: [q for q in ids if md["s3c"][q][0] == g] for g in GROUPS}
out["retrieval_family_ALL_8"] = holm({a: mc(ok["s3c"], ok[a], ids) for a in ARMS if a != "s3c"})
cells = [(mc(ok["s3c"], ok[b], v)["p"], f"{b} {g}") for b in ("chunk", "trag_hetero", "rowcol", "tablerag_path", "tablerag_leaf") for g, v in gids.items()]
out["five_baselines_20_cells"] = {"n_cells": len(cells), "max_p": max(cells), "n_p_lt_05": sum(1 for p, _ in cells if p < .05)}

# 답변 882 (values.py mh882_final 과 같음, randrow 2개 제외)
load = lambda rel: {r["query_id"]: int(r["answer_correct"]) for r in jl(rel)}
ref = load(f"{CAP}/cell_uniq.jsonl")
ids = sorted(ref)
keep = ["cell", "cell_hv33r", "chunk", "fulltable", "rowcol", "rowcol_hv33", "rowcol_values", "trag_hetero"]
arms = {k: load(f"{CAP}/{k}.jsonl") for k in keep} | {f"{a}_official": load(f"{OF}/answers/mh_{a}_official.jsonl") for a in TR}
assert all(set(v) == set(ids) for v in arms.values()) and len(ids) == 882
out["answer882_family_10"] = holm({k: mc(ref, v, ids) | {"correct": sum(v.values())} for k, v in arms.items()})

# 부록 C (values.py appendixF_v1_family8 에서 randrow 2개 제외)
ref1 = load(f"{CAP}/cell.jsonl")
keep6 = ["fulltable", "chunk", "trag_hetero", "rowcol", "rowcol_values", "rowcol_hv33"]
fam6 = holm({k: mc(ref1, load(f"{CAP}/{k}.jsonl"), ids) for k in keep6})
out["appendixC_v1_family6"] = fam6
out["appendixC_max_holm_rowcol_3"] = max((fam6[k]["p_holm"], k) for k in keep6 if k.startswith("rowcol"))

# 이전 값(values.json)과 비교
old = json.loads((ROOT / f"{OF}/replace/values.json").read_text())
cmp = {"retrieval": (old["retrieval"]["mh_doc"]["family_ALL"], out["retrieval_family_ALL_8"]),
       "answer882": (old["answers"]["mh882_final"]["family"], out["answer882_family_10"]),
       "appendixC": (old["answers"]["appendixF_v1_family8"]["family"], fam6)}
out["holm_changed"] = {f: {k: [o[k]["p_holm"], n[k]["p_holm"]] for k in n if abs(o[k]["p_holm"] - n[k]["p_holm"]) > 1e-9 * max(o[k]["p_holm"], 1e-300)}
                       for f, (o, n) in cmp.items()}
out["bc_changed"] = {f: [k for k in n if (o[k]["b"], o[k]["c"]) != (n[k]["b"], n[k]["c"])] for f, (o, n) in cmp.items()}
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps({k: out[k] for k in ("five_baselines_20_cells", "appendixC_max_holm_rowcol_3", "holm_changed", "bc_changed")}, ensure_ascii=False, indent=1))
