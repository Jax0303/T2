"""리더 입력 형식 실험 집계 (PREREG-2026-09-27-reader-format.md). 실행이 끝난 뒤에만 돌린다.

  python3 results/reader_format_20260927/analyze.py dev    -> dev.json
  python3 results/reader_format_20260927/analyze.py test   -> test.json

조건마다 문항 수, 맞힘, 전달 셀 평균, 리더 입력 토큰 평균. 쌍마다 정확 McNemar(양측 이항, b = 앞 조건만 맞힘,
c = 뒤 조건만 맞힘). test 는 데이터셋마다 비교 4개를 한 묶음으로 Holm. MultiHiertt 그룹별 값은 탐색적.
"""
import json
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
GROUPS = ["lookup_m1", "lookup_m2+", "arith_m1", "arith_m2+"]


def jl(p):
    return [json.loads(l) for l in open(ROOT / p, encoding="utf-8") if l.strip()]


def p_exact(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def mh_rows(p):
    return {x["query_id"]: {"ok": x["answer_correct"], "cells": x["cells_in_context"], "tok": x["n_tok"],
                            "layer": x["layer"]} for x in jl(p)}


def hitab_rows(p, arm=None, cells_from=None):
    """HiTab 답변 행. 전달 셀은 행의 cells_delivered, 없으면 검색 레코드(cells_from)의 cells_in_context."""
    rows = [x for x in jl(p) if arm is None or x.get("arm") == arm]
    rec = {x["query_id"]: x.get("cells_in_context") for x in jl(cells_from)} if cells_from else {}
    return {x["query_id"]: {"ok": x["correct_base"], "tok": x.get("reader_input_tokens"),
                            "cells": x.get("cells_delivered", rec.get(x["query_id"]))} for x in rows}


def summary(d, ids):
    cells = [d[q]["cells"] for q in ids if d[q]["cells"] is not None]
    toks = [d[q]["tok"] for q in ids if d[q]["tok"] is not None]
    return {"n": len(ids), "correct": sum(d[q]["ok"] for q in ids),
            "cells_delivered_mean": round(sum(cells) / len(cells), 2) if cells else "검색 없음",
            "input_tokens_mean": round(sum(toks) / len(toks), 1) if toks else None}


def mcnemar(a, b, ids):
    x = sum(1 for q in ids if a[q]["ok"] and not b[q]["ok"])
    y = sum(1 for q in ids if b[q]["ok"] and not a[q]["ok"])
    return {"b": x, "c": y, "p": p_exact(x, y)}


def holm(pairs):
    order = sorted(pairs, key=lambda k: pairs[k]["p"])
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * pairs[k]["p"]))
        pairs[k]["p_holm"] = run


def block(conds, pairs, family_holm, groups):
    ids = sorted(next(iter(conds.values())))
    assert all(sorted(v) == ids for v in conds.values()), "조건마다 문항 집합이 다르다"
    out = {"conditions": {k: summary(v, ids) for k, v in conds.items()},
           "pairs": {f"{a}_vs_{b}": mcnemar(conds[a], conds[b], ids) for a, b in pairs}}
    if family_holm:
        holm(out["pairs"])
    if groups:
        lay = next(iter(conds.values()))
        out["groups_exploratory"] = {}
        for g in GROUPS:
            gi = [q for q in ids if lay[q]["layer"] == g]
            out["groups_exploratory"][g] = {"conditions": {k: summary(v, gi) for k, v in conds.items()},
                                            "pairs": {f"{a}_vs_{b}": mcnemar(conds[a], conds[b], gi) for a, b in pairs}}
    return out


D = "results/reader_format_20260927"
mode = sys.argv[1]
if mode == "dev":
    res = {"multihiertt_dev_primary": block({"rowexp": mh_rows(f"{D}/dev/mh_rowexp.jsonl"),
                                             "cell": mh_rows(f"{D}/dev/mh_cell.jsonl")},
                                            [("rowexp", "cell")], False, True),
           "hitab_dev": block({"rowexp": hitab_rows(f"{D}/dev/hitab_rowexp.jsonl"),
                               "cell": hitab_rows(f"{D}/dev/hitab_cell.jsonl")},
                              [("rowexp", "cell")], False, False)}
elif mode == "test":
    CAP = "results/mh_arms/cap300_20260924"
    TEST_PAIRS = [("rowexp", "cell"), ("rowexp", "fulltable"), ("rowexp", "chunk"), ("cell", "chunk")]
    res = {"multihiertt_test": block({"rowexp": mh_rows(f"{D}/test/mh_rowexp.jsonl"),
                                      "cell": mh_rows(f"{CAP}/cell_uniq.jsonl"),
                                      "fulltable": mh_rows(f"{CAP}/fulltable.jsonl"),
                                      "chunk": mh_rows(f"{D}/test/mh_chunk_final.jsonl")}, TEST_PAIRS, True, True),
           "hitab_test": block({"rowexp": hitab_rows(f"{D}/test/hitab_rowexp.jsonl"),
                                "cell": hitab_rows("results/s3c_answer_hitab300_20260926/rows.jsonl",
                                                   cells_from="results/retrieval_accuracy/t_s3c_gold_labelabl_records.jsonl"),
                                "fulltable": hitab_rows("results/fulltable_20260924/hitab_rows.jsonl"),
                                "chunk": hitab_rows("results/fair_filter_20260921/rows.jsonl", arm="chunk",
                                                    cells_from="results/retrieval_accuracy/t_chunk_s3c_gold_v2_records.jsonl")},
                               TEST_PAIRS, True, False)}
else:
    raise SystemExit("dev 또는 test")
(HERE / f"{mode}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
