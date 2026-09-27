#!/usr/bin/env python3
"""TableRAG 셀 검색 재구현: 원본 숫자 열 판정(official) 대 원고 규칙(infer) — 2026-09-28 (PREREG-2026-09-28-tablerag-official-dtype.md).

재검색 없음. run.sh 출력과 results/rerun_20260926 records 만 읽는다.
- official 대 infer: 같은 질의끼리 정확 McNemar (b = official 만 맞힘, c = infer 만 맞힘).
- official 대 본 방법 s3c: 같은 질의끼리 정확 McNemar (b = s3c 만 맞힘, c = official 만 맞힘) — 원고 표와 같은 방향.
HiTab = test, 질문의 표 안(gold)은 세 유형, 538표(split)는 단일 셀 조회. MultiHiertt = train 문서 안 네 그룹과 전체.

  .venv/bin/python results/tablerag_official_20260928/compare.py
"""
import json
from pathlib import Path

from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
OLD = HERE.parents[0] / "rerun_20260926"


def hitab(stem):
    """{유형: {query_id: 0/1}} — 원고 표 5-2 의 유형 규칙(mode all, aggregation, m)."""
    out = {"single_cell": {}, "multi_cell": {}, "arithmetic": {}}
    for r in map(json.loads, open(f"{stem}_records.jsonl")):
        if "correct" not in r or r["mode"] != "all":
            continue
        t = ("arithmetic" if (r.get("aggregation") or "none") != "none"
             else "single_cell" if r.get("m") == 1 else "multi_cell")
        out[t][r["query_id"]] = int(r["correct"])
    return out


def mh(stem):
    out = {"ALL": {}}
    for r in map(json.loads, open(f"{stem}_records.jsonl")):
        if "doc" in r:
            out.setdefault(r["layer"], {})[r["query_id"]] = out["ALL"][r["query_id"]] = int(r["doc"]["correct"])
    return out


def mc(a, b):
    """a 만 맞힘 : b 만 맞힘, p. 두 dict 의 질의 집합이 같아야 한다."""
    assert a.keys() == b.keys(), "질의 집합이 다르다"
    x = sum(a[q] and not b[q] for q in a)
    y = sum(b[q] and not a[q] for q in a)
    return {"b": x, "c": y, "p": binomtest(min(x, y), x + y, 0.5).pvalue if x + y else 1.0}


res = {}
for scope, types in (("gold", ("single_cell", "multi_cell", "arithmetic")), ("split", ("single_cell",))):
    ours = hitab(OLD / f"hitab/hitab_test_{scope}_s3c")
    for m in ("leaf", "path"):
        new = hitab(HERE / f"hitab/hitab_test_{scope}_tablerag_{m}_official")
        old = hitab(OLD / f"hitab/hitab_test_{scope}_tablerag_{m}")
        for t in types:
            n = len(new[t])
            res[f"hitab {scope} {m} {t}"] = {
                "n": n, "official": sum(new[t].values()), "official_acc": round(sum(new[t].values()) / n, 4),
                "infer": sum(old[t].values()), "infer_acc": round(sum(old[t].values()) / n, 4),
                "official_vs_infer": mc(new[t], old[t]), "s3c_vs_official": mc(ours[t], new[t])}
ours = mh(OLD / "mh/mh_train_s3c")
for m in ("leaf", "path"):
    new, old = mh(HERE / f"mh/mh_train_tablerag_{m}_official"), mh(OLD / f"mh/mh_train_tablerag_{m}")
    for g in ("lookup_m1", "lookup_m2+", "arith_m1", "arith_m2+", "ALL"):
        n = len(new[g])
        res[f"mh doc {m} {g}"] = {
            "n": n, "official": sum(new[g].values()), "official_acc": round(sum(new[g].values()) / n, 4),
            "infer": sum(old[g].values()), "infer_acc": round(sum(old[g].values()) / n, 4),
            "official_vs_infer": mc(new[g], old[g]), "s3c_vs_official": mc(ours[g], new[g])}
for k, v in res.items():
    print(f"{k:32s} n={v['n']:5d}  official {v['official_acc']:.4f}  infer {v['infer_acc']:.4f}  "
          f"off:inf {v['official_vs_infer']['b']}:{v['official_vs_infer']['c']} p={v['official_vs_infer']['p']:.2g}  "
          f"s3c:off {v['s3c_vs_official']['b']}:{v['s3c_vs_official']['c']} p={v['s3c_vs_official']['p']:.2g}")
(HERE / "compare.json").write_text(json.dumps(res, indent=1))
