# SPDX-License-Identifier: MIT
"""2026-09-13 감사 — 표 세 개를 원본 레코드에서 다시 만든다.

  1) 검색-EM 간극 분해 (HiTab): 검색 정확도 / 검색문맥 EM / gold문맥 EM
  2) 검색 범위 사다리: 표 하나 / 538표 / 3,597표
  3) arm 사이 짝지음 McNemar (검색, EM)

요약 JSON 을 읽지 않고 레코드에서 센다 — 요약과 레코드가 어긋난 전과가 있다.

  PYTHONPATH=. .venv/bin/python analysis/audit_2026_09_13.py
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results/evaluation_v2"
ARMS = ["s3c", "mt2net", "chunk1000", "huawei_char", "rowcol"]


def rows(path):
    return [json.loads(l) for l in Path(path).open(encoding="utf-8")]


def stratum(r):
    kind = "조회" if (r.get("aggregation") or "none") == "none" else "산술"
    return f"{kind} m{'1' if r['m'] == 1 else '2+'}"


def gap_table():
    ret = {r["query_id"]: r for r in rows(V2 / "s3c_v2_records.jsonl") if "correct" in r}
    got = {r["query_id"]: r for r in rows(V2 / "s3c_v2_answer_retrieved.jsonl")}
    gold = {r["query_id"]: r for r in rows(V2 / "s3c_v2_answer_gold_modeall.jsonl")}
    cells = {}
    for qid, g in gold.items():
        r = ret[qid]
        k = stratum(r)
        c = cells.setdefault(k, [0, 0, 0, 0])
        c[0] += 1
        c[1] += r["correct"]
        c[2] += got[qid]["answer_correct"]
        c[3] += g["answer_correct"]
    print("## 표 1 — 검색-EM 간극 분해 (HiTab test, mode=all, 본 방법 s3c)\n")
    print("| 층 | query count | 검색 정확도 | EM(검색 문맥) | EM(gold 문맥만) | distractor 비용 | 리더 천장까지 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    tot = [0, 0, 0, 0]
    for k in sorted(cells):
        n, h, e, gc = cells[k]
        tot = [a + b for a, b in zip(tot, cells[k])]
        print(f"| {k} | {n} | {h/n:.3f} | {e/n:.3f} | {gc/n:.3f} | {(gc-e)/n:+.3f} | {1-gc/n:+.3f} |")
    n, h, e, gc = tot
    print(f"| **합계** | {n} | **{h/n:.3f}** | **{e/n:.3f}** | **{gc/n:.3f}** | **{(gc-e)/n:+.3f}** | **{1-gc/n:+.3f}** |")
    print("\ndistractor 비용 = gold 셀만 줬을 때와 검색 문맥을 줬을 때의 EM 차이.")
    print("리더 천장까지 = gold 셀만 줘도 못 맞히는 몫 (4bit 7B 의 산술 한계).\n")


def acc(path):
    rs = [r for r in rows(path) if "correct" in r]
    return {r["query_id"]: r["correct"] for r in rs}


def scope_ladder():
    cand = {
        "표 1개 (자기 범위)": {"s3c": "results/tablerag_fair/g1_s3c_records.jsonl",
                              "mt2net": "results/scope_fair/g_mt2net_records.jsonl",
                              "rowcol": "results/scope_fair/g_rowcol_values_records.jsonl",
                              "chunk1000": "results/scope_fair/g_chunk1000_records.jsonl",
                              "huawei_char": "results/scope_fair/g_trag_hetero_records.jsonl",
                              "tablerag_leaf": "results/tablerag_fair/g2_trag_leaf_records.jsonl",
                              "tablerag_allobj": "results/scope_fair/trag_allobj_leaf_gold_records.jsonl"},
        "538표 (이 분할)": {a: f"results/evaluation_v2/{a}_v2_records.jsonl" for a in ARMS}
        | {"tablerag_leaf": "results/audit_fix_20260910/t_tablerag_leaf_fix_records.jsonl",
           "tablerag_allobj": "results/scope_fair/trag_allobj_leaf_split_records.jsonl"},
        "3,597표 (저장소 전체)": {"s3c": "results/retrieval_accuracy/full_s3c_hybrid_records.jsonl",
                                 "mt2net": "results/retrieval_accuracy/full_mt2net_hybrid_records.jsonl"},
    }
    print("## 표 2 — 검색 범위 사다리 (같은 계측기, 같은 예산 20셀)\n")
    names = ["s3c", "mt2net", "chunk1000", "huawei_char", "rowcol",
             "tablerag_leaf", "tablerag_allobj"]
    print("| 범위 | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 1) + "|")
    for scope, paths in cand.items():
        cells = []
        for a in names:
            p = ROOT / paths[a] if a in paths else None
            if p is None or not p.exists():
                cells.append("—")
                continue
            v = acc(p)
            cells.append(f"{sum(v.values())/len(v):.4f}")
        print(f"| {scope} | " + " | ".join(cells) + " |")
    print()


def mcnemar_block(title, tbl, arms):
    ids = sorted(set.intersection(*[set(tbl[a]) for a in arms]))
    print(f"### {title} (짝지음 n={len(ids)})\n")
    print("| A | B | A | B | 차이 | A승 | B승 | p |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for a, b in itertools.combinations(arms, 2):
        n1 = sum(tbl[a][i] for i in ids) / len(ids)
        n2 = sum(tbl[b][i] for i in ids) / len(ids)
        w = sum(1 for i in ids if tbl[a][i] and not tbl[b][i])
        l = sum(1 for i in ids if not tbl[a][i] and tbl[b][i])
        p = binomtest(w, w + l, 0.5).pvalue if w + l else 1.0
        print(f"| {a} | {b} | {n1:.4f} | {n2:.4f} | {n1-n2:+.4f} | {w} | {l} | {p:.3g} |")
    print()


def significance():
    ret = {a: acc(V2 / f"{a}_v2_records.jsonl") for a in ARMS}
    em = {a: {r["query_id"]: r["answer_correct"]
              for r in rows(V2 / f"{a}_v2_answer_retrieved.jsonl")} for a in ARMS}
    print("## 표 3 — arm 사이 차이가 우연인가 (McNemar, Holm 미적용 원값)\n")
    mcnemar_block("검색 정확도", ret, ARMS)
    mcnemar_block("답변 EM", em, ARMS)


MH = ROOT / "results/mh_arms"
MH_ARMS = [("cell", "본 방법 s3c"), ("mt2net", "MT2Net(원문 문장)"),
           ("chunk", "generic chunk"), ("huawei", "Huawei TableRAG"),
           ("rowcol", "RowCol"), ("tablerag", "TableRAG 셀/스키마")]
LAYERS = ["lookup_m1", "lookup_m2+", "arith_m1", "arith_m2+", "ALL"]


def mh_tables():
    have = [(t, n) for t, n in MH_ARMS if (MH / f"mh_{t}.json").exists()]
    if not have:
        return
    print("## 표 4 — MultiHiertt 검색 정확도 (train, 예산 20셀, 주지표 all)\n")
    for scope in ("doc", "corpus"):
        print(f"### 범위 = {scope}" + (" (MultiHiertt 의 과제 정의)" if scope == "doc"
                                      else " (문서 전부를 한 색인에)") + "\n")
        print("| arm | " + " | ".join(LAYERS) + " |")
        print("|---" * (len(LAYERS) + 1) + "|")
        for tag, name in have:
            d = json.loads((MH / f"mh_{tag}.json").read_text())["by_layer"]
            cells = [f"{d[k][scope]['accuracy_all']:.4f}" if k in d else "—" for k in LAYERS]
            print(f"| {name} | " + " | ".join(cells) + " |")
        print()
    ans = [(t, n) for t, n in have if (MH / f"mh_{t}_answer_doc.json").exists()]
    if not ans:
        return
    print("## 표 5 — MultiHiertt 답변 EM (doc 범위, 같은 리더·같은 질의)\n")
    print("| arm | " + " | ".join(f"{k} 검색/EM" for k in LAYERS) + " |")
    print("|---" * (len(LAYERS) + 1) + "|")
    for tag, name in ans:
        d = json.loads((MH / f"mh_{tag}_answer_doc.json").read_text())["by_layer"]
        cells = [(f"{d[k]['retrieval_accuracy_here']:.3f} / {d[k]['answer_em']:.3f}"
                  if k in d else "—") for k in LAYERS]
        print(f"| {name} | " + " | ".join(cells) + " |")
    g = MH / "mh_GOLD_doc.json"
    if g.exists():
        d = json.loads(g.read_text())["by_layer"]
        cells = [f"— / {d[k]['answer_em']:.3f}" if k in d else "—" for k in LAYERS]
        print("| **gold 셀만(리더 천장)** | " + " | ".join(cells) + " |")
    print()


if __name__ == "__main__":
    gap_table()
    scope_ladder()
    significance()
    mh_tables()
