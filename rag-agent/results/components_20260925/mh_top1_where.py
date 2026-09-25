"""2026-09-26 MultiHiertt 전체 색인, c / e 한쪽만 맞힌 문항에서 틀린 쪽 1위 셀의 위치와 라벨 (리더 없음, 재검색 없음).
채점 = mh_overflow_equiv.py 의 재채점(복사본 문서의 같은 위치 셀도 정답, 판정 불가는 정답 처리). 같은 코드로 다시 계산하고
그 결과 파일의 수(c 801, e 675, c만 285, e만 159)와 같은지 확인한다. 복사본 묶음 = e 문장열이 같은 문서끼리(1,105묶음).
1위 셀 = records 의 corpus 문맥 첫 문장. records 는 문장만 담으므로 그 문장을 가진 색인 단위가 여럿이고 그 단위들의
위치(또는 라벨)가 서로 다르면 '판정 불가'로 센다.
위치: 가 = 정답 셀과 같은 표(복사본 문서의 같은 번호 표 포함), 나 = 정답 문서 묶음의 다른 표, 다 = 다른 문서.
라벨 = 라벨 규칙 L1 로 뽑은 표 라벨 문자열(c 쪽 1위 셀도 같은 표 id 의 L1 라벨). 정답 표가 여럿이면 그중 하나와 같으면 '같음'.
실행: HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python results/components_20260925/mh_top1_where.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.argv = sys.argv[:1]
import mh_arms as mh                                                  # noqa: E402

OUT = Path(__file__).parent
RUNS = {"c": ("none", "mh_cell_hv2"), "e": ("L1", "mh_cell_hv2_L1")}
queries, docs, _ = mh.load_population("train")
recs, idx = {}, {}
for k, (label, stem) in RUNS.items():
    cfg = json.loads((ROOT / f"results/mh_arms/{stem}.json").read_text())
    tables, hdr = mh.build_tables(docs, "v2", label)
    texts, covers, _, unit_tids, _ = mh.build_corpus(
        "", sorted(tables), "s3c", "cell", {}, load=lambda tid, _d: tables.get(tid))
    assert mh.digest(texts) == cfg["corpus_text_sha256"], stem
    live = {(tid, i, j) for tid, tab in tables.items() for i, row in enumerate(tab.table.data)
            for j, v in enumerate(row) if str(v).strip()}
    gold = {q["uid"]: set(q["gold"]) for q in mh.resolve_gold(queries, tables, hdr, live)
            if not q["excluded"] and q["gold"]}
    recs[k] = {r["query_id"]: r for r in map(json.loads, open(ROOT / f"results/mh_arms/{stem}_records.jsonl"))
               if "excluded" not in r}
    assert set(gold) == set(recs[k])
    by_text = defaultdict(list)
    for n, s in enumerate(texts):
        by_text[s].append(n)
    idx[k] = {"tables": tables, "texts": texts, "cells": [next(iter(c)) for c in covers],
              "gold": gold, "by_text": by_text, "unit_of": {next(iter(c)): n for n, c in enumerate(covers)}}
assert idx["c"]["gold"] == idx["e"]["gold"] and sorted(idx["c"]["tables"]) == sorted(idx["e"]["tables"])
GOLD, QIDS = idx["e"]["gold"], sorted(recs["e"])
LABEL = {tid: d.title for tid, d in idx["e"]["tables"].items()}

per_doc = defaultdict(list)
for (tid, _i, _j), s in zip(idx["e"]["cells"], idx["e"]["texts"]):
    per_doc[tid.split("::")[0]].append(s)
groups = defaultdict(list)
for u in sorted(docs):
    groups[tuple(per_doc[u])].append(u)
assert len(groups) == 1105
members = {u: set(g) for g in groups.values() for u in g}


def rescore(k):                                   # mh_overflow_equiv.py 와 같은 규칙
    X, out = idx[k], {}
    count = {s: len(v) for s, v in X["by_text"].items()}
    for u in QIDS:
        r = recs[k][u]["corpus"]
        if r["correct"]:
            out[u] = "yes"
            continue
        ctx, st = Counter(r["context"]), []
        for tid, i, j in GOLD[u]:
            t = tid.split("::")[1]
            eq = [X["unit_of"][c] for u2 in members[u] if (c := (f"{u2}::{t}", i, j)) in X["unit_of"]]
            txt = X["texts"][eq[0]]
            other = count[txt] - len(eq)
            st.append("yes" if ctx[txt] > other else "no" if ctx[txt] == 0 else "amb")
        out[u] = "no" if "no" in st else "yes" if all(s == "yes" for s in st) else "amb"
    return {u: v != "no" for u, v in out.items()}      # 판정 불가 = 정답


ok = {k: rescore(k) for k in RUNS}
assert (sum(ok["c"].values()), sum(ok["e"].values())) == (801, 675)
sets = {"c_right_e_wrong": ("e", [u for u in QIDS if ok["c"][u] and not ok["e"][u]]),
        "c_wrong_e_right": ("c", [u for u in QIDS if ok["e"][u] and not ok["c"][u]])}
assert [len(v[1]) for v in sets.values()] == [285, 159]


def where(u, cell):
    tid, i, j = cell
    uid, t = tid.split("::")
    gold_pos = {(g[0].split("::")[1], g[1], g[2]) for g in GOLD[u]}
    if uid in members[u] and t in {p[0] for p in gold_pos}:
        return "ga_gold_cell_or_copy" if (t, i, j) in gold_pos else "ga_same_table"
    return "na_gold_doc_other_table" if uid in members[u] else "da_other_doc"


def label_status(u, tid):
    top, gold = LABEL[tid], {LABEL[g[0]] for g in GOLD[u]}
    nonempty = gold - {""}
    if not top:
        return "both_empty" if not nonempty else "top1_empty"
    if not nonempty:
        return "gold_empty"
    return "same" if top in nonempty else "different"


result, rows = {}, []
for name, (k, qs) in sets.items():
    X = idx[k]
    w, lab, cross = Counter(), Counter(), Counter()
    for u in qs:
        cand = X["by_text"][recs[k][u]["corpus"]["context"][0]]
        ws = {where(u, X["cells"][n]) for n in cand}
        ls = {label_status(u, X["cells"][n][0]) for n in cand}
        wv = ws.pop() if len(ws) == 1 else "undetermined"
        lv = ls.pop() if len(ls) == 1 else "undetermined"
        w[wv] += 1
        lab[lv] += 1
        cross[f"{wv}|{lv}"] += 1
        rows.append({"set": name, "query_id": u, "run": k, "layer": recs[k][u]["layer"],
                     "n_units_with_top1_text": len(cand), "where": wv, "label": lv})
    result[name] = {"run_whose_top1": k, "n": len(qs), "where": dict(w), "label": dict(lab),
                    "where_x_label": dict(cross)}
out = {"records": {k: f"results/mh_arms/{s}_records.jsonl" for k, (_l, s) in RUNS.items()}, **result}
(OUT / "mh_top1_where.json").write_text(json.dumps(out, indent=1))
with open(OUT / "mh_top1_where_rows.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(json.dumps(out, indent=1))
