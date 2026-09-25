"""2026-09-26 라벨 벡터 섞기 집계 (리더 없음, 재검색 없음 — records 만).
MultiHiertt: c·e = results/recheck_20260926, mixα = 이 폴더. 정답 표 1개 / 2개 이상 / 전체.
  문서 안 = records 판정. 전체 색인 = recheck_20260926/analyze.py 와 같은 재채점(복사본 문서의 같은 위치 셀도 정답,
  판정 불가는 정답). c 대비 = 같은 질의끼리 c만 맞힘 : 이것만 맞힘, 정확 McNemar(이항) p.
HiTab: 단일 셀 조회 991 (type_accuracy query_type == single_cell), c = s2, e = s3c
  (results/retrieval_accuracy/t_{s2,s3c}_{gold,split}_labelabl), mixα = 이 폴더.
실행: HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python results/labelmix_20260926/analyze.py [mix 폴더]
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
MIX = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.argv = sys.argv[:1]
import mh_arms as mh                                                  # noqa: E402

ALPHAS = ("1.0", "0.9", "0.8", "0.7", "0.6", "0.5")
RC = ROOT / "results/recheck_20260926"


def vs(a, b, qs):
    x = sum(a[u] and not b[u] for u in qs)
    y = sum(b[u] and not a[u] for u in qs)
    return {"c_only": x, "this_only": y, "p_exact": binomtest(x, x + y).pvalue if x + y else 1.0}


# ---- HiTab
hitab = {}
for scope in ("gold", "split"):
    runs = {"c": ROOT / f"results/retrieval_accuracy/t_s2_{scope}_labelabl",
            "e": ROOT / f"results/retrieval_accuracy/t_s3c_{scope}_labelabl",
            **{f"mix{m}": MIX / f"t_s2_{scope}_mix{m}" for m in ALPHAS}}
    ok = {}
    for k, stem in runs.items():
        ok[k] = {r["query_id"]: bool(r["retrieval_success"]) for r in map(json.loads, open(f"{stem}_type_accuracy.jsonl"))
                 if r["query_type"] == "single_cell"}
    qs = sorted(ok["c"])
    assert len(qs) == 991 and all(sorted(v) == qs for v in ok.values())
    hitab[scope] = {k: {"n": len(qs), "correct": sum(v.values()), "records": f"{runs[k]}_type_accuracy.jsonl",
                        **({} if k == "c" else {"vs_c": vs(ok["c"], v, qs)})} for k, v in ok.items()}
    for k in runs:
        if k.startswith("mix"):
            hitab[scope][k]["label_mix"] = json.loads(Path(f"{runs[k]}.json").read_text())["label_mix"]

# ---- MultiHiertt
queries, docs, _ = mh.load_population("train")
idx = {}
for k, label in (("c", "none"), ("e", "L1")):
    tables, hdr = mh.build_tables(docs, "v2", label)
    texts, covers, _, _, _ = mh.build_corpus("", sorted(tables), "s3c", "cell", {}, load=lambda tid, _d: tables.get(tid))
    live = {(tid, i, j) for tid, tab in tables.items() for i, row in enumerate(tab.table.data)
            for j, v in enumerate(row) if str(v).strip()}
    by_text = defaultdict(list)
    for n, s in enumerate(texts):
        by_text[s].append(n)
    idx[k] = {"texts": texts, "cells": [next(iter(c)) for c in covers], "sha": mh.digest(texts),
              "unit_of": {next(iter(c)): n for n, c in enumerate(covers)},
              "gold": {q["uid"]: set(q["gold"]) for q in mh.resolve_gold(queries, tables, hdr, live)
                       if not q["excluded"] and q["gold"]}}
    del tables
assert idx["c"]["gold"] == idx["e"]["gold"]
GOLD = idx["e"]["gold"]
QIDS = sorted(GOLD)
per_doc = defaultdict(list)
for (tid, _i, _j), s in zip(idx["e"]["cells"], idx["e"]["texts"]):
    per_doc[tid.split("::")[0]].append(s)
grp = defaultdict(list)
for u in sorted(docs):
    grp[tuple(per_doc[u])].append(u)
members = {u: set(g) for g in grp.values() for u in g}
n_tab = {u: len({g[0] for g in GOLD[u]}) for u in QIDS}
GROUPS = {"gold_tables_1": [u for u in QIDS if n_tab[u] == 1],
          "gold_tables_2plus": [u for u in QIDS if n_tab[u] >= 2], "all": QIDS}


def corpus_rescored(X, r, u):                    # recheck_20260926/analyze.py 와 같은 규칙
    if r["correct"]:
        return True
    ctx = Counter(r["context"])
    for tid, i, j in GOLD[u]:
        t = tid.split("::")[1]
        eq = [X["unit_of"][c] for u2 in members[u] if (c := (f"{u2}::{t}", i, j)) in X["unit_of"]]
        if ctx[X["texts"][eq[0]]] == 0:
            return False
    return True


runs = {"c": (RC / "mh_cell_hv2", "c"), "e": (RC / "mh_cell_hv2_L1", "e"),
        **{f"mix{m}": (MIX / f"mh_cell_hv2_mix{m}", "c") for m in ALPHAS}}
ok, meta = {}, {}
for k, (stem, texts_of) in runs.items():
    summ = json.loads(Path(f"{stem}.json").read_text())
    assert summ["corpus_text_sha256"] == idx[texts_of]["sha"], k
    recs = {r["query_id"]: r for r in map(json.loads, open(f"{stem}_records.jsonl")) if "excluded" not in r}
    assert sorted(recs) == QIDS, k
    ok[k] = {"doc": {u: bool(recs[u]["doc"]["correct"]) for u in QIDS},
             "corpus_rescored": {u: corpus_rescored(idx[texts_of], recs[u]["corpus"], u) for u in QIDS}}
    meta[k] = {"records": f"{stem}_records.jsonl", "summary": f"{stem}.json", "log": f"{stem}.log",
               "label_mix": summ.get("label_mix")}
    del recs
assert [sum(ok[k]["corpus_rescored"].values()) for k in "ce"] == [801, 675]
assert [sum(ok[k]["doc"].values()) for k in "ce"] == [2444, 2389]
multihiertt = {k: {"meta": meta[k], **{f"{scope}|{g}": {"n": len(qs), "correct": sum(ok[k][scope][u] for u in qs),
                                                        **({} if k == "c" else {"vs_c": vs(ok["c"][scope], ok[k][scope], qs)})}
                                       for scope in ("doc", "corpus_rescored") for g, qs in GROUPS.items()}}
               for k in runs}

out = {"hitab_single_cell_991": hitab, "multihiertt": multihiertt}
(MIX / "analyze.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
