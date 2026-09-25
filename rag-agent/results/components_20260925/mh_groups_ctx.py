"""2026-09-26 MultiHiertt c / e (리더 없음, 재검색 없음 — 기존 records 만).
채점 = mh_overflow_equiv.py 와 같은 재채점(전체 색인: 복사본 문서의 같은 위치 셀도 정답, 판정 불가는 정답).
문서 안 검색은 문맥에 복사본 문서가 들어올 수 없어 records 의 판정 그대로 쓴다.
1. 정답 셀이 속한 표 수(1개 / 2개 이상)별 문항 수, c·e 검색 정확도, McNemar(b = c만 맞힘, c = e만 맞힘, 정확 p).
2. 20셀 문맥의 서로 다른 표 수·문서 수. records 는 문장만 담으므로 문장 -> 색인 셀로 되돌린다.
   문서 = 복사본 묶음(1,105), 표 = (묶음, 표 번호). 한 문장이 서로 다른 (묶음, 표) 여럿에 있으면 그 문항은 '판정 불가'로
   빼고 따로 센다. 문서 안 검색은 질문 문서의 셀로만 되돌린다(문서 수는 1).
3. c 정답·e 오답 285건: e 전체 색인 문맥에서 빠진 정답 셀(재채점 판정 'no')이 문맥에 들어간(yes/판정 불가) 다른 정답 셀과
   같은 표인지.
4. 문맥 순서 = 점수 순서: records 에 점수가 없어 직접 비교는 못 한다. records 만으로 되는 두 검사 —
   (i) 같은 문장(같은 점수)이 문맥 안에서 붙어 있는가, (ii) 전체 색인 문맥 중 질문 문서에 있는 문장들의 순서가
   문서 안 검색 문맥(같은 점수 sc 를 문서로 좁혀 정렬, scripts/mh_arms.py)의 앞부분과 같은가.
실행: HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python results/components_20260925/mh_groups_ctx.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

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
    idx[k] = {"texts": texts, "cells": [next(iter(c)) for c in covers], "gold": gold, "by_text": by_text,
              "unit_of": {next(iter(c)): n for n, c in enumerate(covers)}}
    del tables
assert idx["c"]["gold"] == idx["e"]["gold"]
GOLD, QIDS = idx["e"]["gold"], sorted(recs["e"])

per_doc = defaultdict(list)
for (tid, _i, _j), s in zip(idx["e"]["cells"], idx["e"]["texts"]):
    per_doc[tid.split("::")[0]].append(s)
grp = defaultdict(list)
for u in sorted(docs):
    grp[tuple(per_doc[u])].append(u)
assert len(grp) == 1105
members = {u: set(g) for g in grp.values() for u in g}
gid = {u: min(g) for g in grp.values() for u in g}             # 묶음 대표 = 사전순 첫 uid

# 정답 표 수: 해석된 gold 셀 기준. records 의 n_gold_tables(원 좌표 기준)와 같은지 확인
n_tab = {u: len({g[0] for g in GOLD[u]}) for u in QIDS}
assert all(n_tab[u] == recs["e"][u]["n_gold_tables"] == recs["c"][u]["n_gold_tables"] for u in QIDS)
GROUPS = {"all": QIDS, "gold_tables_1": [u for u in QIDS if n_tab[u] == 1],
          "gold_tables_2plus": [u for u in QIDS if n_tab[u] >= 2]}


def cell_status(k, u):                            # mh_overflow_equiv.py 와 같은 규칙, 셀마다
    X, r = idx[k], recs[k][u]["corpus"]
    if r["correct"]:
        return {g: "yes" for g in GOLD[u]}
    ctx, st = Counter(r["context"]), {}
    for g in GOLD[u]:
        tid, i, j = g
        t = tid.split("::")[1]
        eq = [X["unit_of"][c] for u2 in members[u] if (c := (f"{u2}::{t}", i, j)) in X["unit_of"]]
        txt = X["texts"][eq[0]]
        other = len(X["by_text"][txt]) - len(eq)
        st[g] = "yes" if ctx[txt] > other else "no" if ctx[txt] == 0 else "amb"
    return st


status = {k: {u: cell_status(k, u) for u in QIDS} for k in RUNS}
ok = {"corpus": {k: {u: "no" not in status[k][u].values() for u in QIDS} for k in RUNS},
      "doc": {k: {u: bool(recs[k][u]["doc"]["correct"]) for u in QIDS} for k in RUNS}}
assert (sum(ok["corpus"]["c"].values()), sum(ok["corpus"]["e"].values())) == (801, 675)

# ---- 1
part1 = {}
for scope in ("corpus", "doc"):
    for gname, qs in GROUPS.items():
        a, b = ok[scope]["c"], ok[scope]["e"]
        x = sum(a[u] and not b[u] for u in qs)
        y = sum(b[u] and not a[u] for u in qs)
        part1[f"{scope}|{gname}"] = {"n": len(qs), "c_correct": sum(a[u] for u in qs),
                                     "e_correct": sum(b[u] for u in qs), "b_c_only": x, "c_e_only": y,
                                     "p_exact": binomtest(x, x + y).pvalue if x + y else 1.0}

# ---- 2
def ctx_counts(k, u, scope):
    X = idx[k]
    keys_t, keys_d = [], []
    for s in recs[k][u][scope]["context"]:
        cand = X["by_text"][s]
        if scope == "doc":
            cand = [n for n in cand if X["cells"][n][0].split("::")[0] == u]
        kt = {(gid[X["cells"][n][0].split("::")[0]], X["cells"][n][0].split("::")[1]) for n in cand}
        if len(kt) != 1:
            return None
        keys_t.append(kt.pop())
        keys_d.append(keys_t[-1][0])
    return len(set(keys_t)), len(set(keys_d))


part2 = {}
for k in RUNS:
    for scope in ("corpus", "doc"):
        cnt = {u: ctx_counts(k, u, scope) for u in QIDS}
        for gname, qs in GROUPS.items():
            det = [cnt[u] for u in qs if cnt[u] is not None]
            t, d = np.array([x[0] for x in det]), np.array([x[1] for x in det])
            part2[f"{k}|{scope}|{gname}"] = {
                "n": len(qs), "determined": len(det), "undetermined": len(qs) - len(det),
                "tables_median": float(np.median(t)), "tables_mean": float(t.mean()),
                "docs_median": float(np.median(d)), "docs_mean": float(d.mean())}

# ---- 3
q285 = [u for u in QIDS if ok["corpus"]["c"][u] and not ok["corpus"]["e"][u]]
assert len(q285) == 285
cells3, queries3 = Counter(), Counter()
for u in q285:
    st = status["e"][u]
    present = {g[0].split("::")[1] for g, v in st.items() if v != "no"}
    kinds = []
    for g, v in st.items():
        if v != "no":
            continue
        t = g[0].split("::")[1]
        kinds.append("ga_same_table_as_present_gold" if t in present else
                     "na_other_table__some_gold_present" if present else "na_other_table__no_gold_present")
    cells3.update(kinds)
    queries3["+".join(sorted(set(kinds)))] += 1
part3 = {"n_queries": len(q285), "missing_cells": dict(cells3), "n_missing_cells": sum(cells3.values()),
         "queries_by_kinds": dict(queries3),
         "m_of_queries": dict(Counter("1" if len(GOLD[u]) == 1 else "2+" for u in q285))}

# ---- 4
def contiguous(ctx):
    seen, prev = set(), None
    for s in ctx:
        if s != prev and s in seen:
            return False
        seen.add(s)
        prev = s
    return True


def dedupe(xs):
    return list(dict.fromkeys(xs))


part4 = {}
for k in RUNS:
    own = defaultdict(set)
    for (tid, _i, _j), s in zip(idx[k]["cells"], idx[k]["texts"]):
        own[tid.split("::")[0]].add(s)
    part4[k] = {f"same_text_contiguous_{scope}": sum(contiguous(recs[k][u][scope]["context"]) for u in QIDS)
                for scope in ("corpus", "doc")}
    agree, used = 0, 0
    for u in QIDS:
        cd = dedupe(s for s in recs[k][u]["corpus"]["context"] if s in own[u])
        dd = dedupe(recs[k][u]["doc"]["context"])
        used += bool(cd)
        agree += cd == dd[:len(cd)]
    part4[k].update({"n_queries": len(QIDS), "corpus_own_doc_order_is_prefix_of_doc_order": agree,
                     "queries_with_own_doc_text_in_corpus_context": used})

out = {"records": {k: f"results/mh_arms/{s}_records.jsonl" for k, (_l, s) in RUNS.items()},
       "part1": part1, "part2": part2, "part3": part3, "part4": part4}
(OUT / "mh_groups_ctx.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
