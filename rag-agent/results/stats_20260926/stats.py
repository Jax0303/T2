"""2026-09-26 항목 0·3 (리더 생성 없음, 재검색 없음 — 기존 결과와 저장된 임베딩만 읽는다).
0. HiTab 300: s3c 답변(results/s3c_answer_hitab300_20260926/rows.jsonl) 대 표 전체(results/fulltable_20260924/hitab_rows.jsonl)
   — 질의 집합이 같은지, McNemar(정확 이항, b = s3c만 맞힘, c = 표 전체만).
3a. MultiHiertt 셀 쌍 코사인(results/recheck_20260926/analyze.py 와 같은 계산, 같은 임베딩 .cache/recheck_20260926)을
   문서 묶음 1,105개 단위로: 묶음 = scripts/judge_verify.py mh_units 와 같은 키(sha256([문단, 표 HTML])).
   묶음 값 = 묶음에 속한 채점 문항 문서 값의 평균(구성원 간 최대 차이를 함께 적는다). c 대 e 대응 Wilcoxon(scipy 기본).
   그룹 = 전체 / 정답 표 1개 / 2개 이상 — 묶음은 구성원 문항 중 하나라도 그 그룹이면 넣는다.
3b. 리더 EM McNemar + Holm. HiTab 300: 기준 s3c, 비교 = fair_filter 7개 arm(correct_base) + 표 전체.
   MH 882: 기준 cell_uniq(results/mh_arms/cap300_20260924), 비교 = 882행 조건 전부. 묶음(가족)마다 Holm:
   HiTab 전체 / MH 전체(가중 아님, 882 단순) / MH 그룹별.
실행: HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python results/stats_20260926/stats.py
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.argv = sys.argv[:1]
import mh_arms as mh                                                  # noqa: E402


def rows(path, key="correct_base"):
    return {r["query_id"]: int(r[key]) for r in map(json.loads, open(path))}


def mcnemar(ref, x):
    assert set(ref) == set(x)
    b = sum(ref[q] and not x[q] for q in ref)
    c = sum(x[q] and not ref[q] for q in ref)
    return {"n": len(ref), "b": b, "c": c, "p": float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0}


def holm(d):
    order = sorted(d, key=lambda k: d[k]["p"])
    run = 0.0
    for k, name in enumerate(order):
        run = max(run, min(1.0, (len(d) - k) * d[name]["p"]))
        d[name]["p_holm"] = run
    return d


out = {}
# ---- 0
pop = json.load(open(RES / "ksweep_population_300.json"))["query_ids"]
s3c = rows(RES / "s3c_answer_hitab300_20260926/rows.jsonl")
full = rows(RES / "fulltable_20260924/hitab_rows.jsonl")
out["item0"] = {"same_query_set": set(s3c) == set(full) == set(pop), "n_s3c": len(s3c), "n_fulltable": len(full),
                "s3c_correct": sum(s3c.values()), "fulltable_correct": sum(full.values()),
                "both_correct": sum(s3c[q] and full[q] for q in s3c), "both_wrong": sum(not s3c[q] and not full[q] for q in s3c),
                **mcnemar(s3c, full)}

# ---- 3b
ff = defaultdict(dict)
for r in map(json.loads, open(RES / "fair_filter_20260921/rows.jsonl")):
    ff[r["arm"]][r["query_id"]] = int(r["correct_base"])
hit = {("sleaf" if a == "ours" else a): v for a, v in ff.items()}
hit["fulltable"] = full
out["item3b_hitab300"] = holm({a: {"correct": sum(v.values()), **mcnemar(s3c, v)} for a, v in hit.items()})
CAP = RES / "mh_arms/cap300_20260924"
mhrows = {p.stem: {r["query_id"]: r for r in map(json.loads, open(p))} for p in sorted(CAP.glob("*.jsonl"))
          if "." not in p.stem}
mhrows = {k: v for k, v in mhrows.items() if len(v) == 882}
ref = mhrows.pop("cell_uniq")
em = lambda d, qs: {q: int(d[q]["answer_correct"]) for q in qs}
out["item3b_mh882_pooled"] = holm({k: {"correct": sum(em(v, ref).values()), **mcnemar(em(ref, ref), em(v, ref))}
                                   for k, v in mhrows.items()})
layers = sorted({r["layer"] for r in ref.values()})
out["item3b_mh882_by_group"] = {}
for g in layers:
    qs = [q for q in ref if ref[q]["layer"] == g]
    out["item3b_mh882_by_group"][g] = holm({k: {"n_group": len(qs), **mcnemar(em(ref, qs), em(v, qs))}
                                            for k, v in mhrows.items()})
out["sources_3b"] = {"hitab_ref": "results/s3c_answer_hitab300_20260926/rows.jsonl",
                     "hitab_others": ["results/fair_filter_20260921/rows.jsonl (correct_base)",
                                      "results/fulltable_20260924/hitab_rows.jsonl"],
                     "mh": f"results/mh_arms/cap300_20260924/{{cell_uniq,{','.join(mhrows)}}}.jsonl"}
(OUT / "stats.json").write_text(json.dumps(out, indent=1))

# ---- 3a
RUNS = {"c": ("none", "mh_cell_hv2"), "e": ("L1", "mh_cell_hv2_L1")}
queries, docs, _ = mh.load_population("train")
bkey = {u: hashlib.sha256(json.dumps([d[2], d[0]]).encode()).hexdigest() for u, d in docs.items()}
assert len(set(bkey.values())) == 1105
recheck = RES / "recheck_20260926"
idx = {}
for k, (label, stem) in RUNS.items():
    new = json.loads((recheck / f"{stem}.json").read_text())
    tables, hdr = mh.build_tables(docs, "v2", label)
    texts, covers, _, unit_tids, _ = mh.build_corpus("", sorted(tables), "s3c", "cell", {},
                                                     load=lambda tid, _d: tables.get(tid))
    assert mh.digest(texts) == new["corpus_text_sha256"]
    live = {(tid, i, j) for tid, tab in tables.items() for i, row in enumerate(tab.table.data)
            for j, v in enumerate(row) if str(v).strip()}
    gold = {q["uid"]: set(q["gold"]) for q in mh.resolve_gold(queries, tables, hdr, live)
            if not q["excluded"] and q["gold"]}
    key = mh.digest(texts)[:16]
    idx[k] = {"unit_tids": unit_tids, "gold": gold,
              "files": [str(ROOT / new["arguments"]["cache_dir"] / f"cell_{len(texts)}_{key}_{s}.npy")
                        for s in range(0, len(texts), new["arguments"]["shard"])]}
    del tables, texts, covers
assert idx["c"]["unit_tids"] == idx["e"]["unit_tids"] and idx["c"]["gold"] == idx["e"]["gold"]
GOLD = idx["e"]["gold"]
QIDS = sorted(GOLD)
tids = np.array(idx["e"]["unit_tids"])
doc_rows = defaultdict(list)
for n, t in enumerate(tids):
    doc_rows[t.split("::")[0]].append(n)


def pair_means(E, tabs):                          # results/recheck_20260926/analyze.py 와 같은 식
    sq = (E * E).sum(1)
    tot = E.sum(0)
    s_all, n = (tot @ tot - sq.sum()) / 2, len(E)
    s_w = p_w = 0.0
    for t in np.unique(tabs):
        m = tabs == t
        st = E[m].sum(0)
        s_w += (st @ st - sq[m].sum()) / 2
        p_w += m.sum() * (m.sum() - 1) / 2
    p_b = n * (n - 1) / 2 - p_w
    return (s_w / p_w if p_w else np.nan, (s_all - s_w) / p_b if p_b else np.nan)


stats = {}
for k in RUNS:
    emb = np.concatenate([np.load(f) for f in idx[k]["files"]])
    stats[k] = {u: pair_means(emb[doc_rows[u]].astype(np.float64), tids[doc_rows[u]]) for u in QIDS}
    del emb
n_tab = {u: len({g[0] for g in GOLD[u]}) for u in QIDS}
members = defaultdict(list)
for u in QIDS:
    members[bkey[u]].append(u)
spread = {k: max(float(np.nanmax(np.abs(np.array([stats[k][u] for u in m]) - np.array(stats[k][m[0]]))))
                 for m in members.values()) for k in RUNS}
GROUPS = {"all": lambda u: True, "gold_tables_1": lambda u: n_tab[u] == 1, "gold_tables_2plus": lambda u: n_tab[u] >= 2}
part = {"n_bundles_with_scored_queries": len(members), "n_queries": len(QIDS),
        "max_within_bundle_abs_diff": spread}
for gname, f in GROUPS.items():
    bs = [b for b, m in members.items() if any(f(u) for u in m)]
    val = {k: {b: np.nanmean(np.array([stats[k][u] for u in members[b]]), axis=0) for b in bs} for k in RUNS}
    use = [b for b in bs if not np.isnan(val["c"][b]).any()]
    g = {"n_bundles": len(bs), "n_used": len(use), "excluded_nan": len(bs) - len(use)}
    for name, fn in (("same_table", lambda v: v[0]), ("other_table", lambda v: v[1]),
                     ("same_minus_other", lambda v: v[0] - v[1])):
        c = np.array([fn(val["c"][b]) for b in use])
        e = np.array([fn(val["e"][b]) for b in use])
        g[name] = {"c_mean": float(c.mean()), "e_mean": float(e.mean()), "e_minus_c_mean": float((e - c).mean()),
                   "e_gt_c": int((e > c).sum()), "e_lt_c": int((e < c).sum()), "tie": int((e == c).sum()),
                   "wilcoxon_p": float(wilcoxon(c, e).pvalue)}
    part[gname] = g
out["item3a_bundle_wilcoxon"] = part
out["sources_3a"] = {"embeddings": {k: idx[k]["files"][0].replace(str(ROOT) + "/", "") + " …" for k in RUNS},
                     "summaries": [f"results/recheck_20260926/{s}.json" for _l, s in RUNS.values()],
                     "per_query_version": "results/recheck_20260926/analyze.json part3"}
(OUT / "stats.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
