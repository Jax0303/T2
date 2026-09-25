"""2026-09-26 MultiHiertt c·e 재실행 분석 (리더 없음, 재검색 없음 — records 와 저장된 임베딩만).
c = 라벨 없음, e = L1 라벨 (머리글 v2 · s3c · bge-base · α .7 · 20셀).
2. 기존 records(results/mh_arms/mh_cell_hv2{,_L1}_records.jsonl) 대 재실행 records(이 폴더): 맞힌 문항 수, 불일치 문항 수.
   전체 색인 = components_20260925/mh_groups_ctx.py 와 같은 재채점(복사본 문서의 같은 위치 셀도 정답, 판정 불가는 정답).
   records 판정(재채점 전)도 같이 적는다. 문서 안 = records 판정.
3. 재실행이 저장한 셀 임베딩(요약 JSON arguments.cache_dir)으로 질문 문서마다 셀 쌍 코사인 평균:
   (가) 같은 표의 셀 쌍, (나) 다른 표의 셀 쌍, (가)−(나). 쌍 = 문서 안 서로 다른 두 셀 전부(i<j).
   표가 1개인 문서는 (나)가 없어 빼고 수를 적는다. c 대 e 대응 Wilcoxon(scipy 기본: 양측, 차이 0 쌍 제외).
   그룹: 전체 / 정답 표 1개 / 정답 표 2개 이상.
실행: HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python results/recheck_20260926/analyze.py [재실행 폴더]
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
NEW = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.argv = sys.argv[:1]
import mh_arms as mh                                                  # noqa: E402

OUT = NEW
RUNS = {"c": ("none", "mh_cell_hv2"), "e": ("L1", "mh_cell_hv2_L1")}
N_CELLS = 425870


def load_recs(path):
    keep = {}
    for r in map(json.loads, open(path)):
        if "excluded" not in r:
            keep[r["query_id"]] = {"corpus": {"correct": r["corpus"]["correct"], "context": r["corpus"]["context"]},
                                   "doc": r["doc"]["correct"]}
    return keep


queries, docs, _ = mh.load_population("train")
idx, recs, src = {}, {}, {}
for k, (label, stem) in RUNS.items():
    old = json.loads((ROOT / f"results/mh_arms/{stem}.json").read_text())
    new = json.loads((NEW / f"{stem}.json").read_text())
    tables, hdr = mh.build_tables(docs, "v2", label)
    texts, covers, _, unit_tids, _ = mh.build_corpus(
        "", sorted(tables), "s3c", "cell", {}, load=lambda tid, _d: tables.get(tid))
    assert len(texts) == N_CELLS == old["n_units"] == new["n_units"], stem
    assert mh.digest(texts) == old["corpus_text_sha256"] == new["corpus_text_sha256"], stem
    live = {(tid, i, j) for tid, tab in tables.items() for i, row in enumerate(tab.table.data)
            for j, v in enumerate(row) if str(v).strip()}
    gold = {q["uid"]: set(q["gold"]) for q in mh.resolve_gold(queries, tables, hdr, live)
            if not q["excluded"] and q["gold"]}
    src[k] = {"old_records": f"results/mh_arms/{stem}_records.jsonl", "new_records": str(NEW / f"{stem}_records.jsonl"),
              "new_summary": str(NEW / f"{stem}.json"), "new_log": str(NEW / f"{stem}.log"),
              "embedding_input_audit_n_overflow": new["embedding_input_audit"]["documents"]["n_overflow"]}
    recs[k] = {"old": load_recs(ROOT / src[k]["old_records"]), "new": load_recs(NEW / f"{stem}_records.jsonl")}
    assert set(gold) == set(recs[k]["old"]) == set(recs[k]["new"]), stem
    by_text = defaultdict(list)
    for n, s in enumerate(texts):
        by_text[s].append(n)
    key = mh.digest(texts)[:16]
    idx[k] = {"texts": texts, "cells": [next(iter(c)) for c in covers], "gold": gold, "by_text": by_text,
              "unit_of": {next(iter(c)): n for n, c in enumerate(covers)}, "unit_tids": unit_tids,
              "emb_files": [str(Path(new["arguments"]["cache_dir"]) / f"cell_{len(texts)}_{key}_{s}.npy")
                            for s in range(0, len(texts), new["arguments"]["shard"])]}
    del tables
assert idx["c"]["gold"] == idx["e"]["gold"] and idx["c"]["unit_tids"] == idx["e"]["unit_tids"]
GOLD, QIDS = idx["e"]["gold"], sorted(recs["e"]["old"])

# 복사본 묶음 — mh_groups_ctx.py 와 같다
per_doc = defaultdict(list)
for (tid, _i, _j), s in zip(idx["e"]["cells"], idx["e"]["texts"]):
    per_doc[tid.split("::")[0]].append(s)
grp = defaultdict(list)
for u in sorted(docs):
    grp[tuple(per_doc[u])].append(u)
members = {u: set(g) for g in grp.values() for u in g}
n_tab = {u: len({g[0] for g in GOLD[u]}) for u in QIDS}
GROUPS = {"all": QIDS, "gold_tables_1": [u for u in QIDS if n_tab[u] == 1],
          "gold_tables_2plus": [u for u in QIDS if n_tab[u] >= 2]}


def corpus_rescored(k, r, u):                     # mh_groups_ctx.py cell_status 와 같은 규칙
    X = idx[k]
    if r["correct"]:
        return True
    ctx = Counter(r["context"])
    for g in GOLD[u]:
        tid, i, j = g
        t = tid.split("::")[1]
        eq = [X["unit_of"][c] for u2 in members[u] if (c := (f"{u2}::{t}", i, j)) in X["unit_of"]]
        if ctx[X["texts"][eq[0]]] == 0:
            return False
    return True


# ---- 2
ok = {k: {v: {"corpus_rescored": {u: corpus_rescored(k, recs[k][v][u]["corpus"], u) for u in QIDS},
              "corpus_recorded": {u: bool(recs[k][v][u]["corpus"]["correct"]) for u in QIDS},
              "doc": {u: bool(recs[k][v][u]["doc"]) for u in QIDS}} for v in ("old", "new")} for k in RUNS}
assert [sum(ok[k]["old"]["corpus_rescored"].values()) for k in RUNS] == [801, 675]
assert [sum(ok[k]["old"]["doc"].values()) for k in RUNS] == [2444, 2389]
part2 = {}
for k in RUNS:
    for scope in ("corpus_rescored", "corpus_recorded", "doc"):
        a, b = ok[k]["old"][scope], ok[k]["new"][scope]
        part2[f"{k}|{scope}"] = {"n": len(QIDS), "old_correct": sum(a.values()), "new_correct": sum(b.values()),
                                 "disagree": sum(a[u] != b[u] for u in QIDS),
                                 "old_only": sum(a[u] and not b[u] for u in QIDS),
                                 "new_only": sum(b[u] and not a[u] for u in QIDS)}
del recs, ok


# ---- 3
def pair_means(E, tabs):
    """(같은 표 쌍 평균, 다른 표 쌍 평균, 같은 표 쌍 수, 다른 표 쌍 수). 합 = (|Σe|² − Σ|e|²)/2 로 쌍 전부를 센다."""
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
    return (s_w / p_w if p_w else np.nan, (s_all - s_w) / p_b if p_b else np.nan, int(p_w), int(p_b))


tids = np.array(idx["e"]["unit_tids"])
doc_of = np.array([t.split("::")[0] for t in tids])
rows = defaultdict(list)
for n, u in enumerate(doc_of):
    rows[u].append(n)
rows = {u: np.array(rows[u]) for u in set(QIDS)}
stats, emb_src = {}, {}
for k in RUNS:
    emb = np.concatenate([np.load(f) for f in idx[k]["emb_files"]])
    assert emb.shape == (N_CELLS, 768), emb.shape
    emb_src[k] = {"files": idx[k]["emb_files"], "n_cells": int(emb.shape[0]), "dim": int(emb.shape[1])}
    stats[k] = {u: pair_means(emb[rows[u]].astype(np.float64), tids[rows[u]]) for u in rows}
    for u in sorted(rows)[:3]:                    # 쌍 합 공식 대 직접 계산
        E = emb[rows[u]].astype(np.float64)
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        C, same = E @ E.T, tids[rows[u]][:, None] == tids[rows[u]][None, :]
        iu = np.triu_indices(len(E), 1)
        w, b = C[iu][same[iu]], C[iu][~same[iu]]
        assert np.allclose(stats[k][u][:2], (w.mean() if w.size else np.nan, b.mean() if b.size else np.nan),
                           atol=1e-5, equal_nan=True), u
    del emb


def summ(x):
    return {"mean": float(np.mean(x)), "median": float(np.median(x))}


part3 = {}
for gname, qs in GROUPS.items():
    one_table = [u for u in qs if np.isnan(stats["c"][u][1])]
    use = [u for u in qs if not np.isnan(stats["c"][u][0]) and not np.isnan(stats["c"][u][1])]
    g = {"n_queries": len(qs), "excluded_one_table_doc": len(one_table),
         "excluded_no_same_table_pair": sum(1 for u in qs if np.isnan(stats["c"][u][0])),
         "n_used": len(use)}
    for name, f in (("same_table", lambda s: s[0]), ("other_table", lambda s: s[1]),
                    ("same_minus_other", lambda s: s[0] - s[1])):
        c = np.array([f(stats["c"][u]) for u in use])
        e = np.array([f(stats["e"][u]) for u in use])
        g[name] = {"c": summ(c), "e": summ(e), "e_minus_c": summ(e - c),
                   "e_gt_c": int((e > c).sum()), "e_lt_c": int((e < c).sum()), "tie": int((e == c).sum()),
                   "wilcoxon_p": float(wilcoxon(c, e).pvalue)}
    part3[gname] = g
part3["pairs_per_doc"] = {"same_table": summ([stats["c"][u][2] for u in QIDS]),
                          "other_table": summ([stats["c"][u][3] for u in QIDS])}

out = {"sources": src, "embeddings": emb_src, "part2": part2, "part3": part3}
(OUT / "analyze.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
