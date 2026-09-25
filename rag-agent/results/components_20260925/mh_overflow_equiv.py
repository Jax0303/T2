"""2026-09-26 MultiHiertt c / e (리더 없음, 재검색 없음 — 기존 records 만 다시 채점).
c = results/mh_arms/mh_cell_hv2 (라벨 없음), e = results/mh_arms/mh_cell_hv2_L1 (라벨 L1). 머리글 규칙 v2, train.
1. e 색인 셀 문장의 BGE 토큰 수(접두어·특수 토큰 포함, audit_inputs 와 같은 셈), 라벨 문자열만 따로 센 토큰 수,
   정답 셀 문장이 512 를 넘는 문항과 그 문항들의 c·e 검색 정확도.
2. 전체 색인 재채점: 내용이 같은 문서의 같은 위치 셀을 정답과 동일로 인정. records 는 문맥 문장만 담으므로
   문장이 정답과 같은 문맥 단위가 '같은 위치 셀'인지 정해지지 않는 문항은 '미정'으로 따로 센다
   (문맥 속 그 문장 개수 > 같은 문장을 가진 다른 위치 셀 수 이면 확정 인정).
실행: HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python results/components_20260925/mh_overflow_equiv.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.argv = sys.argv[:1]
import mh_arms as mh                                                  # noqa: E402

OUT = Path(__file__).parent
RUNS = {"c": ("none", "mh_cell_hv2"), "e": ("L1", "mh_cell_hv2_L1")}
LIMIT = 512                                   # bge-base max_seq_length

queries, docs, _ = mh.load_population("train")
recs, idx = {}, {}
for k, (label, stem) in RUNS.items():
    cfg = json.loads((ROOT / f"results/mh_arms/{stem}.json").read_text())
    assert cfg["encoder"] == "BAAI/bge-base-en-v1.5" and cfg["header_rule"] == "v2"
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
    assert set(gold) == set(recs[k]) and all(len(gold[u]) == recs[k][u]["m"] for u in gold)
    idx[k] = {"tables": tables, "texts": texts, "unit_tids": unit_tids, "gold": gold,
              "unit_of": {next(iter(c)): n for n, c in enumerate(covers)}}
assert idx["c"]["gold"] == idx["e"]["gold"] and set(recs["c"]) == set(recs["e"])
GOLD, QIDS = idx["e"]["gold"], sorted(recs["e"])
print("corpora rebuilt, hashes match", flush=True)

# ---- 1. e 색인 토큰 수
tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
E = idx["e"]
count = lambda xs, special: [len(v) for s in range(0, len(xs), 20000)
                             for v in tok(xs[s:s + 20000], add_special_tokens=special,
                                          truncation=False)["input_ids"]]
n_tok = np.array(count(E["texts"], True))       # bge 문서 접두어는 "" (encoders.default_prefixes)
tids = sorted(E["tables"])
lab_of = dict(zip(tids, count([E["tables"][t].title for t in tids], False)))
lab_unit = np.array([lab_of[t] for t in E["unit_tids"]])
lab_tab = np.array([lab_of[t] for t in tids])
over_q = [u for u in QIDS if any(n_tok[E["unit_of"][g]] > LIMIT for g in GOLD[u])]


def acc(k, qs, scope):
    c = sum(recs[k][u][scope]["correct"] for u in qs)
    return {"correct": c, "n": len(qs), "rate": c / len(qs) if qs else None}


dist = lambda a: {"median": float(np.median(a)), "mean": float(a.mean()), "max": int(a.max()),
                  "p95": float(np.percentile(a, 95))}
part1 = {
    "tokenizer": "BAAI/bge-base-en-v1.5 (add_special_tokens=True, truncation=False, 문서 접두어 없음)",
    "limit": LIMIT, "n_cells": len(n_tok), "cell_tokens": dist(n_tok),
    "cells_over_limit": int((n_tok > LIMIT).sum()), "cells_over_limit_ratio": float((n_tok > LIMIT).mean()),
    "label_tokens_per_cell": dist(lab_unit), "label_tokens_per_table": dist(lab_tab),
    "cells_with_empty_label": int((lab_unit == 0).sum()), "n_tables": len(tids),
    "tables_with_empty_label": int((lab_tab == 0).sum()),
    "label_share_of_cell_tokens_median": float(np.median(lab_unit / n_tok)),
    "queries_gold_sentence_over_limit": len(over_q), "n_queries": len(QIDS),
    "layers_of_those": dict(Counter(recs["e"][u]["layer"] for u in over_q)),
    "accuracy_on_those": {k: {s: acc(k, over_q, s) for s in ("doc", "corpus")} for k in RUNS},
}
print(json.dumps(part1, ensure_ascii=False, indent=1), flush=True)

# ---- 2. 내용이 같은 문서끼리 묶기. 세 기준(원본 표 HTML, c 문장열, e 문장열)이 같은 묶음인지 확인
uids = sorted(docs)
per_doc = {k: defaultdict(list) for k in RUNS}
for k in RUNS:
    for t, s in zip(idx[k]["unit_tids"], idx[k]["texts"]):
        per_doc[k][t.split("::")[0]].append(s)
partition = lambda key: sorted(sorted(g) for g in _group(key))


def _group(key):
    g = defaultdict(list)
    for u in uids:
        g[key(u)].append(u)
    return g.values()


parts = {"raw_html": partition(lambda u: tuple(docs[u][0])),
         "c_text": partition(lambda u: tuple(per_doc["c"][u])),
         "e_text": partition(lambda u: tuple(per_doc["e"][u]))}
n_groups = {k: len(v) for k, v in parts.items()}
assert parts["c_text"] == parts["e_text"], n_groups
members = {u: g for g in parts["e_text"] for u in g}
print("doc groups", n_groups, flush=True)


def rescore(k):
    X = idx[k]
    by_text = Counter(X["texts"])
    out = {}
    for u in QIDS:
        r = recs[k][u]["corpus"]
        if r["correct"]:
            out[u] = "yes"
            continue
        ctx = Counter(r["context"])
        st = []
        for tid, i, j in GOLD[u]:
            t = tid.split("::")[1]
            eq = [X["unit_of"][c] for u2 in members[u] if (c := (f"{u2}::{t}", i, j)) in X["unit_of"]]
            txt = {X["texts"][n] for n in eq}
            assert len(txt) == 1 and len(eq) == len(members[u]), (k, u)
            txt = txt.pop()
            other = by_text[txt] - len(eq)           # 같은 문장이지만 같은 위치 셀이 아닌 단위 수
            st.append("yes" if ctx[txt] > other else "no" if ctx[txt] == 0 else "amb")
        out[u] = "no" if "no" in st else "yes" if all(s == "yes" for s in st) else "amb"
    return out


res = {k: rescore(k) for k in RUNS}


def mcnemar(a, b):
    x = sum(a[u] and not b[u] for u in QIDS)
    y = sum(b[u] and not a[u] for u in QIDS)
    return {"c_only": x, "e_only": y, "p_exact": binomtest(x, x + y).pvalue if x + y else 1.0}


part2 = {"doc_groups": n_groups, "groups_same_raw_vs_text": parts["raw_html"] == parts["e_text"],
         "docs_in_groups_of_size_ge2": sum(len(g) for g in parts["e_text"] if len(g) > 1),
         "n_queries": len(QIDS),
         "queries_whose_doc_has_copies": sum(len(members[u]) > 1 for u in QIDS)}
for k in RUNS:
    c = Counter(res[k].values())
    part2[k] = {"original_correct": sum(recs[k][u]["corpus"]["correct"] for u in QIDS),
                "rescored_yes": c["yes"], "undetermined": c["amb"], "no": c["no"]}
for name, amb_as in (("undetermined_as_wrong", False), ("undetermined_as_right", True)):
    ok = {k: {u: res[k][u] == "yes" or (amb_as and res[k][u] == "amb") for u in QIDS} for k in RUNS}
    part2[name] = {**{k: sum(ok[k].values()) for k in RUNS}, "mcnemar": mcnemar(ok["c"], ok["e"])}
orig = {k: {u: bool(recs[k][u]["corpus"]["correct"]) for u in QIDS} for k in RUNS}
part2["original_mcnemar"] = mcnemar(orig["c"], orig["e"])
print(json.dumps(part2, indent=1), flush=True)
(OUT / "mh_overflow_equiv.json").write_text(json.dumps(
    {"part1_e_tokens": part1, "part2_equiv_rescore": part2,
     "over_limit_query_ids": over_q}, ensure_ascii=False, indent=1))
