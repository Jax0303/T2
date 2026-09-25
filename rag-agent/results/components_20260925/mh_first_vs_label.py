"""2026-09-26 MultiHiertt 문서 안 검색, 정답 표 2개 이상 582문항, c / e (리더 없음, 재검색 없음 — 기존 records 만).
'가장 먼저 나온 정답 표'(mh_multitable_rank.py 와 같은 정의, 20위 밖 = 21) 가 '질문과 라벨 BGE 코사인이 가장 높은 정답 표'
인지를 c·e 각각 판정하고, 둘 다 판정된 문항으로 McNemar(b = c만 예, c = e만 예, 정확 p).
판정 제외: records 문장을 셀로 못 정함 / 정답 표가 전부 21위 / 정답 표 중 라벨이 빈 표 / 라벨 코사인 최고가 둘 이상(같은 라벨 문자열 포함).
질문 = BGE 질의 접두어 포함, 라벨 = 문서 접두어 없음(bge-base a5beb1e3).
실행: HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python results/components_20260925/mh_first_vs_label.py
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
from rag_agent.retrieve.encoders import default_encoder              # noqa: E402

OUT = Path(__file__).parent
RUNS = {"c": ("none", "mh_cell_hv2"), "e": ("L1", "mh_cell_hv2_L1")}
REV = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
queries, docs, _ = mh.load_population("train")
recs, idx, LABEL = {}, {}, None
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
    doc_text = defaultdict(lambda: defaultdict(list))             # uid -> 문장 -> 셀들
    for c, s in zip((next(iter(c)) for c in covers), texts):
        doc_text[c[0].split("::")[0]][s].append(c)
    idx[k] = {"gold": gold, "doc_text": doc_text}
    if k == "e":
        LABEL = {tid: d.title for tid, d in tables.items()}
    del tables, texts, covers
assert idx["c"]["gold"] == idx["e"]["gold"]
GOLD = idx["e"]["gold"]
QS = sorted(u for u in GOLD if len({g[0] for g in GOLD[u]}) >= 2)
assert len(QS) == 582


def analyse(k, u):
    """None = 판정 불가. 아니면 {정답 표: 최고 순위}, 위치별 표."""
    gold, pos_tab = GOLD[u], []
    best = {g[0]: 21 for g in gold}
    for p, s in enumerate(recs[k][u]["doc"]["context"], 1):
        cand = idx[k]["doc_text"][u][s]
        kinds = {(c[0], c in gold) for c in cand}
        if len(kinds) != 1:
            return None
        tid, is_gold = kinds.pop()
        pos_tab.append(tid)
        if is_gold:
            best[tid] = min(best[tid], p)
    return best, pos_tab


res = {k: {u: analyse(k, u) for u in QS} for k in RUNS}
enc = default_encoder(model_name="BAAI/bge-base-en-v1.5", revision=REV)
qv = dict(zip(QS, enc.encode_query([recs["e"][u]["question"] for u in QS])))
gtids = sorted({g[0] for u in QS for g in GOLD[u]})
lv = dict(zip(gtids, enc.encode([LABEL[t] for t in gtids])))


def argmax_table(u):
    tids = sorted({g[0] for g in GOLD[u]})
    if any(not LABEL[t] for t in tids):
        return None, "empty_label"
    sims = {t: float(qv[u] @ lv[t]) for t in tids}
    top = max(sims.values())
    best = [t for t in tids if sims[t] == top]
    if len(best) > 1 or sum(LABEL[t] == LABEL[best[0]] for t in tids) > 1:
        return None, "label_tie"
    return best[0], None


hit, why = {}, {}
for k in RUNS:
    hit[k], why[k] = {}, defaultdict(int)
    for u in QS:
        r = res[k][u]
        if r is None:
            why[k]["record_undetermined"] += 1
            continue
        b = r[0]
        lo = min(b.values())
        if lo == 21:
            why[k]["no_gold_table_in_top20"] += 1
            continue
        best, reason = argmax_table(u)
        if best is None:
            why[k][reason] += 1
            continue
        hit[k][u] = next(t for t, x in b.items() if x == lo) == best
both = [u for u in QS if u in hit["c"] and u in hit["e"]]
x = sum(hit["c"][u] and not hit["e"][u] for u in both)
y = sum(hit["e"][u] and not hit["c"][u] for u in both)
out = {k: {"judged": len(hit[k]), "first_is_max_label_sim": sum(hit[k].values()), "excluded": dict(why[k])}
       for k in RUNS}
out["paired"] = {"n": len(both), "c_yes": sum(hit["c"][u] for u in both), "e_yes": sum(hit["e"][u] for u in both),
                 "b_c_only": x, "c_e_only": y, "p_exact": binomtest(x, x + y).pvalue if x + y else 1.0}
out["n_gold_tables_hist"] = dict(Counter(len({g[0] for g in GOLD[u]}) for u in QS))
out["encoder"] = {"name": enc.name, "revision": REV, "query_prefix": enc.query_prefix,
                  "passage_prefix": enc.passage_prefix}
out["records"] = {k: f"results/mh_arms/{s}_records.jsonl" for k, (_l, s) in RUNS.items()}
(OUT / "mh_first_vs_label.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
