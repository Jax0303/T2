"""2026-09-26 MultiHiertt 문서 안 검색, 정답 표 2개 이상 582문항, c / e (리더 없음, 재검색 없음 — 기존 records 만).
순위 = records 의 doc 문맥 위치(1..20). records 는 문장만 담으므로 문장 -> 질문 문서의 셀로 되돌린다. 한 문맥 문장이
질문 문서 안에서 서로 다른 (표, 정답 여부)의 셀 여럿과 같으면 그 문항은 '판정 불가'로 빼고 따로 센다.
1. 정답 표마다 그 표 정답 셀의 최고 순위(20위 밖 = 21). 순위 차 = 가장 늦은 표 − 가장 이른 표. c vs e Wilcoxon 부호순위
   (둘 다 판정된 문항, scipy 기본 zero_method='wilcox' = 차 0 인 짝 제외).
2. 가장 이른 정답 표가 20셀 중 차지한 칸 수(그 표의 셀이면 정답 여부 무관).
3. 질문(BGE 질의 접두어 포함)과 정답 표 라벨(L1, 문서 접두어 없음)의 BGE 코사인: 가장 이른 표 vs 가장 늦은 표, Wilcoxon.
   가장 이른 표가 21위(정답 표가 하나도 안 나옴)이거나 가장 늦은 순위를 여러 표가 나눠 가지면 정할 수 없어 빼고 센다.
   라벨이 빈 표가 끼면 빼고 센다.
실행: HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python results/components_20260925/mh_multitable_rank.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

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

stat = lambda a: {"n": len(a), "median": float(np.median(a)), "mean": float(np.mean(a))} if len(a) else {"n": 0}
out, spread, first_cells, cosines = {}, {}, {}, {}
for k in RUNS:
    det = {u: r for u, r in res[k].items() if r is not None}
    spread[k] = {u: max(b.values()) - min(b.values()) for u, (b, _p) in det.items()}
    firsts, cos_pairs, why = {}, {}, defaultdict(int)
    for u, (b, pos) in det.items():
        lo, hi = min(b.values()), max(b.values())
        if lo == 21:
            why["no_gold_table_in_top20"] += 1
            continue
        first = next(t for t, r in b.items() if r == lo)
        firsts[u] = sum(t == first for t in pos)
        lasts = [t for t, r in b.items() if r == hi]
        if len(lasts) > 1:
            why["latest_tied_at_21"] += 1
            continue
        if not LABEL[first] or not LABEL[lasts[0]]:
            why["empty_label"] += 1
            continue
        cos_pairs[u] = (float(qv[u] @ lv[first]), float(qv[u] @ lv[lasts[0]]))
    first_cells[k], cosines[k] = firsts, cos_pairs
    a = [x[0] for x in cos_pairs.values()]
    b = [x[1] for x in cos_pairs.values()]
    out[k] = {
        "n_queries": len(QS), "determined": len(det), "undetermined": len(QS) - len(det),
        "queries_with_a_gold_table_at_21": sum(21 in bb.values() for bb, _p in det.values()),
        "queries_all_gold_tables_at_21": sum(set(bb.values()) == {21} for bb, _p in det.values()),
        "rank_spread": stat(list(spread[k].values())),
        "rank_spread_only_queries_without_21": stat([spread[k][u] for u, (bb, _p) in det.items()
                                                     if 21 not in bb.values()]),
        "first_table_cells_in_20": stat(list(firsts.values())),
        "cos_question_label": {"excluded": dict(why), "first": stat(a), "last": stat(b),
                               "first_minus_last": stat(np.subtract(a, b)),
                               "wilcoxon_p": float(wilcoxon(a, b).pvalue) if a else None},
    }
both = [u for u in QS if u in spread["c"] and u in spread["e"]]
dc = np.array([spread["c"][u] for u in both])
de = np.array([spread["e"][u] for u in both])
out["rank_spread_c_vs_e"] = {"n_pairs": len(both), "zero_diff_pairs": int((dc == de).sum()),
                             "c_larger": int((dc > de).sum()), "e_larger": int((de > dc).sum()),
                             "c": stat(dc), "e": stat(de), "wilcoxon_p": float(wilcoxon(dc, de).pvalue)}
out["encoder"] = {"name": enc.name, "revision": REV, "query_prefix": enc.query_prefix,
                  "passage_prefix": enc.passage_prefix}
out["records"] = {k: f"results/mh_arms/{s}_records.jsonl" for k, (_l, s) in RUNS.items()}
(OUT / "mh_multitable_rank.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
