"""2026-09-26 정답 셀 1개 문항 비교 (리더 없음, 재검색 없음 — 기존 records 만).
HiTab c = results/retrieval_accuracy/t_s2_gold_labelabl (test, 질문의 표 안, query_type single_cell 991).
MultiHiertt c = results/recheck_20260926/mh_cell_hv2 (train, 라벨 없음), 정답 셀 수 m == 1.
  doc = 질문의 문서 안, table = 정답 표 안(m==1 이면 표 1개). 판정 = records 의 correct (예산 20셀, G ⊆ R).
후보 셀 수 = 그 검색 범위의 색인 단위 수(단위 = 비어 있지 않은 셀 1개). 표 수 = 그 범위의 표 수.
n_candidates_le_budget20 = 후보가 예산(20셀) 이하라 범위 전체가 검색 결과가 되는 문항 수.
실행: HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python results/singlecell_compare_20260926/analyze.py
"""
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.argv = sys.argv[:1]
import mh_arms as mh                                                  # noqa: E402
import retrieval_accuracy as ra                                       # noqa: E402


def stats(xs):
    return {"median": st.median(xs), "mean": round(st.fmean(xs), 1), "min": min(xs), "max": max(xs)}


def row(ok, cand, ntab, src):
    return {"n": len(ok), "correct": sum(ok), "accuracy": round(sum(ok) / len(ok), 4),
            "candidate_cells": stats(cand), "n_candidates_le_budget20": sum(c <= 20 for c in cand), "tables_in_scope": stats(ntab), "records": src}


out = {}

# ---- HiTab: 질문의 표 안
HT = ROOT / "results/retrieval_accuracy/t_s2_gold_labelabl"
summ = json.loads(Path(f"{HT}.json").read_text())
recs = [r for r in map(json.loads, open(f"{HT}_type_accuracy.jsonl")) if r["query_type"] == "single_cell"]
tids = sorted({r["gold_cell_ids"][0][0] for r in recs})
assert len(recs) == 991 and all(len({g[0] for g in r["gold_cell_ids"]}) == 1 for r in recs)
texts, covers, _, unit_tids, _ = ra.build_corpus(summ["arguments"]["data_dir"], tids, "s2", "cell", {})
per_tab = Counter(unit_tids)
out["hitab_table"] = row([bool(r["retrieval_success"]) for r in recs],
                         [per_tab[r["gold_cell_ids"][0][0]] for r in recs], [1] * len(recs),
                         f"{HT}_type_accuracy.jsonl")

# ---- MultiHiertt: 질문의 문서 안 / 정답 표 안
queries, docs, _ = mh.load_population("train")
tables, hdr = mh.build_tables(docs, "v2", "none")
texts, covers, _, _, _ = mh.build_corpus("", sorted(tables), "s3c", "cell", {}, load=lambda tid, _d: tables.get(tid))
MH = ROOT / "results/recheck_20260926/mh_cell_hv2"
assert mh.digest(texts) == json.loads(Path(f"{MH}.json").read_text())["corpus_text_sha256"]
cells_tab = Counter(next(iter(c))[0] for c in covers)
cells_doc, tabs_doc = Counter(), Counter()
for t, n in cells_tab.items():
    cells_doc[t.split("::")[0]] += n
    tabs_doc[t.split("::")[0]] += 1
live = {(tid, i, j) for tid, tab in tables.items() for i, r in enumerate(tab.table.data)
        for j, v in enumerate(r) if str(v).strip()}
gold = {q["uid"]: q["gold"] for q in mh.resolve_gold(queries, tables, hdr, live) if not q["excluded"] and q["gold"]}
recs = [r for r in map(json.loads, open(f"{MH}_records.jsonl")) if "excluded" not in r and r["m"] == 1]
for name, keep in (("lookup_m1", lambda r: r["kind"] == "lookup"), ("all_m1", lambda r: True)):
    rs = [r for r in recs if keep(r)]
    for scope in ("doc", "table"):
        if scope == "doc":
            cand = [cells_doc[r["query_id"]] for r in rs]
            ntab = [tabs_doc[r["query_id"]] for r in rs]
        else:
            cand = [cells_tab[next(iter(gold[r["query_id"]]))[0]] for r in rs]
            ntab = [1] * len(rs)
        out[f"mh_{scope}|{name}"] = row([bool(r[scope]["correct"]) for r in rs], cand, ntab, f"{MH}_records.jsonl")

assert out["hitab_table"]["correct"] == 938 and per_tab["100"] == 16   # 표 100: 기록상 16셀 전부 검색됨
(OUT / "analyze.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
