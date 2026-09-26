"""2026-09-27 1단계 검색 실패 진단 (리더 없음, 결과 파일 수정 없음 — rerun_20260926 records 와 저장된 임베딩만 읽는다).

HiTab test 단일 셀 991 (mode all, m=1, aggregation none — compare.py hitab_sc 와 같은 필터), s3c, 두 범위:
  intable = 질문의 표 안에서만 검색(--corpus gold), t538 = 표 538개 한 색인(--corpus split).
MultiHiertt train 문서 안(doc) 2,885 (records 의 doc 채점 문항 전부), s3c, 머리글 v3.3u.

순위: records 의 순위 목록은 20위까지(context_units)뿐이라 hybrid(α=.7)를 저장된 임베딩으로 다시 계산한다.
  HiTab  = retrieval_accuracy.py main 의 점수 식 그대로 (gold 범위는 표로 마스크한 뒤 min-max).
  MH     = mh_arms.py main 의 점수 식 그대로 (코퍼스 전체 min-max 뒤 문서로 마스크, 상위 2048 argpartition).
  재계산 상위 20 이 records(context_units 의 index_unit / doc.context 문장)와 문항 전부 같아야 진단을 쓴다(불일치 0).
path_overlap: results/components_20260925/hitab_path_overlap.py 와 같은 정의(단어 = encoders._tokenize, 불용어 = sklearn
  ENGLISH_STOP_WORDS, 비율 = |질문 단어 ∩ 경로 단어| / |질문 단어|, 경로 = 행 경로 + 열 경로, 제목·값 제외).
judge: results/hitab_e_miss_20260926/judge40_rater_A.csv 의 (가)(다) 값을 그대로 붙인다(새 판정 없음).
실행: HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=. .venv/bin/python results/bottleneck_20260927/diag/diag.py
"""
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.argv = sys.argv[:1]
import retrieval_accuracy as ra                                       # noqa: E402
import mh_arms as mh                                                  # noqa: E402
from rag_agent.eval.artifacts import digest                           # noqa: E402
from rag_agent.retrieve.encoders import _tokenize, default_encoder    # noqa: E402
from rag_agent.retrieve.hybrid_index import _minmax                   # noqa: E402
from rag_agent.retrieve.sparse_bm25 import SparseBM25                 # noqa: E402

R1 = ROOT / "results/rerun_20260926"
ALPHA, BUDGET = 0.7, 20
words = lambda s: set(_tokenize(s)) - ENGLISH_STOP_WORDS
log = open(OUT / "diag.log", "w", encoding="utf-8")


def say(*a):
    print(*a, flush=True)
    print(*a, file=log, flush=True)


def jl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def hitab_sc(stem):
    out = {}
    for r in map(json.loads, open(f"{stem}_records.jsonl")):
        if "correct" in r and r["mode"] == "all" and r.get("m") == 1 and (r.get("aggregation") or "none") == "none":
            out[r["query_id"]] = r
    return out


def rank_bucket(rank):
    return "B_rank21_50" if 21 <= rank <= 50 else "C_rank51plus" if rank > 50 else "D_rank_le20_but_fail"


def dist(vals):
    v = np.array([x for x in vals if x is not None], dtype=float)
    return {"n": len(vals), "n_none": sum(x is None for x in vals),
            "mean": round(float(v.mean()), 4) if len(v) else None,
            "median": round(float(np.median(v)), 4) if len(v) else None,
            "zero": int((v == 0).sum()) if len(v) else 0}


summary = {"inputs": {}, "checks": {}}
enc = default_encoder(model_name="BAAI/bge-base-en-v1.5")

# ======================= HiTab =======================
t0 = time.time()
recs = {"intable": hitab_sc(R1 / "hitab/hitab_test_gold_s3c"), "t538": hitab_sc(R1 / "hitab/hitab_test_split_s3c")}
assert len(recs["intable"]) == len(recs["t538"]) == 991 and set(recs["intable"]) == set(recs["t538"])
assert sum(r["correct"] for r in recs["intable"].values()) == 955
assert sum(r["correct"] for r in recs["t538"].values()) == 906
summ = json.loads((R1 / "hitab/hitab_test_gold_s3c.json").read_text())
page_titles = json.loads(ra.PAGE_TITLES.read_text())
tabs = {}
queries = ra.load_queries("data/hitab", "test", tabs)
tids = sorted({q["table_id"] for q in queries})
texts, covers, _, unit_tids, _ = ra.build_corpus("data/hitab", tids, "s3c", "cell", page_titles, 1000, "leaf",
                                                  "values", 200, None, trag_dtype="infer")
assert digest(texts) == summ["corpus_text_sha256"], "HiTab corpus text differs from the rerun"
unit_tid_arr = np.array(unit_tids)
key = digest({"texts": texts, "encoder": enc.metadata(), "overflow": "error"})[:24]
emb_path = ROOT / f".cache/rerun_20260926/{enc.name.replace('/', '_')}_{len(texts)}_{key}.npy"
assert emb_path.exists(), emb_path
emb = np.load(emb_path)
assert emb.shape == (len(texts), 768)
bm = SparseBM25(_tokenize(t) for t in texts)
cell2unit = {next(iter(c)): i for i, c in enumerate(covers)}
assert len(cell2unit) == len(texts)
summary["inputs"]["hitab"] = {"records": [str(R1 / "hitab/hitab_test_gold_s3c_records.jsonl"),
                                          str(R1 / "hitab/hitab_test_split_s3c_records.jsonl")],
                              "embeddings": str(emb_path), "n_tables": len(tids), "n_units": len(texts),
                              "corpus_text_sha256": digest(texts), "alpha": ALPHA, "budget": BUDGET}
say(f"[hitab] corpus {len(tids)} tables / {len(texts)} units, cache {emb_path.name} ({time.time() - t0:.0f}s)")

judge = {}
jrows = list(csv.DictReader(open(ROOT / "results/hitab_e_miss_20260926/judge40_rater_A.csv", encoding="utf-8-sig")))
ga = next(k for k in jrows[0] if k.startswith("(가)"))
da = next(k for k in jrows[0] if k.startswith("(다)"))
for r in jrows:
    judge[r["query_id"]] = {"rater_A_ga": r[ga], "rater_A_da": r[da]}
assert len(judge) == 40

qmap = {q["query_id"]: q for q in queries}
rows = {"intable": [], "t538": []}
mismatch = Counter()
t0 = time.time()
for n, qid in enumerate(sorted(recs["t538"]), 1):
    q = qmap[qid]
    gold = tuple(recs["t538"][qid]["gold_cells"][0])
    assert q["gold"] == {gold} and recs["intable"][qid]["gold_cells"] == [list(gold)]
    tid = q["table_id"]
    assert gold[0] == tid
    t = tabs[tid].table
    qw = words(q["question"])
    pw = words(" ".join([*t.row_path(gold[1]), *t.col_path(gold[2])]))
    overlap = len(qw & pw) / len(qw) if qw else None
    s = bm.get_scores(_tokenize(q["question"]))
    d = emb @ enc.encode_query([q["question"]])[0].astype(np.float32)
    gu = cell2unit[gold]
    for scope in ("intable", "t538"):
        rec = recs[scope][qid]
        if scope == "intable":
            sel = np.flatnonzero(unit_tid_arr == tid)
            sc = ALPHA * _minmax(d[sel]) + (1 - ALPHA) * _minmax(s[sel])
            order = sel[np.argsort(-sc, kind="stable")]
        else:
            sc = ALPHA * _minmax(d) + (1 - ALPHA) * _minmax(s)
            order = np.argsort(-sc, kind="stable")
        top = [u["index_unit"] for u in rec["context_units"]]
        if order[:len(top)].tolist() != top:
            mismatch[scope] += 1
        rank = int(np.flatnonzero(order == gu)[0]) + 1
        if rec["gold_rank"] is not None and rec["gold_rank"] != rank:
            mismatch[f"{scope}_gold_rank"] += 1
        same_above = int((unit_tid_arr[order[:rank - 1]] == tid).sum())
        row = {"qid": qid, "scope": scope, "correct": rec["correct"], "gold_cell_id": list(gold), "gold_table_id": tid,
               "gold_cell_rank": rank, "records_gold_rank": rec["gold_rank"], "same_table_above": same_above,
               "n_units_in_table": int((unit_tid_arr == tid).sum()),
               "path_overlap": None if overlap is None else round(overlap, 4), "overlap_zero": overlap == 0,
               "n_q_words": len(qw), "n_in_path": len(qw & pw)}
        if scope == "t538":
            row["gold_table_in_top20"] = rec["gold_table_in_context"]
            row["judge_bucket"] = judge.get(qid)
        if rec["correct"]:
            row["bucket"] = "S_success"
            assert rank <= BUDGET
        elif scope == "t538" and not rec["gold_table_in_context"]:
            row["bucket"] = "A_gold_table_not_in_top20"
        else:
            row["bucket"] = rank_bucket(rank)
        rows[scope].append(row)
    if n % 200 == 0:
        say(f"  hitab {n}/991 {time.time() - t0:.0f}s")
summary["checks"]["hitab_top20_mismatch"] = {s: mismatch.get(s, 0) for s in ("intable", "t538")}
summary["checks"]["hitab_records_gold_rank_mismatch"] = {s: mismatch.get(f"{s}_gold_rank", 0) for s in ("intable", "t538")}
say(f"[hitab] top-20 mismatch {dict(mismatch)}")
for scope, name in (("intable", "hitab_intable"), ("t538", "hitab_538")):
    jl(OUT / f"{name}.jsonl", rows[scope])
    rs = rows[scope]
    fail = [r for r in rs if not r["correct"]]
    succ = [r for r in rs if r["correct"]]
    buckets = Counter(r["bucket"] for r in fail)
    xt = {b: {"overlap_zero": sum(r["overlap_zero"] for r in fail if r["bucket"] == b),
              "overlap_nonzero": sum(not r["overlap_zero"] for r in fail if r["bucket"] == b)} for b in sorted(buckets)}
    xt["S_success"] = {"overlap_zero": sum(r["overlap_zero"] for r in succ),
                       "overlap_nonzero": sum(not r["overlap_zero"] for r in succ)}
    o = {"n": len(rs), "correct": len(succ), "fail": len(fail), "fail_buckets": dict(sorted(buckets.items())),
         "bucket_x_overlap_zero": xt,
         "fail_rank_bins": {b: sum(lo <= r["gold_cell_rank"] <= hi for r in fail)
                            for b, lo, hi in (("1-20", 1, 20), ("21-30", 21, 30), ("31-50", 31, 50),
                                              ("51-100", 51, 100), ("101-500", 101, 500), (">500", 501, 10 ** 9))},
         "same_table_above": {"fail": dist([r["same_table_above"] for r in fail]),
                              "success": dist([r["same_table_above"] for r in succ])},
         "path_overlap": {"fail": dist([r["path_overlap"] for r in fail]),
                          "success": dist([r["path_overlap"] for r in succ])}}
    if scope == "t538":
        A = [r for r in fail if r["bucket"] == "A_gold_table_not_in_top20"]
        o["A_rank_bins"] = {b: sum(lo <= r["gold_cell_rank"] <= hi for r in A)
                            for b, lo, hi in (("21-50", 21, 50), ("51-100", 51, 100), ("101-500", 101, 500), (">500", 501, 10 ** 9))}
        o["A_same_table_above"] = dist([r["same_table_above"] for r in A])
        o["judge_rater_A"] = {"n_with_judge": sum(r["judge_bucket"] is not None for r in fail),
                              "n_judge_in_bucket_A": sum(r["judge_bucket"] is not None for r in A),
                              "ga": dict(Counter(r["judge_bucket"]["rater_A_ga"] for r in fail if r["judge_bucket"])),
                              "da": dict(Counter(r["judge_bucket"]["rater_A_da"].split(":")[0] for r in fail if r["judge_bucket"])),
                              "ga_x_da": dict(Counter(f'{r["judge_bucket"]["rater_A_ga"]}|{r["judge_bucket"]["rater_A_da"].split(":")[0]}'
                                                      for r in fail if r["judge_bucket"]))}
        nB = buckets.get("B_rank21_50", 0)
        nA50 = o["A_rank_bins"]["21-50"]
        o["upper_bound"] = {"906_plus_B": f"{906 + nB}/991", "906_plus_B_plus_A_rank_le50": f"{906 + nB + nA50}/991",
                            "906_plus_B_plus_20": f"{906 + nB + 20}/991 (20 = 지시문의 '실제 실패 20', 파일에서 산출한 값이 아님)"}
    else:
        o["upper_bound"] = {"955_plus_B": f"{955 + buckets.get('B_rank21_50', 0)}/991"}
    summary[name] = o
del emb, bm

# ======================= MultiHiertt =======================
t0 = time.time()
mrec = {r["query_id"]: r for r in map(json.loads, open(R1 / "mh/mh_train_s3c_records.jsonl")) if "doc" in r}
assert len(mrec) == 2885 and sum(r["doc"]["correct"] for r in mrec.values()) == 2491
msumm = json.loads((R1 / "mh/mh_train_s3c.json").read_text())
mq, docs, skipped = mh.load_population("train", keep_hybrid=False)
tables, hdr = mh.build_tables(docs, "v3.3u", "none")
texts, covers, _, unit_tids, _ = ra.build_corpus("", sorted(tables), "s3c", "cell", {}, 1000, "leaf", "values", 200,
                                                  None, load=lambda tid, _d: tables.get(tid), trag_dtype="infer")
assert digest(texts) == msumm["corpus_text_sha256"], "MH corpus text differs from the rerun"
live = {(tid, i, j) for tid, tab in tables.items() for i, row in enumerate(tab.table.data)
        for j, v in enumerate(row) if str(v).strip()}
mq = mh.resolve_gold(mq, tables, hdr, live)
uid_arr = np.array([t.split("::")[0] for t in unit_tids])
tid_arr = np.array(unit_tids)
key = digest(texts)[:16]
cache = ROOT / ".cache/rerun_20260926_mh"
emb = np.empty((len(texts), 768), dtype=np.float32)
shard_files = []
for s in range(0, len(texts), 50000):
    f = cache / f"cell_{len(texts)}_{key}_{s}.npy"
    assert f.exists(), f
    v = np.load(f)
    emb[s:s + len(v)] = v
    shard_files.append(str(f))
    del v
bm = SparseBM25(_tokenize(t) for t in texts)
cell2unit = {next(iter(c)): i for i, c in enumerate(covers)}
summary["inputs"]["mh"] = {"records": str(R1 / "mh/mh_train_s3c_records.jsonl"), "embeddings": shard_files,
                           "n_docs": len(docs), "n_tables": len(tables), "n_units": len(texts),
                           "corpus_text_sha256": digest(texts), "alpha": ALPHA, "budget": BUDGET, "header_rule": "v3.3u"}
say(f"[mh] corpus {len(tables)} tables / {len(texts)} units ({time.time() - t0:.0f}s)")

TOP = 2048
mrows, mism, t0 = [], 0, time.time()
scored = [q for q in mq if not q["excluded"] and q["gold"]]
assert len(scored) == 2885 and {q["uid"] for q in scored} == set(mrec)
for n, q in enumerate(scored, 1):
    rec = mrec[q["uid"]]
    sp = bm.get_scores(_tokenize(q["question"]))
    dn = emb @ enc.encode_query([q["question"]])[0].astype(np.float32)
    sc = ALPHA * _minmax(dn) + (1 - ALPHA) * _minmax(sp)
    sel = np.flatnonzero(uid_arr == q["uid"])
    k = min(TOP, len(sel) - 1)
    top = sel[np.argpartition(-sc[sel], k)[:k + 1]]
    order = top[np.argsort(-sc[top], kind="stable")]
    ctx = rec["doc"]["context"]
    if [texts[p] for p in order[:len(ctx)]] != ctx:
        mism += 1
    top20 = set().union(*(covers[p] for p in order[:len(ctx)]))
    assert len(top20) == rec["doc"]["cells_in_context"]
    ctx_tables = {c[0] for c in top20}
    gold = sorted(q["gold"])
    gold_tables = {c[0] for c in gold}
    pos = {int(p): i + 1 for i, p in enumerate(order)}
    missing = []
    for c in gold:
        if c in top20:
            continue
        rk = pos.get(cell2unit[c])
        missing.append({"cell": list(c), "rank": rk if rk is not None and rk <= 50 else ">50",
                        "rank_raw": rk, "table_has_other_cell_in_context": c[0] in ctx_tables})
    found = len(gold) - len(missing)
    n_gt_in = len(gold_tables & ctx_tables)
    if rec["doc"]["correct"]:
        bucket = "S_success"
        assert not missing
    elif n_gt_in == 0:
        bucket = "E_no_gold_table_cell"
    elif found == 0:
        bucket = "H_all_gold_cells_missing"
    elif all(m["table_has_other_cell_in_context"] for m in missing):
        bucket = "F_partial_missing_same_table"
    else:
        bucket = "G_partial_missing_other_table"
    mrows.append({"qid": q["uid"], "correct": rec["doc"]["correct"], "n_gold_cells": len(gold),
                  "n_gold_tables": len(gold_tables), "question_type": q["kind"], "layer": rec["layer"],
                  "n_found_gold_cells": found, "n_gold_tables_with_any_cell_in_top20": n_gt_in,
                  "n_units_in_doc": int(len(sel)), "missing": missing, "bucket": bucket,
                  "all_missing_rank_le50": bool(missing) and all(m["rank_raw"] is not None and m["rank_raw"] <= 50 for m in missing)})
    if n % 500 == 0:
        say(f"  mh {n}/2885 {time.time() - t0:.0f}s")
summary["checks"]["mh_top20_mismatch"] = mism
say(f"[mh] top-20 mismatch {mism}")
jl(OUT / "mh_indoc.jsonl", mrows)
fail = [r for r in mrows if not r["correct"]]
groups = ("lookup_m1", "lookup_m2+", "arith_m1", "arith_m2+")
bins = (("<=30", 1, 30), ("31-50", 31, 50), (">50", 51, 10 ** 9))
mr = [m["rank_raw"] if m["rank_raw"] is not None else 10 ** 9 for r in fail for m in r["missing"]]
summary["mh_indoc"] = {
    "n": len(mrows), "correct": len(mrows) - len(fail), "fail": len(fail),
    "fail_buckets": dict(sorted(Counter(r["bucket"] for r in fail).items())),
    "bucket_by_layer": {g: dict(sorted(Counter(r["bucket"] for r in fail if r["layer"] == g).items())) for g in groups},
    "layer_n_fail": {g: sum(r["layer"] == g for r in fail) for g in groups},
    "layer_n_all": {g: sum(r["layer"] == g for r in mrows) for g in groups},
    "missing_cells_total": len(mr),
    "missing_rank_bins": {b: sum(lo <= x <= hi for x in mr) for b, lo, hi in bins},
    "missing_rank_bins_by_bucket": {bk: {b: sum(lo <= (m["rank_raw"] or 10 ** 9) <= hi for r in fail if r["bucket"] == bk
                                                for m in r["missing"]) for b, lo, hi in bins}
                                    for bk in sorted(Counter(r["bucket"] for r in fail))},
    "fail_all_missing_rank_le50": sum(r["all_missing_rank_le50"] for r in fail),
    "fail_all_missing_rank_le50_by_bucket": dict(sorted(Counter(r["bucket"] for r in fail if r["all_missing_rank_le50"]).items())),
    "fail_all_missing_rank_le50_by_layer": {g: sum(r["all_missing_rank_le50"] for r in fail if r["layer"] == g) for g in groups},
    "bucket_definitions": {
        "E": "상위 20 셀 중 정답 표(들)의 셀이 0개",
        "H": "정답 표 셀은 있으나 정답 셀은 0개 찾음",
        "F": "정답 셀 일부 찾음, 누락 셀의 표가 모두 문맥에 다른 셀로 들어와 있음",
        "G": "정답 셀 일부 찾음, 누락 셀 중 표가 문맥에 없는 것이 있음",
        "missing rank": "문서 안 hybrid 순위(상위 2048 까지), 그 밖은 >50"},
    "upper_bound": {"2491_plus_all_missing_le50": f"{2491 + sum(r['all_missing_rank_le50'] for r in fail)}/2885"}}
(OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
say(json.dumps(summary, indent=1, ensure_ascii=False))
