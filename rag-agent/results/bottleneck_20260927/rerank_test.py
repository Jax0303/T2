"""2026-09-27 PREREG-2026-09-27-rerank.md §9·§11a — test 1회 적용, s3c 단독. dev(rerank_dev.py)와 같은 설정:
bge-reranker-v2-m3, max_length 512, s3c hybrid α=.7 상위 50 -> 재정렬 -> 상위 20 문맥.

범위: HiTab test 질문의 표 안 991 / HiTab test 538표 한 색인 991 / MultiHiertt train 질문의 문서 안 2,885.
재정렬 전 순위 = 저장된 임베딩으로 다시 계산(rerun_20260926 과 같은 식). 재계산 상위 20 이 records 와 문항 전부 같고
맞힘이 955 / 906 / 2,491 이어야 재정렬기를 적재한다(아니면 test/check_fail.json 을 쓰고 멈춘다).
로그에는 재정렬 전 기준선 확인과 처리 문항 수만 출력한다. 결과 표는 끝에 summary.json 으로만 쓴다.
범위별 jsonl 이 이미 있으면 그 범위는 다시 계산하지 않는다(GPU 오류 뒤 이어 돌리기, CLAUDE.md §7).
GATE 1 버킷 = results/bottleneck_20260927/diag/*.jsonl 의 bucket (s3c test 기준).
실행: HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=. .venv/bin/python results/bottleneck_20260927/rerank_test.py
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
OUT, DIAG, R1 = HERE / "test", HERE / "diag", ROOT / "results/rerun_20260926"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.argv = sys.argv[:1]
import retrieval_accuracy as ra                                       # noqa: E402
import mh_arms as mh                                                  # noqa: E402
from rag_agent.eval.artifacts import digest                           # noqa: E402
from rag_agent.retrieve.encoders import _tokenize, default_encoder    # noqa: E402
from rag_agent.retrieve.hybrid_index import _minmax                   # noqa: E402
from rag_agent.retrieve.sparse_bm25 import SparseBM25                 # noqa: E402

ALPHA, POOL, BUDGET, MAXLEN = 0.7, 50, 20, 512
RERANKER = "BAAI/bge-reranker-v2-m3"
EXPECT = {"hitab_intable": 955, "hitab_538": 906, "mh_indoc": 2491}
words = lambda s: set(_tokenize(s)) - ENGLISH_STOP_WORDS
OUT.mkdir(exist_ok=True)
log = open(OUT / "run.log", "a", encoding="utf-8")


def say(*x):
    print(*x, flush=True)
    print(*x, file=log, flush=True)


def overlap(question, paths):
    q = words(question)
    p = set().union(*(words(" ".join(x)) for x in paths))
    return round(len(q & p) / len(q), 4) if q else None


def item(scope, qid, question, layer, gold, order, scores, covers, texts, cid, paths, ref_ok, ref_correct):
    """rerank_dev.item 과 같은 계산. ``ref_ok(order)`` = 재계산 상위 20 이 records 와 같은가."""
    sel = ra.budget_select(order, covers, texts, BUDGET, 0)
    assert ref_ok(order), (scope, qid, "top20 mismatch")
    pos = {int(p): k + 1 for k, p in enumerate(order)}
    ranks = [pos.get(u) for u in gold.values()]
    pre_rank = None if None in ranks else max(ranks)
    correct = int(set(gold) <= sel.cells)
    assert correct == ref_correct, (scope, qid, "correct mismatch")
    assert correct == int(pre_rank is not None and pre_rank <= BUDGET)
    pool = [int(p) for p in order[:POOL]]
    return {"scope": scope, "qid": qid, "question": question, "layer": layer, "n_gold": len(gold),
            "gold": [cid(c) for c in sorted(gold)], "gold_units": [gold[c] for c in sorted(gold)],
            "pre_rank": pre_rank, "pre_correct": correct, "path_overlap": overlap(question, paths),
            "pool": pool, "pre_top50": [[cid(next(iter(covers[p]))), round(float(scores[p]), 6)] for p in pool],
            "_texts": [texts[p] for p in pool]}


def hitab_sc(stem):
    return {r["query_id"]: r for r in map(json.loads, open(f"{stem}_records.jsonl"))
            if "correct" in r and r["mode"] == "all" and r.get("m") == 1 and (r.get("aggregation") or "none") == "none"}


def hitab_items(enc):
    stems = {"hitab_intable": R1 / "hitab/hitab_test_gold_s3c", "hitab_538": R1 / "hitab/hitab_test_split_s3c"}
    rec = {k: hitab_sc(v) for k, v in stems.items()}
    assert set(rec["hitab_intable"]) == set(rec["hitab_538"]) and len(rec["hitab_538"]) == 991
    summ = json.loads(Path(f"{stems['hitab_538']}.json").read_text())
    pages = json.loads(ra.PAGE_TITLES.read_text())
    tabs = {}
    allq = ra.load_queries("data/hitab", "test", tabs)
    tids = sorted({q["table_id"] for q in allq})
    texts, covers, _, unit_tids, _ = ra.build_corpus("data/hitab", tids, "s3c", "cell", pages, 1000, "leaf", "values", 200, None)
    assert digest(texts) == summ["corpus_text_sha256"]
    key = digest({"texts": texts, "encoder": enc.metadata(), "overflow": "error"})[:24]
    f = ROOT / f".cache/rerun_20260926/{enc.name.replace('/', '_')}_{len(texts)}_{key}.npy"
    emb = np.load(f)
    bm = SparseBM25(_tokenize(t) for t in texts)
    tid_arr = np.array(unit_tids)
    unit_of = {next(iter(c)): i for i, c in enumerate(covers)}
    meta = {"records": {k: f"{v}_records.jsonl" for k, v in stems.items()}, "embeddings": str(f.relative_to(ROOT)),
            "n_units": len(texts), "n_tables": len(tids), "corpus_text_sha256": digest(texts)}
    out = {k: [] for k in stems}
    cid = lambda c: f"{c[0]}-{c[1]}-{c[2]}"
    for q in [q for q in allq if q["query_id"] in rec["hitab_538"]]:
        qid = q["query_id"]
        s = bm.get_scores(_tokenize(q["question"]))
        d = emb @ enc.encode_query([q["question"]])[0].astype(np.float32)
        t = tabs[q["table_id"]].table
        gold = {c: unit_of[c] for c in q["gold"]}
        paths = [[*t.row_path(i), *t.col_path(j)] for _, i, j in gold]
        sel = np.flatnonzero(tid_arr == q["table_id"])
        sc_in = np.zeros(len(texts), dtype=np.float64)
        sc_in[sel] = ALPHA * _minmax(d[sel]) + (1 - ALPHA) * _minmax(s[sel])
        sc_all = ALPHA * _minmax(d) + (1 - ALPHA) * _minmax(s)
        for scope, order, sc in (("hitab_intable", sel[np.argsort(-sc_in[sel], kind="stable")], sc_in),
                                 ("hitab_538", np.argsort(-sc_all, kind="stable"), sc_all)):
            r = rec[scope][qid]
            assert r["gold_cells"] == [list(c) for c in gold]
            top = [u["index_unit"] for u in r["context_units"]]
            out[scope].append(item(scope, qid, q["question"], "single_cell", gold, order, sc, covers, texts, cid, paths,
                                   lambda o, top=top: [int(p) for p in o[:len(top)]] == top, r["correct"]))
    return out, meta


def mh_items(enc):
    stem = R1 / "mh/mh_train_s3c"
    summ = json.loads(Path(f"{stem}.json").read_text())
    rec = {r["query_id"]: r for r in map(json.loads, open(f"{stem}_records.jsonl")) if "doc" in r}
    qs, docs, _ = mh.load_population("train", keep_hybrid=False)
    tables, hdr = mh.build_tables(docs, "v3.3u", "none")
    texts, covers, _, unit_tids, _ = ra.build_corpus("", sorted(tables), "s3c", "cell", {}, 1000, "leaf", "values", 200,
                                                      None, load=lambda tid, _d: tables.get(tid))
    assert digest(texts) == summ["corpus_text_sha256"]
    live = {(tid, i, j) for tid, tb in tables.items() for i, row in enumerate(tb.table.data)
            for j, v in enumerate(row) if str(v).strip()}
    qs = [q for q in mh.resolve_gold(qs, tables, hdr, live) if not q["excluded"] and q["gold"]]
    assert {q["uid"] for q in qs} == set(rec) and len(qs) == 2885
    key = digest(texts)[:16]
    emb = np.empty((len(texts), 768), dtype=np.float32)
    files = []
    for s in range(0, len(texts), 50000):
        f = ROOT / f".cache/rerun_20260926_mh/cell_{len(texts)}_{key}_{s}.npy"
        v = np.load(f)
        emb[s:s + len(v)] = v
        files.append(str(f.relative_to(ROOT)))
        del v
    bm = SparseBM25(_tokenize(t) for t in texts)
    uid_arr = np.array([t.split("::")[0] for t in unit_tids])
    unit_of = {next(iter(c)): i for i, c in enumerate(covers)}
    meta = {"records": f"{stem}_records.jsonl", "embeddings": files, "n_units": len(texts), "n_tables": len(tables),
            "corpus_text_sha256": digest(texts)}
    out = []
    for n, q in enumerate(qs, 1):
        sc = ALPHA * _minmax(emb @ enc.encode_query([q["question"]])[0].astype(np.float32)) + \
            (1 - ALPHA) * _minmax(bm.get_scores(_tokenize(q["question"])))
        sel = np.flatnonzero(uid_arr == q["uid"])
        k = min(2048, len(sel) - 1)
        top = sel[np.argpartition(-sc[sel], k)[:k + 1]]
        order = top[np.argsort(-sc[top], kind="stable")]
        r = rec[q["uid"]]
        ctx = r["doc"]["context"]
        gold = {c: unit_of[c] for c in q["gold"]}
        paths = [[*tables[c[0]].table.row_path(c[1]), *tables[c[0]].table.col_path(c[2])] for c in gold]
        out.append(item("mh_indoc", q["uid"], q["question"], r["layer"], gold, order, sc, covers, texts,
                        lambda c: mh.cell_id(c, hdr), paths,
                        lambda o, ctx=ctx: [texts[p] for p in o[:len(ctx)]] == ctx, r["doc"]["correct"]))
        if n % 500 == 0:
            say(f"  mh_indoc pre-rank {n}/{len(qs)}")
    return out, meta


def mcnemar(rows):
    b = sum(r["pre_correct"] and not r["post_correct"] for r in rows)
    c = sum(r["post_correct"] and not r["pre_correct"] for r in rows)
    return {"n": len(rows), "pre_correct": sum(r["pre_correct"] for r in rows),
            "post_correct": sum(r["post_correct"] for r in rows), "b_pre_only": b, "c_post_only": c,
            "p_two_sided": float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0}


def moves(rows):
    mid = [r for r in rows if r["pre_rank"] is not None and 21 <= r["pre_rank"] <= POOL]
    return {"n": len(rows), "pre_le20_to_post_gt20": sum(r["pre_correct"] and not r["post_correct"] for r in rows),
            "pre_21_50": len(mid), "pre_21_50_to_post_le20": sum(r["post_correct"] for r in mid),
            "pre_gt50_unrecoverable": sum(r["pre_rank"] is None or r["pre_rank"] > POOL for r in rows)}


def buckets(rows):
    """GATE 1 버킷별 n, 전/후 맞힘, 회복(틀림->맞힘), 새로 틀림(맞힘->틀림). 대상 = 사용자 지정 대상 버킷."""
    blk = lambda v: {"n": len(v), "pre_correct": sum(r["pre_correct"] for r in v), "post_correct": sum(r["post_correct"] for r in v),
                     "recovered": sum(r["post_correct"] and not r["pre_correct"] for r in v),
                     "newly_wrong": sum(r["pre_correct"] and not r["post_correct"] for r in v)}
    out = {b: blk([r for r in rows if r["gate1_bucket"] == b]) for b in sorted({r["gate1_bucket"] for r in rows})}
    out["target"] = blk([r for r in rows if r["gate1_target"]])
    return out


t0 = time.time()
enc = default_encoder(model_name="BAAI/bge-base-en-v1.5")
diag = {s: {r["qid"]: r for r in map(json.loads, open(DIAG / f"{f}.jsonl"))}
        for s, f in (("hitab_intable", "hitab_intable"), ("hitab_538", "hitab_538"), ("mh_indoc", "mh_indoc"))}
target = {"hitab_intable": lambda d: d["bucket"] == "B_rank21_50",
          "hitab_538": lambda d: d["bucket"] == "B_rank21_50" or (d["bucket"] == "A_gold_table_not_in_top20" and d["gold_cell_rank"] <= 50),
          "mh_indoc": lambda d: d["bucket"][0] in "FGH" and d["all_missing_rank_le50"]}
done = {s: [json.loads(l) for l in open(OUT / f"{s}.jsonl")] for s in EXPECT if (OUT / f"{s}.jsonl").exists()}
items, meta = {}, {}
if not {"hitab_intable", "hitab_538"} <= set(done):
    h, meta["hitab"] = hitab_items(enc)
    items.update(h)
if "mh_indoc" not in done:
    items["mh_indoc"], meta["mh"] = mh_items(enc)
pre = {s: sum(r["pre_correct"] for r in v) for s, v in {**items, **done}.items()}
say(f"[pre-rerank check] {pre} expected {EXPECT} top20 check ok ({time.time() - t0:.0f}s)")
if pre != EXPECT:
    (OUT / "check_fail.json").write_text(json.dumps({"pre_correct": pre, "expected": EXPECT}, indent=1))
    raise SystemExit("pre-rerank correct differs — stop")
for s, rows in items.items():
    for r in rows:
        d = diag[s][r["qid"]]
        r["gate1_bucket"], r["gate1_target"] = d["bucket"], bool(target[s](d))
        if s.startswith("hitab"):
            assert d["gold_cell_rank"] == r["pre_rank"], (s, r["qid"])

from sentence_transformers.cross_encoder import CrossEncoder   # noqa: E402
from transformers import AutoTokenizer                         # noqa: E402
ce = CrossEncoder(RERANKER, device="cuda", max_length=MAXLEN, local_files_only=True)
tok = AutoTokenizer.from_pretrained(RERANKER, local_files_only=True)
secs = {}
for s, rows in items.items():
    t1 = time.time()
    for n, r in enumerate(rows, 1):
        pairs = [(r["question"], t) for t in r.pop("_texts")]
        ids = tok([p[0] for p in pairs], [p[1] for p in pairs], truncation=False)["input_ids"]
        r["n_pairs"], r["n_truncated"] = len(pairs), sum(len(x) > MAXLEN for x in ids)
        sc = np.asarray(ce.predict(pairs, batch_size=POOL, show_progress_bar=False), dtype=np.float64)
        k = np.argsort(-sc, kind="stable")
        post = [r["pool"][i] for i in k]
        r["post_top50"] = [[r["pre_top50"][i][0], round(float(sc[i]), 6)] for i in k]
        pos = {u: j + 1 for j, u in enumerate(post)}
        ranks = [pos.get(u) for u in r["gold_units"]]
        r["post_rank"] = None if None in ranks else max(ranks)
        r["post_correct"] = int(r["post_rank"] is not None and r["post_rank"] <= BUDGET)
        if n % 200 == 0:
            say(f"  {s} {n}/{len(rows)} {time.time() - t1:.0f}s")
    secs[s] = round(time.time() - t1, 1)
    with open(OUT / f"{s}.jsonl", "x", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    say(f"  {s} done {len(rows)} items")
    done[s] = rows

res = {"reranker": {"model": RERANKER, "max_length": MAXLEN, "batch_size": POOL, "dtype": "fp32", "device": "cuda"},
       "meta": meta, "seconds_rerank": secs}
for s in EXPECT:
    rows = done[s]
    z = [r for r in rows if r["path_overlap"] == 0]
    res[s] = {"pairs": sum(r["n_pairs"] for r in rows), "truncated_pairs": sum(r["n_truncated"] for r in rows),
              "mcnemar": mcnemar(rows), "moves": moves(rows), "gate1_buckets": buckets(rows),
              "overlap_zero": {"mcnemar": mcnemar(z), "moves": moves(z), "gate1_buckets": buckets(z)}}
order = sorted(EXPECT, key=lambda s: res[s]["mcnemar"]["p_two_sided"])
run = 0.0
for k, s in enumerate(order):                                     # Holm, 묶음 = 세 범위 (PREREG §9)
    run = max(run, min(1.0, (len(order) - k) * res[s]["mcnemar"]["p_two_sided"]))
    res[s]["mcnemar"]["p_holm_3scopes"] = run
res["mh_indoc"]["by_layer_EXPLORATORY"] = {
    g: {"mcnemar": mcnemar(v), "moves": moves(v)} for g in ("lookup_m1", "lookup_m2+", "arith_m1", "arith_m2+")
    if (v := [r for r in done["mh_indoc"] if r["layer"] == g])}
res["elapsed_s_total_this_run"] = round(time.time() - t0, 1)
(OUT / "summary.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
say(f"[done] summary.json written ({res['elapsed_s_total_this_run']}s)")
