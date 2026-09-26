"""2026-09-27 PREREG-2026-09-27-rerank.md — 진단 실험: s3c 하이브리드 상위 50 을 bge-reranker-v2-m3 로 재정렬해 상위 20 문맥.

dev 두 집합만 (test 는 이 스크립트가 읽지 않는다):
  HiTab dev 질문의 표 안, 단일 셀 1,062 — 기준선 results/dev_alpha_20260926/hitab_dev/hitab_dev_gold_prefix (s3c)
  MultiHiertt validation 문서 안 911 — 기준선 results/dev_alpha_20260926/mh_dev/mh_dev_a1.0 (s3c, 라벨 없음)
재정렬 전 순위 = retrieval_accuracy.py / mh_arms.py 의 hybrid(α=.7) 점수 식 그대로, 저장된 임베딩으로 다시 계산.
재계산 상위 20 이 기준선 records 와 문항 전부 같고 맞힘 수가 같아야 진행한다(아니면 멈춘다).

모드:
  --check           재정렬 없이 기준선 재현, 쌍 수, 절단 건수(토크나이저만) -> dev/check.json
  --throughput 100  HiTab dev 첫 100쌍만 재정렬기에 넣어 초당 쌍 수 -> dev/throughput.json
  (없음)            dev 전체 실행 -> dev/{hitab_dev,mh_dev}.jsonl, dev/summary.json
실행: HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=. .venv/bin/python results/bottleneck_20260927/rerank_dev.py [모드]
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / "dev"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
ap = argparse.ArgumentParser()
ap.add_argument("--check", action="store_true")
ap.add_argument("--throughput", type=int, default=0)
a = ap.parse_args()
sys.argv = sys.argv[:1]
import retrieval_accuracy as ra                                       # noqa: E402
import mh_arms as mh                                                  # noqa: E402
from rag_agent.eval.artifacts import digest                           # noqa: E402
from rag_agent.retrieve.encoders import _tokenize, default_encoder    # noqa: E402
from rag_agent.retrieve.hybrid_index import _minmax                   # noqa: E402
from rag_agent.retrieve.sparse_bm25 import SparseBM25                 # noqa: E402

ALPHA, POOL, BUDGET, MAXLEN = 0.7, 50, 20, 512
RERANKER = "BAAI/bge-reranker-v2-m3"
DEV = ROOT / "results/dev_alpha_20260926"
words = lambda s: set(_tokenize(s)) - ENGLISH_STOP_WORDS
OUT.mkdir(exist_ok=True)
log = open(OUT / ("check.log" if a.check else "throughput.log" if a.throughput else "run.log"), "a", encoding="utf-8")


def say(*x):
    print(*x, flush=True)
    print(*x, file=log, flush=True)


def overlap(question, paths):
    q = words(question)
    p = set().union(*(words(" ".join(x)) for x in paths))
    return round(len(q & p) / len(q), 4) if q else None


def item(ds, qid, question, layer, gold, order, scores, covers, texts, cid, paths, ref_top, ref_correct):
    """재정렬 전 순위(order: 전체 또는 상위 2048)에서 상위 50 후보와 진단값. 기준선 records 와 대조한다."""
    sel = ra.budget_select(order, covers, texts, BUDGET, 0)
    assert ref_top is None or [int(p) for p in order[:len(ref_top)]] == ref_top, (ds, qid, "top20 mismatch")
    pos = {int(p): k + 1 for k, p in enumerate(order)}
    ranks = [pos.get(u) for u in gold.values()]
    pre_rank = None if None in ranks else max(ranks)
    correct = int(set(gold) <= sel.cells)
    assert correct == ref_correct, (ds, qid, "correct mismatch")
    assert correct == int(pre_rank is not None and pre_rank <= BUDGET)
    pool = [int(p) for p in order[:POOL]]
    return {"dataset": ds, "qid": qid, "question": question, "layer": layer, "n_gold": len(gold),
            "gold": [cid(c) for c in sorted(gold)], "gold_units": [gold[c] for c in sorted(gold)],
            "pre_rank": pre_rank, "pre_correct": correct, "path_overlap": overlap(question, paths),
            "pool": pool, "pre_top50": [[cid(next(iter(covers[p]))), round(float(scores[p]), 6)] for p in pool],
            "_texts": [texts[p] for p in pool]}


def hitab_items(enc):
    stem = DEV / "hitab_dev/hitab_dev_gold_prefix"
    summ = json.loads(Path(f"{stem}.json").read_text())
    ta = {r["query_id"]: r for r in map(json.loads, open(f"{stem}_type_accuracy.jsonl")) if r["query_type"] == "single_cell"}
    rec = {r["query_id"]: r for r in map(json.loads, open(f"{stem}_records.jsonl")) if r["query_id"] in ta}
    assert len(ta) == len(rec) == 1062 and sum(r["retrieval_success"] for r in ta.values()) == 1005
    pages = json.loads(ra.PAGE_TITLES.read_text())
    tabs = {}
    allq = ra.load_queries("data/hitab", "dev", tabs)
    tids = sorted({q["table_id"] for q in allq})
    queries = [q for q in allq if q["query_id"] in ta]
    texts, covers, _, unit_tids, _ = ra.build_corpus("data/hitab", tids, "s3c", "cell", pages, 1000, "leaf", "values", 200, None)
    assert digest(texts) == summ["corpus_text_sha256"]
    key = digest({"texts": texts, "encoder": enc.metadata(), "overflow": "error"})[:24]
    f = ROOT / f".cache/dev_alpha_20260926/{enc.name.replace('/', '_')}_{len(texts)}_{key}.npy"
    emb = np.load(f)
    bm = SparseBM25(_tokenize(t) for t in texts)
    tid_arr = np.array(unit_tids)
    unit_of = {next(iter(c)): i for i, c in enumerate(covers)}
    meta = {"records": f"{stem}_records.jsonl", "embeddings": str(f.relative_to(ROOT)), "n_units": len(texts),
            "n_tables": len(tids), "corpus_text_sha256": digest(texts)}
    out = []
    for q in queries:
        qid = q["query_id"]
        sel = np.flatnonzero(tid_arr == q["table_id"])
        s = bm.get_scores(_tokenize(q["question"]))[sel]
        d = (emb @ enc.encode_query([q["question"]])[0].astype(np.float32))[sel]
        sc = np.zeros(len(texts), dtype=np.float64)
        sc[sel] = ALPHA * _minmax(d) + (1 - ALPHA) * _minmax(s)
        order = sel[np.argsort(-sc[sel], kind="stable")]
        t = tabs[q["table_id"]].table
        gold = {c: unit_of[c] for c in q["gold"]}
        assert sorted(map(list, gold)) == sorted(ta[qid]["gold_cell_ids"])
        out.append(item("hitab_dev_intable", qid, q["question"], "single_cell", gold, order, sc, covers, texts,
                        lambda c: f"{c[0]}-{c[1]}-{c[2]}", [[*t.row_path(i), *t.col_path(j)] for _, i, j in gold],
                        [u["index_unit"] for u in rec[qid]["context_units"]], ta[qid]["retrieval_success"]))
    return out, meta


def mh_items(enc):
    stem = DEV / "mh_dev/mh_dev_a1.0"
    summ = json.loads(Path(f"{stem}.json").read_text())
    rec = {r["query_id"]: r for r in map(json.loads, open(f"{stem}_records.jsonl")) if "doc" in r}
    assert len(rec) == 911 and sum(r["doc"]["correct"] for r in rec.values()) == 774
    qs, docs, _ = mh.load_population("validation", keep_hybrid=True)
    tables, hdr = mh.build_tables(docs, "v3.3u", "none")
    texts, covers, _, unit_tids, _ = ra.build_corpus("", sorted(tables), "s3c", "cell", {}, 1000, "leaf", "values", 200,
                                                      None, load=lambda tid, _d: tables.get(tid))
    assert digest(texts) == summ["corpus_text_sha256"]
    live = {(tid, i, j) for tid, tb in tables.items() for i, row in enumerate(tb.table.data)
            for j, v in enumerate(row) if str(v).strip()}
    qs = [q for q in mh.resolve_gold(qs, tables, hdr, live) if not q["excluded"] and q["gold"]]
    assert {q["uid"] for q in qs} == set(rec)
    key = digest(texts)[:16]
    emb = np.empty((len(texts), 768), dtype=np.float32)
    files = []
    for s in range(0, len(texts), 50000):
        f = ROOT / f".cache/dev_alpha_20260926/cell_{len(texts)}_{key}_{s}.npy"
        v = np.load(f)
        emb[s:s + len(v)] = v
        files.append(str(f.relative_to(ROOT)))
    bm = SparseBM25(_tokenize(t) for t in texts)
    uid_arr = np.array([t.split("::")[0] for t in unit_tids])
    unit_of = {next(iter(c)): i for i, c in enumerate(covers)}
    meta = {"records": f"{stem}_records.jsonl", "embeddings": files, "n_units": len(texts), "n_tables": len(tables),
            "corpus_text_sha256": digest(texts), "label_mix": "none (기준선 a1.0 의 α=1.0 섞기는 적용하지 않음 — 상위 20 대조로 확인)"}
    out = []
    for q in qs:
        sc = ALPHA * _minmax(emb @ enc.encode_query([q["question"]])[0].astype(np.float32)) + \
            (1 - ALPHA) * _minmax(bm.get_scores(_tokenize(q["question"])))
        sel = np.flatnonzero(uid_arr == q["uid"])
        k = min(2048, len(sel) - 1)
        top = sel[np.argpartition(-sc[sel], k)[:k + 1]]
        order = top[np.argsort(-sc[top], kind="stable")]
        r = rec[q["uid"]]
        assert [texts[p] for p in order[:len(r["doc"]["context"])]] == r["doc"]["context"], (q["uid"], "top20 mismatch")
        gold = {c: unit_of[c] for c in q["gold"]}
        paths = [[*tables[c[0]].table.row_path(c[1]), *tables[c[0]].table.col_path(c[2])] for c in gold]
        out.append(item("mh_dev_indoc", q["uid"], q["question"], r["layer"], gold, order, sc, covers, texts,
                        lambda c: mh.cell_id(c, hdr), paths, None, r["doc"]["correct"]))
    return out, meta


def load_ce():
    from sentence_transformers.cross_encoder import CrossEncoder
    from transformers import AutoTokenizer
    ce = CrossEncoder(RERANKER, device="cuda", max_length=MAXLEN, local_files_only=True)
    return ce, AutoTokenizer.from_pretrained(RERANKER, local_files_only=True)


def n_truncated(tok, pairs):
    enc = tok([p[0] for p in pairs], [p[1] for p in pairs], truncation=False)
    return sum(len(x) > MAXLEN for x in enc["input_ids"])


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


t0 = time.time()
enc = default_encoder(model_name="BAAI/bge-base-en-v1.5")
if a.throughput:
    items, _ = hitab_items(enc)
    ce, _tok = load_ce()
    pairs = [(it["question"], t) for it in items for t in it["_texts"]][:a.throughput]
    t1 = time.time()
    ce.predict(pairs, batch_size=POOL, show_progress_bar=False)
    dt = time.time() - t1
    chk = json.loads((OUT / "check.json").read_text())
    rate = len(pairs) / dt
    res = {"pairs": len(pairs), "seconds": round(dt, 3), "pairs_per_second": round(rate, 2),
           "estimate_hours_98650_pairs": round(98650 / rate / 3600, 3),
           "actual_dev_pairs": chk["pairs_total"], "estimate_hours_actual_pairs": round(chk["pairs_total"] / rate / 3600, 3),
           "note": "모델 적재 뒤 첫 predict 호출 1회(예열 없음), batch_size 50, fp32, cuda"}
    (OUT / "throughput.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    say(json.dumps(res, ensure_ascii=False))
    raise SystemExit(0)

hi, hmeta = hitab_items(enc)
say(f"[hitab dev] {len(hi)} items, pre correct {sum(r['pre_correct'] for r in hi)}, top20 check ok ({time.time() - t0:.0f}s)")
mi, mmeta = mh_items(enc)
say(f"[mh dev] {len(mi)} items, pre correct {sum(r['pre_correct'] for r in mi)}, top20 check ok ({time.time() - t0:.0f}s)")
ce, tok = (None, None) if a.check else load_ce()
if a.check:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(RERANKER, local_files_only=True)
res = {}
for name, rows, meta in (("hitab_dev", hi, hmeta), ("mh_dev", mi, mmeta)):
    t1, trunc = time.time(), 0
    for n, r in enumerate(rows, 1):
        texts = r.pop("_texts")
        pairs = [(r["question"], t) for t in texts]
        r["n_pairs"], r["n_truncated"] = len(pairs), n_truncated(tok, pairs)
        trunc += r["n_truncated"]
        if a.check:
            continue
        s = np.asarray(ce.predict(pairs, batch_size=POOL, show_progress_bar=False), dtype=np.float64)
        k = np.argsort(-s, kind="stable")
        post = [r["pool"][i] for i in k]
        r["post_top50"] = [[r["pre_top50"][i][0], round(float(s[i]), 6)] for i in k]
        pos = {u: j + 1 for j, u in enumerate(post)}
        ranks = [pos.get(u) for u in r["gold_units"]]
        r["post_rank"] = None if None in ranks else max(ranks)
        r["post_correct"] = int(r["post_rank"] is not None and r["post_rank"] <= BUDGET)
        if n % 200 == 0:
            say(f"  {name} {n}/{len(rows)} {time.time() - t1:.0f}s")
    res[name] = {"meta": meta, "pairs": sum(r["n_pairs"] for r in rows), "truncated_pairs": trunc,
                 "seconds": round(time.time() - t1, 1)}
    if a.check:
        continue
    with open(OUT / f"{name}.jsonl", "x", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    z = [r for r in rows if r["path_overlap"] == 0]
    res[name].update(mcnemar=mcnemar(rows), moves=moves(rows), overlap_zero={"mcnemar": mcnemar(z), "moves": moves(z)})
    if name == "mh_dev":
        res[name]["by_layer"] = {g: {"mcnemar": mcnemar(v), "moves": moves(v)} for g in ("lookup_m1", "lookup_m2+", "arith_m1", "arith_m2+")
                                 if (v := [r for r in rows if r["layer"] == g])}
if a.check:
    chk = {"hitab_dev": res["hitab_dev"], "mh_dev": res["mh_dev"],
           "pairs_total": res["hitab_dev"]["pairs"] + res["mh_dev"]["pairs"],
           "truncated_total": res["hitab_dev"]["truncated_pairs"] + res["mh_dev"]["truncated_pairs"],
           "pre_correct": {"hitab_dev": sum(r["pre_correct"] for r in hi), "mh_dev": sum(r["pre_correct"] for r in mi)},
           "layers_mh": dict(Counter(r["layer"] for r in mi))}
    (OUT / "check.json").write_text(json.dumps(chk, indent=1, ensure_ascii=False))
    say(json.dumps(chk, indent=1, ensure_ascii=False))
    raise SystemExit(0)
h, m = res["hitab_dev"]["mcnemar"], res["mh_dev"]["mcnemar"]
res["acceptance"] = {
    "rule": "두 dev 모두 post_correct >= pre_correct, 그리고 적어도 한 dev 에서 c > b 이고 McNemar 양측 p < .05",
    "both_not_lower": h["post_correct"] >= h["pre_correct"] and m["post_correct"] >= m["pre_correct"],
    "one_sig_increase": any(x["c_post_only"] > x["b_pre_only"] and x["p_two_sided"] < .05 for x in (h, m))}
res["acceptance"]["met"] = res["acceptance"]["both_not_lower"] and res["acceptance"]["one_sig_increase"]
res["reranker"] = {"model": RERANKER, "max_length": MAXLEN, "batch_size": POOL, "dtype": "fp32", "device": "cuda"}
res["elapsed_s_total"] = round(time.time() - t0, 1)
(OUT / "summary.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
say(json.dumps(res, indent=1, ensure_ascii=False))
