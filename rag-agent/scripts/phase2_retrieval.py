#!/usr/bin/env python3
"""Phase 2 — 검색 베이스라인 (라우팅 없이), 전체 3597표 풀.

베이스라인:
  bm25     : rank_bm25 BM25Okapi, k1×b grid 튜닝 (plain_markdown 텍스트)
  dense    : bge-large-en-v1.5, 3 직렬화 각각 (GPU 인코딩, cosine/IP, 표당 max-pool)
  hybrid   : RRF(k=60)  bm25 + best-dense  (Cormack et al. 2009)
검색 풀 = 전체 3597 고유 표.

지표: R@1, R@5, R@10, MRR, nDCG@10 (binary single-gold).
통계: dense-best 대비 paired bootstrap 95% CI (1000 resample, seed=42).

오프라인 1회 인코딩 후 .npy 캐시. seed=42 고정.
사용: python scripts/phase2_retrieval.py [--split dev] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
EMB_MODEL = "BAAI/bge-large-en-v1.5"
SERIALIZERS = ["plain_markdown", "json_kv", "header_path"]
_TOK = re.compile(r"[a-z0-9]+")


def tok(s: str):
    return _TOK.findall(s.lower())


def load_tables():
    return [json.loads(l)["table_id"] for l in open(ROOT / "corpus" / "tables.jsonl")]


def load_serialization(name):
    d = defaultdict(list)
    for l in open(ROOT / "corpus" / "serialized" / f"{name}.records.jsonl"):
        r = json.loads(l)
        d[r["table_id"]].append(r["text"])
    return d


def load_queries(split, limit=None):
    qs = [json.loads(l) for l in open(ROOT / "queries.jsonl")]
    qs = [q for q in qs if q["split"] == split]
    return qs[:limit] if limit else qs


# ---------- metrics ----------
def metrics_from_ranks(ranks, n):
    def at(k):
        return sum(1 for r in ranks if r is not None and r <= k) / n
    mrr = sum(1.0 / r for r in ranks if r is not None) / n
    ndcg = sum(1.0 / math.log2(r + 1) for r in ranks if r is not None and r <= 10) / n
    return {"r1": at(1), "r5": at(5), "r10": at(10), "mrr": mrr, "ndcg": ndcg}


def paired_bootstrap(a_hit, b_hit, B=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    a = np.asarray(a_hit, float); b = np.asarray(b_hit, float)
    n = len(a)
    base = float(b.mean() - a.mean())
    idx = rng.integers(0, n, size=(B, n))
    diffs = np.sort((b[idx] - a[idx]).mean(axis=1))
    lo, hi = float(diffs[int(0.025 * B)]), float(diffs[int(0.975 * B)])
    return {"delta": base, "ci": [lo, hi], "sig": bool(lo > 0 or hi < 0)}


# ---------- dense (vectorized scatter-max) ----------
def encode_corpus(name, table_ids, model):
    cache = ROOT / "data" / "emb_cache"
    cache.mkdir(parents=True, exist_ok=True)
    tag = EMB_MODEL.split("/")[-1].replace("-", "_").replace(".", "_")
    f_emb, f_ids = cache / f"{name}_{tag}.npy", cache / f"{name}_{tag}.ids.json"
    if f_emb.exists() and f_ids.exists():
        return np.load(f_emb), json.loads(f_ids.read_text())
    ser = load_serialization(name)            # only read the corpus file on a cache miss
    texts, owners = [], []
    for tid in table_ids:
        for t in ser[tid]:
            texts.append(t); owners.append(tid)
    t0 = time.time()
    emb = model.encode(texts, batch_size=128, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    print(f"  [{name}] encoded {len(texts)} chunks in {time.time()-t0:.1f}s", flush=True)
    np.save(f_emb, emb); f_ids.write_text(json.dumps(owners))
    return emb, owners


def table_score_matrix(q_embs, doc_emb, owners, table_index):
    own_idx = np.array([table_index[t] for t in owners], dtype=np.int64)
    n_tables, n_q = len(table_index), q_embs.shape[0]
    scores = np.full((n_tables, n_q), -1e9, dtype=np.float32)
    B = 8192
    qT = q_embs.T.astype(np.float32)
    for s in range(0, doc_emb.shape[0], B):
        sims = doc_emb[s:s + B] @ qT          # [b, n_q]
        np.maximum.at(scores, own_idx[s:s + B], sims)
    return scores  # [n_tables, n_q]


def ranks_from_scores(scores, table_index, golds):
    order = np.argsort(-scores, axis=0)        # [n_tables, n_q]
    gold_idx = np.array([table_index[g] for g in golds])
    # position of gold row within each column's order
    ranks = []
    for qi in range(scores.shape[1]):
        pos = np.where(order[:, qi] == gold_idx[qi])[0]
        ranks.append(int(pos[0]) + 1 if len(pos) else None)
    return ranks, order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "phase2_retrieval.json"))
    args = ap.parse_args()
    random.seed(SEED); np.random.seed(SEED)

    table_ids = load_tables()
    table_index = {t: i for i, t in enumerate(table_ids)}
    queries = load_queries(args.split, args.limit)
    golds = [q["gold_table_id"] for q in queries]
    questions = [q["question"] or "" for q in queries]
    n = len(queries)
    print(f"pool={len(table_ids)} tables, eval={n} {args.split} queries", flush=True)

    out = {"config": {"pool_size": len(table_ids), "n_eval": n, "split": args.split,
                      "embedder": EMB_MODEL, "seed": SEED,
                      "protocol": "full-corpus DTR R@k; binary single-gold nDCG; max-pool chunks"},
           "overall": {}, "grids": {}, "per_query_r1": {}}

    # ---- BM25 (tuned grid) ----
    ser_pm = load_serialization("plain_markdown")
    corpus_tok = [tok(ser_pm[t][0]) for t in table_ids]
    q_tok = [tok(q) for q in questions]
    from rank_bm25 import BM25Okapi
    best = {"mrr": -1}
    grid = []
    t0 = time.time()
    for k1 in (0.9, 1.2, 1.6):
        for b in (0.4, 0.75):
            bm = BM25Okapi(corpus_tok, k1=k1, b=b)
            ranks, top1000 = [], []
            for qt, g in zip(q_tok, golds):
                order = np.argsort(-bm.get_scores(qt))
                top1000.append(order[:1000])      # kept for the winning config's RRF pass
                gi = table_index[g]
                pos = np.where(order == gi)[0]
                ranks.append(int(pos[0]) + 1 if len(pos) else None)
            m = metrics_from_ranks(ranks, n)
            grid.append({"k1": k1, "b": b, **m})
            print(f"  bm25 k1={k1} b={b} mrr={m['mrr']:.4f} r1={m['r1']:.4f}", flush=True)
            if m["mrr"] > best["mrr"]:
                best = {"mrr": m["mrr"], "k1": k1, "b": b, "ranks": ranks, "top1000": top1000}
    out["grids"]["bm25"] = grid
    bm25_ranks, bm25_top1000 = best["ranks"], best["top1000"]
    out["overall"]["bm25"] = {"best_k1": best["k1"], "best_b": best["b"],
                              **metrics_from_ranks(bm25_ranks, n)}
    out["per_query_r1"]["bm25"] = [int(r is not None and r <= 1) for r in bm25_ranks]
    print(f"  bm25 grid {time.time()-t0:.1f}s; best k1={best['k1']} b={best['b']}", flush=True)

    # ---- dense (3 serializations) ----
    from sentence_transformers import SentenceTransformer
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMB_MODEL, device=dev)
    print(f"  dense device={dev}", flush=True)
    q_embs = model.encode(questions, batch_size=128, convert_to_numpy=True,
                          normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    dense_order = {}
    for name in SERIALIZERS:
        doc_emb, owners = encode_corpus(name, table_ids, model)
        scores = table_score_matrix(q_embs, doc_emb, owners, table_index)
        ranks, order = ranks_from_scores(scores, table_index, golds)
        dense_order[name] = order
        out["overall"][f"dense_{name}"] = metrics_from_ranks(ranks, n)
        out["per_query_r1"][f"dense_{name}"] = [int(r is not None and r <= 1) for r in ranks]
        print(f"  dense[{name}] r1={out['overall'][f'dense_{name}']['r1']:.4f} "
              f"mrr={out['overall'][f'dense_{name}']['mrr']:.4f}", flush=True)

    best_dense = max(SERIALIZERS, key=lambda s: out["overall"][f"dense_{s}"]["mrr"])
    out["config"]["best_dense_serializer"] = best_dense

    # ---- hybrid RRF(k=60): bm25 + best dense ----
    RRFK = 60
    d_order = dense_order[best_dense]  # [n_tables, n_q]
    hyb_ranks = []
    for i in range(n):
        rrf = defaultdict(float)
        bo = bm25_top1000[i]               # reuse winning grid pass; no re-scoring
        for rank, ti in enumerate(bo, 1):
            rrf[int(ti)] += 1.0 / (RRFK + rank)
        for rank, ti in enumerate(d_order[:1000, i], 1):
            rrf[int(ti)] += 1.0 / (RRFK + rank)
        ranked = sorted(rrf, key=lambda t: -rrf[t])
        gi = table_index[golds[i]]
        hyb_ranks.append(ranked.index(gi) + 1 if gi in ranked else None)
    out["overall"]["hybrid_rrf"] = metrics_from_ranks(hyb_ranks, n)
    out["per_query_r1"]["hybrid_rrf"] = [int(r is not None and r <= 1) for r in hyb_ranks]
    print(f"  hybrid r1={out['overall']['hybrid_rrf']['r1']:.4f} "
          f"mrr={out['overall']['hybrid_rrf']['mrr']:.4f}", flush=True)

    # ---- paired stats vs best-dense ----
    ref = out["per_query_r1"][f"dense_{best_dense}"]
    out["paired_r1_vs_best_dense"] = {
        name: paired_bootstrap(ref, hits)
        for name, hits in out["per_query_r1"].items() if name != f"dense_{best_dense}"
    }

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
