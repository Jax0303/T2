#!/usr/bin/env python3
"""All-datasets retrieval-only sweep: BM25 vs dense vs hybrid (RRF), no LLM.

For each benchmark (hitab / finqa / wikisql) loaded through the unified
``rag_agent.bench.registry``:
  * pool   = every unique table in the benchmark (the gold table is in-pool)
  * each table is serialized once (title + header paths + rows) → one chunk
  * three retrievers rank the whole pool per query:
      bm25    : BM25Okapi (k1=1.5, b=0.75) over the serialized text
      dense   : BAAI/bge-large-en-v1.5, cosine over normalized embeddings (GPU)
      hybrid  : RRF(k=60) of the bm25 + dense rankings (Cormack et al. 2009)
  * metrics : R@1, R@5, R@10, MRR, nDCG@10 (binary single-gold)
  * query-type breakdown: by ``aggregation`` label and by gold-operand-count
    bucket (1 / 2 / 3+), so retrieval is reported across diverse query types.

Corpus embeddings are cached per benchmark in data/emb_cache/md_<bench>_*.npy.
Usage: .venv/bin/python scripts/multidataset_retrieval.py [--benches hitab,finqa,wikisql] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rag_agent.bench import registry  # noqa: E402

EMB_MODEL = "BAAI/bge-large-en-v1.5"
SEED = 42
RRFK = 60
MAX_ROWS = 50          # cap rows per table so the embedder stays within budget
_TOK = re.compile(r"[a-z0-9]+")


def tok(s: str):
    return _TOK.findall(s.lower())


def serialize(t) -> str:
    """Plain serialization shared by BM25 and dense: title + headers + rows."""
    parts = []
    if t.title:
        parts.append(str(t.title))
    cols = [" > ".join(p) if p else "" for p in t.top_paths] or [
        f"col{c}" for c in range(t.n_cols)
    ]
    parts.append(" | ".join(cols))
    for r in range(min(t.n_rows, MAX_ROWS)):
        rowhdr = " > ".join(t.row_path(r)) if t.left_paths else ""
        cells = [str(t.cell(r, c)) for c in range(t.n_cols)]
        line = (rowhdr + " : " if rowhdr else "") + " | ".join(cells)
        parts.append(line)
    return "\n".join(parts)


# ---------- metrics ----------
def metrics_from_ranks(ranks, n):
    def at(k):
        return sum(1 for r in ranks if r is not None and r <= k) / n
    mrr = sum(1.0 / r for r in ranks if r is not None) / n
    ndcg = sum(1.0 / math.log2(r + 1) for r in ranks if r is not None and r <= 10) / n
    return {"r1": at(1), "r5": at(5), "r10": at(10), "mrr": mrr, "ndcg": ndcg, "n": n}


def rank_of(order, gold_idx):
    pos = np.where(order == gold_idx)[0]
    return int(pos[0]) + 1 if len(pos) else None


def breakdown(ranks, keys):
    """Group per-query ranks by a categorical key → metrics dict per group."""
    g = defaultdict(list)
    for r, k in zip(ranks, keys):
        g[k].append(r)
    return {str(k): metrics_from_ranks(rs, len(rs)) for k, rs in sorted(g.items())}


def operand_bucket(q):
    n = len(getattr(q, "gold_operands", []) or [])
    if n <= 1:
        return "1_operand"
    if n == 2:
        return "2_operands"
    return "3+_operands"


def encode_cached(tag, texts, model):
    cache = ROOT / "data" / "emb_cache"
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"{tag}.npy"
    if f.exists():
        emb = np.load(f)
        if emb.shape[0] == len(texts):
            return emb
    t0 = time.time()
    emb = model.encode(texts, batch_size=128, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    np.save(f, emb)
    print(f"    encoded {len(texts)} docs in {time.time()-t0:.1f}s", flush=True)
    return emb


def run_bench(name, model, limit=None):
    print(f"\n{'='*70}\n{name}\n{'='*70}", flush=True)
    queries, tables = registry.load(name)
    if limit:
        queries = queries[:limit]
    table_ids = list(tables.keys())
    tindex = {t: i for i, t in enumerate(table_ids)}
    # keep only queries whose gold is in pool
    queries = [q for q in queries if q.gold_table_id in tindex]
    golds = [tindex[q.gold_table_id] for q in queries]
    questions = [q.question or "" for q in queries]
    n = len(queries)
    print(f"  pool={len(table_ids)} tables | eval={n} queries", flush=True)

    corpus_text = [serialize(tables[t]) for t in table_ids]

    # ---- BM25 ----
    from rank_bm25 import BM25Okapi
    t0 = time.time()
    bm = BM25Okapi([tok(c) for c in corpus_text], k1=1.5, b=0.75)
    bm_order, bm_ranks = [], []
    for q, g in zip(questions, golds):
        order = np.argsort(-bm.get_scores(tok(q)))
        bm_order.append(order)
        bm_ranks.append(rank_of(order, g))
    print(f"  bm25 done {time.time()-t0:.1f}s", flush=True)

    # ---- dense ----
    tag_corpus = f"md_{name}_bge_large"
    doc_emb = encode_cached(tag_corpus, corpus_text, model)
    q_emb = model.encode(questions, batch_size=128, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    sims = q_emb @ doc_emb.T               # [n_q, n_tables]
    dn_order = np.argsort(-sims, axis=1)
    dn_ranks = [rank_of(dn_order[i], golds[i]) for i in range(n)]

    # ---- hybrid RRF ----
    hy_ranks = []
    for i in range(n):
        rrf = defaultdict(float)
        for rank, ti in enumerate(bm_order[i], 1):
            rrf[int(ti)] += 1.0 / (RRFK + rank)
        for rank, ti in enumerate(dn_order[i], 1):
            rrf[int(ti)] += 1.0 / (RRFK + rank)
        ranked = sorted(rrf, key=lambda t: -rrf[t])
        hy_ranks.append(ranked.index(golds[i]) + 1 if golds[i] in ranked else None)

    methods = {"bm25": bm_ranks, "dense": dn_ranks, "hybrid": hy_ranks}
    agg_keys = [str(q.aggregation) for q in queries]
    op_keys = [operand_bucket(q) for q in queries]

    out = {
        "pool_size": len(table_ids), "n_eval": n, "embedder": EMB_MODEL,
        "overall": {m: metrics_from_ranks(r, n) for m, r in methods.items()},
        "by_aggregation": {m: breakdown(r, agg_keys) for m, r in methods.items()},
        "by_operand_count": {m: breakdown(r, op_keys) for m, r in methods.items()},
    }
    # console table
    hdr = f"  {'method':<8}" + "".join(f"{c:>9}" for c in ["R@1", "R@5", "R@10", "MRR", "nDCG"])
    print(hdr)
    for m in methods:
        o = out["overall"][m]
        print(f"  {m:<8}" + "".join(f"{o[k]:>9.3f}" for k in ["r1", "r5", "r10", "mrr", "ndcg"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", default="hitab,finqa,wikisql")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "multidataset_retrieval.json"))
    args = ap.parse_args()
    np.random.seed(SEED)

    from sentence_transformers import SentenceTransformer
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {EMB_MODEL} on {dev}", flush=True)
    model = SentenceTransformer(EMB_MODEL, device=dev)

    results = {}
    for name in [b.strip() for b in args.benches.split(",") if b.strip()]:
        results[name] = run_bench(name, model, args.limit)

    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
