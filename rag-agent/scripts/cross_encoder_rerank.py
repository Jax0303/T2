#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan item 0b / item 3 (2026-09-17): off-the-shelf cross-encoder rerank of
the deployed hybrid's top-K pool. [[leaf-boost-rerank-closed-2026-09-17]]
closed item 0a (a flat literal-leaf-token bonus is too non-selective to
help, p=0.69 at the sane bonus, harmful only once cranked up); this escalates
to a model that reads the full candidate text pairwise against the query,
instead of a fixed rule or a single pooled vector. BAAI/bge-reranker-v2-m3
(already cached locally) via sentence_transformers.cross_encoder.CrossEncoder
-- no training, inference only.

Caveat flagged in the plan up front: MS-MARCO-trained cross-encoders
reportedly underweight compressed table-row text (arXiv:2604.01733) --
expectations are lowered accordingly; this checks the claim, not assumes a
win.

Pool = hybrid top-50 (same deployed structural_leaf hybrid, ALPHA=0.7, as
every other script in this investigation -- see leaf_boost_rerank.
build_corpus, reused as-is). The cross-encoder scores all 50 candidates ONCE
per query; K=20 and K=50 rerank are both read off that same pass (K=20 =
hybrid's own top-20 slice of the pool, reranked; K=50 = the full pool,
reranked), so there is no dev-side K selection to prereg or overfit -- both
numbers are reported together, nothing is chosen after the fact.

  PYTHONPATH=. .venv/bin/python scripts/cross_encoder_rerank.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                                                    # noqa: E402
from scipy.stats import binomtest                                     # noqa: E402

from rag_agent.retrieve.encoders import default_encoder                # noqa: E402
from rag_agent.retrieve.hybrid_index import _minmax, _tokenize         # noqa: E402
from rag_agent.retrieve.sparse_bm25 import SparseBM25                  # noqa: E402
from bottleneck_diagnosis import load_primary_population, split_corpus_table_ids  # noqa: E402
from leaf_boost_rerank import build_corpus                            # noqa: E402
from sentence_disambiguation_eval import encode_corpus                # noqa: E402

OUT_DIR = ROOT / "results/cross_encoder_rerank_20260917"
ALPHA = 0.7                     # same deployed hybrid weight as embedding_fusion_diagnosis.py
POOL_SIZE = 50
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


def has_cuda() -> bool:
    import torch
    return torch.cuda.is_available()


def rerank_hits(pop: dict, texts: list, coord_index: dict, emb, bm, enc, ce,
                limit: int = 0) -> dict:
    """(base_hits, k20_hits, k50_hits) -- per-query hybrid-top1 vs cross-encoder
    -reranked-top1 correctness, keyed by query_id, same population for all
    three so McNemar can pair them directly."""
    base_hits, k20_hits, k50_hits = {}, {}, {}
    ordered = sorted(pop.items())
    if limit:
        ordered = ordered[:limit]
    for qid, r in ordered:
        gold = tuple(sorted(r["gold_cells"])[0])
        gidx = coord_index.get(gold)
        if gidx is None:
            continue
        q = r["question"]
        dense = emb @ enc.encode_query([q])[0].astype(np.float32)
        sparse = bm.get_scores(_tokenize(q))
        hybrid = ALPHA * _minmax(dense) + (1 - ALPHA) * _minmax(sparse)
        pool = np.argsort(-hybrid, kind="stable")[:POOL_SIZE]
        ce_scores = np.asarray(ce.predict([(q, texts[i]) for i in pool],
                                          batch_size=POOL_SIZE, show_progress_bar=False))
        base_hits[qid] = bool(pool[0] == gidx)
        k20_hits[qid] = bool(pool[:20][int(np.argmax(ce_scores[:20]))] == gidx)
        k50_hits[qid] = bool(pool[int(np.argmax(ce_scores))] == gidx)
    return base_hits, k20_hits, k50_hits


def mcnemar(a: dict, b: dict, a_name: str, b_name: str) -> dict:
    qids = sorted(set(a) & set(b))
    a_only = sum(a[q] and not b[q] for q in qids)
    b_only = sum(b[q] and not a[q] for q in qids)
    n_disc = a_only + b_only
    p = binomtest(min(a_only, b_only), n_disc, 0.5).pvalue if n_disc else 1.0
    return {"n": len(qids), f"{a_name}_r1": round(sum(a.values()) / len(qids), 4),
           f"{b_name}_r1": round(sum(b.values()) / len(qids), 4),
           f"{a_name}_only": a_only, f"{b_name}_only": b_only,
           "discordant": n_disc, "p_value": round(p, 4)}


def run(data_dir: str = "data/hitab", limit: int = 0) -> dict:
    from sentence_transformers.cross_encoder import CrossEncoder

    pop = load_primary_population()
    tids = split_corpus_table_ids()
    texts, _row_leaf, _col_leaf, coords = build_corpus(tids, data_dir)
    coord_index = {c: i for i, c in enumerate(coords)}

    enc = default_encoder()
    emb, cache_hit = encode_corpus(texts, enc)
    bm = SparseBM25(_tokenize(t) for t in texts)
    ce = CrossEncoder(RERANK_MODEL, device="cuda" if has_cuda() else "cpu", max_length=512)

    t0 = time.time()
    base_hits, k20_hits, k50_hits = rerank_hits(pop, texts, coord_index, emb, bm, enc, ce, limit)
    elapsed = time.time() - t0

    summary = {
        "n": len(base_hits), "pool_size": POOL_SIZE, "rerank_model": RERANK_MODEL,
        "elapsed_s": round(elapsed, 1), "embed_cache_hit": cache_hit,
        "baseline_vs_k20": mcnemar(base_hits, k20_hits, "baseline", "k20"),
        "baseline_vs_k50": mcnemar(base_hits, k50_hits, "baseline", "k50"),
        "k20_vs_k50": mcnemar(k20_hits, k50_hits, "k20", "k50"),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    n = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
    raise SystemExit(0 if run(limit=n) else 1)
