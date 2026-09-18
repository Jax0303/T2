#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Two checks on the structural_leaf hybrid retriever (results/sentence_disambiguation_20260916/):

1. Verify the BGE encoder's ACTUAL pooling/normalization, instead of trusting
   templates.py's STRUCTURAL_LEAF docstring assumption ("mean-pooled embeddings
   weight repeated tokens more") -- read it off the loaded SentenceTransformer's
   own Pooling/Normalize modules.
2. For every hybrid top-1 miss, decompose it by asking each channel alone
   ("if only dense ranked, would gold be #1? if only sparse ranked, would gold
   be #1?"):
     - combination_failure: at least one channel alone already ranks gold #1,
       but the alpha=0.7 fused score does not -- a fusion-weight problem.
     - shared_representation_failure: NEITHER channel alone ranks gold #1 --
       both the embedded sentence and its BM25 tokens fail to distinguish gold
       from the wrong cell, a representation problem, not a fusion one.

  PYTHONPATH=. .venv/bin/python scripts/embedding_fusion_diagnosis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                                                    # noqa: E402

from rag_agent.retrieve.encoders import default_encoder                # noqa: E402
from rag_agent.retrieve.hybrid_index import _minmax, _tokenize         # noqa: E402
from rag_agent.retrieve.sparse_bm25 import SparseBM25                  # noqa: E402
from bottleneck_diagnosis import load_primary_population               # noqa: E402
from sentence_disambiguation_eval import build_leaf_corpus, encode_corpus  # noqa: E402

OUT_DIR = ROOT / "results/embedding_fusion_diagnosis_20260917"
ALPHA = 0.7


def verify_pooling(encoder) -> dict:
    pool, norm = encoder.model[1], encoder.model[2]
    info = {
        "pooling_mode": getattr(pool, "pooling_mode", None),
        "pooling_class": type(pool).__name__,
        "normalize_class": type(norm).__name__,
        "normalize_embeddings_flag_in_encode_call": True,  # rag_agent/retrieve/encoders.py:111
    }
    if info["pooling_mode"] != "mean":
        info["WARNING"] = ("pooling is NOT mean -- templates.py's STRUCTURAL_LEAF/"
                            "STRUCTURAL_LEAF_X2 docstrings justify leaf-repetition via "
                            "'mean-pooled embeddings weight repeated tokens more', which "
                            "does not describe this encoder")
    return info


def classify_miss(dense_alone_correct: bool, sparse_alone_correct: bool) -> str:
    """A hybrid top-1 miss is a fusion problem only if some channel, run alone,
    already ranked gold #1; otherwise neither representation carries enough
    signal and fusion weighting cannot be the cause."""
    if dense_alone_correct or sparse_alone_correct:
        return "combination_failure"
    return "shared_representation_failure"


def run(template_name: str = "structural_leaf", data_dir: str = "data/hitab") -> dict:
    pop = load_primary_population()
    texts, coords = build_leaf_corpus(data_dir, template_name)
    coord_index = {c: idx for idx, c in enumerate(coords)}

    enc = default_encoder()
    pooling_info = verify_pooling(enc)
    emb, cache_hit = encode_corpus(texts, enc)
    bm = SparseBM25(_tokenize(t) for t in texts)

    rows = []
    for qid, r in sorted(pop.items()):
        gold = tuple(sorted(r["gold_cells"])[0])
        gidx = coord_index.get(gold)
        if gidx is None:
            continue
        q = r["question"]
        dense = (emb @ enc.encode_query([q])[0].astype(np.float32))
        sparse = bm.get_scores(_tokenize(q))
        hybrid = ALPHA * _minmax(dense) + (1 - ALPHA) * _minmax(sparse)

        def rank_of(scores):
            order = np.argsort(-scores, kind="stable")
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(order) + 1)
            return int(ranks[gidx]), int(order[0])

        hyb_rank, hyb_top1 = rank_of(hybrid)
        if hyb_rank == 1:
            continue  # only decomposing misses
        dense_rank, dense_top1 = rank_of(dense)
        sparse_rank, sparse_top1 = rank_of(sparse)
        dense_alone_correct = dense_rank == 1
        sparse_alone_correct = sparse_rank == 1
        cls = classify_miss(dense_alone_correct, sparse_alone_correct)
        rows.append({
            "query_id": qid, "hybrid_gold_rank": hyb_rank,
            "dense_gold_rank": dense_rank, "sparse_gold_rank": sparse_rank,
            "dense_alone_correct": dense_alone_correct, "sparse_alone_correct": sparse_alone_correct,
            "dense_top1_cell": list(coords[dense_top1]), "sparse_top1_cell": list(coords[sparse_top1]),
            "same_antagonist": dense_top1 == sparse_top1,
            "dense_margin_gold_minus_top1": float(dense[gidx] - dense[hyb_top1]),
            "sparse_margin_gold_minus_top1": float(sparse[gidx] - sparse[hyb_top1]),
            "class": cls,
        })

    n_miss = len(rows)
    n_comb = sum(r["class"] == "combination_failure" for r in rows)
    n_shared = n_miss - n_comb
    shared_rows = [r for r in rows if r["class"] == "shared_representation_failure"]
    n_shared_same = sum(r["same_antagonist"] for r in shared_rows)
    summary = {
        "template": template_name, "alpha": ALPHA,
        "pooling_verified": pooling_info,
        "n_population": len(pop), "n_hybrid_miss": n_miss,
        "combination_failure": {
            "n": n_comb, "pct_of_misses": round(n_comb / n_miss, 4) if n_miss else None,
        },
        "shared_representation_failure": {
            "n": n_shared, "pct_of_misses": round(n_shared / n_miss, 4) if n_miss else None,
            "same_antagonist_in_both_channels": {
                "n": n_shared_same,
                "pct_of_shared_representation_failures":
                    round(n_shared_same / n_shared, 4) if n_shared else None,
            },
        },
        "mean_dense_margin_gold_minus_top1": round(
            float(np.mean([r["dense_margin_gold_minus_top1"] for r in rows])), 4) if rows else None,
        "mean_sparse_margin_gold_minus_top1": round(
            float(np.mean([r["sparse_margin_gold_minus_top1"] for r in rows])), 4) if rows else None,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"misses_{template_name}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    (OUT_DIR / f"summary_{template_name}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="structural_leaf",
                    choices=("structural_leaf", "structural_leaf_x2"))
    args = ap.parse_args()
    raise SystemExit(0 if run(args.template) else 1)
