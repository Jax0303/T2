#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Priority (b) of the 2026-09-17 post-leaf-repeat investigation: does a
bigger dense encoder move R@1 past the ~.62 ceiling every structural_leaf
text variant hit, holding the text (structural_leaf) and hybrid formula
(alpha=0.7, same BM25 index -- BM25 doesn't depend on the dense encoder)
fixed? BAAI/bge-large-en-v1.5 is already cached locally (no download);
BAAI/bge-base-en-v1.5 is the deployed baseline.

  PYTHONPATH=. .venv/bin/python scripts/encoder_swap_eval.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                                                    # noqa: E402

from rag_agent.retrieve.encoders import default_encoder                # noqa: E402
from rag_agent.retrieve.hybrid_index import _minmax, _tokenize         # noqa: E402
from rag_agent.retrieve.sparse_bm25 import SparseBM25                  # noqa: E402
from bottleneck_diagnosis import classify_top1_error, load_primary_population  # noqa: E402
from sentence_disambiguation_eval import build_leaf_corpus, encode_corpus  # noqa: E402

OUT_DIR = ROOT / "results/embedding_fusion_diagnosis_20260917"
ALPHA = 0.7
ENCODERS = ("BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5")


def run(data_dir: str = "data/hitab") -> dict:
    pop = load_primary_population()
    texts, coords = build_leaf_corpus(data_dir, "structural_leaf")
    coord_index = {c: idx for idx, c in enumerate(coords)}
    bm = SparseBM25(_tokenize(t) for t in texts)  # encoder-independent, built once

    ordered = sorted(pop.items())
    tabs: dict = {}
    results = {}
    for model_name in ENCODERS:
        enc = default_encoder(model_name=model_name)
        emb, cache_hit = encode_corpus(texts, enc)
        rows = []
        for qid, r in ordered:
            gold = tuple(sorted(r["gold_cells"])[0])
            gidx = coord_index.get(gold)
            if gidx is None:
                continue
            dense = emb @ enc.encode_query([r["question"]])[0].astype(np.float32)
            sparse = bm.get_scores(_tokenize(r["question"]))
            hybrid = ALPHA * _minmax(dense) + (1 - ALPHA) * _minmax(sparse)
            order = np.argsort(-hybrid, kind="stable")
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(order) + 1)
            gold_rank = int(ranks[gidx])
            top1 = coords[int(order[0])]
            cls = "exact_match" if top1 == gold else classify_top1_error(top1, gold, tabs, data_dir)
            rows.append({"gold_rank": gold_rank, "error_class": cls})
        ranks_arr = [r["gold_rank"] for r in rows]
        counts = Counter(r["error_class"] for r in rows)
        results[model_name] = {
            "recall_at_1": round(sum(x == 1 for x in ranks_arr) / len(ranks_arr), 4),
            "n": len(ranks_arr), "embed_cache_hit": cache_hit,
            "error_class_counts": dict(counts),
        }
        print(model_name, json.dumps(results[model_name], ensure_ascii=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "encoder_swap_eval.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
