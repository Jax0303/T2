#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""2x2 ablation: does the STRUCTURAL_LEAF row/col leaf-prefix repeat help via
the dense channel, the sparse (BM25) channel, both, or neither? Each channel
independently sees either the plain STRUCTURAL_COMPACT text or the
STRUCTURAL_LEAF text; the other channel's choice is held fixed while the 2x2
grid varies both. Same population/alpha/corpus split as
results/sentence_disambiguation_20260916/ -- only which text each channel
scores changes.

  PYTHONPATH=. .venv/bin/python scripts/leaf_channel_ablation.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                                                    # noqa: E402

from rag_agent.bench import hitab_grid as hg                          # noqa: E402
from rag_agent.retrieve.encoders import default_encoder                # noqa: E402
from rag_agent.retrieve.hybrid_index import _minmax, _tokenize         # noqa: E402
from rag_agent.retrieve.sparse_bm25 import SparseBM25                  # noqa: E402
from rag_agent.serialization.caption import caption_sentence, with_page_title  # noqa: E402
from rag_agent.serialization.templates import STRUCTURAL_COMPACT, STRUCTURAL_LEAF  # noqa: E402
from scripts.retrieval_accuracy import PAGE_TITLES                    # noqa: E402
from bottleneck_diagnosis import classify_top1_error, load_primary_population, split_corpus_table_ids  # noqa: E402
from sentence_disambiguation_eval import encode_corpus                # noqa: E402

OUT_DIR = ROOT / "results/embedding_fusion_diagnosis_20260917"
ALPHA = 0.7
_PAGE_TITLES = json.loads(PAGE_TITLES.read_text()) if PAGE_TITLES.exists() else {}


def build_dual_corpus(data_dir: str = "data/hitab"):
    """Same cell/coordinate order for both variants -- only the template differs."""
    plain, leaf, coords = [], [], []
    for tid in split_corpus_table_ids():
        tab = hg.load_table(tid, data_dir)
        if tab is None:
            continue
        t = tab.table
        title = with_page_title(tab.title, _PAGE_TITLES.get(tid))
        for i in range(t.n_rows):
            for j in range(t.n_cols):
                v = t.data[i][j]
                if not str(v).strip():
                    continue
                rp, cp = t.row_path(i), t.col_path(j)
                plain.append(caption_sentence(title, rp, cp, value=v, template=STRUCTURAL_COMPACT))
                leaf.append(caption_sentence(title, rp, cp, value=v, template=STRUCTURAL_LEAF))
                coords.append((tid, i, j))
    return plain, leaf, coords


def run(data_dir: str = "data/hitab") -> dict:
    pop = load_primary_population()
    plain_texts, leaf_texts, coords = build_dual_corpus(data_dir)
    coord_index = {c: idx for idx, c in enumerate(coords)}

    enc = default_encoder()
    emb = {"plain": encode_corpus(plain_texts, enc)[0], "leaf": encode_corpus(leaf_texts, enc)[0]}
    bm = {"plain": SparseBM25(_tokenize(t) for t in plain_texts),
         "leaf": SparseBM25(_tokenize(t) for t in leaf_texts)}

    ordered = sorted(pop.items())
    qvecs = {qid: enc.encode_query([r["question"]])[0].astype(np.float32) for qid, r in ordered}
    qtoks = {qid: _tokenize(r["question"]) for qid, r in ordered}

    tabs: dict = {}
    grid = {}
    for dense_choice in ("plain", "leaf"):
        for sparse_choice in ("plain", "leaf"):
            cell_rows = []
            for qid, r in ordered:
                gold = tuple(sorted(r["gold_cells"])[0])
                gidx = coord_index.get(gold)
                if gidx is None:
                    continue
                dense = emb[dense_choice] @ qvecs[qid]
                sparse = bm[sparse_choice].get_scores(qtoks[qid])
                hybrid = ALPHA * _minmax(dense) + (1 - ALPHA) * _minmax(sparse)
                order = np.argsort(-hybrid, kind="stable")
                ranks = np.empty_like(order)
                ranks[order] = np.arange(1, len(order) + 1)
                gold_rank = int(ranks[gidx])
                top1 = coords[int(order[0])]
                cls = ("exact_match" if top1 == gold
                      else classify_top1_error(top1, gold, tabs, data_dir))
                cell_rows.append({"query_id": qid, "gold_rank": gold_rank, "error_class": cls})

            ranks_arr = [r["gold_rank"] for r in cell_rows]
            counts = Counter(r["error_class"] for r in cell_rows)
            key = f"dense={dense_choice},sparse={sparse_choice}"
            grid[key] = {
                "recall_at_1": round(sum(x == 1 for x in ranks_arr) / len(ranks_arr), 4),
                "n": len(ranks_arr), "error_class_counts": dict(counts),
            }
            print(key, json.dumps(grid[key], ensure_ascii=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "leaf_channel_ablation.json").write_text(
        json.dumps(grid, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(grid, indent=2, ensure_ascii=False))
    return grid


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
