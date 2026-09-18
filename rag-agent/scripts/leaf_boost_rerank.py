#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan item 0a (2026-09-17): rule-based leaf-token boost rerank. [[real-ceiling
-not-data-2026-09-17]] found the miss mechanism is structural, not capacity --
a row/col leaf token gets outvoted by the shared title/header text it sits
inside, in both dense pooling and BM25's flat bag-of-words. This tests the
cheapest possible fix for that: no training, no model, just a fixed additive
bonus on the deployed structural_leaf hybrid score (embedding_fusion_
diagnosis.py's ALPHA=0.7 setup) when the query contains a candidate cell's own
row-leaf or column-leaf label VERBATIM (case-insensitive substring).

Bonus size is the only free parameter and is picked on HiTab dev (prereg
convention already used by every weight in retrieval_improvement.py: choose
on dev, score test once), then applied once to the primary test population
(bottleneck_diagnosis.load_primary_population, n=991) and checked against the
deployed hybrid with McNemar (leaf_significance_check.py's pattern).

Reuses (no reimplementation): bottleneck_diagnosis.load_primary_population/
split_corpus_table_ids, retrieval_improvement.primary_population (dev split,
generalizes load_primary_population to dev), sentence_disambiguation_eval.
encode_corpus (doc-embedding cache), hybrid_index._minmax/_tokenize,
sparse_bm25.SparseBM25, encoders.default_encoder, serialization.caption.
caption_sentence/with_page_title, serialization.base.fmt_value.

  PYTHONPATH=. .venv/bin/python scripts/leaf_boost_rerank.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                                                    # noqa: E402
from scipy.stats import binomtest                                     # noqa: E402

from rag_agent.bench import hitab_grid as hg                          # noqa: E402
from rag_agent.retrieve.encoders import default_encoder                # noqa: E402
from rag_agent.retrieve.hybrid_index import _minmax, _tokenize         # noqa: E402
from rag_agent.retrieve.sparse_bm25 import SparseBM25                  # noqa: E402
from rag_agent.serialization.base import fmt_value                    # noqa: E402
from rag_agent.serialization.caption import caption_sentence, with_page_title  # noqa: E402
from rag_agent.serialization.templates import STRUCTURAL_LEAF          # noqa: E402
from scripts.retrieval_accuracy import PAGE_TITLES                    # noqa: E402
from bottleneck_diagnosis import load_primary_population, split_corpus_table_ids  # noqa: E402
from retrieval_improvement import primary_population as dev_primary_population  # noqa: E402
from sentence_disambiguation_eval import encode_corpus                # noqa: E402

OUT_DIR = ROOT / "results/leaf_boost_rerank_20260917"
ALPHA = 0.7                     # same deployed hybrid weight as embedding_fusion_diagnosis.py
# 0.0 included deliberately: an earlier pass swept only 0.05..0.5, which never
# priced in bonus=0 (the true baseline) and made the sweep look monotonically
# decreasing when it is actually flat/noisy below ~0.02 and only clearly
# harmful from ~0.05 up -- see coverage_stats(), the reason why.
BONUS_GRID = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
_PAGE_TITLES = json.loads(PAGE_TITLES.read_text()) if PAGE_TITLES.exists() else {}


def build_corpus(tids, data_dir: str = "data/hitab"):
    """(texts, row_leaf, col_leaf, coords) over ``tids`` under the deployed
    STRUCTURAL_LEAF template -- same shape as dense_representation_sweep.
    build_corpus, generalized to any table-id list so it also serves dev."""
    texts, row_leaf, col_leaf, coords = [], [], [], []
    for tid in tids:
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
                texts.append(caption_sentence(title, rp, cp, value=v, template=STRUCTURAL_LEAF))
                row_leaf.append(fmt_value(rp[-1]) if rp else "")
                col_leaf.append(fmt_value(cp[-1]) if cp else "")
                coords.append((tid, i, j))
    return texts, row_leaf, col_leaf, coords


def leaf_ids(row_leaf: list, col_leaf: list):
    """Dedup row/col leaf strings once so a query only needs to check
    membership against the distinct labels, not every cell -- most leaf
    labels (e.g. "Total", "2020") repeat across many cells."""
    uniq = sorted({s.lower() for s in row_leaf + col_leaf if s})
    idx = {s: i for i, s in enumerate(uniq)}
    row_ids = np.array([idx.get(s.lower(), -1) for s in row_leaf], dtype=np.int64)
    col_ids = np.array([idx.get(s.lower(), -1) for s in col_leaf], dtype=np.int64)
    return row_ids, col_ids, uniq


def boost_mask(question: str, row_ids: np.ndarray, col_ids: np.ndarray, uniq: list) -> np.ndarray:
    """Per-cell match count (0, 1, or 2) -- +1 if the question contains that
    cell's row leaf verbatim, +1 if it contains the column leaf verbatim.
    ``row_ids``/``col_ids`` use -1 for "no leaf"; ``hit`` has one extra
    trailing 0.0 entry so index -1 resolves to "never matches"."""
    q = question.lower()
    hit = np.array([1.0 if s in q else 0.0 for s in uniq] + [0.0], dtype=np.float32)
    return hit[row_ids] + hit[col_ids]


def eval_population(ordered, gold_key, coord_index, emb, bm, row_ids, col_ids, uniq,
                    enc, bonus: float) -> dict:
    """R@1 (baseline hybrid vs boosted) and per-query hit dicts for McNemar."""
    base_hits, boost_hits = {}, {}
    for qid, r in ordered:
        gold = tuple(sorted(gold_key(r))[0])
        gidx = coord_index.get(gold)
        if gidx is None:
            continue
        q = r["question"]
        dense = emb @ enc.encode_query([q])[0].astype(np.float32)
        sparse = bm.get_scores(_tokenize(q))
        hybrid = ALPHA * _minmax(dense) + (1 - ALPHA) * _minmax(sparse)
        mask = boost_mask(q, row_ids, col_ids, uniq)
        base_hits[qid] = bool(np.argmax(hybrid) == gidx)
        boost_hits[qid] = bool(np.argmax(hybrid + bonus * mask) == gidx)
    return base_hits, boost_hits


def coverage_stats(ordered, row_ids: np.ndarray, col_ids: np.ndarray, uniq: list,
                   n_cells: int) -> dict:
    """How many corpus cells get ANY boost per query -- the leaf label alone
    (last path element only) is often shared by thousands of cells (a common
    year, "Total", a repeated category name), so this checks whether the
    literal-match signal is actually selective before trusting a bonus sweep
    on top of it."""
    counts = [int((boost_mask(r["question"], row_ids, col_ids, uniq) > 0).sum())
             for _qid, r in ordered]
    return {"n_corpus_cells": n_cells, "mean_cells_boosted": round(sum(counts) / len(counts), 1),
           "mean_share_of_corpus": round(sum(counts) / len(counts) / n_cells, 4),
           "max_cells_boosted": max(counts), "min_cells_boosted": min(counts)}


def mcnemar(a: dict, b: dict) -> dict:
    qids = sorted(set(a) & set(b))
    a_only = sum(a[q] and not b[q] for q in qids)
    b_only = sum(b[q] and not a[q] for q in qids)
    n_disc = a_only + b_only
    p = binomtest(min(a_only, b_only), n_disc, 0.5).pvalue if n_disc else 1.0
    return {"n": len(qids), "baseline_r1": round(sum(a.values()) / len(qids), 4),
           "boosted_r1": round(sum(b.values()) / len(qids), 4),
           "baseline_only": a_only, "boosted_only": b_only, "discordant": n_disc,
           "p_value": round(p, 4)}


def run(data_dir: str = "data/hitab") -> dict:
    enc = default_encoder()

    # ---- dev: pick the bonus (never look at test for this) ----
    tabs_dev: dict = {}
    dev_queries, dev_primary = dev_primary_population("dev", tabs_dev)
    dev_tids = sorted({q["table_id"] for q in dev_queries})
    dev_texts, dev_row_leaf, dev_col_leaf, dev_coords = build_corpus(dev_tids, data_dir)
    dev_coord_index = {c: i for i, c in enumerate(dev_coords)}
    dev_emb, dev_hit = encode_corpus(dev_texts, enc)
    dev_bm = SparseBM25(_tokenize(t) for t in dev_texts)
    dev_row_ids, dev_col_ids, dev_uniq = leaf_ids(dev_row_leaf, dev_col_leaf)
    dev_ordered = sorted(dev_primary.items())

    dev_coverage = coverage_stats(dev_ordered, dev_row_ids, dev_col_ids, dev_uniq, len(dev_texts))

    dev_sweep = {}
    for bonus in BONUS_GRID:
        _, boost_hits = eval_population(dev_ordered, lambda r: r["gold"], dev_coord_index,
                                        dev_emb, dev_bm, dev_row_ids, dev_col_ids, dev_uniq,
                                        enc, bonus)
        dev_sweep[bonus] = round(sum(boost_hits.values()) / len(boost_hits), 4)
    best_bonus = max(dev_sweep, key=dev_sweep.get)

    # ---- test: score once with the dev-picked bonus ----
    pop = load_primary_population()
    test_tids = split_corpus_table_ids()
    test_texts, test_row_leaf, test_col_leaf, test_coords = build_corpus(test_tids, data_dir)
    test_coord_index = {c: i for i, c in enumerate(test_coords)}
    test_emb, test_hit = encode_corpus(test_texts, enc)
    test_bm = SparseBM25(_tokenize(t) for t in test_texts)
    test_row_ids, test_col_ids, test_uniq = leaf_ids(test_row_leaf, test_col_leaf)
    test_ordered = sorted(pop.items())

    base_hits, boost_hits = eval_population(test_ordered, lambda r: r["gold_cells"],
                                            test_coord_index, test_emb, test_bm,
                                            test_row_ids, test_col_ids, test_uniq,
                                            enc, best_bonus)
    check = mcnemar(base_hits, boost_hits)

    summary = {"dev_n": len(dev_ordered), "dev_leaf_match_coverage": dev_coverage,
              "dev_bonus_sweep": dev_sweep, "best_bonus": best_bonus,
              "test_n": len(test_ordered), "test": check,
              "embed_cache_hit_dev": dev_hit, "embed_cache_hit_test": test_hit}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
