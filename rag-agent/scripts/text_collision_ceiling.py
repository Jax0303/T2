#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The real ceiling behind [[leaf-repeat-axis-closed-2026-09-17]]'s .58-.62 R@1
plateau: not "is the encoder good enough" but "does the index unit's own text,
with the value removed, even carry enough information to tell gold apart from
the model's actual top-1 pick?"

For every hybrid top-1 miss on the deployed structural_leaf retriever (same
population/encoder/alpha=0.7 as embedding_fusion_diagnosis.py), render BOTH
gold's cell and the model's top-1 antagonist with ``value=None`` (render()'s
own no-leak header-scope form -- no placeholder hack, no value substring can
appear either side). If those two context strings are byte-identical, NO
similarity function over this text -- not this encoder, not a bigger one, not
BM25 -- could ever have ranked them apart: the ceiling is the corpus's own
labeling, not the model. If they differ, the text does carry a distinguishing
signal and the miss is a genuine, in-principle-fixable representation gap.

    real_ceiling_R@1 = 1 - (hard_collision_misses / n_population)

Also renders both under plain structural_compact (no leaf-repeat), reported for
reference only -- NOTE this comparison is near-tautological, not independent
evidence about leaf-repeat's mechanism: leaf-repeat only prepends a prefix to
the compact body, and prepending an identical string to two already-identical
strings cannot make them differ, so compact_collision implies leaf_collision by
construction and the two counts are structurally very likely to match
regardless of any real mechanism. Caught 2026-09-17 when asked "are you sure?"
-- do not cite equal counts here as showing repeat "didn't resolve collisions".

  PYTHONPATH=. .venv/bin/python scripts/text_collision_ceiling.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                                                    # noqa: E402

from rag_agent.bench import hitab_grid as hg                          # noqa: E402
from rag_agent.retrieve.encoders import default_encoder               # noqa: E402
from rag_agent.retrieve.hybrid_index import _minmax, _tokenize        # noqa: E402
from rag_agent.retrieve.sparse_bm25 import SparseBM25                 # noqa: E402
from rag_agent.serialization.caption import caption_sentence, with_page_title  # noqa: E402
from rag_agent.serialization.templates import STRUCTURAL_COMPACT, STRUCTURAL_LEAF  # noqa: E402
from scripts.retrieval_accuracy import PAGE_TITLES                    # noqa: E402
from bottleneck_diagnosis import load_primary_population, split_corpus_table_ids  # noqa: E402
from embedding_fusion_diagnosis import classify_miss                  # noqa: E402
from sentence_disambiguation_eval import encode_corpus                # noqa: E402

OUT_DIR = ROOT / "results/text_collision_ceiling_20260917"
ALPHA = 0.7
_PAGE_TITLES = json.loads(PAGE_TITLES.read_text()) if PAGE_TITLES.exists() else {}


def build_corpus_with_context(data_dir: str = "data/hitab"):
    """(full_texts, context_leaf, context_compact, coords) for the split corpus.

    ``full_texts`` is the deployed structural_leaf index (with value -- what
    gets encoded/BM25'd). ``context_leaf``/``context_compact`` are the SAME
    cell rendered with ``value=None``: render()'s own header-scope form, which
    never inserts a value token, under structural_leaf and plain
    structural_compact respectively. One table load serves all three.
    """
    full_texts, context_leaf, context_compact, coords = [], [], [], []
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
                full_texts.append(caption_sentence(title, rp, cp, value=v,
                                                   template=STRUCTURAL_LEAF))
                context_leaf.append(caption_sentence(title, rp, cp, value=None,
                                                      template=STRUCTURAL_LEAF))
                context_compact.append(caption_sentence(title, rp, cp, value=None,
                                                         template=STRUCTURAL_COMPACT))
                coords.append((tid, i, j))
    return full_texts, context_leaf, context_compact, coords


def ceiling_stats(rows: list, n_pop: int) -> dict:
    """Pure aggregation over one ``run()`` miss list -- no I/O, unit-testable."""
    n_miss = len(rows)

    def rate(hits, n):
        return round(hits / n, 4) if n else None

    def block(collision_key):
        n_col = sum(r[collision_key] for r in rows)
        return {"hard_collision_misses": n_col, "pct_of_misses": rate(n_col, n_miss),
               "real_ceiling_R@1": round(1 - n_col / n_pop, 4) if n_pop else None,
               "fixable_gap_misses": n_miss - n_col}

    return {
        "structural_leaf": block("hybrid_antagonist_collision_leaf"),
        "structural_compact_no_repeat": block("hybrid_antagonist_collision_compact"),
        "by_class": {
            cls: {"n": sum(r["class"] == cls for r in rows),
                 "collision_leaf": sum(r["class"] == cls and r["hybrid_antagonist_collision_leaf"]
                                       for r in rows)}
            for cls in ("combination_failure", "shared_representation_failure")
        },
    }


def run(data_dir: str = "data/hitab") -> dict:
    pop = load_primary_population()
    full_texts, ctx_leaf, ctx_compact, coords = build_corpus_with_context(data_dir)
    coord_index = {c: idx for idx, c in enumerate(coords)}

    enc = default_encoder()
    emb, cache_hit = encode_corpus(full_texts, enc)
    bm = SparseBM25(_tokenize(t) for t in full_texts)

    rows = []
    for qid, r in sorted(pop.items()):
        gold = tuple(sorted(r["gold_cells"])[0])
        gidx = coord_index.get(gold)
        if gidx is None:
            continue
        q = r["question"]
        dense = emb @ enc.encode_query([q])[0].astype(np.float32)
        sparse = bm.get_scores(_tokenize(q))
        hybrid = ALPHA * _minmax(dense) + (1 - ALPHA) * _minmax(sparse)

        def top1(scores):
            return int(np.argmax(scores))

        hyb_top1, dense_top1, sparse_top1 = top1(hybrid), top1(dense), top1(sparse)
        if hyb_top1 == gidx:
            continue  # only misses carry a ceiling question
        cls = classify_miss(dense_top1 == gidx, sparse_top1 == gidx)
        rows.append({
            "query_id": qid,
            "class": cls,
            "hybrid_antagonist_collision_leaf": ctx_leaf[hyb_top1] == ctx_leaf[gidx],
            "hybrid_antagonist_collision_compact": ctx_compact[hyb_top1] == ctx_compact[gidx],
            "dense_antagonist_collision_leaf": ctx_leaf[dense_top1] == ctx_leaf[gidx],
            "sparse_antagonist_collision_leaf": ctx_leaf[sparse_top1] == ctx_leaf[gidx],
        })

    summary = {"n_population": len(pop), "n_hybrid_miss": len(rows),
              "embed_cache_hit": cache_hit, **ceiling_stats(rows, len(pop))}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "misses.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
