#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Three dense-side-only comparisons, BM25 input held fixed at plain
STRUCTURAL_COMPACT throughout so every R@1/error-class delta is attributable
to the dense text alone (the leaf_channel_ablation.py 2x2 showed the two
channels are ~additive with no interaction, so pinning sparse=plain isolates
the dense effect cleanly):

1. repeat_intensity  -- leaf prefix repeated n=0..4 times. Where does R@1
   saturate or reverse? (STRUCTURAL_LEAF_X2 already showed n=2 reverses
   slightly relative to n=1; this adds n=3,4 to confirm the trend.)
2. axis_emphasis     -- row-leaf-only / col-leaf-only / both (n=1 each).
   Does wrong_row drop specifically under row-only, wrong_column under
   col-only?
3. explicit_labels   -- row path and column path stated as separate tagged
   fields ("row path: ... | column path: ... | value: ...") instead of
   repeating the leaf token. Only structural roles the code already has
   (row path, column path, value, table) are used as tags -- no per-domain
   semantic label ("metric", "target", ...) is invented, since nothing in
   the table data names one.

  PYTHONPATH=. .venv/bin/python scripts/dense_representation_sweep.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                                                    # noqa: E402

from rag_agent.bench import hitab_grid as hg                          # noqa: E402
from rag_agent.retrieve.encoders import default_encoder                # noqa: E402
from rag_agent.retrieve.hybrid_index import _minmax, _tokenize         # noqa: E402
from rag_agent.retrieve.sparse_bm25 import SparseBM25                  # noqa: E402
from rag_agent.serialization.base import fmt_value, join_path          # noqa: E402
from rag_agent.serialization.caption import caption_sentence, with_page_title  # noqa: E402
from rag_agent.serialization.templates import STRUCTURAL_COMPACT       # noqa: E402
from scripts.retrieval_accuracy import PAGE_TITLES                    # noqa: E402
from bottleneck_diagnosis import classify_top1_error, load_primary_population, split_corpus_table_ids  # noqa: E402
from sentence_disambiguation_eval import encode_corpus                # noqa: E402

OUT_DIR = ROOT / "results/embedding_fusion_diagnosis_20260917"
ALPHA = 0.7
_PAGE_TITLES = json.loads(PAGE_TITLES.read_text()) if PAGE_TITLES.exists() else {}


def leaf_repeat_text(body: str, row_leaf: str, col_leaf: str, axes: str, n: int) -> str:
    """``n`` copies of a leaf prefix over ``body`` (STRUCTURAL_COMPACT text).
    ``axes`` selects which leaf(s) go in the prefix: 'row', 'col', or 'both'
    (n=1,'both' == STRUCTURAL_LEAF; n=2,'both' == STRUCTURAL_LEAF_X2)."""
    parts = {"row": [row_leaf], "col": [col_leaf], "both": [row_leaf, col_leaf]}[axes]
    prefix = " / ".join(p for p in parts if p)
    if not prefix or n == 0:
        return body
    return f"{prefix}: " * n + body


def explicit_labels_text(title_s: str, row_path: Sequence[str], col_path: Sequence[str],
                         val_s: str) -> str:
    fields = []
    if title_s:
        fields.append(f"table: {title_s}")
    rp, cp = join_path(list(row_path)), join_path(list(col_path))
    if rp:
        fields.append(f"row path: {rp}")
    if cp:
        fields.append(f"column path: {cp}")
    fields.append(f"value: {val_s}")
    return " | ".join(fields)


VARIANTS = (
    ("plain", "compact", "both", 0),
    ("leaf_row_x1", "compact", "row", 1),
    ("leaf_col_x1", "compact", "col", 1),
    ("leaf_both_x1", "compact", "both", 1),
    ("leaf_both_x2", "compact", "both", 2),
    ("leaf_both_x3", "compact", "both", 3),
    ("leaf_both_x4", "compact", "both", 4),
    ("explicit_labels", "explicit", "both", 0),
)


def build_corpus(data_dir: str = "data/hitab"):
    """One pass over every cell, all VARIANTS' texts aligned to the same coords."""
    texts = {name: [] for name, *_ in VARIANTS}
    coords = []
    for tid in split_corpus_table_ids():
        tab = hg.load_table(tid, data_dir)
        if tab is None:
            continue
        t = tab.table
        title = with_page_title(tab.title, _PAGE_TITLES.get(tid))
        title_s = fmt_value(title) if title else ""
        for i in range(t.n_rows):
            for j in range(t.n_cols):
                v = t.data[i][j]
                if not str(v).strip():
                    continue
                row_path, col_path = t.row_path(i), t.col_path(j)
                body = caption_sentence(title, row_path, col_path, value=v, template=STRUCTURAL_COMPACT)
                row_leaf = fmt_value(row_path[-1]) if row_path else ""
                col_leaf = fmt_value(col_path[-1]) if col_path else ""
                val_s = fmt_value(v)
                for name, kind, axes, n in VARIANTS:
                    if kind == "compact":
                        texts[name].append(leaf_repeat_text(body, row_leaf, col_leaf, axes, n))
                    else:
                        texts[name].append(explicit_labels_text(title_s, row_path, col_path, val_s))
                coords.append((tid, i, j))
    return texts, coords


def run(data_dir: str = "data/hitab") -> dict:
    pop = load_primary_population()
    texts, coords = build_corpus(data_dir)
    coord_index = {c: idx for idx, c in enumerate(coords)}

    enc = default_encoder()
    emb = {name: encode_corpus(t, enc)[0] for name, t in texts.items()}
    bm_plain = SparseBM25(_tokenize(t) for t in texts["plain"])  # BM25 input pinned to plain

    ordered = sorted(pop.items())
    qvecs = {qid: enc.encode_query([r["question"]])[0].astype(np.float32) for qid, r in ordered}
    sparse_by_q = {qid: bm_plain.get_scores(_tokenize(r["question"])) for qid, r in ordered}

    tabs: dict = {}
    results = {}
    for name in texts:
        rows = []
        for qid, r in ordered:
            gold = tuple(sorted(r["gold_cells"])[0])
            gidx = coord_index.get(gold)
            if gidx is None:
                continue
            dense = emb[name] @ qvecs[qid]
            hybrid = ALPHA * _minmax(dense) + (1 - ALPHA) * _minmax(sparse_by_q[qid])
            order = np.argsort(-hybrid, kind="stable")
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(order) + 1)
            gold_rank = int(ranks[gidx])
            top1 = coords[int(order[0])]
            cls = "exact_match" if top1 == gold else classify_top1_error(top1, gold, tabs, data_dir)
            rows.append({"gold_rank": gold_rank, "error_class": cls})
        ranks_arr = [r["gold_rank"] for r in rows]
        counts = Counter(r["error_class"] for r in rows)
        results[name] = {
            "recall_at_1": round(sum(x == 1 for x in ranks_arr) / len(ranks_arr), 4),
            "n": len(ranks_arr), "error_class_counts": dict(counts),
        }
        print(name, json.dumps(results[name], ensure_ascii=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "dense_representation_sweep.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
