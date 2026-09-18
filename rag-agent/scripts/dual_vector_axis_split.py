#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Every dense variant tried so far pools ROW and COLUMN condition into ONE
sentence -> ONE CLS vector, so the two axes' signal is mixed by the same
pooling that mixes everything else in the sentence. This tests keeping them
apart instead: two short texts per cell -- a row-clause and a column-clause,
each the corresponding half of the already-validated STRUCTURAL sentence
("In the table 'T', among {row}." / "In the table 'T', the value of {col} is
{val}.") -- each encoded to its OWN vector, then combined per query by mean,
max, or min over the two per-axis similarities. min is the theoretically
interesting one: a wrong_row/wrong_column distractor matches on ONE axis but
not the other, so min (both axes must agree) should suppress it where max (or
a single mixed vector) would not.

BM25 input is pinned to plain STRUCTURAL_COMPACT throughout (same discipline
as scripts/dense_representation_sweep.py) so every delta is attributable to
the dense representation/combination change, not the sparse side.

  PYTHONPATH=. .venv/bin/python scripts/dual_vector_axis_split.py
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
COMBINERS = {
    "mean": lambda a, b: (a + b) / 2,
    "max": np.maximum,
    "min": np.minimum,
}


def row_col_clauses(title_s: str, row_path, col_path, val_s: str) -> tuple:
    row_join, col_join = join_path(list(row_path)), join_path(list(col_path))
    if title_s:
        row_text = f"In the table '{title_s}', among {row_join}." if row_join else f"In the table '{title_s}'."
        col_text = (f"In the table '{title_s}', the value of {col_join} is {val_s}." if col_join
                   else f"In the table '{title_s}', the value is {val_s}.")
    else:
        row_text = row_join
        col_text = f"{col_join}: {val_s}" if col_join else val_s
    return row_text, col_text


def build_corpus(data_dir: str = "data/hitab"):
    plain, row_texts, col_texts, coords = [], [], [], []
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
                plain.append(caption_sentence(title, row_path, col_path, value=v, template=STRUCTURAL_COMPACT))
                rt, ct = row_col_clauses(title_s, row_path, col_path, fmt_value(v))
                row_texts.append(rt)
                col_texts.append(ct)
                coords.append((tid, i, j))
    return plain, row_texts, col_texts, coords


def run(data_dir: str = "data/hitab") -> dict:
    pop = load_primary_population()
    plain, row_texts, col_texts, coords = build_corpus(data_dir)
    coord_index = {c: idx for idx, c in enumerate(coords)}

    enc = default_encoder()
    emb_plain, _ = encode_corpus(plain, enc)
    emb_row, _ = encode_corpus(row_texts, enc)
    emb_col, _ = encode_corpus(col_texts, enc)
    bm_plain = SparseBM25(_tokenize(t) for t in plain)

    ordered = sorted(pop.items())
    qvecs = {qid: enc.encode_query([r["question"]])[0].astype(np.float32) for qid, r in ordered}
    sparse_by_q = {qid: bm_plain.get_scores(_tokenize(r["question"])) for qid, r in ordered}

    tabs: dict = {}
    results = {}
    conditions = {"plain (single-vector baseline)": None, **{f"dual_vector_{k}": k for k in COMBINERS}}
    for label, combiner_key in conditions.items():
        rows = []
        for qid, r in ordered:
            gold = tuple(sorted(r["gold_cells"])[0])
            gidx = coord_index.get(gold)
            if gidx is None:
                continue
            if combiner_key is None:
                dense = emb_plain @ qvecs[qid]
            else:
                sim_row = emb_row @ qvecs[qid]
                sim_col = emb_col @ qvecs[qid]
                dense = COMBINERS[combiner_key](sim_row, sim_col)
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
        results[label] = {
            "recall_at_1": round(sum(x == 1 for x in ranks_arr) / len(ranks_arr), 4),
            "n": len(ranks_arr), "error_class_counts": dict(counts),
        }
        print(label, json.dumps(results[label], ensure_ascii=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "dual_vector_axis_split.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
