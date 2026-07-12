#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""S3 caption + 1-table-1-chunk baseline on MultiHiertt (교수님 3-step spec, real raw data).

``scripts/s3_table_chunk_baseline.py`` runs this same pipeline on HiTab, whose
header tree is either read from the gold annotation or synthetically
re-derived from it (``--reconstruct``) -- in both cases the underlying table
started life as a clean, already-tabular gold structure. MultiHiertt tables
are genuinely raw: scraped HTML with ``rowspan``/``colspan`` and no header
tree at all. Running the identical 3-step pipeline here is the version of
the experiment that cannot lean on HiTab's annotation in any way:

  1. Structure is parsed BEFORE anything is embedded, and genuinely
     RECONSTRUCTED (not read off an answer key -- none exists for this
     dataset): ``rag_agent.reconstruct.parse_html_table`` turns the raw HTML
     into a blank-after-first grid, then ``reconstruct_col_paths`` /
     ``reconstruct_row_paths`` forward-fill it into header paths (same
     algorithm validated in ``scripts/tree_reconstruct_multihiertt.py``).
  2. Cells are rendered as S3 caption sentences from those reconstructed
     paths, with the same short/medium/long length presets.
  3. Retrieval is table-granular (1 table = 1 chunk), two-stage: which table,
     then which cell -- identical cascade to the HiTab script.

Population: MultiHiertt (HF ``bevaya/MultiHiertt``, train split) questions
with an empty ``program`` (no arithmetic) and exactly one ``table_evidence``
cell and no ``text_evidence`` -- the direct single-cell-lookup analogue of
HiTab's ``aggregation in (None, "none")`` filter. Scored with a tolerant
numeric/string match (``rag_agent.eval.metrics.numeric_match``), NOT HiTab's
official scorer -- MultiHiertt has no equivalent official scorer in this repo,
so this is a diagnostic number, not literature-comparable.

Run: PYTHONPATH=. python scripts/s3_table_chunk_baseline_multihiertt.py --max-queries 150
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from rag_agent.eval.metrics import mrr, ndcg_at_k, numeric_match, recall_at_k
from rag_agent.query.operand_decomposer import Embedder
from rag_agent.reconstruct import guess_n_header_rows, parse_html_table, reconstruct_col_paths, reconstruct_row_paths
from rag_agent.serialization import serialize
from rag_agent.serialization.caption import LENGTHS


@dataclass
class ReconstructedTable:
    """Minimal TableView over a MultiHiertt raw-HTML table, using SELF-
    RECONSTRUCTED header paths only (no gold tree exists for this dataset)."""

    table_id: str
    title: str
    grid: List[List[str]]
    nhr: int
    nhc: int = 1
    _col_paths: List[List[str]] = field(default_factory=list)
    _row_paths: List[List[str]] = field(default_factory=list)

    def __post_init__(self):
        self._col_paths = reconstruct_col_paths(self.grid, self.nhr, self.nhc)
        self._row_paths = reconstruct_row_paths(self.grid, self.nhr, self.nhc)

    @property
    def n_rows(self) -> int:
        return len(self.grid) - self.nhr

    @property
    def n_cols(self) -> int:
        return (len(self.grid[0]) if self.grid else 0) - self.nhc

    def cell(self, row: int, col: int) -> Any:
        r, c = row + self.nhr, col + self.nhc
        if 0 <= r < len(self.grid) and 0 <= c < len(self.grid[r]):
            return self.grid[r][c]
        return None

    def col_path(self, col: int) -> List[str]:
        return self._col_paths[col] if 0 <= col < len(self._col_paths) else []

    def row_path(self, row: int) -> List[str]:
        return self._row_paths[row] if 0 <= row < len(self._row_paths) else []


def load_clean_lookup_queries(max_queries: int):
    """MultiHiertt questions with no arithmetic, one gold cell, no text
    evidence -- the single-cell-lookup analogue of HiTab's direct-lookup
    population. Returns (queries, docs_by_uid)."""
    from datasets import load_dataset

    ds = load_dataset("bevaya/MultiHiertt", split="train")
    docs = ds.to_list()  # `for row in ds` is >100x slower on this dataset
    queries = []
    docs_by_uid = {}
    for row in docs:
        prog = (row.get("program") or "").strip()
        te = row.get("table_evidence") or []
        txte = row.get("text_evidence") or []
        if prog or len(te) != 1 or txte:
            continue
        key = te[0]
        parts = key.split("-")
        if len(parts) != 3:
            continue
        t_idx, r, c = (int(x) for x in parts)
        queries.append({
            "uid": row["uid"], "table_idx": t_idx, "gold_row": r, "gold_col": c,
            "question": row["question"], "answer": row["answer"],
        })
        docs_by_uid[row["uid"]] = row
        if len(queries) >= max_queries:
            break
    return queries, docs_by_uid


def build_tables(queries, docs_by_uid):
    """One ReconstructedTable per unique (uid, table_idx) referenced by the
    sampled queries -- mirrors the HiTab script's 'only the gold tables in
    the eval sample' corpus, not the full 31k-table pool."""
    tables = {}
    kept = []
    for q in queries:
        tid = f"{q['uid']}::{q['table_idx']}"
        if tid not in tables:
            html = docs_by_uid[q["uid"]]["tables"][q["table_idx"]]
            grid = parse_html_table(html)
            if len(grid) < 3 or len(grid[0]) < 2:
                continue
            nhr = max(1, min(guess_n_header_rows(grid, n_header_cols=1), len(grid) - 1))
            tables[tid] = ReconstructedTable(table_id=tid, title="", grid=grid, nhr=nhr)
        rt = tables[tid]
        # gold_row/gold_col are raw-grid coordinates; convert to the
        # data-row/data-col indices ReconstructedTable.cell() expects.
        dr, dc = q["gold_row"] - rt.nhr, q["gold_col"] - rt.nhc
        if not (0 <= dr < rt.n_rows and 0 <= dc < rt.n_cols):
            continue
        kept.append({**q, "table_id": tid, "data_row": dr, "data_col": dc})
    return kept, tables


def _serialize_kwargs(scheme: str, length: str) -> dict:
    """S3 (caption) takes a length preset; S2 (header-path / tree-mapping)
    has no such axis -- omit it so header_path.serialize() doesn't choke on
    an unknown kwarg."""
    return {"length": length} if scheme == "S3" else {}


def build_table_corpus(tables: dict, scheme: str, length: str):
    ids, texts = [], []
    for tid, rt in tables.items():
        chunk = serialize(rt, scheme=scheme, granularity="table", **_serialize_kwargs(scheme, length))[0]
        ids.append(tid)
        texts.append(chunk.text)
    return ids, texts


def build_cell_candidates(rt, scheme: str, length: str):
    chunks = serialize(rt, scheme=scheme, granularity="cell", include_title=False,
                        **_serialize_kwargs(scheme, length))
    texts = [c.text for c in chunks]
    values = [rt.cell(c.row_index, c.col_index) for c in chunks]
    return texts, values


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--max-queries", type=int, default=150,
                     help="cap on clean single-cell-lookup queries scanned (speed)")
    ap.add_argument("--rerank-k", type=int, default=1,
                     help="coarse-to-fine cascade: take the top-K tables by table-chunk "
                          "similarity, then re-pick the table by its BEST per-cell caption "
                          "similarity instead of the coarse table-chunk score. K=1 reproduces "
                          "the plain 1-table-1-chunk baseline (no rerank). Diagnostic finding: "
                          "median gold-table rank is 2/150 and most misses are near-duplicate "
                          "tables (e.g. 'X-1' vs 'X-2' sections of the same filing) that the "
                          "coarse whole-table embedding can't tell apart but per-cell captions can.")
    ap.add_argument("--scheme", choices=["S2", "S3"], default="S3",
                     help="S2 = tree-mapping (header-path prefix, no length axis) -- the "
                          "student's original method. S3 = natural-language caption "
                          "(the advisor's alternative, with a length axis). Run both with "
                          "identical population/pool/metrics/rerank-k for a head-to-head.")
    ap.add_argument("--out", default="results/s3_table_chunk_baseline_multihiertt.json")
    args = ap.parse_args()

    raw_queries, docs_by_uid = load_clean_lookup_queries(args.max_queries)
    pop, tables = build_tables(raw_queries, docs_by_uid)
    n = len(pop)
    print(f"[pop] multihiertt clean single-cell-lookup queries: {n}  |  unique gold tables: {len(tables)}  "
          f"|  hierarchy: self-reconstructed (no gold tree exists for this dataset)  |  scheme={args.scheme}")

    emb = Embedder(args.embed_model, device="cpu")
    results = {}
    lengths = LENGTHS if args.scheme == "S3" else ("n/a",)

    for length in lengths:
        print(f"\n=== scheme={args.scheme} length={length} ===")
        table_ids, table_texts = build_table_corpus(tables, args.scheme, length)
        table_vecs = np.asarray(emb.encode(table_texts))
        avg_chars = round(sum(len(t) for t in table_texts) / len(table_texts), 1) if table_texts else 0

        cell_cache = {}
        table_hits = answer_hits = table_recall_answer_hits = 0
        r5_sum = r10_sum = mrr_sum = ndcg10_sum = 0.0

        def cell_lookup(tid, qv):
            if tid not in cell_cache:
                texts, values = build_cell_candidates(tables[tid], args.scheme, length)
                vecs = np.asarray(emb.encode(texts)) if texts else np.zeros((0, 1))
                cell_cache[tid] = (texts, values, vecs)
            texts, values, vecs = cell_cache[tid]
            if not len(texts):
                return None, -1.0
            csims = vecs @ qv
            j = int(np.argmax(csims))
            return values[j], float(csims[j])

        for i, q in enumerate(pop):
            qv = np.asarray(emb.encode([q["question"]])[0])
            sims = table_vecs @ qv
            order = np.argsort(-sims)  # full ranking, best first
            ranked_ids = [table_ids[j] for j in order]
            r5_sum += recall_at_k(ranked_ids, q["table_id"], 5)
            r10_sum += recall_at_k(ranked_ids, q["table_id"], 10)
            mrr_sum += mrr(ranked_ids, q["table_id"])
            ndcg10_sum += ndcg_at_k(ranked_ids, q["table_id"], 10)

            # Coarse-to-fine: rerank the top-K table-chunk candidates by their
            # BEST per-cell caption similarity, not the coarse whole-table score.
            # K=1 collapses to the plain baseline (pred/retrieved_tid unchanged).
            candidates = ranked_ids[: max(args.rerank_k, 1)]
            best_tid, pred, best_cell_sim = candidates[0], None, -1.0
            for cand in candidates:
                cand_pred, cand_sim = cell_lookup(cand, qv)
                if cand_sim > best_cell_sim:
                    best_tid, pred, best_cell_sim = cand, cand_pred, cand_sim

            retrieved_tid = best_tid
            table_hit = retrieved_tid == q["table_id"]
            table_hits += int(table_hit)

            ok = numeric_match(pred, q["answer"])
            answer_hits += int(ok)
            if ok and table_hit:
                table_recall_answer_hits += 1

            if (i + 1) % 50 == 0:
                print(f"  ... {i + 1}/{n}")

        results[length] = {
            "avg_table_chunk_chars": avg_chars,
            "rerank_k": args.rerank_k,
            "table_recall@1": round(table_hits / n, 4) if n else None,
            "table_recall@5": round(r5_sum / n, 4) if n else None,
            "table_recall@10": round(r10_sum / n, 4) if n else None,
            "table_mrr": round(mrr_sum / n, 4) if n else None,
            "table_ndcg@10": round(ndcg10_sum / n, 4) if n else None,
            "answer_match": round(answer_hits / n, 4) if n else None,
            "answer_match_given_correct_table": (
                round(table_recall_answer_hits / table_hits, 4) if table_hits else None
            ),
        }
        r = results[length]
        print(f"  avg table-chunk length : {r['avg_table_chunk_chars']} chars")
        print(f"  table_recall@1/5/10    : {r['table_recall@1']} / {r['table_recall@5']} / {r['table_recall@10']}")
        print(f"  table_mrr / ndcg@10    : {r['table_mrr']} / {r['table_ndcg@10']}")
        print(f"  answer_match           : {r['answer_match']}  "
              f"(numeric_match, tolerant -- not an official scorer)")

    scheme_desc = {
        "S2": "S2 (tree-mapping: header-path prefix, the student's original method)",
        "S3": "S3 (natural-language caption, the advisor's alternative)",
    }[args.scheme]
    out = {
        "population": {"name": "multihiertt_clean_single_cell_lookup", "n": n,
                        "n_unique_tables": len(tables)},
        "embed_model": args.embed_model,
        "scheme": f"{scheme_desc} + 1-table-1-chunk retrieval, "
                  "2-stage: table chunk -> cell-sentence within retrieved table",
        "hierarchy_source": "self-reconstructed (rag_agent.reconstruct, "
                             "no gold tree exists for MultiHiertt)",
        "scorer": "numeric_match (tolerant, rel_tol=0.02) -- diagnostic, not literature-comparable",
        "by_length": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
