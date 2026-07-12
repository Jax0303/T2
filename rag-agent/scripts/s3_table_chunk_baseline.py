#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""S3 caption + 1-table-1-chunk baseline (교수님 3-step spec).

The three things this script demonstrates end-to-end, matching the spec:

  1. Structure is parsed BEFORE anything is embedded. By default this reads
     HiTab's own gold header tree (``BenchTable.top_paths``/``left_paths``)
     -- which means step 1 ("figure out the table's hierarchy") is NOT
     actually exercised, since HiTab hands the tree to us pre-solved. Pass
     ``--reconstruct`` to instead SELF-RECONSTRUCT the hierarchy from a
     blank-after-first grid (the same ``rag_agent.reconstruct`` algorithm
     validated in ``scripts/tree_reconstruct_hitab.py``), so step 1 is
     genuinely tested rather than read off an answer key. Either way, the
     raw table is never discarded, only its *serialized text* goes into the
     index.
  2. Cells are rendered as natural-language caption sentences (S3, see
     ``rag_agent/serialization/caption.py``), e.g. "For Seoul, Population is
     950.", with three ``length`` presets so sentence length is an explicit,
     sweepable variable.
  3. Retrieval is table-granular: one embedding per WHOLE table (S3,
     granularity="table"). A query is answered in two stages —
     (a) which table? (cosine over table-chunk embeddings), then
     (b) which cell in that table? (cosine over the retrieved table's
     per-cell caption sentences) — and the matched sentence's cell value is
     the answer, scored with HiTab's own exact-match scorer.

This is a retrieval BASELINE (no LLM, no arithmetic) — it only answers direct
cell-lookup questions (``aggregation in (None, "none")``), which is the
"Seoul is 950"-style query the spec describes. It is meant to be compared
against the S1/S2 row-chunk schemes already in the repo, not against the
OSC/operand-decomposition results.

Run: PYTHONPATH=. python scripts/s3_table_chunk_baseline.py --split dev --max-queries 150
     PYTHONPATH=. python scripts/s3_table_chunk_baseline.py --reconstruct --out results/s3_table_chunk_baseline_reconstructed.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from rag_agent.bench.hitab import load_queries
from rag_agent.eval.metrics import hitab_exact_match, mrr, ndcg_at_k, recall_at_k
from rag_agent.query.operand_decomposer import Embedder
from rag_agent.reconstruct import reconstruct_col_paths, reconstruct_row_paths
from rag_agent.serialization import serialize
from rag_agent.serialization.caption import LENGTHS


def flatten_to_grid(top_paths, left_paths):
    """Render known gold header paths into a synthetic blank-after-first grid
    -- the way a merged-cell spreadsheet looks once copy-pasted into a flat
    CSV. Mirrors ``scripts/tree_reconstruct_hitab.py``; duplicated here (not
    imported) since ``scripts/`` isn't a package."""
    n_header_rows = max((len(p) for p in top_paths), default=0)
    n_header_cols = max((len(p) for p in left_paths), default=0)
    n_cols, n_rows = len(top_paths), len(left_paths)
    width = n_header_cols + n_cols
    height = n_header_rows + n_rows
    grid = [["" for _ in range(width)] for _ in range(height)]

    for c, path in enumerate(top_paths):
        for d in range(n_header_rows):
            label = path[d] if d < len(path) else ""
            prev = top_paths[c - 1][d] if c > 0 and d < len(top_paths[c - 1]) else None
            if label and label != prev:
                grid[d][n_header_cols + c] = label

    for r, path in enumerate(left_paths):
        for d in range(n_header_cols):
            label = path[d] if d < len(path) else ""
            prev = left_paths[r - 1][d] if r > 0 and d < len(left_paths[r - 1]) else None
            if label and label != prev:
                grid[n_header_rows + r][d] = label

    for r in range(n_rows):
        for c in range(n_cols):
            grid[n_header_rows + r][n_header_cols + c] = "1"

    return grid, n_header_rows, n_header_cols


def reconstruct_table(bt):
    """Return a copy of ``bt`` whose ``top_paths``/``left_paths`` come from
    self-reconstruction (no gold tree read at inference time), not HiTab's
    gold tree. Cell values/title/table_id are untouched -- only the header
    paths fed to the caption serializer change."""
    grid, nhr, nhc = flatten_to_grid(bt.top_paths, bt.left_paths)
    rec_cols = reconstruct_col_paths(grid, nhr, nhc)
    rec_rows = reconstruct_row_paths(grid, nhr, nhc)
    return replace(bt, top_paths=rec_cols, left_paths=rec_rows)


def build_table_corpus(tables: dict, length: str):
    """One S3 chunk per table (granularity='table')."""
    ids, texts = [], []
    for tid, bt in tables.items():
        chunk = serialize(bt, scheme="S3", length=length, granularity="table")[0]
        ids.append(tid)
        texts.append(chunk.text)
    return ids, texts


def build_cell_candidates(bt, length: str):
    """Per-cell S3 sentences + the value each sentence names, for a single table."""
    chunks = serialize(bt, scheme="S3", length=length, granularity="cell", include_title=False)
    texts = [c.text for c in chunks]
    values = [bt.cell(c.row_index, c.col_index) for c in chunks]
    return texts, values


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--max-queries", type=int, default=150,
                     help="cap on direct-lookup queries evaluated (speed)")
    ap.add_argument("--reconstruct", action="store_true",
                     help="self-reconstruct row/col header paths (rag_agent.reconstruct) "
                          "instead of reading HiTab's gold tree -- actually exercises "
                          "step 1 ('figure out the hierarchy') instead of reading it off "
                          "the dataset's answer key")
    ap.add_argument("--rerank-k", type=int, default=1,
                     help="coarse-to-fine cascade: take the top-K tables by table-chunk "
                          "similarity, then re-pick the table by its BEST per-cell caption "
                          "similarity instead of the coarse table-chunk score. K=1 reproduces "
                          "the plain 1-table-1-chunk baseline (no rerank).")
    ap.add_argument("--out", default="results/s3_table_chunk_baseline.json")
    args = ap.parse_args()

    queries, tables = load_queries(args.data_dir, args.split)
    pop = [q for q in queries if (q.aggregation or "none") in (None, "none")]
    if args.max_queries:
        pop = pop[: args.max_queries]
    n = len(pop)
    eval_table_ids = {q.gold_table_id for q in pop}
    eval_tables = {tid: tables[tid] for tid in eval_table_ids if tid in tables}
    if args.reconstruct:
        eval_tables = {tid: reconstruct_table(bt) for tid, bt in eval_tables.items()}
    print(f"[pop] direct-lookup queries: {n}  |  unique gold tables: {len(eval_tables)}  "
          f"|  hierarchy: {'self-reconstructed' if args.reconstruct else 'HiTab gold tree (given)'}")

    emb = Embedder(args.embed_model, device="cpu")
    results = {}

    for length in LENGTHS:
        print(f"\n=== length={length} ===")
        table_ids, table_texts = build_table_corpus(eval_tables, length)
        table_vecs = np.asarray(emb.encode(table_texts))
        avg_chars = round(sum(len(t) for t in table_texts) / len(table_texts), 1)
        avg_cell_words = round(
            sum(len(l.split()) for t in table_texts for l in t.splitlines()[1:])
            / sum(max(len(t.splitlines()) - 1, 1) for t in table_texts), 2
        )

        cell_cache = {}  # table_id -> (texts, values, vecs)
        table_hits = 0
        answer_hits = 0
        table_recall_answer_hits = 0  # answer correct AND table retrieval was correct
        r5_sum = r10_sum = mrr_sum = ndcg10_sum = 0.0

        def cell_lookup(tid, qv):
            if tid not in cell_cache:
                bt = eval_tables[tid]
                texts, values = build_cell_candidates(bt, length)
                vecs = np.asarray(emb.encode(texts)) if texts else np.zeros((0, 1))
                cell_cache[tid] = (texts, values, vecs)
            texts, values, vecs = cell_cache[tid]
            if not len(texts):
                return None, -1.0
            csims = vecs @ qv
            j = int(np.argmax(csims))
            return values[j], float(csims[j])

        for i, q in enumerate(pop):
            qv = np.asarray(emb.encode([q.question])[0])
            sims = table_vecs @ qv
            order = np.argsort(-sims)  # full ranking, best first
            ranked_ids = [table_ids[j] for j in order]
            r5_sum += recall_at_k(ranked_ids, q.gold_table_id, 5)
            r10_sum += recall_at_k(ranked_ids, q.gold_table_id, 10)
            mrr_sum += mrr(ranked_ids, q.gold_table_id)
            ndcg10_sum += ndcg_at_k(ranked_ids, q.gold_table_id, 10)

            candidates = ranked_ids[: max(args.rerank_k, 1)]
            best_tid, pred, best_cell_sim = candidates[0], None, -1.0
            for cand in candidates:
                cand_pred, cand_sim = cell_lookup(cand, qv)
                if cand_sim > best_cell_sim:
                    best_tid, pred, best_cell_sim = cand, cand_pred, cand_sim

            retrieved_tid = best_tid
            table_hit = retrieved_tid == q.gold_table_id
            table_hits += int(table_hit)

            ok = hitab_exact_match(pred, q.answer)
            answer_hits += int(ok)
            if ok and table_hit:
                table_recall_answer_hits += 1

            if (i + 1) % 50 == 0:
                print(f"  ... {i + 1}/{n}")

        results[length] = {
            "avg_table_chunk_chars": avg_chars,
            "avg_words_per_cell_sentence": avg_cell_words,
            "rerank_k": args.rerank_k,
            "table_recall@1": round(table_hits / n, 4),
            "table_recall@5": round(r5_sum / n, 4),
            "table_recall@10": round(r10_sum / n, 4),
            "table_mrr": round(mrr_sum / n, 4),
            "table_ndcg@10": round(ndcg10_sum / n, 4),
            "answer_exact_match": round(answer_hits / n, 4),
            "answer_exact_match_given_correct_table": (
                round(table_recall_answer_hits / table_hits, 4) if table_hits else None
            ),
        }
        r = results[length]
        print(f"  avg table-chunk length : {r['avg_table_chunk_chars']} chars "
              f"({r['avg_words_per_cell_sentence']} words/cell-sentence)")
        print(f"  table_recall@1/5/10    : {r['table_recall@1']} / {r['table_recall@5']} / {r['table_recall@10']}")
        print(f"  table_mrr / ndcg@10    : {r['table_mrr']} / {r['table_ndcg@10']}")
        print(f"  answer_exact_match     : {r['answer_exact_match']}  "
              f"(hitab_exact_match, official scorer)")

    out = {
        "population": {"name": "hitab_direct_lookup", "split": args.split, "n": n,
                        "n_unique_tables": len(eval_tables)},
        "embed_model": args.embed_model,
        "scheme": "S3 (natural-language caption) + 1-table-1-chunk retrieval, "
                  "2-stage: table chunk -> cell-sentence within retrieved table",
        "hierarchy_source": "self-reconstructed (rag_agent.reconstruct, gold tree NOT used)"
                             if args.reconstruct else "HiTab gold tree (given, not reconstructed)",
        "by_length": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
