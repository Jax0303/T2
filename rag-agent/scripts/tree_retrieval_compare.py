#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Re-test the lab-meeting claim ("tree-structure serialization beats keyword /
dense / hybrid retrieval, up to 10x10 tables") on a dataset where the tree is
NOT given for free.

The original claim was measured on HiTab, whose header tree is pre-parsed
annotation — so "we mapped the tree" cost nothing and the comparison doesn't
show whether tree-structure serialization still wins once the tree has to be
built by the system itself (`rag_agent.reconstruct`, from raw 2D/HTML, with
its own reconstruction error). This script runs the IDENTICAL retrieval
pipeline (BM25 / dense / hybrid-RRF via `HybridIndex`, alpha=0/1/0.5) on two
serializations of the same table pool — "flat" (values only, no header path)
vs "tree" (every cell prefixed by its row/col header path) — for BOTH:

  --dataset hitab        tree = HiTab's own GOLD header path (upper bound)
  --dataset multihiertt  tree = OUR reconstruction from raw HTML (the real test)

Tables are capped at --max-rows x --max-cols (default 10x10, matching the
original claim's stated scope). Run both and diff the "tree - flat" gap.

Run:
  PYTHONPATH=. python scripts/tree_retrieval_compare.py --dataset hitab
  PYTHONPATH=. python scripts/tree_retrieval_compare.py --dataset multihiertt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.serialization.base import Chunk
from rag_agent.retrieve.hybrid_index import HybridIndex
from rag_agent.retrieve.encoders import default_encoder

RETRIEVERS = [("bm25 (keyword)", 0.0), ("dense (embedding)", 1.0), ("hybrid (RRF-style avg)", 0.5)]

BUCKETS = {"<=10x10": (0, 10), "10-20": (11, 20), ">20": (21, 10 ** 9), "any": (0, 10 ** 9)}


def in_bucket(n_rows: int, n_cols: int, bucket: str) -> bool:
    """Same definition as scripts/tree_reconstruct_{hitab,multihiertt}.py's
    size_bucket(): bucketed by the LARGER of the two dimensions, so a table
    only needs ONE side to exceed 10 to count as beyond the original claim's
    stated 10x10 scope."""
    lo, hi = BUCKETS[bucket]
    return lo <= max(n_rows, n_cols) <= hi


def metrics(ranks, n):
    def at(k):
        return round(sum(1 for r in ranks if r is not None and r <= k) / n, 4)
    mrr = round(sum(1.0 / r for r in ranks if r is not None) / n, 4)
    return {"r1": at(1), "r5": at(5), "r10": at(10), "mrr": mrr, "n": n}


def eval_pool(table_ids, texts, queries, golds, encoder):
    """queries/golds are parallel lists; golds are table_ids (must be in table_ids)."""
    chunks = [Chunk(table_id=tid, chunk_id=f"{tid}::pool", text=txt, scheme="pool")
              for tid, txt in zip(table_ids, texts)]
    out = {}
    for label, alpha in RETRIEVERS:
        hidx = HybridIndex(chunks, encoder=encoder, alpha=alpha)
        ranks = []
        for q, g in zip(queries, golds):
            hits = hidx.search(q, k=len(chunks))
            rank = next((i + 1 for i, h in enumerate(hits) if h.chunk.table_id == g), None)
            ranks.append(rank)
        out[label] = metrics(ranks, len(queries))
    return out


# ---------------------------------------------------------------------------
# HiTab: gold tree (upper bound / sanity check against the original claim)
# ---------------------------------------------------------------------------

def run_hitab(args, encoder):
    from rag_agent.bench.hitab import load_queries

    queries, tables = load_queries(args.data_dir, args.split)
    small = {tid: bt for tid, bt in tables.items() if in_bucket(bt.n_rows, bt.n_cols, args.bucket)}
    pop = [q for q in queries
           if q.gold_table_id in small and (q.aggregation or "none") in (None, "none")]
    if args.max_queries:
        pop = pop[: args.max_queries]
    print(f"[hitab] tables in size bucket {args.bucket}: {len(small)}  | eval queries: {len(pop)}")

    table_ids = list(small.keys())
    flat_texts, tree_texts = [], []
    for tid in table_ids:
        bt = small[tid]
        title = [str(bt.title)] if bt.title else []
        flat_lines, tree_lines = [], []
        for r in range(bt.n_rows):
            flat_lines.append(" | ".join(str(bt.cell(r, c)) for c in range(bt.n_cols)))
            rp = " > ".join(bt.row_path(r))
            for c in range(bt.n_cols):
                cp = " > ".join(bt.col_path(c))
                path = " > ".join(x for x in (rp, cp) if x)
                tree_lines.append(f"{path}: {bt.cell(r, c)}" if path else str(bt.cell(r, c)))
        flat_texts.append("\n".join(title + flat_lines))
        tree_texts.append("\n".join(title + tree_lines))

    golds = [q.gold_table_id for q in pop]
    questions = [q.question for q in pop]
    return {
        "pool_size": len(table_ids), "n_eval": len(pop),
        "flat": eval_pool(table_ids, flat_texts, questions, golds, encoder),
        "tree": eval_pool(table_ids, tree_texts, questions, golds, encoder),
        "tree_source": "HiTab gold header tree (given, not reconstructed)",
    }


# ---------------------------------------------------------------------------
# MultiHiertt: self-reconstructed tree (the real test)
# ---------------------------------------------------------------------------

def run_multihiertt(args, encoder):
    from datasets import load_dataset
    from rag_agent.reconstruct import parse_html_table, guess_n_header_rows, \
        reconstruct_col_paths, reconstruct_row_paths

    ds = load_dataset(args.hf_dataset, split="train")
    # `for row in ds` decodes Arrow -> Python one row at a time and is very
    # slow on this dataset (long list/string fields); to_list() batch-decodes
    # once and is >100x faster in practice.
    rows = ds.to_list()
    print(f"[multihiertt] dataset loaded: {len(rows)} rows (1 doc = 1 row here)")

    pool = {}   # key -> dict(grid, nhr, rec_cols, rec_rows)
    n_docs = min(args.max_docs, len(rows)) if args.max_docs else len(rows)
    for i, row in enumerate(rows[:n_docs]):
        if i and i % 1000 == 0:
            print(f"  ...parsed {i}/{n_docs} docs, pool so far: {len(pool)} tables", flush=True)
        uid = row["uid"]
        for t_idx, html in enumerate(row["tables"]):
            if args.require_rowspan and "rowspan" not in html:
                continue
            grid = parse_html_table(html)
            if len(grid) < 2 or len(grid[0]) < 2:
                continue
            if not in_bucket(len(grid), len(grid[0]), args.bucket):
                continue
            nhr = max(1, min(guess_n_header_rows(grid, n_header_cols=1), len(grid) - 1))
            rec_cols = reconstruct_col_paths(grid, nhr, n_header_cols=1)
            rec_rows = reconstruct_row_paths(grid, nhr, n_header_cols=1)
            pool[f"{uid}::{t_idx}"] = {
                "grid": grid, "nhr": nhr, "rec_cols": rec_cols, "rec_rows": rec_rows,
            }
    print(f"[multihiertt] docs scanned: {n_docs}  "
          f"| tables in size bucket {args.bucket}: {len(pool)}")

    table_ids = list(pool.keys())
    flat_texts, tree_texts = [], []
    for tid in table_ids:
        t = pool[tid]
        grid, nhr = t["grid"], t["nhr"]
        flat_lines, tree_lines = [], []
        for r in range(nhr, len(grid)):
            flat_lines.append(" | ".join(v for v in grid[r][1:] if True))
            rp = " > ".join(t["rec_rows"][r - nhr]) if (r - nhr) < len(t["rec_rows"]) else ""
            for c in range(1, len(grid[0])):
                cp = " > ".join(t["rec_cols"][c - 1]) if (c - 1) < len(t["rec_cols"]) else ""
                path = " > ".join(x for x in (rp, cp) if x)
                v = grid[r][c]
                tree_lines.append(f"{path}: {v}" if path else v)
        flat_texts.append("\n".join(flat_lines))
        tree_texts.append("\n".join(tree_lines))

    # QA candidates: pure single-table questions (no text evidence) whose gold
    # table is in the size-filtered pool. Same 1 row = 1 doc = 1 question.
    questions, golds = [], []
    for row in rows[:n_docs]:
        if row.get("text_evidence"):
            continue
        ev = row.get("table_evidence") or []
        if not ev:
            continue
        t_idxs = {int(e.split("-")[0]) for e in ev}
        if len(t_idxs) != 1:
            continue
        key = f"{row['uid']}::{next(iter(t_idxs))}"
        if key not in pool:
            continue
        questions.append(row["question"])
        golds.append(key)
        if args.max_queries and len(questions) >= args.max_queries:
            break
    print(f"[multihiertt] eval queries (table-only, single-table gold): {len(questions)}")

    return {
        "pool_size": len(table_ids), "n_eval": len(questions),
        "flat": eval_pool(table_ids, flat_texts, questions, golds, encoder),
        "tree": eval_pool(table_ids, tree_texts, questions, golds, encoder),
        "tree_source": "self-reconstructed from raw HTML (rag_agent.reconstruct, "
                        "table_description NOT used)",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["hitab", "multihiertt"], required=True)
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--hf-dataset", default="bevaya/MultiHiertt")
    ap.add_argument("--max-docs", type=int, default=0, help="0 = all documents")
    ap.add_argument("--bucket", choices=list(BUCKETS), default="<=10x10",
                     help="size bucket by max(n_rows, n_cols), matching "
                          "scripts/tree_reconstruct_*.py's stratification")
    ap.add_argument("--require-rowspan", action="store_true",
                     help="multihiertt only: restrict pool to tables whose raw HTML "
                          "uses rowspan (the structurally-complex, harder-to-reconstruct subset)")
    ap.add_argument("--max-queries", type=int, default=200)
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    encoder = default_encoder(model_name=args.embed_model)
    out = run_hitab(args, encoder) if args.dataset == "hitab" else run_multihiertt(args, encoder)
    out["dataset"] = args.dataset
    out["bucket"] = args.bucket
    out["require_rowspan"] = args.require_rowspan

    out_path = args.out or f"results/tree_retrieval_{args.dataset}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"\npool={out['pool_size']} tables | eval={out['n_eval']} queries | "
          f"tree source: {out['tree_source']}")
    print(f"{'retriever':<24}{'flat R@1':>10}{'tree R@1':>10}{'  |':>4}"
          f"{'flat MRR':>10}{'tree MRR':>10}")
    for label, _ in RETRIEVERS:
        f, t = out["flat"][label], out["tree"][label]
        print(f"{label:<24}{f['r1']:>10.3f}{t['r1']:>10.3f}{'  |':>4}"
              f"{f['mrr']:>10.3f}{t['mrr']:>10.3f}")
    print(f"\nwrote -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
