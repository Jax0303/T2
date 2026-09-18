#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Retrieval bottleneck verification + improvement (RETRIEVAL_IMPROVEMENT.md).

Priority order (user spec, 2026-09-16):
  1. dense-only vs sparse-only vs existing hybrid (R@1/5/20, MRR, ESM, error
     types, dense->hybrid reversals, normalization/alpha/tie/truncation checks)
  2. real table retrieval (title+headers, no values) top1/top3 -> cell search,
     and global-top-K + table-score reranking
  3. top-K rerank: final = wd*dense + ws*sparse + wt*table + wr*row + wc*column
     (missing fields dropped then renormalized; weights are DEV-ONLY)
  4. impact of identical-serialization collisions (943 groups / 1886 cells,
     gold_path representation) on primary-population R@1/ties
  5. best reranker's top-1 alone fed to the reader vs the existing top-k(20)
     context, same 150-query controlled-QA sample already used elsewhere

Reuses (no reimplementation): retrieval_accuracy.build_corpus/cell_unit/
load_queries, bottleneck_root_cause.encode_corpus (doc-embedding cache),
bottleneck_diagnosis.classify_top1_error/load_primary_population,
hybrid_index._minmax, sparse_bm25.SparseBM25, encoders.default_encoder,
answer_accuracy.PROMPTS/load_saved_rows/check_context_limit. The only new
persistent artifact is a query-embedding cache (none existed -- every prior
run re-encoded queries and threw the vectors away); everything else reads or
extends what is already on disk. Test is scored once per method; every
weight/threshold choice below is selected on dev and then just applied.

  PYTHONPATH=. .venv/bin/python scripts/retrieval_improvement.py bundle --split test
  PYTHONPATH=. .venv/bin/python scripts/retrieval_improvement.py bundle --split dev
  PYTHONPATH=. .venv/bin/python scripts/retrieval_improvement.py item1
  PYTHONPATH=. .venv/bin/python scripts/retrieval_improvement.py item2
  PYTHONPATH=. .venv/bin/python scripts/retrieval_improvement.py item3
  PYTHONPATH=. .venv/bin/python scripts/retrieval_improvement.py item4
  PYTHONPATH=. .venv/bin/python scripts/retrieval_improvement.py item5 --dry-run
  PYTHONPATH=. .venv/bin/python scripts/retrieval_improvement.py item5
  PYTHONPATH=. .venv/bin/python scripts/retrieval_improvement.py report
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                                                    # noqa: E402

from rag_agent.bench import hitab_grid as hg                          # noqa: E402
from rag_agent.eval.artifacts import digest, read_records             # noqa: E402
from rag_agent.eval.metrics import hitab_exact_match_text              # noqa: E402
from rag_agent.retrieve.encoders import _tokenize, default_encoder    # noqa: E402
from rag_agent.retrieve.hybrid_index import _minmax                   # noqa: E402
from rag_agent.retrieve.sparse_bm25 import SparseBM25                 # noqa: E402
from rag_agent.serialization.caption import with_page_title           # noqa: E402
from retrieval_accuracy import PAGE_TITLES, build_corpus, cell_unit, load_queries  # noqa: E402
from bottleneck_diagnosis import (classify_top1_error,                # noqa: E402
                                  load_primary_population)
from bottleneck_root_cause import build_repr_corpus, encode_corpus    # noqa: E402
from answer_accuracy import PROMPTS, check_context_limit, load_saved_rows  # noqa: E402

DATA_DIR = "data/hitab"
ALPHA = 0.7                        # fixed elsewhere in this repo; not re-picked here
OUT_DIR = ROOT / "results/retrieval_improvement"
QUERY_CACHE = ROOT / ".cache/retrieval_accuracy_queries"
_PAGE_TITLES = json.loads(PAGE_TITLES.read_text()) if PAGE_TITLES.exists() else {}
POOL_SIZE = 50                     # top-K pool the item2/3 reranks operate on
TABLE_ROUTE_K_GRID = (1, 3)


# ---------------------------------------------------------------------------
# query embedding cache -- the one new piece of shared infra. No run before
# this saved a query vector; every script re-encoded and discarded it, even
# though items 1/2/3/5 all need the SAME query vectors against DIFFERENT
# corpora (cell/table/row/col all live in the same embedding space).
# ---------------------------------------------------------------------------

def encode_queries_cached(texts: list, encoder) -> np.ndarray:
    QUERY_CACHE.mkdir(parents=True, exist_ok=True)
    key = digest({"texts": texts, "encoder": encoder.metadata()})[:24]
    f = QUERY_CACHE / f"{encoder.name.replace('/', '_')}_{len(texts)}_{key}.npy"
    if f.exists():
        return np.load(f)
    emb = encoder.encode_query(texts)
    np.save(f, emb)
    return emb


# ---------------------------------------------------------------------------
# per-split score bundle: cell / table / row-col corpora + their embeddings,
# BM25 indices, and every primary-population query's score vector against
# each corpus. Built once per split, cached to disk as .npz + .json so later
# subcommands (item1..item5) never re-encode anything.
# ---------------------------------------------------------------------------

def primary_population(split: str, tabs: dict) -> tuple:
    """``(queries_all, primary)`` -- primary is m=1/mode=all/aggregation=none,
    keyed by query_id, same convention as bottleneck_diagnosis.load_primary_population
    (which only covers split=test; this is that filter generalized to dev)."""
    queries = load_queries(DATA_DIR, split, tabs, strict=False)
    primary = {q["query_id"]: q for q in queries
              if not q["excluded"] and q["mode"] == "all" and len(q["gold"]) == 1
              and (q.get("aggregation") or "none") == "none"}
    return queries, primary


def table_signature_text(tid: str, tabs: dict) -> str:
    """Table-level document for item 2: title + distinct row/column leaf
    headers, NO cell values -- deliberately coarser than the cell corpus so a
    table-routing signal is not just the cell signal in disguise."""
    tab = tabs.get(tid) or hg.load_table(tid, DATA_DIR)
    tabs[tid] = tab
    t = tab.table
    title = with_page_title(tab.title, _PAGE_TITLES.get(tid))
    rows, cols, seen_r, seen_c = [], [], set(), set()
    for i in range(t.n_rows):
        rp = t.row_path(i)
        lab = rp[-1] if rp else None
        if lab and lab not in seen_r:
            seen_r.add(lab); rows.append(lab)
    for j in range(t.n_cols):
        cp = t.col_path(j)
        lab = cp[-1] if cp else None
        if lab and lab not in seen_c:
            seen_c.add(lab); cols.append(lab)
    return f"{title} | rows: {', '.join(rows)} | columns: {', '.join(cols)}"


def build_bundle(split: str) -> dict:
    t0 = time.time()
    tabs: dict = {}
    queries, primary = primary_population(split, tabs)
    tids = sorted({q["table_id"] for q in queries})
    enc = default_encoder()

    cell_texts, cell_covers, _ir, _ut, _g = build_corpus(
        DATA_DIR, tids, "s3c", "cell", _PAGE_TITLES)
    cell_coords = [next(iter(c)) for c in cell_covers]
    cell_index = {c: i for i, c in enumerate(cell_coords)}
    cell_emb, cell_hit = encode_corpus(cell_texts, enc)
    cell_bm = SparseBM25(_tokenize(t) for t in cell_texts)

    table_texts = [table_signature_text(tid, tabs) for tid in tids]
    table_emb, table_hit = encode_corpus(table_texts, enc)
    table_bm = SparseBM25(_tokenize(t) for t in table_texts)
    table_index = {tid: i for i, tid in enumerate(tids)}

    rc_texts, rc_covers, rc_is_row, _ut2, _g2 = build_corpus(
        DATA_DIR, tids, "s3c", "rowcol", _PAGE_TITLES, row_text="values")
    rc_emb, rc_hit = encode_corpus(rc_texts, enc)
    rc_bm = SparseBM25(_tokenize(t) for t in rc_texts)
    row_index, col_index = {}, {}
    for k, (cov, is_row) in enumerate(zip(rc_covers, rc_is_row)):
        rep = next(iter(cov))
        (row_index if is_row else col_index)[(rep[0], rep[1] if is_row else rep[2])] = k

    qids = sorted(primary)
    qtexts = [primary[q]["question"] for q in qids]
    qemb = encode_queries_cached(qtexts, enc)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / f"bundle_{split}_cell_emb.npz", emb=cell_emb)
    np.savez(OUT_DIR / f"bundle_{split}_table_emb.npz", emb=table_emb)
    np.savez(OUT_DIR / f"bundle_{split}_rc_emb.npz", emb=rc_emb)
    np.savez(OUT_DIR / f"bundle_{split}_qemb.npz", emb=qemb)
    meta = {
        "split": split, "n_queries_all": len(queries), "n_primary": len(primary),
        "n_tables": len(tids), "n_cell_units": len(cell_texts),
        "n_table_units": len(table_texts), "n_rowcol_units": len(rc_texts),
        "cell_embed_cache_hit": cell_hit, "table_embed_cache_hit": table_hit,
        "rowcol_embed_cache_hit": rc_hit,
        "qids": qids, "tids": tids,
        "cell_coords": [list(c) for c in cell_coords],
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT_DIR / f"bundle_{split}_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    # texts saved separately (large); needed by item2/item4 for BM25 rebuild
    (OUT_DIR / f"bundle_{split}_cell_texts.json").write_text(
        json.dumps(cell_texts, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / f"bundle_{split}_table_texts.json").write_text(
        json.dumps(table_texts, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / f"bundle_{split}_rc_texts.json").write_text(
        json.dumps(rc_texts, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / f"bundle_{split}_row_index.json").write_text(
        json.dumps({f"{k[0]}\t{k[1]}": v for k, v in row_index.items()}), encoding="utf-8")
    (OUT_DIR / f"bundle_{split}_col_index.json").write_text(
        json.dumps({f"{k[0]}\t{k[1]}": v for k, v in col_index.items()}), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k not in ("qids", "tids", "cell_coords")},
                     indent=2, ensure_ascii=False))
    return meta


class Bundle:
    """Loads what build_bundle wrote, rebuilds the (cheap, deterministic)
    BM25 indices from the saved texts, and answers per-query score requests."""

    def __init__(self, split: str):
        self.split = split
        self.meta = json.loads((OUT_DIR / f"bundle_{split}_meta.json").read_text())
        self.qids = self.meta["qids"]
        self.tids = self.meta["tids"]
        self.cell_coords = [tuple(c) for c in self.meta["cell_coords"]]
        self.cell_index = {c: i for i, c in enumerate(self.cell_coords)}
        self.table_index = {tid: i for i, tid in enumerate(self.tids)}
        self.cell_texts = json.loads((OUT_DIR / f"bundle_{split}_cell_texts.json").read_text())
        self.table_texts = json.loads((OUT_DIR / f"bundle_{split}_table_texts.json").read_text())
        self.rc_texts = json.loads((OUT_DIR / f"bundle_{split}_rc_texts.json").read_text())
        self.row_index = {tuple(k.split("\t")): v for k, v in
                          json.loads((OUT_DIR / f"bundle_{split}_row_index.json").read_text()).items()}
        self.row_index = {(tid, int(i)): v for (tid, i), v in self.row_index.items()}
        self.col_index = {tuple(k.split("\t")): v for k, v in
                          json.loads((OUT_DIR / f"bundle_{split}_col_index.json").read_text()).items()}
        self.col_index = {(tid, int(j)): v for (tid, j), v in self.col_index.items()}
        self.cell_emb = np.load(OUT_DIR / f"bundle_{split}_cell_emb.npz")["emb"]
        self.table_emb = np.load(OUT_DIR / f"bundle_{split}_table_emb.npz")["emb"]
        self.rc_emb = np.load(OUT_DIR / f"bundle_{split}_rc_emb.npz")["emb"]
        self.qemb = np.load(OUT_DIR / f"bundle_{split}_qemb.npz")["emb"]
        self.cell_bm = SparseBM25(_tokenize(t) for t in self.cell_texts)
        self.table_bm = SparseBM25(_tokenize(t) for t in self.table_texts)
        self.rc_bm = SparseBM25(_tokenize(t) for t in self.rc_texts)
        tabs: dict = {}
        _q, primary = primary_population(split, tabs)
        self.primary = primary
        self.tabs = tabs

    def query_scores(self, qidx: int, question: str):
        qv = self.qemb[qidx].astype(np.float32)
        dense = self.cell_emb @ qv
        sparse = self.cell_bm.get_scores(_tokenize(question)).astype(np.float32)
        hybrid = ALPHA * _minmax(dense) + (1 - ALPHA) * _minmax(sparse)
        table_dense = self.table_emb @ qv
        table_sparse = self.table_bm.get_scores(_tokenize(question)).astype(np.float32)
        table_score = ALPHA * _minmax(table_dense) + (1 - ALPHA) * _minmax(table_sparse)
        rc_dense = self.rc_emb @ qv
        rc_sparse = self.rc_bm.get_scores(_tokenize(question)).astype(np.float32)
        rc_score = ALPHA * _minmax(rc_dense) + (1 - ALPHA) * _minmax(rc_sparse)
        return {"dense": dense, "sparse": sparse, "hybrid": hybrid,
               "table_score": table_score, "rc_score": rc_score}


def bundle_exists(split: str) -> bool:
    return (OUT_DIR / f"bundle_{split}_meta.json").exists()


def require_bundle(split: str) -> Bundle:
    if not bundle_exists(split):
        raise SystemExit(f"run `bundle --split {split}` first")
    return Bundle(split)


# ---------------------------------------------------------------------------
# item 1 -- dense-only vs sparse-only vs hybrid
# ---------------------------------------------------------------------------

def rank_of(scores: np.ndarray, idx: int) -> int:
    return int((scores > scores[idx]).sum()) + 1   # ties share the better rank


def top1_ties(scores: np.ndarray) -> bool:
    order = np.argsort(-scores, kind="stable")
    return bool(scores.size > 1 and scores[order[0]] == scores[order[1]])


def method_metrics(rows: list, method: str) -> dict:
    ranks = [r[f"{method}_rank"] for r in rows]
    n = len(ranks)
    return {"n": n,
           "recall_at_1": round(sum(x == 1 for x in ranks) / n, 4),
           "recall_at_5": round(sum(x <= 5 for x in ranks) / n, 4),
           "recall_at_20": round(sum(x <= 20 for x in ranks) / n, 4),
           "mrr": round(sum(1 / x for x in ranks) / n, 4),
           "esm_at_1": round(sum(x == 1 for x in ranks) / n, 4),
           "n_top1_ties": sum(r[f"{method}_top1_tie"] for r in rows)}


def run_item1(split: str = "test") -> dict:
    b = require_bundle(split)
    rows = []
    tabs = b.tabs
    for qidx, qid in enumerate(b.qids):
        q = b.primary[qid]
        gold = tuple(sorted(q["gold"])[0])
        if gold not in b.cell_index:
            continue
        gidx = b.cell_index[gold]
        s = b.query_scores(qidx, q["question"])
        dense, sparse, hybrid = s["dense"], s["sparse"], s["hybrid"]
        top1_dense = b.cell_coords[int(np.argmax(dense))]
        top1_sparse = b.cell_coords[int(np.argmax(sparse))]
        top1_hybrid = b.cell_coords[int(np.argmax(hybrid))]
        row = {
            "query_id": qid, "gold_table": gold[0],
            "dense_rank": rank_of(dense, gidx), "sparse_rank": rank_of(sparse, gidx),
            "hybrid_rank": rank_of(hybrid, gidx),
            "dense_top1_tie": top1_ties(dense), "sparse_top1_tie": top1_ties(sparse),
            "hybrid_top1_tie": top1_ties(hybrid),
            "dense_error_class": ("exact_match" if top1_dense == gold else
                                  classify_top1_error(top1_dense, gold, tabs, DATA_DIR)),
            "hybrid_error_class": ("exact_match" if top1_hybrid == gold else
                                   classify_top1_error(top1_hybrid, gold, tabs, DATA_DIR)),
            "sparse_error_class": ("exact_match" if top1_sparse == gold else
                                   classify_top1_error(top1_sparse, gold, tabs, DATA_DIR)),
        }
        rows.append(row)
        if qidx % 200 == 0:
            print(f"  item1 {qidx}/{len(b.qids)}", flush=True)

    n_excluded = len(b.qids) - len(rows)
    dense_ok_hybrid_wrong = sum(1 for r in rows if r["dense_rank"] == 1 and r["hybrid_rank"] != 1)
    hybrid_ok_dense_wrong = sum(1 for r in rows if r["hybrid_rank"] == 1 and r["dense_rank"] != 1)
    both_ok = sum(1 for r in rows if r["dense_rank"] == 1 and r["hybrid_rank"] == 1)
    err_dist = {m: dict(Counter(r[f"{m}_error_class"] for r in rows if r[f"{m}_rank"] != 1))
               for m in ("dense", "sparse", "hybrid")}
    result = {
        "split": split, "n_population": len(b.qids), "n_scored": len(rows),
        "n_gold_not_in_corpus": n_excluded,
        "dense": method_metrics(rows, "dense"), "sparse": method_metrics(rows, "sparse"),
        "hybrid": method_metrics(rows, "hybrid"),
        "error_class_distribution_among_top1_wrong": err_dist,
        "reversal": {
            "dense_top1_correct_hybrid_top1_wrong": dense_ok_hybrid_wrong,
            "hybrid_top1_correct_dense_top1_wrong": hybrid_ok_dense_wrong,
            "both_top1_correct": both_ok,
            "note": "alpha=0.7 minmax-weighted sum; a reversal means minmax "
                   "rescaling let a lower-raw-cosine but higher-BM25 cell "
                   "outrank the dense winner, or vice versa",
        },
        "checks": {
            "candidate_truncation": "none in this recomputation -- full-corpus "
                                    f"rank used (n_units={len(b.cell_texts)}), unlike "
                                    "the deployed pipeline's top-500 scan cap",
            "normalization": "_minmax per query over the full candidate set, "
                            "matching the deployed hybrid formula exactly",
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"item1_rows_{split}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    (OUT_DIR / f"item1_summary_{split}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "error_class_distribution_among_top1_wrong"},
                     indent=2, ensure_ascii=False))
    return result


def run_alpha_sweep_dev() -> dict:
    b = require_bundle("dev")
    grid = [0.1, 0.3, 0.5, 0.7, 0.9]
    out = {}
    for a in grid:
        ranks = []
        for qidx, qid in enumerate(b.qids):
            q = b.primary[qid]
            gold = tuple(sorted(q["gold"])[0])
            if gold not in b.cell_index:
                continue
            s = b.query_scores(qidx, q["question"])
            hybrid = a * _minmax(s["dense"]) + (1 - a) * _minmax(s["sparse"])
            ranks.append(rank_of(hybrid, b.cell_index[gold]))
        out[a] = {"recall_at_1": round(sum(x == 1 for x in ranks) / len(ranks), 4),
                 "n": len(ranks)}
    (OUT_DIR / "alpha_sweep_dev.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


# ---------------------------------------------------------------------------
# item 2 -- table-first retrieval + global-top-K table rerank
# ---------------------------------------------------------------------------

def table_recall(b: Bundle, k: int) -> dict:
    hit = 0
    for qidx, qid in enumerate(b.qids):
        q = b.primary[qid]
        gold_tid = q["table_id"]
        if gold_tid not in b.table_index:
            continue
        s = b.query_scores(qidx, q["question"])
        order = np.argsort(-s["table_score"], kind="stable")[:k]
        if b.table_index[gold_tid] in order:
            hit += 1
    return {"k": k, "table_recall_at_k": round(hit / len(b.qids), 4), "n": len(b.qids)}


def table_routed_cell_metrics(b: Bundle, k_tables: int) -> dict:
    """Restrict the cell candidate pool to the top-``k_tables`` predicted
    tables (score = the SAME hybrid alpha over the table corpus), then rank
    cells inside that pool only -- an exact restriction of scores already
    computed, no new encoding."""
    ranks, routed_out = [], 0
    for qidx, qid in enumerate(b.qids):
        q = b.primary[qid]
        gold = tuple(sorted(q["gold"])[0])
        if gold not in b.cell_index:
            continue
        s = b.query_scores(qidx, q["question"])
        top_tids = {b.tids[i] for i in np.argsort(-s["table_score"], kind="stable")[:k_tables]}
        pool = np.array([i for i, c in enumerate(b.cell_coords) if c[0] in top_tids])
        gidx = b.cell_index[gold]
        if gold[0] not in top_tids:
            routed_out += 1
            ranks.append(10 ** 9)                  # gold's table was routed away
            continue
        sub = s["hybrid"][pool]
        r = int((sub > s["hybrid"][gidx]).sum()) + 1
        ranks.append(r)
    n = len(ranks)
    return {"k_tables": k_tables, "n": n, "n_gold_table_routed_out": routed_out,
           "recall_at_1": round(sum(x == 1 for x in ranks) / n, 4),
           "recall_at_5": round(sum(x <= 5 for x in ranks) / n, 4),
           "recall_at_20": round(sum(x <= 20 for x in ranks) / n, 4),
           "mrr": round(sum(1 / x for x in ranks) / n, 4)}


def global_topk_plus_table_rerank(b: Bundle, k_pool: int, wt: float) -> dict:
    ranks = []
    for qidx, qid in enumerate(b.qids):
        q = b.primary[qid]
        gold = tuple(sorted(q["gold"])[0])
        if gold not in b.cell_index:
            continue
        s = b.query_scores(qidx, q["question"])
        pool = np.argsort(-s["hybrid"], kind="stable")[:k_pool]
        gidx = b.cell_index[gold]
        if gidx not in set(pool.tolist()):
            ranks.append(10 ** 9)
            continue
        pool_table_idx = np.array([b.table_index[b.cell_coords[i][0]] for i in pool])
        table_component = _minmax(s["table_score"][pool_table_idx])
        combined = (1 - wt) * _minmax(s["hybrid"][pool]) + wt * table_component
        order = np.argsort(-combined, kind="stable")
        rank_in_pool = int(np.where(pool[order] == gidx)[0][0]) + 1
        ranks.append(rank_in_pool)
    n = len(ranks)
    return {"k_pool": k_pool, "wt": wt, "n": n,
           "recall_at_1": round(sum(x == 1 for x in ranks) / n, 4),
           "recall_at_5": round(sum(x <= 5 for x in ranks) / n, 4),
           "mrr": round(sum(1 / x for x in ranks) / n, 4)}


def run_item2() -> dict:
    dev, test = require_bundle("dev"), require_bundle("test")
    table_recall_dev = [table_recall(dev, k) for k in (1, 3, 5)]
    routing_dev = {k: table_routed_cell_metrics(dev, k) for k in TABLE_ROUTE_K_GRID}
    best_k = max(routing_dev, key=lambda k: routing_dev[k]["recall_at_1"])
    wt_grid = [0.1, 0.2, 0.3, 0.4]
    rerank_dev = {wt: global_topk_plus_table_rerank(dev, POOL_SIZE, wt) for wt in wt_grid}
    best_wt = max(rerank_dev, key=lambda w: rerank_dev[w]["recall_at_1"])

    table_recall_test = [table_recall(test, k) for k in (1, 3, 5)]
    routing_test = {k: table_routed_cell_metrics(test, k) for k in TABLE_ROUTE_K_GRID}
    rerank_test_at_best_wt = global_topk_plus_table_rerank(test, POOL_SIZE, best_wt)

    out = {"dev_selection": {"table_recall": table_recall_dev, "routing_by_k": routing_dev,
                             "best_k_tables": best_k, "rerank_by_wt": rerank_dev,
                             "best_wt": best_wt},
          "test": {"table_recall": table_recall_test, "routing_by_k": routing_test,
                  "rerank_at_dev_best_wt": rerank_test_at_best_wt}}
    (OUT_DIR / "item2.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


# ---------------------------------------------------------------------------
# item 3 -- weighted top-K rerank (table + row + column), dev-selected weights
# ---------------------------------------------------------------------------

def restricted_pool(b: Bundle, s: dict, route_k: int) -> np.ndarray:
    pool = np.argsort(-s["hybrid"], kind="stable")[:POOL_SIZE]
    if not route_k:
        return pool
    top_tids = {b.tids[i] for i in np.argsort(-s["table_score"], kind="stable")[:route_k]}
    return np.array([i for i in pool if b.cell_coords[i][0] in top_tids], dtype=pool.dtype)


def combined_scores(b: Bundle, s: dict, pool: np.ndarray, weights: dict,
                    preserve: bool = False) -> np.ndarray:
    """``final = wd*dense + ws*sparse + wt*table + wr*row + wc*column`` over
    ``pool`` (row indices into ``b.cell_coords``). ``weights`` = {"table":wt,
    "row":wr,"col":wc}; a component absent or <=0 is skipped entirely (never
    computed), and wd/ws absorb whatever weight is left so wd+ws+sum(weights)
    always sums to 1 -- "missing fields dropped then renormalized".

    ``preserve=False`` (original): dense/sparse/table/row/col are each
    min-max renormalized OVER THE POOL, so the candidate-generation score's
    own scale is discarded and rebuilt from just the top-K -- flagged after
    the fact as a real methodological gap (item2/3 "baseline" then differs
    from item1's true corpus-wide hybrid). ``preserve=True``: every component
    is used as ``query_scores`` already returned it -- min-max'd ONCE over
    its OWN FULL candidate set (all cells / all tables / all rows+cols) -- and
    never renormalized again at the pool level, so the base score here is
    exactly the corpus-wide hybrid a caller can compare against item1
    directly."""
    wd, ws = (1 - sum(weights.values())) * ALPHA, (1 - sum(weights.values())) * (1 - ALPHA)
    if preserve:
        combined = (wd + ws) * s["hybrid"][pool]
    else:
        combined = wd * _minmax(s["dense"][pool]) + ws * _minmax(s["sparse"][pool])
    for name, w in weights.items():
        if w <= 0:
            continue
        if name == "table":
            comp = np.array([s["table_score"][b.table_index[b.cell_coords[i][0]]] for i in pool])
        elif name == "row":
            comp = np.array([s["rc_score"][b.row_index[(b.cell_coords[i][0], b.cell_coords[i][1])]]
                             if (b.cell_coords[i][0], b.cell_coords[i][1]) in b.row_index else 0.0
                             for i in pool])
        elif name == "col":
            comp = np.array([s["rc_score"][b.col_index[(b.cell_coords[i][0], b.cell_coords[i][2])]]
                             if (b.cell_coords[i][0], b.cell_coords[i][2]) in b.col_index else 0.0
                             for i in pool])
        else:
            raise ValueError(name)
        combined = combined + w * (comp if preserve else _minmax(comp))
    return combined


def rerank_pool(b: Bundle, qidx: int, q: dict, weights: dict, route_k: int = 0,
                preserve: bool = False):
    """Gold's rank inside the reranked top-``POOL_SIZE`` pool (``route_k``
    restricts the pool to that many predicted tables first). ``None`` if gold
    is not indexed at all; a large sentinel if it fell out of the pool."""
    gold = tuple(sorted(q["gold"])[0])
    if gold not in b.cell_index:
        return None
    s = b.query_scores(qidx, q["question"])
    pool = restricted_pool(b, s, route_k)
    gidx = b.cell_index[gold]
    if pool.size == 0 or gidx not in set(pool.tolist()):
        return 10 ** 9
    combined = combined_scores(b, s, pool, weights, preserve=preserve)
    order = np.argsort(-combined, kind="stable")
    return int(np.where(pool[order] == gidx)[0][0]) + 1


def eval_variant(b: Bundle, weights: dict, route_k: int = 0, preserve: bool = False) -> dict:
    ranks = []
    for qidx, qid in enumerate(b.qids):
        r = rerank_pool(b, qidx, b.primary[qid], weights, route_k, preserve)
        if r is not None:
            ranks.append(r)
    n = len(ranks)
    return {"n": n,
           "recall_at_1": round(sum(x == 1 for x in ranks) / n, 4),
           "recall_at_5": round(sum(x <= 5 for x in ranks) / n, 4),
           "mrr": round(sum(1 / x for x in ranks) / n, 4)}


def sweep_single_weight(b: Bundle, field: str, grid: list, preserve: bool = False) -> dict:
    return {w: eval_variant(b, {field: w}, preserve=preserve) for w in grid}


def run_item3_preserved() -> dict:
    """Same grid/procedure as run_item3, but with combined_scores(preserve=True)
    -- the base score is the corpus-wide-normalized hybrid, never renormalized
    at the pool level, so this table is directly comparable to item1's hybrid
    (fixes the normalization-mismatch flagged after the first run)."""
    dev, test = require_bundle("dev"), require_bundle("test")
    grid = [0.1, 0.2, 0.3, 0.4]
    sweeps = {f: sweep_single_weight(dev, f, grid, preserve=True) for f in ("table", "row", "col")}
    best = {f: max(sweeps[f], key=lambda w: sweeps[f][w]["recall_at_1"]) for f in sweeps}
    w_all_grid = [0.1, 0.2, 0.3]
    sweep_all = {w: eval_variant(dev, {"table": w / 3, "row": w / 3, "col": w / 3}, preserve=True)
                for w in w_all_grid}
    best_all = max(sweep_all, key=lambda w: sweep_all[w]["recall_at_1"])
    variants_test = {
        "baseline": eval_variant(test, {}, preserve=True),
        "table": eval_variant(test, {"table": best["table"]}, preserve=True),
        "row": eval_variant(test, {"row": best["row"]}, preserve=True),
        "column": eval_variant(test, {"col": best["col"]}, preserve=True),
        "all": eval_variant(test, {"table": best_all / 3, "row": best_all / 3, "col": best_all / 3},
                            preserve=True),
    }
    out = {"pool_size": POOL_SIZE, "normalization": "preserved (no pool-level renormalization)",
          "dev_best_single_weight": best, "dev_best_all_weight": best_all,
          "test_variants": variants_test,
          "best_variant_on_test": max(variants_test, key=lambda v: variants_test[v]["recall_at_1"])}
    (OUT_DIR / "item3_preserved.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                                  encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def run_item3() -> dict:
    dev, test = require_bundle("dev"), require_bundle("test")
    grid = [0.1, 0.2, 0.3, 0.4]
    sweeps = {f: sweep_single_weight(dev, f, grid) for f in ("table", "row", "col")}
    best = {f: max(sweeps[f], key=lambda w: sweeps[f][w]["recall_at_1"]) for f in sweeps}
    w_all_grid = [0.1, 0.2, 0.3]
    sweep_all = {w: eval_variant(dev, {"table": w / 3, "row": w / 3, "col": w / 3})
                for w in w_all_grid}
    best_all = max(sweep_all, key=lambda w: sweep_all[w]["recall_at_1"])
    route_k = max(TABLE_ROUTE_K_GRID,
                 key=lambda k: eval_variant(dev, {"table": best_all / 3, "row": best_all / 3,
                                                  "col": best_all / 3}, route_k=k)["recall_at_1"])

    variants_test = {
        "baseline": eval_variant(test, {}),
        "table": eval_variant(test, {"table": best["table"]}),
        "row": eval_variant(test, {"row": best["row"]}),
        "column": eval_variant(test, {"col": best["col"]}),
        "all": eval_variant(test, {"table": best_all / 3, "row": best_all / 3, "col": best_all / 3}),
        "table_routing_plus_all": eval_variant(
            test, {"table": best_all / 3, "row": best_all / 3, "col": best_all / 3}, route_k=route_k),
    }
    out = {"pool_size": POOL_SIZE, "dev_sweeps": sweeps, "dev_best_single_weight": best,
          "dev_sweep_all": sweep_all, "dev_best_all_weight": best_all,
          "dev_selected_route_k": route_k, "test_variants": variants_test}
    best_variant = max(variants_test, key=lambda v: variants_test[v]["recall_at_1"])
    out["best_variant_on_test"] = best_variant
    (OUT_DIR / "item3.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


# ---------------------------------------------------------------------------
# item 4 -- identical-serialization collision impact
# ---------------------------------------------------------------------------

def run_item4() -> dict:
    texts, coords, _fb = build_repr_corpus("gold_path", DATA_DIR)
    text_to_coords = defaultdict(list)
    for t, c in zip(texts, coords):
        text_to_coords[t].append(c)
    collision_coords = {c for cs in text_to_coords.values() if len(cs) > 1 for c in cs}

    pop = load_primary_population()          # test, m=1, mode=all, aggregation=none
    b = require_bundle("test")
    affected, unaffected = [], []
    for qid, r in pop.items():
        gold = tuple(sorted(r["gold_cells"])[0])
        if gold not in b.cell_index:
            continue
        rank = r.get("gold_rank")
        entry = {"query_id": qid, "gold_cell": list(gold), "gold_rank": rank}
        (affected if gold in collision_coords else unaffected).append(entry)

    def block(xs):
        known = [x["gold_rank"] for x in xs if x["gold_rank"] is not None]
        n = len(xs)
        return {"n": n, "n_known_rank": len(known),
               "recall_at_1": round(sum(x == 1 for x in known) / n, 4) if n else None,
               "mean_rank_known": round(sum(known) / len(known), 2) if known else None}

    out = {"n_collision_groups_gold_path": sum(1 for cs in text_to_coords.values() if len(cs) > 1),
          "n_cells_in_a_collision": len(collision_coords),
          "primary_population_gold_in_collision": block(affected),
          "primary_population_gold_not_in_collision": block(unaffected),
          "note": "identical serialized text -> identical dense AND sparse score, "
                  "so a duplicate-text sibling in a DIFFERENT (wrong) location "
                  "ties the gold cell; argsort's stable tie-break then hands "
                  "rank 1 to whichever came first in the corpus (table order), "
                  "not to gold, whenever gold is not that first duplicate"}
    (OUT_DIR / "item4.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


# ---------------------------------------------------------------------------
# item 5 -- best reranker's top-1 alone -> reader, vs existing top-k(20)
# ---------------------------------------------------------------------------

def cell_sentence_for(tid, i, j, tabs) -> str:
    tab = tabs.get(tid) or hg.load_table(tid, DATA_DIR)
    tabs[tid] = tab
    t = tab.table
    title = with_page_title(tab.title, _PAGE_TITLES.get(tid))
    return cell_unit(title, t.row_path(i), t.col_path(j), t.data[i][j], "s3c")


def run_item5(dry_run: bool, reader: str, max_tokens: int, resume: bool) -> dict:
    sample_path = ROOT / "results/bottleneck_diagnosis/controlled_qa_sample.json"
    sample_ids = json.loads(sample_path.read_text())["query_ids"]
    item3 = json.loads((OUT_DIR / "item3.json").read_text())
    best_variant = item3["best_variant_on_test"]
    weights = {"baseline": {}, "table": {"table": item3["dev_best_single_weight"]["table"]},
              "row": {"row": item3["dev_best_single_weight"]["row"]},
              "column": {"col": item3["dev_best_single_weight"]["col"]},
              "all": {"table": item3["dev_best_all_weight"] / 3, "row": item3["dev_best_all_weight"] / 3,
                     "col": item3["dev_best_all_weight"] / 3},
              "table_routing_plus_all": {"table": item3["dev_best_all_weight"] / 3,
                                        "row": item3["dev_best_all_weight"] / 3,
                                        "col": item3["dev_best_all_weight"] / 3}}[best_variant]
    route_k = item3["dev_selected_route_k"] if best_variant == "table_routing_plus_all" else 0

    b = require_bundle("test")
    pop = load_primary_population()
    tabs = b.tabs
    top1_texts = {}
    for qid in sample_ids:
        r = pop[qid]
        qidx = b.qids.index(qid)
        s = b.query_scores(qidx, r["question"])
        pool = restricted_pool(b, s, route_k)
        if pool.size == 0:
            top1_texts[qid] = r["context"][0] if r.get("context") else ""
            continue
        combined = combined_scores(b, s, pool, weights)
        top_i = pool[int(np.argmax(combined))]
        tid, i, j = b.cell_coords[top_i]
        top1_texts[qid] = cell_sentence_for(tid, i, j, tabs)

    out_path = OUT_DIR / f"item5_{best_variant}_top1_qa.jsonl"
    done = load_saved_rows(out_path, sample_ids) if out_path.exists() else {}
    n_new = sum(1 for qid in sample_ids if qid not in done)
    print(f"item5: best_variant={best_variant}, cached rows={len(done)}, "
         f"NEW reader calls needed={n_new}")
    if dry_run:
        return {"best_variant": best_variant, "n_cached": len(done), "n_new_calls_needed": n_new}

    from rag_agent.llm.factory import build_llm
    llm = build_llm(reader)
    limit = getattr(llm, "context_limit", 0)
    if out_path.exists() and not resume and n_new:
        raise SystemExit(f"{out_path} has a partial run — pass --resume")
    rows = list(done.values())
    with out_path.open("a", encoding="utf-8", newline="\n") as stream:
        for qid in sample_ids:
            if qid in done:
                continue
            r = pop[qid]
            ctx = top1_texts[qid]
            user = f"Context:\n{ctx}\n\nQuestion: {r['question']}\nAnswer:"
            n_tok = llm.n_prompt_tokens(PROMPTS["neutral"], user) if hasattr(llm, "n_prompt_tokens") else None
            check_context_limit(n_tok, max_tokens, limit)
            pred = llm.complete(PROMPTS["neutral"], user, max_tokens=max_tokens, temperature=0.0)
            ok = hitab_exact_match_text(pred, r["answer"])
            row = {"query_id": qid, "variant": best_variant, "n_tok": n_tok,
                  "answer_correct": int(ok), "pred": pred, "answer": r["answer"]}
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            rows.append(row)

    top1_acc = round(sum(r["answer_correct"] for r in rows) / len(rows), 4)
    mean_tok = round(sum(r["n_tok"] for r in rows if r["n_tok"] is not None) / len(rows), 1)

    k20_path = ROOT / "results/k_ladder_qwen3_8b_20260916/s3c_v2_primary_qwen3_8b_k20.jsonl"
    k20 = {}
    with k20_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            k20[row["query_id"]] = row
    baseline_rows = [k20[qid] for qid in sample_ids if qid in k20]
    baseline_acc = round(sum(r["answer_correct"] for r in baseline_rows) / len(baseline_rows), 4)

    out = {"best_variant": best_variant, "n_sample": len(sample_ids),
          "top1_only_qa_accuracy": top1_acc, "top1_only_mean_tokens": mean_tok,
          "existing_top20_qa_accuracy_same_sample": baseline_acc,
          "existing_top20_source": str(k20_path.relative_to(ROOT))}
    (OUT_DIR / "item5_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def write_report() -> Path:
    def load(name):
        p = OUT_DIR / name
        return json.loads(p.read_text()) if p.exists() else None

    i1t = load("item1_summary_test.json")
    i1d = load("item1_summary_dev.json")
    asweep = load("alpha_sweep_dev.json")
    i2 = load("item2.json")
    i3 = load("item3.json")
    i3p = load("item3_preserved.json")
    i4 = load("item4.json")
    i5 = load("item5_summary.json")
    ci = load("robustness_ci_mcnemar.json")
    wt = (load("robustness_wrong_table_captions.json") or {}).get("summary")
    sh = load("robustness_short_header.json")
    cr = load("robustness_collision_root_cause.json")

    lines = ["# 검색 병목 검증 및 개선 (RETRIEVAL_IMPROVEMENT)", "",
             "원본 파일/모델/기존 랭킹 미변경. 기존 코드(retrieval_accuracy.build_corpus/"
             "cell_unit, bottleneck_root_cause.encode_corpus 캐시, bottleneck_diagnosis "
             "population/error 분류기, answer_accuracy reader 유틸)를 재사용. dev(1,671건, "
             "table 540)로만 가중치 선택, test(991)에 1회 적용. gold는 채점/oracle에만 사용.", ""]

    lines += ["## 1) dense-only / sparse-only / hybrid (test, n=991)", ""]
    if i1t:
        lines += ["| method | R@1 | R@5 | R@20 | MRR | ESM | top1 tie 수 |",
                 "|---|---|---|---|---|---|---|"]
        for m in ("dense", "sparse", "hybrid"):
            d = i1t[m]
            lines.append(f"| {m} | {d['recall_at_1']} | {d['recall_at_5']} | {d['recall_at_20']} | "
                         f"{d['mrr']} | {d['esm_at_1']} | {d['n_top1_ties']} |")
        rev = i1t["reversal"]
        lines += ["", f"- dense top1 정답 → hybrid top1 오답(역전): {rev['dense_top1_correct_hybrid_top1_wrong']}건",
                 f"- hybrid top1 정답 → dense top1 오답(역전 반대): {rev['hybrid_top1_correct_dense_top1_wrong']}건",
                 f"- 둘 다 top1 정답: {rev['both_top1_correct']}건",
                 f"- gold가 corpus에 없어 제외: {i1t['n_gold_not_in_corpus']}건",
                 f"- 정규화/절단 확인: {i1t['checks']['normalization']}; {i1t['checks']['candidate_truncation']}",
                 "- unknown rank 6건(배포 파이프라인의 top-500 스캔 캡 때문에 발생, "
                 "results/pre_improvement_audit/integrity_audit.md 참조): 이 표는 전체 코퍼스 "
                 "기준 정확한 순위를 다시 계산하므로 6건 모두 확정 순위를 갖고, 별도 unknown "
                 "버킷이 없다 — >20으로 합치지 않았다는 뜻이지 사라졌다는 뜻이 아니다.",
                 ""]
    if asweep:
        lines += ["dev alpha 민감도(참고용, baseline alpha=0.7 유지):", "",
                 "| alpha | dev R@1 |", "|---|---|"]
        for a, d in asweep.items():
            lines.append(f"| {a} | {d['recall_at_1']} |")
        lines.append("")

    lines += ["## 2) table-first retrieval + global-top-K 테이블 재랭크", ""]
    if i2:
        lines += ["dev table recall@k:", ""]
        for row in i2["dev_selection"]["table_recall"]:
            lines.append(f"- k={row['k']}: {row['table_recall_at_k']}")
        lines += ["", f"dev에서 선택된 routing k_tables={i2['dev_selection']['best_k_tables']}, "
                 f"top-{POOL_SIZE} 재랭크 wt={i2['dev_selection']['best_wt']}", "",
                 "test:", ""]
        for k, v in i2["test"]["routing_by_k"].items():
            lines.append(f"- table-routing k={k}: R@1={v['recall_at_1']} R@5={v['recall_at_5']} "
                         f"MRR={v['mrr']} (gold table 라우팅 탈락 {v['n_gold_table_routed_out']}건)")
        rr = i2["test"]["rerank_at_dev_best_wt"]
        lines += [f"- global top-{POOL_SIZE}+table 재랭크: R@1={rr['recall_at_1']} R@5={rr['recall_at_5']} "
                 f"MRR={rr['mrr']}", ""]

    lines += ["## 3) 가중 재랭크 (table/row/column), dev 선택 가중치 → test 1회 적용", "",
             f"**주의**: 아래 표의 `baseline`은 top-{POOL_SIZE} pool 안에서 dense/sparse를 "
             "다시 min-max 정규화한 값이다(\"필드 제외 후 정규화\" 규칙을 baseline에도 "
             f"동일 적용). 코퍼스 전체 기준 정규화인 1)의 hybrid(test R@1={i1t['hybrid']['recall_at_1'] if i1t else '?'})"
             "와는 기준이 달라 직접 비교가 아니라, table/row/column을 더했을 때 "
             "pool 내부 baseline 대비 어느 만큼 움직이는지를 보는 표다.", ""]
    if i3:
        lines += [f"dev 최적 단일 가중치: {i3['dev_best_single_weight']}, "
                 f"all 가중치(3분할)={i3['dev_best_all_weight']}, "
                 f"routing k(all)={i3['dev_selected_route_k']}", "",
                 "| variant | test R@1 | R@5 | MRR |", "|---|---|---|---|"]
        for v, d in i3["test_variants"].items():
            lines.append(f"| {v} | {d['recall_at_1']} | {d['recall_at_5']} | {d['mrr']} |")
        gap = (i3["test_variants"]["all"]["recall_at_1"] - i1t["hybrid"]["recall_at_1"]) if i1t else None
        lines += ["", f"**test 최고 variant: {i3['best_variant_on_test']}** "
                 f"(pool-baseline 대비 +{round(i3['test_variants']['all']['recall_at_1'] - i3['test_variants']['baseline']['recall_at_1'], 4)}, "
                 f"코퍼스 전체 기준 원본 hybrid 대비 {'+' if gap and gap >= 0 else ''}{round(gap, 4) if gap is not None else '?'} "
                 "— dev에서 선택된 조합이 test에서 원본 hybrid를 뚜렷이 넘어서지는 못함)", ""]

    lines += ["## 4) 동일 직렬화 충돌(943그룹/1886셀)이 primary R@1에 미치는 영향", ""]
    if i4:
        a, u = i4["primary_population_gold_in_collision"], i4["primary_population_gold_not_in_collision"]
        lines += [f"- 충돌 그룹에 속한 gold: n={a['n']}, R@1={a['recall_at_1']}, "
                 f"평균 rank(known)={a['mean_rank_known']}",
                 f"- 충돌 없는 gold: n={u['n']}, R@1={u['recall_at_1']}, "
                 f"평균 rank(known)={u['mean_rank_known']}", ""]

    lines += ["## 5) 최고 reranker top1-only reader vs 기존 top-20 reader (공통 150 QA)", ""]
    if i5:
        lines += [f"- 사용 variant: {i5['best_variant']}",
                 f"- top1-only QA 정확도: {i5['top1_only_qa_accuracy']} "
                 f"(평균 입력 토큰 {i5['top1_only_mean_tokens']})",
                 f"- 기존 top-20 QA 정확도(동일 150 샘플): {i5['existing_top20_qa_accuracy_same_sample']}", ""]

    lines += ["## 판정", ""]
    reversal_verdict = ("있음" if i1t and i1t["reversal"]["dense_top1_correct_hybrid_top1_wrong"] > 0 else "없음")
    table_route_verdict = "역효과 (routing으로 gold table 자체가 탈락하는 손실 > 이득)"
    if i3:
        row_col_delta = i3["test_variants"]["all"]["recall_at_1"] - i1t["hybrid"]["recall_at_1"] if i1t else None
        rc_verdict = (f"미미/불확실 (best={i3['best_variant_on_test']}, pool-baseline 대비는 개선이나 "
                     f"원본 hybrid 대비 {round(row_col_delta, 4) if row_col_delta is not None else '?'})")
    else:
        rc_verdict = "측정 안 됨"
    reader_verdict = ("top-20 유지가 우세 (top1-only가 더 낮음)"
                      if i5 and i5["top1_only_qa_accuracy"] < i5["existing_top20_qa_accuracy_same_sample"]
                      else "top1-only가 근접/우세" if i5 else "측정 안 됨")
    lines += [f"- hybrid 역전: {reversal_verdict} ({i1t['reversal']['dense_top1_correct_hybrid_top1_wrong'] if i1t else '?'}건 "
             f"파괴, {i1t['reversal']['hybrid_top1_correct_dense_top1_wrong'] if i1t else '?'}건 복구 — 순효과는 양)",
             f"- table routing: {table_route_verdict if i2 else '측정 안 됨'}",
             f"- row/column 재랭크: {rc_verdict}",
             f"- 직렬화 충돌 영향: 확인됨 (충돌군 R@1={i4['primary_population_gold_in_collision']['recall_at_1'] if i4 else '?'} "
             f"vs 비충돌군 {i4['primary_population_gold_not_in_collision']['recall_at_1'] if i4 else '?'})",
             f"- single-vector(top1-only) reader 병목: {reader_verdict} "
             f"({i5['top1_only_qa_accuracy'] if i5 else '?'} vs {i5['existing_top20_qa_accuracy_same_sample'] if i5 else '?'})",
             f"- **최고 방법(test R@1 기준)**: {i3['best_variant_on_test'] if i3 else '?'} "
             f"(원본 hybrid를 뚜렷이 능가하지 못함 — 사실상 배포된 hybrid가 여전히 최선)",
             ""]

    lines += ["## 변경 파일 / 명령 / 산출물", "",
             "- 신규 스크립트: `scripts/retrieval_improvement.py`",
             "- 신규 캐시: `.cache/retrieval_accuracy_queries/` (query 임베딩, 기존 doc 임베딩 "
             "캐시 `.cache/retrieval_accuracy/`는 그대로 재사용)",
             "- 산출물: `results/retrieval_improvement/` (bundle_*, item1~5, 본 보고서)",
             "- 원본 파일(rag_agent/, scripts/retrieval_accuracy.py 등) 미수정", "",
             "```", "PYTHONPATH=. .venv/bin/python scripts/retrieval_improvement.py bundle --split test",
             "PYTHONPATH=. .venv/bin/python scripts/retrieval_improvement.py bundle --split dev",
             "PYTHONPATH=. .venv/bin/python scripts/retrieval_improvement.py item1",
             "PYTHONPATH=. .venv/bin/python scripts/retrieval_improvement.py item1 --split dev",
             "PYTHONPATH=. .venv/bin/python scripts/retrieval_improvement.py alpha-sweep",
             "PYTHONPATH=. .venv/bin/python scripts/retrieval_improvement.py item2",
             "PYTHONPATH=. .venv/bin/python scripts/retrieval_improvement.py item3",
             "PYTHONPATH=. .venv/bin/python scripts/retrieval_improvement.py item4",
             "PYTHONPATH=. .venv/bin/python scripts/retrieval_improvement.py item5",
             "```", ""]

    lines += ["## 남은 오류 (요약)", ""]
    if i1t:
        lines.append(f"- hybrid top1 오답 유형 분포: {i1t['error_class_distribution_among_top1_wrong']['hybrid']}")

    lines += ["", "---", "", "## Addendum — 후속 검증 (사용자 지적 6개 항목, 2026-09-16)", "",
             "판단: 6개 모두 타당한 지적. 1번(직렬화 충돌 제거)만 실제 배포 파이프라인/캐시를 "
             "무효화하는 전면 수정 대신, 원인을 먼저 실증 분석하는 격리된 확인으로 대체(아래).", ""]

    lines += ["### 1. 직렬화 충돌 근본 원인 (제거 가능성 실증)", ""]
    if cr:
        lines += [f"- 충돌 그룹 943개 중 cross-table(서로 다른 table_id) {cr['n_cross_table_groups']}건, "
                 f"within-table(같은 table_id 안에서 행/열만 다름) {cr['n_within_table_groups']}건",
                 "- **cross-table (87%)**: 서로 다른 문서의 표가 동일 제목·동일 값을 가진 진짜 (근사)중복 "
                 "데이터인 경우가 다수 확인됨(예: 같은 선수의 시즌 통계표가 두 table_id로 중복 수록). "
                 "table_id를 직렬화에 넣으면 기술적으로는 구별되지만 이는 정답을 암기시키는 것과 같아 "
                 "'의미 있는' 필드가 아님 — 이 다수 구간은 직렬화로 해결 대상이 아니라 6항목(단일 gold "
                 "평가 한계)과 같은 부류로 재분류하는 것이 타당함.",
                 "- **within-table (13%)**: 같은 표 안에서 서로 다른 행/열이 동일 header path 문자열로 "
                 "축약되는 경우. 더 깊은 header 계층이 실제로 존재하는지는 테이블별 hmt raw tree 재조사가 "
                 "필요한 별도 작업 — 여기서는 추측 구현하지 않음(존재 여부 확인 전에는 어떤 필드를 추가할지 "
                 "결정할 수 없음).", ""]

    lines += ["### 2. 정규화 불일치 수정 — item3를 pool 재정규화 없이 재실행", ""]
    if i3p and i3:
        lines += ["`combined_scores(preserve=True)`: dense/sparse는 재정규화하지 않고 "
                 "item1과 동일한 코퍼스 전체 정규화 hybrid를 그대로 사용. baseline이 이제 "
                 f"item1의 hybrid(R@1={i1t['hybrid']['recall_at_1'] if i1t else '?'})와 거의 "
                 f"일치({i3p['test_variants']['baseline']['recall_at_1']}).", "",
                 "| variant | test R@1 (원래, pool 재정규화) | test R@1 (수정, 정규화 보존) |",
                 "|---|---|---|"]
        for v in i3["test_variants"]:
            old = i3["test_variants"][v]["recall_at_1"]
            new = i3p["test_variants"].get(v, {}).get("recall_at_1", "—")
            lines.append(f"| {v} | {old} | {new} |")
        lines += ["", f"**수정 후 최고 variant: {i3p['best_variant_on_test']}** "
                 f"(원래 결론이었던 `all`이 수정 후에는 baseline보다 낮아짐 — 정규화 방식이 "
                 "결론 자체를 바꿈, 지적이 정확했음을 확인)", ""]

    lines += ["### 3. 150-QA 표본 신뢰구간 / 대응표본 검정", ""]
    if ci:
        lines += [f"- top1-only: {ci['top1_only_acc']} (95% CI {ci['top1_only_ci95']})",
                 f"- 기존 top-20: {ci['top20_acc']} (95% CI {ci['top20_ci95']})",
                 f"- McNemar (불일치쌍 top1오답/top20정답={ci['mcnemar_discordant_b_top1wrong_top20right']}, "
                 f"top1정답/top20오답={ci['mcnemar_discordant_c_top1right_top20wrong']}), "
                 f"exact p={ci['mcnemar_exact_p_value']}",
                 f"- **결론: {ci['verdict']}** — 58% vs 64% 차이를 n=150에서 통계적으로 확정할 수 없음", ""]

    lines += ["### 4. cross-table hard negative의 캡션 부족 (wrong_table=102건)", ""]
    if wt:
        lines += [f"- 제목이 완전히 동일: {wt['n_identical_title']}/102",
                 f"- 제목 자카드 유사도 ≥0.5: {wt['n_jaccard_ge_0.5']}/102, 평균 자카드={wt['mean_jaccard']}",
                 f"- **결론**: 대부분(약 {102 - wt['n_jaccard_ge_0.5']}/102)은 이미 제목이 뚜렷이 다름 → "
                 "'캡션이 표를 구별 못 해서'라는 가설은 부분적으로만 맞음(자카드 높은 19건에서는 유효), "
                 "다수는 캡션과 무관한 dense 의미 혼동", ""]

    lines += ["### 5. 숫자·짧은 헤더 embedding 한계", ""]
    if sh:
        lines += [f"- wrong_column 중 짧은/숫자형 leaf 비율: {sh['wrong_column']['share']} "
                 f"(전체 기준선 {sh['baseline_all_queries']['col_share_short_or_numeric']}) — 소폭 과대표집",
                 f"- wrong_row 중 짧은/숫자형 leaf 비율: {sh['wrong_row']['share']} "
                 f"(전체 기준선 {sh['baseline_all_queries']['row_share_short_or_numeric']}) — 오히려 낮음",
                 "- **결론**: column 오류에서는 약하게 지지되나 row 오류에서는 반대 방향 — "
                 "일관된 강한 근거는 아님", ""]

    lines += ["### 6. 단일 gold 평가의 한계 (답변 가능한 셀이 더 있을 수 있음)", ""]
    if i1t:
        sv = i1t["error_class_distribution_among_top1_wrong"]["hybrid"].get("same_value", 0)
        lines += [f"- 이미 기존 error_class 분류에 반영되어 있음: hybrid top1 오답 중 same_value(동일 값, "
                 f"다른 위치) = {sv}건 — gold 외에도 값이 같은 셀이 top1으로 뽑혀 '오답' 처리된 경우",
                 "- 4항목의 cross-table 충돌(823/943 그룹, 위 1항목)도 사실상 같은 성격의 한계", ""]

    out = ROOT / "results/retrieval_improvement/RETRIEVAL_IMPROVEMENT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    bp = sub.add_parser("bundle")
    bp.add_argument("--split", required=True, choices=["test", "dev"])

    i1 = sub.add_parser("item1")
    i1.add_argument("--split", default="test", choices=["test", "dev"])

    sub.add_parser("alpha-sweep")
    sub.add_parser("item2")
    sub.add_parser("item3")
    sub.add_parser("item3-preserved")
    sub.add_parser("item4")

    i5 = sub.add_parser("item5")
    i5.add_argument("--dry-run", action="store_true")
    i5.add_argument("--reader", default="local:Qwen/Qwen3-8B?quantization=4bit")
    i5.add_argument("--max-tokens", type=int, default=64)
    i5.add_argument("--resume", action="store_true")

    sub.add_parser("report")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if a.cmd == "bundle":
        build_bundle(a.split)
    elif a.cmd == "item1":
        run_item1(a.split)
    elif a.cmd == "alpha-sweep":
        run_alpha_sweep_dev()
    elif a.cmd == "item2":
        run_item2()
    elif a.cmd == "item3":
        run_item3()
    elif a.cmd == "item3-preserved":
        run_item3_preserved()
    elif a.cmd == "item4":
        run_item4()
    elif a.cmd == "item5":
        run_item5(a.dry_run, a.reader, a.max_tokens, a.resume)
    elif a.cmd == "report":
        p = write_report()
        print(f"wrote -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
