#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Retrieval accuracy on HiTab: per query, was the retrieval right or wrong?

One number, defined the way the dataset lets it be defined. HiTab says which
cells each answer is read from, so a query is not scored on "how much of the
gold was covered" or "how deep the first hit was" — it is scored PASS or FAIL,
and the accuracy is the pass rate over the whole split:

    accuracy = (# queries whose retrieved context contains the answer evidence)
               / (# queries)

The retrieved context is a fixed CELL BUDGET, identical for every arm, so a
baseline that retrieves whole rows or whole tables is not quietly handed more
of the table than the cell index gets. Two gold shapes exist, both from the
annotation and neither chosen by us:

  all  the answer is read off specific data cells (``answer_formulas``) — every
       one of them must be in the context.
  any  the answer IS a header ("which province had the highest rate"), which no
       cell index holds as a unit; the header rides in the sentence of every
       cell it labels, so any one of those cells serves. Reported as its own
       row, never pooled into the headline, because its bar is lower.

Queries whose gold cannot be resolved are EXCLUDED BY NAME with a reason and
counted in the summary. They are not dropped into the denominator's shadow.

  PYTHONPATH=. .venv/bin/python scripts/retrieval_accuracy.py --arm hybrid
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

import numpy as np                                                    # noqa: E402

from rag_agent.bench import hitab_grid as hg                          # noqa: E402
from rag_agent.data.loader import load_samples                        # noqa: E402
from rag_agent.retrieve.encoders import default_encoder               # noqa: E402
from rag_agent.retrieve.hybrid_index import _minmax, _tokenize        # noqa: E402
from rag_agent.retrieve.sparse_bm25 import SparseBM25                 # noqa: E402
from rag_agent.serialization.caption import caption_sentence          # noqa: E402
from rag_agent.serialization.templates import (MT2NET, STRUCTURAL,    # noqa: E402
                                               STRUCTURAL_COMPACT)

TEMPLATES = {"s3c": STRUCTURAL_COMPACT, "s3": STRUCTURAL, "mt2net": MT2NET}
# Two ablations of the index unit itself, not templates: they answer "what does
# the header path buy, and what does the table's own label buy" by removing one
# at a time. Byte-identical to point3_reconstruction_cost.cell_text(..., "flat")
# and (..., "S2"), so they stay comparable with the older numbers on disk.
ABLATIONS = ("flat", "s2")
UNITS = ("cell", "row", "table")
PAGE_TITLES = ROOT / "results/tableconf/totto_page_titles.json"


def cell_unit(title, row_path, col_path, value, template: str) -> str:
    """One index unit's text under ``template``.

    ``flat`` keeps only the LEAF labels -- the hierarchy is gone, and so is the
    table's own label. ``s2`` keeps the full header path but no table label.
    Together they separate the two things the deployed unit carries.
    """
    if template == "flat":
        lab = " ".join(x for x in ((row_path[-1] if row_path else ""),
                                   (col_path[-1] if col_path else "")) if x)
        return f"{lab}: {value}" if lab else str(value)
    if template == "s2":
        path = " > ".join([*row_path, *col_path])
        return f"{path}: {value}" if path else str(value)
    return caption_sentence(title, row_path, col_path, value=value,
                            template=TEMPLATES[template])


def build_corpus(data_dir: str, tids, template: str, unit: str, page_titles: dict):
    """(texts, cell_sets) — one index unit per entry, and the cells it delivers."""
    texts, covers = [], []
    for tid in tids:
        tab = hg.load_table(tid, data_dir)
        if tab is None:
            continue
        t = tab.table
        title = tab.title
        pg = (page_titles.get(tid) or {}).get("page_title", "").strip()
        if pg:
            title = f"{pg}: {title}" if title else pg
        live = [(i, j) for i in range(t.n_rows) for j in range(t.n_cols)
                if str(t.data[i][j]).strip()]
        if unit == "cell":
            for i, j in live:
                texts.append(cell_unit(title, t.row_path(i), t.col_path(j),
                                       t.data[i][j], template))
                covers.append(frozenset([(tid, i, j)]))
        elif unit == "row":
            for i in range(t.n_rows):
                cs = [(i, j) for (a, j) in live if a == i]
                if not cs:
                    continue
                texts.append(" | ".join(
                    cell_unit(title if k == 0 else "", t.row_path(i),
                              t.col_path(j), t.data[i][j], template)
                    for k, (i_, j) in enumerate(cs)))
                covers.append(frozenset((tid, i, j) for _i, j in cs))
        else:                                     # whole table as one unit
            texts.append(" | ".join(
                cell_unit(title if k == 0 else "", t.row_path(i),
                          t.col_path(j), t.data[i][j], template)
                for k, (i, j) in enumerate(live)))
            covers.append(frozenset((tid, i, j) for i, j in live))
    return texts, covers


def load_queries(data_dir: str, split: str, tabs):
    """Every question in the split, with its gold target and why it is excluded."""
    out = []
    for s in load_samples(data_dir, split):
        tid = s["table_id"]
        tab = tabs.get(tid) or hg.load_table(tid, data_dir)
        tabs[tid] = tab
        if tab is None:
            out.append({"query_id": s["id"], "question": s["question"],
                        "answer": s.get("answer"), "table_id": tid,
                        "gold": set(), "mode": "all", "excluded": "table_missing"})
            continue
        gold, mode, why = hg.gold_target(s, tab)
        out.append({"query_id": s["id"], "question": s["question"],
                    "answer": s.get("answer"), "table_id": tid,
                    "aggregation": (s.get("aggregation") or [None])[0]
                    if isinstance(s.get("aggregation"), list) else s.get("aggregation"),
                    "gold": gold, "mode": mode, "excluded": why})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--corpus", default="all", choices=["all", "split"],
                    help="all = every table in the store (the real haystack); "
                         "split = only tables this split's questions touch")
    ap.add_argument("--template", default="s3c",
                    choices=list(TEMPLATES) + list(ABLATIONS))
    ap.add_argument("--unit", default="cell", choices=UNITS)
    ap.add_argument("--embed-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--alpha", type=float, default=0.7,
                    help="dense weight. 0 = BM25 only, 1 = dense only. The "
                         "default was fixed on the dev split in an earlier "
                         "phase and is not re-tuned here.")
    ap.add_argument("--budget", type=int, default=20,
                    help="context size in CELLS — the same budget for every "
                         "arm, so a row/table unit cannot buy coverage with a "
                         "bigger context")
    ap.add_argument("--no-query-prefix", action="store_true",
                    help="encode the question with no instruction prefix — what "
                         "the pipeline did before the fix, kept so the cost of "
                         "that bug is a measured number rather than a claim")
    ap.add_argument("--cache-dir", default=".cache/retrieval_accuracy")
    ap.add_argument("--dump-context", type=int, default=0,
                    help="also write the top-N context sentences per query, "
                         "for the reader stage")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default="results/retrieval_accuracy")
    a = ap.parse_args()

    t0 = time.time()
    page_titles = json.loads(PAGE_TITLES.read_text()) if PAGE_TITLES.exists() else {}
    tabs: dict = {}
    queries = load_queries(a.data_dir, a.split, tabs)
    tids = (hg.table_ids(a.data_dir) if a.corpus == "all"
            else sorted({q["table_id"] for q in queries}))
    texts, covers = build_corpus(a.data_dir, tids, a.template, a.unit, page_titles)
    print(f"[corpus] {len(tids)} tables / {len(texts)} {a.unit} units "
          f"({time.time() - t0:.0f}s)", flush=True)

    t0 = time.time()
    bm = SparseBM25(_tokenize(t) for t in texts)
    print(f"[bm25] {len(bm.vocab)} terms in {time.time() - t0:.0f}s", flush=True)

    enc = default_encoder(model_name=a.embed_model) if a.alpha > 0 else None
    if enc is not None and a.no_query_prefix:
        enc.query_prefix = ""
    emb = None
    if a.alpha > 0:
        cache = Path(a.cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        from hashlib import md5
        key = md5("\x00".join(texts).encode()).hexdigest()[:16]
        f = cache / f"{enc.name.replace('/', '_')}_{len(texts)}_{key}.npy"
        if f.exists():
            emb = np.load(f)
            print(f"[dense] cache hit {f.name}", flush=True)
        else:
            t0 = time.time()
            emb = enc.encode(texts)
            np.save(f, emb)
            print(f"[dense] encoded {len(texts)} in {time.time() - t0:.0f}s", flush=True)

    recs, t0 = [], time.time()
    for k, q in enumerate(queries, 1):
        if q["excluded"]:
            recs.append({"query_id": q["query_id"], "excluded": q["excluded"],
                         "table_id": q["table_id"], "mode": q["mode"]})
            continue
        s = bm.get_scores(_tokenize(q["question"]))
        if emb is not None:
            d = emb @ enc.encode_query([q["question"]])[0].astype(np.float32)
            s = a.alpha * _minmax(d) + (1 - a.alpha) * _minmax(s) if a.alpha < 1 else d
        order = np.argsort(-s)
        got, n_cells, ctx = set(), 0, []
        for p in order:
            if n_cells >= a.budget:
                break
            got |= covers[p]
            n_cells += len(covers[p])
            if a.dump_context and len(ctx) < a.dump_context:
                ctx.append(texts[p])
        gold = q["gold"]
        hit = (gold <= got) if q["mode"] == "all" else bool(gold & got)
        r = {"query_id": q["query_id"], "table_id": q["table_id"],
             "mode": q["mode"], "m": len(gold), "correct": int(hit),
             "aggregation": q.get("aggregation"), "cells_in_context": n_cells,
             "gold_table_in_context": int(any(c[0] == q["table_id"] for c in got))}
        if a.dump_context:
            r["question"] = q["question"]
            r["answer"] = q["answer"]
            r["context"] = ctx
            r["gold_cells"] = sorted(list(gold))[:64]
        recs.append(r)
        if k % 200 == 0:
            print(f"  {k}/{len(queries)}  {time.time() - t0:.0f}s", flush=True)

    scored = [r for r in recs if "correct" in r]
    excl = Counter(r["excluded"] for r in recs if "excluded" in r)
    by_mode = defaultdict(list)
    for r in scored:
        by_mode[r["mode"]].append(r["correct"])

    def acc(v):
        return round(sum(v) / len(v), 4) if v else None

    summary = {
        "split": a.split, "corpus": a.corpus, "n_tables": len(tids),
        "n_units": len(texts), "unit": a.unit, "template": a.template,
        "encoder": enc.name if enc else "none (bm25 only)", "alpha": a.alpha,
        "query_prefix": (enc.query_prefix if enc else ""), "budget_cells": a.budget,
        "n_queries_in_split": len(queries), "n_scored": len(scored),
        "n_excluded": sum(excl.values()), "excluded_by_reason": dict(excl),
        "accuracy_all_mode": acc(by_mode["all"]), "n_all_mode": len(by_mode["all"]),
        "accuracy_any_mode": acc(by_mode["any"]), "n_any_mode": len(by_mode["any"]),
        "accuracy_scored": acc([r["correct"] for r in scored]),
        "accuracy_over_full_split": round(
            sum(r["correct"] for r in scored) / len(queries), 4),
        "gold_table_in_context": round(
            sum(r["gold_table_in_context"] for r in scored) / max(len(scored), 1), 4),
    }
    tag = a.tag or f"{a.split}_{a.corpus}_{a.unit}_{a.template}_a{a.alpha}_k{a.budget}"
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps(summary, indent=2))
    with open(out / f"{tag}_records.jsonl", "w") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote -> {out / tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
