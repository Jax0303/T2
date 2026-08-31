#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Where every gold cell lands in the corpus-wide ranking. No token budget.

`corpus_dump_vs_cell.py` reports OSC, which is set-EM@C where C is however many
cells a token budget happened to buy. That conflates two questions -- did the
ranker put the gold cell high, and did the budget reach it -- and stage 1 of
this work is only the first one. So this ranks and records, and stops.

One JSONL line per query::

    {query_id, m, gold_table, table_rank_cellvote, ranks: [...], qtype, subqtype}

``ranks`` are 0-based positions of each gold cell in the full ranking, so
Recall@k, set-EM@k, hit@k and MRR all read off them at any k, and a reader is
never called.

  PYTHONPATH=. .venv/bin/python analysis/cell_rank_dump.py --dataset hitab \
      --population hitab_dev_lookup_all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                                                   # noqa: E402
import corpus_dump_vs_cell as cdv                                    # noqa: E402
from header_path_coverage import load_corpus                         # noqa: E402
from rag_agent.retrieve.encoders import default_encoder              # noqa: E402
from rag_agent.retrieve.hybrid_index import HybridIndex, _minmax     # noqa: E402
from rag_agent.serialization.base import Chunk                       # noqa: E402
from rag_agent.serialization.caption import caption_sentence         # noqa: E402
from rag_agent.serialization.templates import (MT2NET, STRUCTURAL,   # noqa: E402
                                               STRUCTURAL_COMPACT)

TEMPLATE = {"S3": STRUCTURAL, "S3c": STRUCTURAL_COMPACT, "mt2net": MT2NET}
KS = (1, 5, 10, 20, 50, 100, 200, 500)


def cell_texts(C, scheme):
    if scheme == "S2":
        return [cdv.cell_text(rp, cp, v, "S2") for rp, cp, v in C.cell_paths]
    return [caption_sentence(C.title.get(t, ""), *C.cell_paths[n],
                             template=TEMPLATE[scheme])
            for n, (t, _i, _j) in enumerate(C.cell_owner)]


def report(recs, title):
    n = len(recs)
    if not n:
        return
    print(f"\n{title}  n={n}")
    print(f"{'k':>6}{'Recall@k':>10}{'setEM@k':>10}{'hit@k':>8}")
    for k in KS:
        rec = sum(sum(1 for r in q["ranks"] if r < k) / len(q["ranks"]) for q in recs) / n
        st = sum(1 for q in recs if max(q["ranks"]) < k) / n
        hit = sum(1 for q in recs if min(q["ranks"]) < k) / n
        print(f"{k:>6}{rec:>10.4f}{st:>10.4f}{hit:>8.4f}")
    miss = [q for q in recs if max(q["ranks"]) >= 10]
    if miss:
        tot = sum(q["above"] for q in miss)
        sm = sum(q["above_same_table"] for q in miss)
        print(f"  setEM@10 miss n={len(miss)}: of the cells outranking gold, "
              f"{sm / tot:.3f} sit in the gold table itself "
              f"(distinct tables above gold: median "
              f"{sorted(q['above_tables'] for q in miss)[len(miss) // 2]})")
    if all("ranks_in_table" in q for q in recs):
        print("  ORACLE table gating (rank inside the gold table only):")
        line = "   " + "".join(f"@{k}={sum(1 for q in recs if max(q['ranks_in_table']) < k) / n:.4f} "
                               for k in (1, 5, 10, 20, 50))
        print(line + f"  (gold table holds median "
              f"{sorted(q['gold_table_cells'] for q in recs)[n // 2]} cells)")
    mrr = sum(1.0 / (min(q["ranks"]) + 1) for q in recs) / n
    tr = [q["table_rank_cellvote"] for q in recs]
    print(f"  MRR(first gold cell)={mrr:.4f}   "
          f"table recall@1={sum(1 for r in tr if r == 1) / n:.4f} "
          f"@10={sum(1 for r in tr if r <= 10) / n:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="hitab",
                    choices=["hitab", "multihiertt", "aitqa", "realhitbench"])
    ap.add_argument("--cell-scheme", default="S3c",
                    choices=["S2", "S3", "S3c", "mt2net"])
    ap.add_argument("--retriever", default="hybrid", choices=list(cdv.ALPHA))
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--population", default="hitab_dev_lookup_all")
    ap.add_argument("--rhb-question-types", nargs="*", default=[])
    ap.add_argument("--rhb-em-only", action="store_true")
    ap.add_argument("--mh-queries", type=int, default=400)
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--cache-dir", default=".cache/corpus_dump_vs_cell")
    ap.add_argument("--table-prior", type=float, nargs="*", default=[0.0],
                    help="add lambda * (its table's score) to every cell score "
                         "before ranking. 0 reproduces the plain cell ranking "
                         "bit for bit. Several values run in one pass -- the "
                         "scores are computed once per query and only the sort "
                         "changes. PREREG-2026-08-30-table-prior.md")
    ap.add_argument("--table-shortlist", type=int, nargs="*", default=[],
                    help="keep only cells of the top-T tables, tables ordered by "
                         "CELL VOTE (a table's rank is its best cell's). 0 means "
                         "no restriction and reproduces the plain ranking. T=1 is "
                         "the `cascade` arm. PREREG-2026-08-30-table-shortlist.md")
    ap.add_argument("--out-dir", default="results/rank")
    a = ap.parse_args()

    t0 = time.time()
    C = load_corpus(a)
    pop = C.queries[:a.max_queries] if a.max_queries else C.queries
    print(f"[corpus] {len(C.tids)} tables / {len(C.cell_owner)} cells | "
          f"[pop] {len(pop)} queries | {time.time() - t0:.0f}s", flush=True)

    chunks = [Chunk(table_id=t, chunk_id=f"c::{t}::{i}:{j}", text=x,
                    scheme=a.cell_scheme, kind="cell")
              for x, (t, i, j) in zip(cell_texts(C, a.cell_scheme), C.cell_owner)]
    enc = cdv._CachedEncoder(default_encoder(model_name=a.embed_model), a.cache_dir,
                             f"{a.dataset}_{a.split}_{a.embed_model}")
    t0 = time.time()
    ix = HybridIndex(chunks, encoder=enc, alpha=0.5)
    print(f"[index] built in {time.time() - t0:.0f}s", flush=True)

    # the table-level index the prior reads: title/caption plus header labels,
    # exactly what corpus_dump_vs_cell's `dump` arm ranks tables with
    t_ix = None
    if any(lam > 0 for lam in a.table_prior):
        t_ix = HybridIndex([Chunk(table_id=t, chunk_id=f"t::{t}",
                                  text=C.table_text[t], scheme="table",
                                  kind="table") for t in C.tids],
                           encoder=enc, alpha=0.5)
        of_tid = {t: k for k, t in enumerate(C.tids)}
        cell_tid = np.array([of_tid[t] for t, _i, _j in C.cell_owner],
                            dtype=np.int64)

    def scores(ix, question):
        bm = ix._bm25_scores(question)
        dn = ix._dense_scores(question) if a.alpha > 0 else np.zeros_like(bm)
        if a.alpha == 0.0:
            return bm
        if a.alpha == 1.0:
            return dn
        return a.alpha * _minmax(dn) + (1 - a.alpha) * _minmax(bm)

    pos_of = {c: n for n, c in enumerate(C.cell_owner)}
    by_table = defaultdict(list)
    for n, (t, _i, _j) in enumerate(C.cell_owner):
        by_table[t].append(n)
    # the population is part of the identity: HiTab ships several, and one
    # overwriting another is how two runs silently become one file
    tag = a.population if a.dataset == "hitab" else a.dataset
    out = Path(a.out_dir) / (f"{tag}_{a.cell_scheme}_"
                             f"{a.retriever}{a.alpha}_ranks.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.table_shortlist:
        a.table_prior = [float(t) for t in a.table_shortlist]
    all_recs = {lam: [] for lam in a.table_prior}
    t0 = time.time()
    for k, q in enumerate(pop, 1):
        cs = scores(ix, q["question"])
        ts = scores(t_ix, q["question"]) if t_ix is not None else None
        base = list(np.argsort(-cs))
        for lam in a.table_prior:
            recs = all_recs[lam]
            if a.table_shortlist:
                # lam is a table count here; the cell order inside the kept
                # tables is untouched, only other tables' cells are dropped
                T = int(lam)
                if T:
                    keep, seen_t = [], {}
                    for p_ in base:
                        t_ = C.cell_owner[p_][0]
                        if t_ not in seen_t:
                            if len(seen_t) >= T:
                                continue
                            seen_t[t_] = len(seen_t)
                        keep.append(p_)
                    order = keep
                else:
                    order = base
            else:
                order = (base if lam == 0 else
                         list(np.argsort(-(cs + lam * ts[cell_tid]))))
            rank = np.full(len(C.cell_owner), 10 ** 9, dtype=np.int64)
            rank[np.asarray(order, dtype=np.int64)] = np.arange(len(order))
            ranks = [int(rank[pos_of[g]]) for g in sorted(q["gold_cells"])
                     if g in pos_of]
            if not ranks or min(ranks) >= 10 ** 9:
                # every gold cell was shortlisted away: rank is "not retrieved",
                # which the 1e9 sentinel already encodes for every @k
                pass
            if not ranks:
                continue                 # no resolvable gold cell; nothing to rank
            voted, seen = [], set()
            for p in order[:5000]:
                t = C.cell_owner[p][0]
                if t not in seen:
                    seen.add(t)
                    voted.append(t)
            # What outranks the gold cell -- siblings inside the gold table, or
            # cells of other tables? The two call for opposite fixes and OSC
            # cannot tell them apart. Capped: a gold cell at rank 40k would cost
            # a 40k scan for a number that is already "hopeless".
            # CEILING for anything that pushes the gold table's cells up: where
            # the gold cell sits among its OWN table's cells. Any table prior,
            # cascade or reranker is bounded by this, because none of them can
            # reorder cells inside a table.
            # each gold cell against ITS OWN table -- a MultiHiertt question
            # draws operands from several tables of one document, so gating on
            # `gold_table` alone would drop the cells that live elsewhere
            own_at, in_tab, n_own = {}, [], 0
            for g in sorted(q["gold_cells"]):
                if g not in pos_of:
                    continue
                if g[0] not in own_at:
                    own_at[g[0]] = {p_: r_ for r_, p_ in enumerate(
                        sorted(by_table[g[0]], key=lambda x: rank[x]))}
                    n_own += len(own_at[g[0]])
                in_tab.append(own_at[g[0]][pos_of[g]])
            worst = max(ranks)
            above = order[:min(worst, 2000)]
            same = sum(1 for p_ in above if C.cell_owner[p_][0] == q["gold_table"])
            r = {"query_id": q["query_id"], "m": len(ranks),
                 "above": len(above), "above_same_table": same,
                 "above_tables": len({C.cell_owner[p_][0] for p_ in above}),
                 "above_capped": int(worst > 2000),
                 "gold_table": q["gold_table"],
                 "table_rank_cellvote": (voted.index(q["gold_table"]) + 1
                                         if q["gold_table"] in voted else 10 ** 6),
                 "ranks": ranks, "ranks_in_table": in_tab,
                 "gold_table_cells": n_own,
                 "qtype": q.get("qtype"), "subqtype": q.get("subqtype")}
            recs.append(r)
        if k % 100 == 0:
            print(f"  {k}/{len(pop)}  {time.time() - t0:.0f}s", flush=True)

    for lam in a.table_prior:
        recs = all_recs[lam]
        knob = f"T{int(lam)}" if a.table_shortlist else f"tp{lam}"
        f = out.with_name(out.name.replace("_ranks.jsonl", f"_{knob}_ranks.jsonl"))
        with open(f, "w") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        report(recs, f"{a.dataset} / {a.cell_scheme} / {a.retriever} "
                     f"a={a.alpha} / {'shortlist T=' if a.table_shortlist else 'table_prior='}"
                     f"{int(lam) if a.table_shortlist else lam}")
        for m_lab, sel in (("m=1", [r for r in recs if r["m"] == 1]),
                           ("m>=2", [r for r in recs if r["m"] >= 2])):
            report(sel, f"  split {m_lab}")
        print(f"wrote -> {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
