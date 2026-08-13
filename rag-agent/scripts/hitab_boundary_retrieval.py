#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Step 1 — how much of the cell-retrieval loss is the reconstruction's fault?

MultiHiertt cannot answer this: it ships gold *evidence cells* but no gold
header tree, so a wrong header path is invisible there. HiTab ships both, so the
same query set can be retrieved three times over the same cells, varying only
the header paths the index unit is built from:

  gold        — ``bt.row_path``/``bt.col_path``, the published tree. The ceiling
                a perfect reconstructor would reach.
  goldbound   — paths rebuilt by ``reconstruct_{row,col}_paths`` but *given* the
                header/data boundary from the gold trees. Isolates the path
                builder from the boundary guesser.
  guessed     — boundary guessed the way the deployed MultiHiertt corpus guesses
                it (``operand_collision_multihiertt.py``: rows first at
                ``n_header_cols=1``, then cols from that). What is actually shipped.

The pool is held fixed at the gold data region in all three arms, so the only
moving part is path quality; a wrong boundary also leaks header rows into the
pool in deployment, and that second effect is NOT measured here.

Run: PYTHONPATH=. .venv/bin/python scripts/hitab_boundary_retrieval.py --split dev
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from rag_agent.bench.hitab import load_queries
from rag_agent.reconstruct import (guess_n_header_cols, guess_n_header_rows,
                                   reconstruct_col_paths, reconstruct_row_paths)
from rag_agent.retrieve.encoders import default_encoder, default_prefixes
from rag_agent.runenv import run_env
from rag_agent.serialization.templates import STRUCTURAL, render
from tree_reconstruct_hitab_raw import align, tree_lines

ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}
ARMS = ("gold", "goldbound", "guessed")


def table_arms(raw: dict, bt):
    """Per-arm (row_paths, col_paths) over the SAME gold data region.

    Returns ``(rows_c, cols_c, {arm: (row_paths, col_paths)})`` in data
    coordinates, or ``None`` if the table does not align against its gold trees.
    """
    texts = raw.get("texts") or []
    if not texts:
        return None
    cols_c, top_rows = tree_lines(raw.get("top_root") or {}, "top")
    rows_c, left_cols = tree_lines(raw.get("left_root") or {}, "left")
    if not cols_c or not rows_c or not top_rows or not left_cols:
        return None
    nhr_gold, nhc_gold = min(rows_c), min(cols_c)
    if nhr_gold <= 0 or nhc_gold <= 0:
        return None
    al = align(texts, rows_c, cols_c, nhr_gold, nhc_gold, bt)
    if al is None:
        return None
    rows_c, cols_c, _ = al

    # deployed guess: rows first with the 1-column default, then cols from it
    nhr_g = max(1, min(guess_n_header_rows(texts, n_header_cols=1), len(texts) - 1))
    nhc_g = max(1, min(guess_n_header_cols(texts, n_header_rows=nhr_g),
                       len(texts[0]) - 1))

    def rebuilt(nhr, nhc):
        rr = reconstruct_row_paths(texts, nhr, nhc)
        cc = reconstruct_col_paths(texts, nhr, nhc)
        rows = [rr[r - nhr] if 0 <= r - nhr < len(rr) else [] for r in rows_c]
        cols = [cc[c - nhc] if 0 <= c - nhc < len(cc) else [] for c in cols_c]
        return rows, cols

    arms = {
        "gold": ([bt.row_path(i) for i in range(len(rows_c))],
                 [bt.col_path(j) for j in range(len(cols_c))]),
        "goldbound": rebuilt(nhr_gold, nhc_gold),
        "guessed": rebuilt(nhr_g, nhc_g),
    }
    # The guessed/goldbound contrast is only worth reading where the guess is
    # actually wrong. On HiTab dev the guesser is right on ~92% of tables, so
    # the caller MUST report this count next to the scores: a null result over
    # two disagreeing tables is not evidence that boundary errors are cheap.
    boundary = {"nhr_gold": nhr_gold, "nhr_guessed": nhr_g,
                "nhc_gold": nhc_gold, "nhc_guessed": nhc_g,
                "agrees": (nhr_g, nhc_g) == (nhr_gold, nhc_gold)}
    return rows_c, cols_c, arms, boundary


def summarize(per_query):
    ks = (1, 5, 10, 20)
    out = {"n_queries": len(per_query),
           "mrr": round(float(np.mean([1.0 / min(r) for r in per_query])), 4)}
    for k in ks:
        out[f"hit@{k}"] = round(float(np.mean([min(r) <= k for r in per_query])), 4)
        out[f"set_em@{k}"] = round(float(np.mean([max(r) <= k for r in per_query])), 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--raw-dir", default=None,
                    help="the source grids; defaults to <data-dir>/data/tables/raw")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--min-operands", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/hitab_boundary_retrieval.json")
    args = ap.parse_args()
    env = run_env(args.seed, args.embed_model)

    raw_dir = Path(args.raw_dir) if args.raw_dir else Path(args.data_dir) / "data/tables/raw"
    queries, tables = load_queries(args.data_dir, args.split)
    pop = [q for q in queries
           if (q.aggregation or "") in ARITH
           and len({(g.row, g.col) for g in q.gold_operands}) >= args.min_operands]
    print(f"[pop] HiTab {args.split} arith m>={args.min_operands}: {len(pop)}", flush=True)

    # build the per-table corpora once, then drop queries whose table cannot align
    built, dropped, boundary = {}, collections.Counter(), {}
    for q in pop:
        tid = q.gold_table_id
        if tid in built:
            continue
        p = raw_dir / f"{tid}.json"
        bt = tables.get(tid)
        if bt is None or not p.exists():
            dropped["no_table"] += 1
            built[tid] = None
            continue
        raw = json.load(open(p))
        got = table_arms(raw, bt)
        if got is None:
            dropped["unaligned"] += 1
            built[tid] = None
            continue
        rows_c, cols_c, arms, bnd = got
        boundary[tid] = bnd
        texts = raw["texts"]
        cells, index = [], {}
        for i, r in enumerate(rows_c):
            for j, c in enumerate(cols_c):
                v = (texts[r][c] or "").strip() if r < len(texts) and c < len(texts[r]) else ""
                if not v:
                    continue
                index[(i, j)] = len(cells)
                cells.append((i, j, v))
        built[tid] = {"cells": cells, "index": index, "arms": arms,
                      "title": bt.title} if cells else None
        if not cells:
            dropped["no_cells"] += 1

    kept = []
    for q in pop:
        b = built.get(q.gold_table_id)
        if b is None:
            continue
        gold = {b["index"].get((g.row, g.col)) for g in q.gold_operands}
        if None in gold or len(gold) < args.min_operands:
            dropped["operand_off_grid"] += 1
            continue
        kept.append((q, b, sorted(gold)))
    print(f"[pop] scored: {len(kept)}  dropped: {dict(dropped)}", flush=True)
    if not kept:
        return 1

    enc = default_encoder(model_name=args.embed_model)
    qpre, ppre = default_prefixes(enc.name)
    qv = np.asarray(enc.encode([qpre + q.question for q, _, _ in kept]))

    results, dup, best_rank = {}, {}, {}
    for arm in ARMS:
        # one encode pass over every table's cells for this arm
        texts_by_tid, offset = {}, {}
        flat = []
        for tid, b in built.items():
            if b is None:
                continue
            rows, cols = b["arms"][arm]
            offset[tid] = len(flat)
            for i, j, v in b["cells"]:
                flat.append(render(STRUCTURAL, b["title"], rows[i], cols[j], v))
            texts_by_tid[tid] = None
        vecs = np.asarray(enc.encode([ppre + t for t in flat]))

        # sentence ambiguity: cells whose value-stripped sentence has a twin
        amb_cells = amb_tot = 0
        for tid, b in built.items():
            if b is None:
                continue
            o = offset[tid]
            cnt = collections.Counter(
                flat[o + n].replace(v, "", 1) for n, (_, _, v) in enumerate(b["cells"]))
            amb_cells += sum(c for c in cnt.values() if c > 1)
            amb_tot += len(b["cells"])
        dup[arm] = round(amb_cells / amb_tot, 4) if amb_tot else 0.0

        per_query = []
        for qi, (q, b, gold) in enumerate(kept):
            o = offset[q.gold_table_id]
            n = len(b["cells"])
            s = vecs[o:o + n] @ qv[qi]
            order = np.argsort(-s)
            rank = {int(x): p for p, x in enumerate(order, 1)}
            per_query.append([rank[g] for g in gold])
        best_rank[arm] = [min(r) for r in per_query]
        results[arm] = summarize(per_query)
        print(f"[{arm:9}] mrr={results[arm]['mrr']:.4f} "
              f"hit@1={results[arm]['hit@1']:.4f} hit@5={results[arm]['hit@5']:.4f} "
              f"dup={dup[arm]:.3f}", flush=True)

    # Is "better structure" actually worth anything? Paired over the same queries.
    from scipy.stats import wilcoxon
    contrasts = {}
    for a, b in (("guessed", "goldbound"), ("guessed", "gold"), ("goldbound", "gold")):
        xa, xb = best_rank[a], best_rank[b]
        d = [u - v for u, v in zip(xa, xb) if u != v]
        contrasts[f"{a}->{b}"] = {
            "n_pairs": len(xa), "n_changed": len(d),
            "improved": sum(1 for x in d if x > 0),   # rank got smaller under b
            "worsened": sum(1 for x in d if x < 0),
            "p_two_sided": round(float(wilcoxon(xa, xb).pvalue), 4) if d else 1.0,
        }

    scored_tids = {q.gold_table_id for q, _, _ in kept}
    disagree = sorted(t for t in scored_tids if not boundary[t]["agrees"])
    print(f"\n[contrast] boundary guess disagrees with gold on "
          f"{len(disagree)}/{len(scored_tids)} scored tables "
          f"-- guessed vs goldbound is uninformative below a handful", flush=True)

    out = {
        "env": env,
        "boundary_guess_contrast_size": {
            "scored_tables": len(scored_tids),
            "tables_where_guess_differs": len(disagree),
            "tables": disagree,
            "caveat": "guessed-vs-goldbound is only interpretable in proportion "
                      "to this count; a null result over a couple of tables says "
                      "nothing about what a boundary error costs",
        },
        "boundary_contrasts_on_best_gold_rank": contrasts,
        "population": {"name": f"hitab_{args.split}_arith_m_ge_{args.min_operands}",
                       "n_queries": len(kept), "dropped": dict(dropped)},
        "pool": "gold data region of the query's own table, identical in all arms",
        "note": "arms vary ONLY the header paths the index unit is built from; a "
                "wrong boundary also leaks header rows into the deployed pool, "
                "which this design deliberately holds out",
        "encoder": enc.name, "retriever": "dense",
        "ambiguous_cell_rate": dup,
        "by_arm": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
