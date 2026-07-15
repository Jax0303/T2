#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""THE thesis problem, measured on the reference dataset: do operand cells that
exist identically in OTHER tables (e.g. "Total") collide under similarity
retrieval over a multi-table corpus — and which of the three proposed methods
(tree-mapping / caption sentences / 1-table-1-chunk cascade) mitigates it?

Setting (mirrors the problem figure):
  * Corpus = every parseable table from the sampled MultiHiertt docs (real
    filings, so near-duplicate "twin" tables and repeated row labels like
    "total" occur naturally).
  * Queries = arithmetic questions (non-empty ``program``) whose gold operand
    cells (``table_evidence``) all sit in ONE table and need >= 2 cells, with
    no text evidence — the "Table A needs Seoul + Total, but every table has
    a Total" scenario.
  * Retrieval unit = individual DATA cells across the whole corpus
    (flat / S2 / S3 serializations x BM25 / dense / hybrid), plus the
    1-table-1-chunk CASCADE (pick table first, then cells inside it), which
    by construction excludes other tables' Total cells.

Metrics — the paper's axis is all-or-nothing operand completeness:
  * all_covered@k (k=10/20/50): every gold operand cell ranked <= k.
  * per-operand median rank / reached@50, stratified by CROSS-TABLE LABEL
    COLLISION: an operand's row-leaf label occurring in >= --collision-min
    other tables (the literal "타 표에도 동일하게 존재" condition), and by a
    "total"-keyword flag for interpretability.

Scores use the same math as rag_agent.retrieve.HybridIndex (per-query min-max
normalization, score = alpha*dense + (1-alpha)*bm25; alpha 0/1/0.5) but encode
each serialization's corpus ONCE and reuse the arrays across the three alphas.

Run: PYTHONPATH=. python scripts/operand_collision_multihiertt.py --max-queries 60
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from rag_agent.reconstruct import guess_n_header_rows, parse_html_table, \
    reconstruct_col_paths, reconstruct_row_paths
from rag_agent.retrieve.encoders import _tokenize, default_encoder

RETRIEVERS = [("bm25", 0.0), ("dense", 1.0), ("hybrid", 0.5)]
KS = (10, 20, 50)

_TOTAL_RE = re.compile(r"\btotal\b", re.IGNORECASE)


def _norm_label(s: str) -> str:
    return " ".join((s or "").lower().split())


def _minmax(a: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return a
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def load_population(max_queries: int, population: str = "arith_multi"):
    """Query populations over MultiHiertt train, table-only evidence.

    arith_multi   — arithmetic, single-table, multi-operand (>= 2 cells): the
                    paper's collision population.
    lookup_single — no program (pure lookup), exactly ONE gold cell: the
                    scope-1 control slice (does the serialization gain persist
                    where completeness is trivial?).
    """
    from datasets import load_dataset
    rows = load_dataset("bevaya/MultiHiertt", split="train").to_list()

    queries, docs = [], {}
    for row in rows:
        has_program = bool((row.get("program") or "").strip())
        if population == "arith_multi" and not has_program:
            continue                          # arithmetic only
        if population == "lookup_single" and has_program:
            continue                          # pure lookup only
        if row.get("text_evidence"):
            continue                          # table-only evidence
        ev = row.get("table_evidence") or []
        want = 2 if population == "arith_multi" else 1
        if (len(ev) < 2 and population == "arith_multi") or \
           (len(ev) != want and population == "lookup_single"):
            continue
        coords = []
        ok = True
        for e in ev:
            parts = e.split("-")
            if len(parts) != 3:
                ok = False
                break
            coords.append(tuple(int(x) for x in parts))
        if not ok:
            continue
        if len({c[0] for c in coords}) != 1:
            continue                          # all operands in ONE table
        queries.append({"uid": row["uid"], "question": row["question"],
                        "t_idx": coords[0][0], "cells": [(r, c) for _, r, c in coords]})
        docs[row["uid"]] = row
        if len(queries) >= max_queries:
            break
    return queries, docs


def build_corpus(queries, docs):
    """Reconstruct EVERY parseable table of the sampled docs; return per-cell
    records + gold (query -> global cell indices), dropping queries whose
    operands don't land on data cells."""
    tables = {}
    for uid, row in docs.items():
        for t_idx, html in enumerate(row["tables"]):
            grid = parse_html_table(html)
            if len(grid) < 3 or len(grid[0]) < 2:
                continue
            nhr = max(1, min(guess_n_header_rows(grid, n_header_cols=1), len(grid) - 1))
            tables[(uid, t_idx)] = {
                "grid": grid, "nhr": nhr,
                "rows": reconstruct_row_paths(grid, nhr, 1),
                "cols": reconstruct_col_paths(grid, nhr, 1),
            }

    cells = []               # dicts: table, r, c (grid coords), row_path, col_path, value
    cell_index = {}          # (uid, t_idx, r, c) -> global index
    for key, t in tables.items():
        grid, nhr = t["grid"], t["nhr"]
        for r in range(nhr, len(grid)):
            rp = t["rows"][r - nhr] if (r - nhr) < len(t["rows"]) else []
            for c in range(1, len(grid[0])):
                v = (grid[r][c] or "").strip()
                if not v:
                    continue
                cp = t["cols"][c - 1] if (c - 1) < len(t["cols"]) else []
                cell_index[key + (r, c)] = len(cells)
                cells.append({"table": key, "row_path": rp, "col_path": cp, "value": v})

    kept = []
    for q in queries:
        gold = [cell_index.get((q["uid"], q["t_idx"], r, c)) for r, c in q["cells"]]
        if any(g is None for g in gold):
            continue                          # operand in header region / unparsed
        kept.append({**q, "gold": gold, "table": (q["uid"], q["t_idx"])})

    # cross-table label collision: for each row-leaf label, in how many tables?
    label_tables = {}
    for cell in cells:
        leaf = _norm_label(cell["row_path"][-1] if cell["row_path"] else "")
        if leaf:
            label_tables.setdefault(leaf, set()).add(cell["table"])
    for cell in cells:
        leaf = _norm_label(cell["row_path"][-1] if cell["row_path"] else "")
        cell["n_tables_with_label"] = len(label_tables.get(leaf, set()))
        cell["is_total_like"] = bool(_TOTAL_RE.search(leaf))

    return tables, cells, kept


def cell_text(cell, scheme: str) -> str:
    rp, cp, v = cell["row_path"], cell["col_path"], cell["value"]
    if scheme == "flat":                      # leaf labels only, no hierarchy
        leaf_r = rp[-1] if rp else ""
        leaf_c = cp[-1] if cp else ""
        lab = " ".join(x for x in (leaf_r, leaf_c) if x)
        return f"{lab}: {v}" if lab else v
    if scheme == "S2":                        # full tree-mapped header path
        path = " > ".join([*rp, *cp])
        return f"{path}: {v}" if path else v
    if scheme == "S3":                        # caption sentence (medium preset)
        row = " > ".join(rp)
        col = " > ".join(cp)
        if row and col:
            return f"For {row}, {col} is {v}."
        return f"{col or row} is {v}." if (col or row) else f"The value is {v}."
    raise ValueError(scheme)


def summarize(all_ranks, gold_sets, n_q, cells):
    """all_ranks: global-index -> rank arrays are too big; instead we get, per
    query, {global_cell_idx: rank}. Compute completeness + stratified stats."""
    out = {}
    for k in KS:
        out[f"all_covered@{k}"] = round(
            sum(1 for ranks in all_ranks if all(r is not None and r <= k for r in ranks.values()))
            / n_q, 4)
    dup_ranks, uniq_ranks, total_ranks, ord_ranks = [], [], [], []
    for ranks in all_ranks:
        for gi, r in ranks.items():
            tgt = dup_ranks if cells[gi]["n_tables_with_label"] >= ARGS.collision_min else uniq_ranks
            tgt.append(r)
            (total_ranks if cells[gi]["is_total_like"] else ord_ranks).append(r)
    med = lambda xs: (round(statistics.median([x for x in xs if x is not None]), 1)
                      if any(x is not None for x in xs) else None)
    reach = lambda xs: (round(sum(1 for x in xs if x is not None and x <= 50) / len(xs), 4)
                        if xs else None)
    out["operand_median_rank"] = {
        "colliding_label": med(dup_ranks), "unique_label": med(uniq_ranks),
        "total_like": med(total_ranks), "ordinary": med(ord_ranks),
    }
    out["operand_reached@50"] = {
        "colliding_label": reach(dup_ranks), "unique_label": reach(uniq_ranks),
        "total_like": reach(total_ranks), "ordinary": reach(ord_ranks),
    }
    out["n_operands"] = {"colliding_label": len(dup_ranks), "unique_label": len(uniq_ranks),
                          "total_like": len(total_ranks), "ordinary": len(ord_ranks)}
    return out


RECORDS = []


def main() -> int:
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-queries", type=int, default=60)
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--population", default="arith_multi",
                    choices=["arith_multi", "lookup_single"])
    ap.add_argument("--query-prefix", default="",
                    help='e.g. "query: " for e5-family embedders')
    ap.add_argument("--passage-prefix", default="",
                    help='e.g. "passage: " for e5-family embedders')
    ap.add_argument("--collision-min", type=int, default=5,
                     help="operand counts as 'colliding' if its row-leaf label "
                          "occurs in >= this many distinct tables")
    ap.add_argument("--cascade-tables", type=int, nargs="+", default=[1, 3],
                     help="cascade variants: pool the cells of the top-M tables")
    ap.add_argument("--out", default="results/operand_collision_multihiertt.json")
    ARGS = ap.parse_args()

    from rank_bm25 import BM25Okapi

    queries, docs = load_population(ARGS.max_queries, ARGS.population)
    tables, cells, pop = build_corpus(queries, docs)
    n_q = len(pop)
    n_cells = len(cells)
    print(f"[pop] {ARGS.population} queries: {n_q} "
          f"(of {len(queries)} scanned) | corpus: {len(tables)} tables, {n_cells} cells")
    if not n_q:
        print("no evaluable queries — abort")
        return 1

    encoder = default_encoder(model_name=ARGS.embed_model)
    q_texts = [q["question"] for q in pop]
    q_vecs = np.asarray(encoder.encode([ARGS.query_prefix + t for t in q_texts]))

    results = {}
    for scheme in ("flat", "S2", "S3"):
        print(f"\n=== scheme={scheme} ===", flush=True)
        texts = [cell_text(c, scheme) for c in cells]
        vecs = np.asarray(encoder.encode([ARGS.passage_prefix + t for t in texts]))
        bm25 = BM25Okapi([_tokenize(t) for t in texts])

        # ---- global cell retrieval: bm25 / dense / hybrid --------------------
        per_retriever = {name: [] for name, _ in RETRIEVERS}
        for qi, q in enumerate(pop):
            dn = _minmax(vecs @ q_vecs[qi])
            bm = _minmax(np.asarray(bm25.get_scores(_tokenize(q_texts[qi])), dtype=np.float32))
            for name, alpha in RETRIEVERS:
                combined = alpha * dn + (1.0 - alpha) * bm
                order = np.argsort(-combined)
                rank_of = {int(g): None for g in q["gold"]}
                gold_left = set(rank_of)
                for pos, idx in enumerate(order, 1):
                    ii = int(idx)
                    if ii in gold_left:
                        rank_of[ii] = pos
                        gold_left.discard(ii)
                        if not gold_left:
                            break
                per_retriever[name].append(rank_of)
        for name, _ in RETRIEVERS:
            results.setdefault(scheme, {})[name] = summarize(per_retriever[name], None, n_q, cells)
            for qi, ranks in enumerate(per_retriever[name]):
                for gi, r in ranks.items():
                    RECORDS.append({"scheme": scheme, "retriever": name, "query": qi,
                                     "cell": gi, "rank": r,
                                     "colliding": cells[gi]["n_tables_with_label"] >= ARGS.collision_min,
                                     "total_like": cells[gi]["is_total_like"]})

        # ---- cascade: 1 table = 1 chunk, then cells inside -------------------
        table_keys = list(tables.keys())
        cell_idx_by_table = {}
        for gi, c in enumerate(cells):
            cell_idx_by_table.setdefault(c["table"], []).append(gi)
        t_texts = ["\n".join(texts[gi] for gi in cell_idx_by_table.get(k, [])) for k in table_keys]
        t_vecs = np.asarray(encoder.encode([ARGS.passage_prefix + t for t in t_texts]))

        for m in ARGS.cascade_tables:
            cas_ranks = []
            table_hit = 0
            for qi, q in enumerate(pop):
                order_t = np.argsort(-(t_vecs @ q_vecs[qi]))[:m]
                chosen = [table_keys[int(i)] for i in order_t]
                if q["table"] in chosen:
                    table_hit += 1
                cand = [gi for key in chosen for gi in cell_idx_by_table.get(key, [])]
                if cand:
                    sub = np.asarray([vecs[gi] for gi in cand])
                    order_c = np.argsort(-(sub @ q_vecs[qi]))
                    pos_of = {cand[int(idx)]: pos for pos, idx in enumerate(order_c, 1)}
                else:
                    pos_of = {}
                cas_ranks.append({int(g): pos_of.get(int(g)) for g in q["gold"]})
            key = f"cascade_top{m}_tables"
            results[scheme][key] = summarize(cas_ranks, None, n_q, cells)
            results[scheme][key]["gold_table_in_top_m"] = round(table_hit / n_q, 4)

        for name in [n for n, _ in RETRIEVERS] + [f"cascade_top{m}_tables" for m in ARGS.cascade_tables]:
            r = results[scheme][name]
            print(f"  {name:<22} all@10={r['all_covered@10']:.3f} all@20={r['all_covered@20']:.3f} "
                  f"all@50={r['all_covered@50']:.3f}  med_rank collide/unique="
                  f"{r['operand_median_rank']['colliding_label']}/{r['operand_median_rank']['unique_label']}")

    n_collide = sum(1 for q in pop for g in q["gold"]
                    if cells[g]["n_tables_with_label"] >= ARGS.collision_min)
    n_gold = sum(len(q["gold"]) for q in pop)
    out = {
        "population": {"name": "multihiertt_arithmetic_single_table_multi_operand",
                        "n_queries": n_q, "n_gold_operand_cells": n_gold,
                        "pct_operands_with_colliding_label": round(n_collide / n_gold, 4),
                        "collision_min_tables": ARGS.collision_min},
        "corpus": {"n_tables": len(tables), "n_cells": n_cells},
        "embed_model": ARGS.embed_model,
        "hierarchy_source": "self-reconstructed (rag_agent.reconstruct)",
        "score_math": "per-query min-max, alpha*dense+(1-alpha)*bm25 (HybridIndex convention)",
        "by_scheme": results,
    }
    Path(ARGS.out).parent.mkdir(parents=True, exist_ok=True)
    with open(ARGS.out, "w") as fh:
        json.dump(out, fh, indent=2)
    rec_path = str(Path(ARGS.out).with_suffix("")) + "_records.jsonl"
    with open(rec_path, "w") as fh:
        for rec in RECORDS:
            fh.write(json.dumps(rec) + "\n")
    print(f"\nwrote -> {ARGS.out}  (+ per-operand ranks -> {rec_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
