#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""E9 — OSC under a fixed TOKEN budget: retrieval+injection vs OHD-style whole-table.

E8 established feasibility (whole-table serialization is ~9x larger and often exceeds
the context limit). This closes the loop with the *same-metric, same-setting* frontier
the reviewers will ask for: at each token budget B, what operand-set completeness does
each strategy actually deliver?

Arms (identical per-cell line format `row-path | col-path = value`, tokens=chars//3,
same population, LLM-free):
  - bm25/dense/hybrid plain      : top-k row chunks, largest k whose context fits B
  - bm25/dense/hybrid +inject    : same, unioned with cross-encoder-column-resolver
                                   total-row cells (the §5.10 winning config)
  - ohd_strict                   : whole table (row-major, single serialization,
                                   *charitable* — OHD itself doubles it); OSC only if
                                   the WHOLE table fits in B, else 0 (no selection
                                   mechanism exists to pick a subset)
  - ohd_trunc                    : generous variant — row-major prefix of the table
                                   up to B (a truncation heuristic OHD does not have)
  - ohd_dual_*                   : same two, with OHD's faithful dual (row-major +
                                   column-major) serialization cost

Deployment-honest: for retrieval arms the k chosen at budget B is the largest k that
fits — no peeking at gold to pick the best-scoring k.

Population: HiTab dev arithmetic m>=2 (n=161). LLM-free.
Run: PYTHONPATH=. python scripts/e9_osc_token_budget.py --split dev
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from rag_agent.bench.hitab import load_queries
from rag_agent.data.loader import load_table
from rag_agent.eval.operand_set import operand_set_completeness
from rag_agent.retrieve.header_enum import total_like_rows_hybrid
from rag_agent.retrieve.operand_retriever import HybridRetriever, _tok
from rag_agent.query.operand_decomposer import Embedder
from rag_agent.serialize import S2, serialize_table
from rag_agent.stores.original_store import build_original_table

ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}
KGRID = (1, 2, 3, 5, 7, 10, 15, 20, 30, 40)
BUDGETS = (250, 500, 1000, 2000, 4000, 8000, 16000, 32000)


def numeric_cells(ot, rows, cols):
    return {(r, c) for r in rows for c in cols if ot.cell_num(r, c) is not None}


def cells_of_chunks(ot, retriever, idxs):
    out = set()
    for i in idxs:
        ch = retriever.chunks[i]
        out |= numeric_cells(ot, ch.rows, ch.cols)
    return out


def cell_line(ot, r, c) -> str:
    row = " > ".join(s for s in ot.row_path(r) if s) or f"row{r}"
    col = " > ".join(s for s in ot.col_path(c) if s) or f"col{c}"
    return f"{row} | {col} = {ot.cell(r, c)}"


def tokens_of(ot, cells) -> int:
    """Rough token count of a cell set rendered one line per cell (chars//3, as E7)."""
    return sum(len(cell_line(ot, r, c)) + 1 for (r, c) in cells) // 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--cross-encoder", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--out", default="results/e9_osc_token_budget.json")
    args = ap.parse_args()

    queries, tables = load_queries(args.data_dir, args.split)
    pop = [q for q in queries
           if (q.aggregation or "none") in ARITH
           and len({(o.row, o.col) for o in q.gold_operands}) >= 2]
    n = len(pop)
    print(f"[pop] arithmetic m>=2: {n}")

    emb = Embedder(args.embed_model, device="cpu")
    from sentence_transformers import CrossEncoder
    from rag_agent.query.header_embed_resolver import EmbedResolver
    col_resolver = EmbedResolver(emb, col_mode="cross",
                                 cross_encoder=CrossEncoder(args.cross_encoder),
                                 top_n_cross=2)

    need = {q.gold_table_id for q in pop}
    retr, ots, total_rows = {}, {}, {}
    for tid in need:
        retr[tid] = HybridRetriever(serialize_table(tables[tid], S2), emb)
        ot = build_original_table(load_table(tid, args.data_dir))
        ots[tid] = ot
        total_rows[tid] = total_like_rows_hybrid(ot)

    methods = ("bm25", "dense", "hybrid")
    # arm -> budget -> per-query OSC list (paired across arms)
    arms = ([f"{m}_plain" for m in methods] + [f"{m}_inject" for m in methods]
            + ["ohd_strict", "ohd_trunc", "ohd_dual_strict", "ohd_dual_trunc"])
    osc_at = {a: {B: [] for B in BUDGETS} for a in arms}
    whole_tok_single, whole_tok_dual = [], []

    for q in pop:
        ot = ots[q.gold_table_id]
        R = retr[q.gold_table_id]
        gold = q.gold_operands
        trows = total_rows[q.gold_table_id]

        # winning §5.10 injection config: total rows x resolver-picked columns
        intent = col_resolver.resolve(q.question, ot)
        cidx = set()
        for p in intent.col_paths:
            cidx.update(ot.find_cols_by_header(" > ".join(p)))
        resolver_tcells = {(r, c) for r in trows for c in cidx
                           if ot.cell_num(r, c) is not None}

        qv = np.asarray(emb.encode([q.question])[0])
        bm25_rank = R._rank(R._bm25.get_scores(_tok(q.question)))
        dense_rank = R._rank(np.asarray(R._emb) @ qv) if R._emb is not None else bm25_rank
        fused = {}
        for rank, i in enumerate(bm25_rank):
            fused[i] = fused.get(i, 0.0) + 1.0 / (R.rrf_k + rank)
        for rank, i in enumerate(dense_rank):
            fused[i] = fused.get(i, 0.0) + 1.0 / (R.rrf_k + rank)
        hybrid_rank = sorted(fused, key=lambda i: -fused[i])
        ranks = {"bm25": bm25_rank, "dense": dense_rank, "hybrid": hybrid_rank}

        # retrieval arms: (tokens, osc) point per k; at budget B use the LARGEST k
        # that fits (deployment rule — no gold peeking)
        for m in methods:
            pts_p, pts_i = [], []
            for k in KGRID:
                base = cells_of_chunks(ot, R, ranks[m][:k])
                aug = base | resolver_tcells
                pts_p.append((tokens_of(ot, base), operand_set_completeness(gold, base)))
                pts_i.append((tokens_of(ot, aug), operand_set_completeness(gold, aug)))
            for B in BUDGETS:
                fit_p = [o for (t, o) in pts_p if t <= B]
                fit_i = [o for (t, o) in pts_i if t <= B]
                osc_at[f"{m}_plain"][B].append(fit_p[-1] if fit_p else 0.0)
                osc_at[f"{m}_inject"][B].append(fit_i[-1] if fit_i else 0.0)

        # OHD arms: whole numeric table, row-major cell order (their serialization)
        cells_rm = sorted(numeric_cells(ot, range(ot.n_rows), range(ot.n_cols)))
        line_costs = [len(cell_line(ot, r, c)) + 1 for (r, c) in cells_rm]
        tot_chars = sum(line_costs)
        t_single, t_dual = tot_chars // 3, (2 * tot_chars) // 3
        whole_tok_single.append(t_single)
        whole_tok_dual.append(t_dual)
        osc_whole = operand_set_completeness(gold, set(cells_rm))
        for B in BUDGETS:
            osc_at["ohd_strict"][B].append(osc_whole if t_single <= B else 0.0)
            osc_at["ohd_dual_strict"][B].append(osc_whole if t_dual <= B else 0.0)
            for arm, mult in (("ohd_trunc", 1), ("ohd_dual_trunc", 2)):
                budget_chars, cum, pref = B * 3 // mult, 0, []
                for (rc, cost) in zip(cells_rm, line_costs):
                    cum += cost
                    if cum > budget_chars:
                        break
                    pref.append(rc)
                osc_at[arm][B].append(operand_set_completeness(gold, set(pref)))

    # ---- aggregate + paired tests ----------------------------------------
    from scipy.stats import binomtest
    out = {"population": {"name": "arithmetic_m>=2", "n": n},
           "config": {"kgrid": list(KGRID), "budgets": list(BUDGETS),
                      "injection": "total_rows x cross-encoder resolver cols (§5.10)",
                      "token_estimate": "chars//3 (as E7)"},
           "whole_table_tokens": {
               "single": {"mean": round(float(np.mean(whole_tok_single)), 0),
                          "median": float(np.median(whole_tok_single)),
                          "p90": float(np.percentile(whole_tok_single, 90))},
               "dual": {"mean": round(float(np.mean(whole_tok_dual)), 0),
                        "median": float(np.median(whole_tok_dual)),
                        "p90": float(np.percentile(whole_tok_dual, 90))}},
           "osc_at_budget": {a: {str(B): round(sum(v) / n, 4)
                                 for B, v in osc_at[a].items()} for a in arms},
           "paired_inject_vs_ohd_trunc": {}}

    # paired significance: hybrid_inject vs the generous ohd_trunc, per budget
    for B in BUDGETS:
        a, b = osc_at["hybrid_inject"][B], osc_at["ohd_trunc"][B]
        a_only = sum(1 for x, y in zip(a, b) if x > y)
        b_only = sum(1 for x, y in zip(a, b) if y > x)
        pv = binomtest(a_only, a_only + b_only, 0.5).pvalue if (a_only + b_only) else 1.0
        out["paired_inject_vs_ohd_trunc"][str(B)] = {
            "osc_inject": round(sum(a) / n, 4), "osc_ohd_trunc": round(sum(b) / n, 4),
            "delta": round((sum(a) - sum(b)) / n, 4),
            "inject_only": a_only, "ohd_only": b_only, "mcnemar_p": round(float(pv), 6)}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    wt = out["whole_table_tokens"]
    print(f"\nwhole-table tokens: single mean={wt['single']['mean']:.0f} "
          f"median={wt['single']['median']:.0f} p90={wt['single']['p90']:.0f}  "
          f"(dual = 2x)")
    hdr = f"{'arm':<18}" + "".join(f"{('@'+str(B)):>8}" for B in BUDGETS)
    print("\nOSC @ token budget B\n" + hdr)
    for a in arms:
        row = "".join(f"{out['osc_at_budget'][a][str(B)]:>8.3f}" for B in BUDGETS)
        print(f"{a:<18}{row}")
    print("\npaired hybrid_inject vs ohd_trunc (generous OHD):")
    print(f"{'B':>7}{'inj':>8}{'ohd':>8}{'Δ':>8}{'inj>':>6}{'ohd>':>6}{'p':>10}")
    for B in BUDGETS:
        t = out["paired_inject_vs_ohd_trunc"][str(B)]
        print(f"{B:>7}{t['osc_inject']:>8.3f}{t['osc_ohd_trunc']:>8.3f}"
              f"{t['delta']:>+8.3f}{t['inject_only']:>6}{t['ohd_only']:>6}"
              f"{t['mcnemar_p']:>10.5f}")
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
