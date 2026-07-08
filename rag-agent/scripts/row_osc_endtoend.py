#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Row-axis cross-encoder, end-to-end on OSC (the payoff test).

row_select_bench shows the cross-encoder beats the embedding matcher on row-recall.
This asks the question that matters: does that row-recall lift convert into raw
**OSC** (= operand-set completeness = decomposition success)? The row axis is the
measured decomposition bottleneck, so improving row node-resolution should raise
the share of queries whose full operand set is recovered.

Paired, same queries, identical col axis (lexical) — only the row matcher differs:
  * baseline  = EmbedResolver(row_mode="embed", col_mode="lexical")   [production hybrid]
  * treatment = EmbedResolver(row_mode="cross", col_mode="lexical")   [cross-encoder rows]
Both pick the same number of row scope-nodes (top_n_rows == top_n_cross) so the
only difference is *which* nodes, not how many. Enumeration expands each node to its
child rows. Reports mean ΔOSC (paired bootstrap 95% CI) and McNemar on the
OSC==1.0 (fully-complete) indicator — "decomposition succeeded" per query.

Population: HiTab dev arithmetic m>=2 (the bottleneck population). LLM-free.
Run: PYTHONPATH=. python scripts/row_osc_endtoend.py --split dev \
        --cross-encoder cross-encoder/ms-marco-MiniLM-L-6-v2
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.bench.hitab import load_queries
from rag_agent.data.loader import load_table
from rag_agent.eval.operand_set import operand_set_completeness, per_cell_recall
from rag_agent.query.header_embed_resolver import EmbedResolver
from rag_agent.query.operand_decomposer import Embedder
from rag_agent.retrieve.header_enum import enumerate_scope
from rag_agent.stores.original_store import build_original_table

SEED = 42
ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}


def _bootstrap_diff_ci(pairs, n_boot=2000, seed=SEED):
    """Paired bootstrap CI of mean(treatment - baseline)."""
    if not pairs:
        return [float("nan")] * 2
    rng = random.Random(seed)
    n = len(pairs)
    diffs = []
    for _ in range(n_boot):
        s = sum((lambda p: p[0] - p[1])(pairs[rng.randrange(n)]) for _ in range(n))
        diffs.append(s / n)
    diffs.sort()
    return [round(diffs[int(0.025 * n_boot)], 4), round(diffs[int(0.975 * n_boot)], 4)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--cross-encoder", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--top-n-rows", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/row_osc_endtoend.json")
    args = ap.parse_args()

    queries, _ = load_queries(args.data_dir, args.split)
    pop = [q for q in queries
           if (q.aggregation or "none") in ARITH
           and len({(o.row, o.col) for o in q.gold_operands}) >= 2]
    n = len(pop)
    print(f"[pop] arithmetic m>=2: {n}  |  ce={args.cross_encoder}  top_n_rows={args.top_n_rows}")

    embedder = Embedder(args.embed_model, device=args.device)
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(args.cross_encoder)

    base = EmbedResolver(embedder, row_mode="embed", col_mode="lexical",
                         top_n_rows=args.top_n_rows)
    treat = EmbedResolver(embedder, row_mode="cross", col_mode="lexical",
                          top_n_rows=args.top_n_rows, cross_encoder=ce,
                          top_n_cross=args.top_n_rows)
    ots = {t: build_original_table(load_table(t, args.data_dir))
           for t in {q.gold_table_id for q in pop}}

    recs = []
    for i, q in enumerate(pop):
        ot = ots[q.gold_table_id]
        gold = q.gold_operands
        gold_rows = {o.row for o in gold}
        gold_cols = {o.col for o in gold}

        def osc_of(resolver):
            intent = resolver.resolve(q.question, ot)
            enum = enumerate_scope(ot, intent.row_paths, intent.col_paths)
            return (operand_set_completeness(gold, enum.cells),
                    per_cell_recall(gold, enum.cells),
                    int(gold_rows <= enum.rows), int(gold_cols <= enum.cols),
                    len(enum.cells))

        ob, pb, rrb, rcb, cb = osc_of(base)
        oc, pc, rrc, rcc, cc = osc_of(treat)
        recs.append({"query_id": q.query_id,
                     "osc_base": ob, "osc_cross": oc,
                     "pcr_base": pb, "pcr_cross": pc,
                     "rowcov_base": rrb, "rowcov_cross": rrc,
                     "colcov_base": rcb, "colcov_cross": rcc,
                     "cells_base": cb, "cells_cross": cc})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n}")

    def mean(key):
        return sum(r[key] for r in recs) / n

    # McNemar on the OSC==1.0 (fully complete) indicator: decomposition succeeded
    sb = [r["osc_base"] >= 0.999 for r in recs]
    sc = [r["osc_cross"] >= 0.999 for r in recs]
    b = sum(1 for x, y in zip(sb, sc) if x and not y)   # base succeeds, cross fails
    c = sum(1 for x, y in zip(sb, sc) if y and not x)   # cross succeeds, base fails (win)
    from scipy.stats import binomtest
    p_full = binomtest(c, b + c, 0.5, alternative="two-sided").pvalue if (b + c) else float("nan")

    pairs_osc = [(r["osc_cross"], r["osc_base"]) for r in recs]

    out = {
        "experiment": "row_osc_endtoend",
        "population": {"name": "arithmetic_m>=2", "n": n},
        "cross_encoder": args.cross_encoder, "top_n_rows": args.top_n_rows,
        "col_mode": "lexical (identical both arms)",
        "osc_base": round(mean("osc_base"), 4),
        "osc_cross": round(mean("osc_cross"), 4),
        "delta_osc": round(mean("osc_cross") - mean("osc_base"), 4),
        "delta_osc_ci95": _bootstrap_diff_ci(pairs_osc),
        "pcr_base": round(mean("pcr_base"), 4),
        "pcr_cross": round(mean("pcr_cross"), 4),
        "row_cov_base": round(mean("rowcov_base"), 4),
        "row_cov_cross": round(mean("rowcov_cross"), 4),
        "col_cov_base": round(mean("colcov_base"), 4),
        "col_cov_cross": round(mean("colcov_cross"), 4),
        "mean_cells_base": round(mean("cells_base"), 1),
        "mean_cells_cross": round(mean("cells_cross"), 1),
        "full_osc_base": round(sum(sb) / n, 4),
        "full_osc_cross": round(sum(sc) / n, 4),
        "mcnemar_full_osc": {"base_only": b, "cross_only": c, "n_discordant": b + c,
                             "exact_binom_p": round(float(p_full), 5)},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"\n{'arm':<10}{'OSC':>8}{'fullOSC':>10}{'rowCov':>9}{'colCov':>9}{'cells':>8}")
    print(f"{'embed':<10}{out['osc_base']:>8.3f}{out['full_osc_base']:>10.3f}"
          f"{out['row_cov_base']:>9.3f}{out['col_cov_base']:>9.3f}{out['mean_cells_base']:>8.1f}")
    print(f"{'cross':<10}{out['osc_cross']:>8.3f}{out['full_osc_cross']:>10.3f}"
          f"{out['row_cov_cross']:>9.3f}{out['col_cov_cross']:>9.3f}{out['mean_cells_cross']:>8.1f}")
    print(f"\nΔOSC = {out['delta_osc']:+.4f}  (95% CI {out['delta_osc_ci95']})")
    print(f"McNemar (fullOSC): base_only={b}, cross_only={c}, p={p_full:.5f}  "
          f"{'(significant)' if p_full < 0.05 else '(n.s.)'}")
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
