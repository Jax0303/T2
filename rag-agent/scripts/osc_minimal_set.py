#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""How few cells can carry OSC=1.0?

E5 established that completeness is reachable -- ``axis_complete | dense_k20``
hits OSC 1.000 -- but at ~123 cells against a gold set of ~3, and E7/E8 show
that handing the reader that many cells costs answer accuracy (whole-table
.068 vs oracle .230). So the open quantity is not *whether* completeness is
reachable but *how cheaply*.

This measures the completeness/cost frontier directly:

* **dense_k**            similarity alone, k swept -- the cheap baseline.
* **precise|dense_k**    matched-scope enumeration plus similarity.
* **rowc|dense_k**       complete the row axis only (R x all numeric cols).
* **colc|dense_k**       complete the column axis only (all numeric rows x C).
* **axisc|dense_k**      both axes -- E5's B, the current recipe at k=20.

Each dense arm is run at two granularities, because the unit of retrieval sets
a floor on the cell count independently of the ranking. E5 retrieved *row*
chunks, so k=20 admits twenty whole rows -- every column of each, ~105 cells --
even when the query needed three. The ``cellk`` arms index one chunk per
numeric cell instead, where k is literally k cells. Comparing the two answers
whether the 123-cell recipe is expensive because retrieval is imprecise or
merely because the chunk is coarse.

Two summary quantities beyond the per-config means:

* **min_cells@OSC=1** -- per query, the smallest config on the ladder that
  still contains every gold operand. Averaging it gives the budget an
  *adaptive* policy would need if it always picked the cheapest sufficient
  config: a lower bound on what a selector could buy, and the number to beat.
* **gold_cells** -- |gold| itself, the information-theoretic floor.

Dense is retrieved once at max k and truncated, so the k sweep is free and
every k sees the identical ranking.

Run:
    PYTHONPATH=. python scripts/osc_minimal_set.py --split dev
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.bench.hitab import load_queries
from rag_agent.data.loader import load_table
from rag_agent.eval.operand_set import bin_scope, operand_set_completeness
from rag_agent.query.header_embed_resolver import EmbedResolver
from rag_agent.query.operand_decomposer import Embedder
from rag_agent.retrieve.header_enum import enumerate_scope
from rag_agent.retrieve.operand_retriever import HybridRetriever, retrieve
from rag_agent.serialize import S2, serialize_table

SEED = 42
ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}
K_GRID = [1, 2, 3, 5, 8, 12, 20, 30]


def _numeric_cells(ot):
    return {(r, c) for r in range(ot.n_rows) for c in range(ot.n_cols)
            if ot.cell_num(r, c) is not None}


def _cell_chunks(bt, ot):
    """One S2 chunk per numeric cell: ``row path > col path: value``.

    Same text convention as the row serializer, minus the row-level packing, so
    the only variable against ``serialize_table`` is the retrieval unit.
    """
    from rag_agent.bench.schema import Chunk
    out = []
    title = f"{bt.title} | " if getattr(bt, "title", "") else ""
    for r in range(ot.n_rows):
        rp = [p for p in bt.row_path(r) if p]
        for c in range(ot.n_cols):
            if ot.cell_num(r, c) is None:
                continue
            path = rp + [p for p in bt.col_path(c) if p]
            v = bt.cell(r, c)
            text = title + ((" > ".join(path) + f": {v}") if path else str(v))
            out.append(Chunk(table_id=bt.table_id, chunk_id=f"{bt.table_id}#c{r}_{c}",
                             text=text, rows=[r], cols=[c]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/osc_minimal_set.json")
    ap.add_argument("--dump-records", default=None,
                    help="jsonl of per-query features + per-config (osc, cells), "
                         "for fitting a budget selector")
    args = ap.parse_args()

    queries, tables = load_queries(args.data_dir, args.split, args.max)
    pop = [q for q in queries
           if (q.aggregation or "none") in ARITH and len(q.gold_operands) >= 2]
    print(f"[pop] arithmetic m>=2: {len(pop)}")

    embedder = Embedder(args.embed_model, device=args.device)
    resolver = EmbedResolver(embedder, row_mode="embed", col_mode="lexical")
    from rag_agent.stores.original_store import build_original_table
    ots, retr = {}, {}
    cell_retr = {}
    for tid in {q.gold_table_id for q in pop}:
        ots[tid] = build_original_table(load_table(tid, args.data_dir))
        retr[tid] = HybridRetriever(serialize_table(tables[tid], S2), embedder)
        cell_retr[tid] = HybridRetriever(_cell_chunks(tables[tid], ots[tid]), embedder)

    k_max = max(K_GRID)
    recs = []
    for i, q in enumerate(pop):
        ot = ots[q.gold_table_id]
        gold = {(o.row, o.col) for o in q.gold_operands}
        all_rows, all_cols = set(range(ot.n_rows)), set(range(ot.n_cols))

        intent = resolver.resolve(q.question, ot)
        e = enumerate_scope(ot, intent.row_paths, intent.col_paths)
        R, C = e.rows, e.cols
        precise = set(e.cells)
        rowc = {(r, c) for r in R for c in all_cols if ot.cell_num(r, c) is not None}
        colc = {(r, c) for r in all_rows for c in C if ot.cell_num(r, c) is not None}

        # one retrieval at k_max; smaller k = a prefix of the same ranking
        res = retrieve(q.question, tables[q.gold_table_id], None, mode="plain",
                       k=k_max, scheme=S2, embedder=embedder,
                       retriever=retr[q.gold_table_id])
        dense_by_k = {}
        acc = set()
        for rank, ch in enumerate(res.retrieved, 1):
            for r in ch.rows:
                for c in ch.cols:
                    if ot.cell_num(r, c) is not None:
                        acc.add((r, c))
            if rank in K_GRID:
                dense_by_k[rank] = set(acc)
        for k in K_GRID:                       # fewer chunks than k were returned
            dense_by_k.setdefault(k, set(acc))

        # same sweep at cell granularity: k chunks == k cells
        cres = retrieve(q.question, tables[q.gold_table_id], None, mode="plain",
                        k=k_max, scheme=S2, embedder=embedder,
                        retriever=cell_retr[q.gold_table_id])
        cell_by_k, cacc = {}, set()
        for rank, ch in enumerate(cres.retrieved, 1):
            for r in ch.rows:
                for c in ch.cols:
                    if ot.cell_num(r, c) is not None:
                        cacc.add((r, c))
            if rank in K_GRID:
                cell_by_k[rank] = set(cacc)
        for k in K_GRID:
            cell_by_k.setdefault(k, set(cacc))

        sets = {}
        for k in K_GRID:
            d, cd = dense_by_k[k], cell_by_k[k]
            sets[f"dense_k{k}"] = d
            sets[f"precise|dense_k{k}"] = precise | d
            sets[f"rowc|dense_k{k}"] = rowc | d
            sets[f"colc|dense_k{k}"] = colc | d
            sets[f"axisc|dense_k{k}"] = rowc | colc | d
            sets[f"cellk{k}"] = cd
            sets[f"precise|cellk{k}"] = precise | cd

        # --- nested escalation ladder ------------------------------------
        # The named configs above are not nested (cellk5 is not a superset of
        # dense_k1), so "spend one rung more" does not reliably buy coverage --
        # which is what a policy needs. This ladder is cumulative by
        # construction, so both cells and OSC are monotone in the rung index
        # and a safety margin genuinely converts into completeness.
        esc, acc_set = [], set()
        for k in K_GRID:                       # cheapest first: single cells
            acc_set |= cell_by_k[k]
            esc.append((f"esc_cell{k}", set(acc_set)))
        for k in K_GRID:                       # then whole rows
            acc_set |= dense_by_k[k]
            esc.append((f"esc_+dense{k}", set(acc_set)))
        for nm, s in (("+precise", precise), ("+rowc", rowc), ("+colc", colc)):
            acc_set |= s
            esc.append((f"esc_{nm}", set(acc_set)))

        rec = {"m": len(gold), "gold_cells": len(gold)}
        for nm, s in esc:
            rec[f"{nm}__osc"] = operand_set_completeness(gold, s)
            rec[f"{nm}__cells"] = len(s)
        esc_hit = [i for i, (_, s) in enumerate(esc) if gold <= s]
        rec["esc_min_rung"] = esc_hit[0] if esc_hit else None
        rec["esc_min_cells"] = len(esc[esc_hit[0]][1]) if esc_hit else None

        if args.dump_records:
            # Features a live system can compute: resolver output, table shape,
            # question surface. `m` and the gold cells are NOT features -- they
            # are what we are trying to avoid needing. q.aggregation is a HiTab
            # annotation, so it is excluded too.
            ql = q.question.lower()
            rec["feat"] = {
                "n_row_paths": len(intent.row_paths),
                "n_col_paths": len(intent.col_paths),
                "n_scope_rows": len(R),
                "n_scope_cols": len(C),
                "n_precise": len(precise),
                "n_rowc": len(rowc),
                "n_colc": len(colc),
                "tbl_rows": ot.n_rows,
                "tbl_cols": ot.n_cols,
                "tbl_numeric": len(_numeric_cells(ot)),
                "q_tokens": len(ql.split()),
                "q_has_total": int(any(w in ql for w in ("total", "overall", "all "))),
                "q_has_avg": int(any(w in ql for w in ("average", "mean", "per "))),
                "q_has_diff": int(any(w in ql for w in ("difference", "change", "increase",
                                                        "decrease", "more than", "less than"))),
                "q_has_ratio": int(any(w in ql for w in ("percent", "proportion", "share",
                                                         "ratio", "%"))),
                "q_n_numerals": sum(ch.isdigit() for ch in ql),
                "dense_top1_cells": len(dense_by_k[1]),
                "cell_top1_hit": len(cell_by_k[1]),
            }
        for name, s in sets.items():
            rec[f"{name}__osc"] = operand_set_completeness(gold, s)
            rec[f"{name}__cells"] = len(s)
        # cheapest config on the ladder that is still complete for THIS query
        complete = [(len(s), n) for n, s in sets.items() if gold <= s]
        rec["min_cells_at_osc1"] = min(complete)[0] if complete else None
        rec["min_config_at_osc1"] = min(complete)[1] if complete else None
        recs.append(rec)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(pop)}", flush=True)

    names = [n[:-5] for n in recs[0] if n.endswith("__osc")]
    esc_names = [n for n in names if n.startswith("esc_")]
    n = len(recs)

    def summ(sub):
        if not sub:
            return {}
        out = {}
        for c in names:
            out[c] = {
                "osc": round(sum(r[f"{c}__osc"] for r in sub) / len(sub), 4),
                "pct_complete": round(sum(1 for r in sub if r[f"{c}__osc"] == 1.0) / len(sub), 4),
                "mean_cells": round(sum(r[f"{c}__cells"] for r in sub) / len(sub), 1),
                "median_cells": round(statistics.median(r[f"{c}__cells"] for r in sub), 1),
            }
        return out

    solved = [r for r in recs if r["min_cells_at_osc1"] is not None]
    frontier = {
        "n_queries": n,
        "n_reaching_osc1_somewhere": len(solved),
        "gold_cells_mean": round(sum(r["gold_cells"] for r in recs) / n, 2),
        "adaptive_min_cells_mean": round(
            sum(r["min_cells_at_osc1"] for r in solved) / len(solved), 1) if solved else None,
        "adaptive_min_cells_median": round(
            statistics.median(r["min_cells_at_osc1"] for r in solved), 1) if solved else None,
        "cheapest_config_histogram": {
            k: sum(1 for r in solved if r["min_config_at_osc1"] == k)
            for k in sorted({r["min_config_at_osc1"] for r in solved})},
    }
    esc_solved = [r for r in recs if r["esc_min_rung"] is not None]
    frontier["escalation"] = {
        "ladder": esc_names,
        "n_reaching_osc1": len(esc_solved),
        "min_cells_mean": round(sum(r["esc_min_cells"] for r in esc_solved) / len(esc_solved), 1)
        if esc_solved else None,
        "min_cells_median": round(statistics.median(r["esc_min_cells"] for r in esc_solved), 1)
        if esc_solved else None,
        "rung_histogram": {esc_names[i]: sum(1 for r in esc_solved if r["esc_min_rung"] == i)
                           for i in sorted({r["esc_min_rung"] for r in esc_solved})},
    }

    out = {
        "experiment": "osc_minimal_set", "split": args.split, "seed": SEED,
        "embed_model": args.embed_model, "encoder": getattr(embedder, "name", args.embed_model),
        "k_grid": K_GRID,
        "note": ("OSC=1 means every gold operand is inside the returned set. "
                 "min_cells_at_osc1 is the cheapest ladder config that is complete "
                 "for that query -- an adaptive-policy lower bound, not a system."),
        "frontier": frontier,
        "overall": summ(recs),
        "by_scope": {b: summ([r for r in recs if bin_scope(r["m"]) == b])
                     for b in ("2", "3-4", "5-8", "9+")},
    }
    out["by_scope"] = {b: v for b, v in out["by_scope"].items() if v}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    if args.dump_records:
        os.makedirs(os.path.dirname(args.dump_records) or ".", exist_ok=True)
        with open(args.dump_records, "w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        print(f"records -> {args.dump_records}")

    print(f"\n{'config':<24} {'OSC':>7} {'%complete':>10} {'mean_cells':>11}")
    for c in names:
        v = out["overall"][c]
        print(f"  {c:<22} {v['osc']:>7.3f} {v['pct_complete']:>10.3f} {v['mean_cells']:>11.1f}")
    f = frontier
    print(f"\ngold cells (floor)          : {f['gold_cells_mean']}")
    print(f"adaptive min cells @ OSC=1  : mean {f['adaptive_min_cells_mean']} / "
          f"median {f['adaptive_min_cells_median']}  "
          f"({f['n_reaching_osc1_somewhere']}/{f['n_queries']} queries reachable)")
    print(f"cheapest config histogram   : {f['cheapest_config_histogram']}")
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
