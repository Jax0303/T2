#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit the open questions about HiTab raw-grid reconstruction, in one pass.

Every number the reconstruction story rests on has a caveat that was never
checked. This checks them, so each one either stops being a caveat or becomes a
stated limitation:

A. **Depth inflation.** The section-row scope rule stacks ancestors. Does it
   stack MORE than gold does? (RealHiTBench looked alarming at "row_depth 4.2"
   until that field turned out to be the per-table MAX, not the mean — it is
   named mean_MAX_row_depth now. HiTab has gold, so measure it properly here.)
   ANSWER: no. Reconstruction runs SHALLOWER than gold (1.96 vs 2.15 mean row
   depth; 1,201 lines shallower against 81 deeper). The failure mode is dropped
   ancestors, never invented ones.

B. **Survivorship.** 116 of 540 dev tables fail value-equality alignment and are
   dropped from every raw-grid number. If the dropped ones are structurally
   harder, .8202 is measured on the easy half.

C. **Value disagreement.** 935 cells differ between the raw grid and the hmt
   parse even after numeric normalisation. Dataset noise, or a bad alignment?

D. **Per-line vs per-cell.** row_exact is per LINE, sentence path accuracy is per
   CELL, and the two disagree in opposite directions at the two header-boundary
   settings. Cell-weighting is the suspect; this measures it.

E. **Why reconstructed paths out-retrieve gold ones.** S2_recon beats S2_gold at
   set-EM. If reconstructed paths simply carry more question vocabulary, that is
   the whole story and it is a lexical-overlap effect, not a correctness one.
   ANSWER: they carry LESS (.2366 vs .2419 question-token recall, 7.36 vs 7.87
   path tokens) — they are shorter, and being shorter is the advantage. Confirmed
   by the S2_gold_trunc arm in point3_reconstruction_cost.json, which cuts GOLD
   to the reconstructed depth and matches or beats S2_recon at every k.

LLM-free, CPU-only, no embeddings.
Run: PYTHONPATH=. python scripts/recon_audit_hitab.py --split dev
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_agent.bench.hitab import load_queries
from rag_agent.reconstruct import reconstruct_col_paths, reconstruct_row_paths
from tree_reconstruct_hitab_raw import _cell, _norm_val, align, norm, tree_lines

_TOK = re.compile(r"[a-z0-9]+")


def toks(s) -> set:
    return set(_TOK.findall(str(s).lower()))


def path_toks(path) -> set:
    out = set()
    for p in path:
        out |= toks(p)
    return out


def mean(xs):
    return round(st.mean(xs), 4) if xs else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default="results/recon_audit_hitab.json")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir) if args.raw_dir else Path(args.data_dir) / "data/tables/raw"
    queries, tables = load_queries(args.data_dir, args.split)

    # A / D
    depth = {"row_rec": [], "row_gold": [], "col_rec": [], "col_gold": []}
    depth_cmp = {"row": Counter(), "col": Counter()}
    # D: cells behind right vs wrong lines
    width_ok, width_bad = [], []
    line_hit = line_tot = 0
    cell_hit = cell_tot = 0
    # B
    aligned_stats, dropped_stats = [], []
    # C
    val_bad = 0
    val_tot = 0
    val_examples = []
    val_kinds = Counter()
    # E
    ov_rec, ov_gold, len_rec, len_gold = [], [], [], []

    per_table = {}

    for tid, bt in tables.items():
        p = raw_dir / f"{tid}.json"
        if not p.exists():
            continue
        try:
            raw = json.load(open(p))
        except Exception:
            continue
        texts = raw.get("texts") or []
        cols_c, _ = tree_lines(raw.get("top_root") or {}, "top")
        rows_c, _ = tree_lines(raw.get("left_root") or {}, "left")
        if not texts or not cols_c or not rows_c:
            continue
        nhr, nhc = min(rows_c), min(cols_c)
        if nhr <= 0 or nhc <= 0:
            continue

        prof = {
            "grid_rows": len(texts),
            "grid_cols": max((len(r) for r in texts), default=0),
            "n_merges": len(raw.get("merged_regions") or []),
            "n_header_rows": nhr,
            "n_header_cols": nhc,
            "gold_row_depth": mean([len(bt.row_path(i)) for i in range(bt.n_rows)]) or 0,
            "gold_col_depth": mean([len(bt.col_path(j)) for j in range(bt.n_cols)]) or 0,
        }

        al = align(texts, rows_c, cols_c, nhr, nhc, bt)
        if al is None:
            dropped_stats.append(prof)          # B: the survivorship question
            continue
        aligned_stats.append(prof)
        g_rows, g_cols, _rate = al

        rec_cols = reconstruct_col_paths(texts, nhr, nhc)
        rec_rows = reconstruct_row_paths(texts, nhr, nhc)

        def rp(line):
            i = line - nhr
            return list(rec_rows[i]) if 0 <= i < len(rec_rows) else []

        def cp(line):
            j = line - nhc
            return list(rec_cols[j]) if 0 <= j < len(rec_cols) else []

        row_ok, col_ok = {}, {}
        for i, r in enumerate(g_rows):
            rec, gold = rp(r), list(bt.row_path(i))
            row_ok[i] = norm(rec) == norm(gold)
            depth["row_rec"].append(len(rec)); depth["row_gold"].append(len(gold))
            depth_cmp["row"]["deeper" if len(rec) > len(gold) else
                              "shallower" if len(rec) < len(gold) else "same"] += 1
        for j, c in enumerate(g_cols):
            rec, gold = cp(c), list(bt.col_path(j))
            col_ok[j] = norm(rec) == norm(gold)
            depth["col_rec"].append(len(rec)); depth["col_gold"].append(len(gold))
            depth_cmp["col"]["deeper" if len(rec) > len(gold) else
                              "shallower" if len(rec) < len(gold) else "same"] += 1

        # D: how many indexed cells sit behind a right vs a wrong row line
        for i, r in enumerate(g_rows):
            w = sum(1 for j in range(bt.n_cols)
                    if ("" if bt.cell(i, j) is None else str(bt.cell(i, j)).strip()) != "")
            (width_ok if row_ok[i] else width_bad).append(w)
            line_tot += 1; line_hit += int(row_ok[i])

        for i in range(bt.n_rows):
            for j in range(bt.n_cols):
                gv = bt.cell(i, j)
                if (""  if gv is None else str(gv).strip()) == "":
                    continue
                cell_tot += 1
                cell_hit += int(row_ok[i] and col_ok[j])
                # C: value agreement between raw grid and hmt parse
                rv = _cell(texts, g_rows[i], g_cols[j])
                val_tot += 1
                if _norm_val(rv) != _norm_val(gv):
                    val_bad += 1
                    kind = ("raw_blank" if not str(rv).strip() else
                            "gold_blank" if not str(gv).strip() else
                            "both_numeric" if (toks(rv) and toks(gv) and
                                               str(rv).strip()[:1].isdigit() and
                                               str(gv).strip()[:1].isdigit())
                            else "text_differs")
                    val_kinds[kind] += 1
                    if len(val_examples) < 20:
                        val_examples.append({"table_id": tid, "cell": [i, j],
                                             "raw_grid": str(rv)[:60],
                                             "hmt_parse": str(gv)[:60], "kind": kind})

        per_table[tid] = {"g_rows": g_rows, "g_cols": g_cols,
                          "rec_rows": rec_rows, "rec_cols": rec_cols,
                          "nhr": nhr, "nhc": nhc}

    # E: does the reconstructed path carry more of the question's words than gold?
    for q in queries:
        pt = per_table.get(q.gold_table_id)
        if pt is None:
            continue
        bt = tables[q.gold_table_id]
        qt = toks(q.question)
        if not qt:
            continue
        for op in q.gold_operands:
            i, j = op.row, op.col
            if not (0 <= i < bt.n_rows and 0 <= j < bt.n_cols):
                continue
            ri, cj = pt["g_rows"][i] - pt["nhr"], pt["g_cols"][j] - pt["nhc"]
            rec = ((list(pt["rec_rows"][ri]) if 0 <= ri < len(pt["rec_rows"]) else [])
                   + (list(pt["rec_cols"][cj]) if 0 <= cj < len(pt["rec_cols"]) else []))
            gold = list(bt.row_path(i)) + list(bt.col_path(j))
            ov_rec.append(len(qt & path_toks(rec)) / len(qt))
            ov_gold.append(len(qt & path_toks(gold)) / len(qt))
            len_rec.append(len(path_toks(rec)))
            len_gold.append(len(path_toks(gold)))

    def prof_summary(rows):
        if not rows:
            return {}
        keys = rows[0].keys()
        return {"n": len(rows), **{k: mean([r[k] for r in rows]) for k in keys}}

    out = {
        "population": {"dataset": "hitab", "split": args.split,
                       "aligned": len(aligned_stats), "dropped_unaligned": len(dropped_stats)},
        "A_depth_inflation": {
            "row_mean_depth_recon": mean(depth["row_rec"]),
            "row_mean_depth_gold": mean(depth["row_gold"]),
            "col_mean_depth_recon": mean(depth["col_rec"]),
            "col_mean_depth_gold": mean(depth["col_gold"]),
            "row_vs_gold": dict(depth_cmp["row"]),
            "col_vs_gold": dict(depth_cmp["col"]),
        },
        "B_survivorship": {"aligned": prof_summary(aligned_stats),
                           "dropped": prof_summary(dropped_stats)},
        "C_value_disagreement": {
            "cells_compared": val_tot,
            "cells_disagreeing": val_bad,
            "rate": round(val_bad / max(val_tot, 1), 4),
            "kinds": dict(val_kinds),
            "examples": val_examples,
        },
        "D_line_vs_cell": {
            "row_exact_per_line": round(line_hit / max(line_tot, 1), 4),
            "path_exact_per_cell": round(cell_hit / max(cell_tot, 1), 4),
            "mean_indexed_cells_behind_a_CORRECT_row": mean(width_ok),
            "mean_indexed_cells_behind_a_WRONG_row": mean(width_bad),
        },
        "E_question_overlap_on_gold_operands": {
            "n_operand_cells": len(ov_rec),
            "mean_question_token_recall_recon": mean(ov_rec),
            "mean_question_token_recall_gold": mean(ov_gold),
            "mean_path_tokens_recon": mean(len_rec),
            "mean_path_tokens_gold": mean(len_gold),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)

    a, b = out["A_depth_inflation"], out["B_survivorship"]
    print("=== A. depth: reconstructed vs gold ===")
    print(f"  row  recon {a['row_mean_depth_recon']}  gold {a['row_mean_depth_gold']}  {a['row_vs_gold']}")
    print(f"  col  recon {a['col_mean_depth_recon']}  gold {a['col_mean_depth_gold']}  {a['col_vs_gold']}")
    print("\n=== B. are the dropped tables harder? ===")
    for k in ("aligned", "dropped"):
        print(f"  {k:8} {b[k]}")
    c = out["C_value_disagreement"]
    print(f"\n=== C. value disagreement: {c['cells_disagreeing']}/{c['cells_compared']} "
          f"({c['rate']:.2%}) {c['kinds']}")
    d = out["D_line_vs_cell"]
    print(f"\n=== D. per-line {d['row_exact_per_line']} vs per-cell {d['path_exact_per_cell']}"
          f"  | cells behind correct row {d['mean_indexed_cells_behind_a_CORRECT_row']}"
          f"  wrong row {d['mean_indexed_cells_behind_a_WRONG_row']}")
    e = out["E_question_overlap_on_gold_operands"]
    print(f"\n=== E. question-token recall  recon {e['mean_question_token_recall_recon']}"
          f"  gold {e['mean_question_token_recall_gold']}"
          f"  | path tokens recon {e['mean_path_tokens_recon']} gold {e['mean_path_tokens_gold']}")
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
