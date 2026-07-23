#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Tree-reconstruction experiment C — genuinely raw AND complex tables (RealHiTBench).

RealHiTBench (Zhang et al., ACL 2025; arXiv:2506.13405; HF `spzy/RealHiTBench`)
is the dataset that resolves the HiTab-vs-MultiHiertt dilemma
(`scripts/tree_reconstruct_multihiertt.py`): HiTab is complex but ships
pre-built gold header trees (the "raw" case is only simulated), while
MultiHiertt is genuinely raw HTML but mostly flat. RealHiTBench ships tables
as PhpSpreadsheet-exported HTML (Excel -> HTML, `rowspan`/`colspan` preserved)
that carry HiTab-grade nested column headers AND merged/indented row headers —
complex AND raw at once.

What this script CAN and CANNOT measure (be honest about it):
  * CAN (CPU-only): size the aggregation-question population, run the markup
    reconstruction front-end (`parse_html_table_with_merges` +
    `reconstruct_paths_with_merges`) over every table those questions touch,
    and characterise its behaviour sliced by `CompStrucCata` (the dataset's
    own structure-complexity label): parse-success rate, guessed header depth,
    reconstructed column/row tree depth, how many merged regions are consumed,
    degenerate-output rate.
  * CANNOT: reconstruction ACCURACY. RealHiTBench ships NO gold header tree and
    NO per-cell description (unlike MultiHiertt's `table_description`), so there
    is nothing to score reconstructed paths against. This is descriptive
    characterisation, not an accuracy gate. End-to-end ANSWER accuracy (the
    reason RealHiTBench is in the 2-dataset strategy) needs the solver and is a
    separate, GROQ-gated step.

Aggregation subset = the arithmetic-aggregation analogue of HiTab's arith(m>=2):
SubQType in {Calculation, Multi-hop Numerical Reasoning} — 334 queries over 258
tables, spread across all 7 CompStrucCata types.

Run: PYTHONPATH=. python scripts/realhitbench_ingest.py --max-tables 0
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.reconstruct import (guess_n_header_cols, guess_n_header_rows,
                                   parse_html_table_with_merges,
                                   reconstruct_paths_with_merges)

HF_REPO = "spzy/RealHiTBench"
AGG_SUBQTYPES = {"Calculation", "Multi-hop Numerical Reasoning"}


def _depth(paths) -> int:
    """Max segment count across a list of header paths (tree depth)."""
    return max((len(p) for p in paths), default=0)


def _mean(xs):
    return round(statistics.fmean(xs), 3) if xs else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-repo", default=HF_REPO)
    ap.add_argument("--subqtypes", nargs="*", default=sorted(AGG_SUBQTYPES),
                    help="SubQType values that define the aggregation subset")
    ap.add_argument("--max-tables", type=int, default=0, help="0 = all tables in the subset")
    ap.add_argument("--out", default="results/realhitbench_ingest.json")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    qa = json.load(open(hf_hub_download(args.hf_repo, "QA_final.json", repo_type="dataset")))["queries"]
    want = set(args.subqtypes)
    agg = [q for q in qa if q.get("SubQType") in want]

    # Map each table to its structure-complexity label (constant per FileName).
    table_cata = {}
    for q in agg:
        table_cata.setdefault(q["FileName"], q["CompStrucCata"])
    tables = sorted(table_cata)
    if args.max_tables:
        tables = tables[:args.max_tables]

    per_cata = defaultdict(lambda: {
        "n_tables": 0, "n_parsed": 0, "n_with_merges": 0, "n_degenerate": 0,
        "grid_rows": [], "grid_cols": [], "nhr": [], "nhc": [],
        "col_depth": [], "row_depth": [], "n_merges": [],
    })
    examples = []
    n_fail = 0

    for fname in tables:
        cata = table_cata[fname]
        s = per_cata[cata]
        s["n_tables"] += 1
        try:
            html = open(hf_hub_download(args.hf_repo, f"html/{fname}.html", repo_type="dataset")).read()
            grid, merges = parse_html_table_with_merges(html)
        except Exception as e:  # noqa: BLE001 — record and move on
            n_fail += 1
            if len(examples) < 8:
                examples.append({"FileName": fname, "CompStrucCata": cata, "error": repr(e)[:200]})
            continue
        if not grid or len(grid) < 3 or len(grid[0]) < 2:
            n_fail += 1
            continue
        s["n_parsed"] += 1
        nhc = guess_n_header_cols(grid)
        nhr = guess_n_header_rows(grid, n_header_cols=nhc)
        nhr = max(1, min(nhr, len(grid) - 1))
        cols, rows = reconstruct_paths_with_merges(grid, merges, nhr, n_header_cols=nhc)

        cd, rd = _depth(cols), _depth(rows)
        # Degenerate = reconstruction produced no usable hierarchy on either axis
        # (every column path length <=1 AND every row path length <=1), i.e. the
        # front-end recovered nothing structural beyond the flat grid.
        degenerate = cd <= 1 and rd <= 1
        s["n_degenerate"] += int(degenerate)
        s["n_with_merges"] += int(bool(merges))
        s["grid_rows"].append(len(grid))
        s["grid_cols"].append(len(grid[0]))
        s["nhr"].append(nhr)
        s["nhc"].append(nhc)
        s["col_depth"].append(cd)
        s["row_depth"].append(rd)
        s["n_merges"].append(len(merges))

        if len(examples) < 8:
            examples.append({
                "FileName": fname, "CompStrucCata": cata,
                "grid": [len(grid), len(grid[0])], "nhr": nhr, "nhc": nhc,
                "n_merges": len(merges), "col_depth": cd, "row_depth": rd,
                "col_paths_head": cols[:4], "row_paths_head": rows[:3],
            })

    def summarize(s):
        n = s["n_parsed"]
        return {
            "n_tables": s["n_tables"],
            "n_parsed": n,
            "parse_rate": round(n / s["n_tables"], 3) if s["n_tables"] else None,
            "n_with_merges": s["n_with_merges"],
            "merge_rate": round(s["n_with_merges"] / n, 3) if n else None,
            "n_degenerate": s["n_degenerate"],
            "degenerate_rate": round(s["n_degenerate"] / n, 3) if n else None,
            "mean_grid": [_mean(s["grid_rows"]), _mean(s["grid_cols"])],
            "mean_nhr": _mean(s["nhr"]), "mean_nhc": _mean(s["nhc"]),
            "mean_col_depth": _mean(s["col_depth"]),
            "mean_row_depth": _mean(s["row_depth"]),
            "col_depth_ge2_rate": round(sum(d >= 2 for d in s["col_depth"]) / n, 3) if n else None,
            "row_depth_ge2_rate": round(sum(d >= 2 for d in s["row_depth"]) / n, 3) if n else None,
            "mean_n_merges": _mean(s["n_merges"]),
        }

    by_cata = {k: summarize(v) for k, v in sorted(per_cata.items())}
    # Overall roll-up
    alls = defaultdict(list)
    tot = {"n_tables": 0, "n_parsed": 0, "n_with_merges": 0, "n_degenerate": 0}
    for v in per_cata.values():
        for k in tot:
            tot[k] += v[k]
        for k in ("col_depth", "row_depth", "nhr", "n_merges"):
            alls[k] += v[k]

    out = {
        "population": {
            "hf_repo": args.hf_repo,
            "subqtypes": sorted(want),
            "n_agg_queries": len(agg),
            "n_unique_tables": len(table_cata),
            "n_tables_processed": len(tables),
            "n_parse_fail": n_fail,
        },
        "note": ("descriptive characterisation only — RealHiTBench ships no gold "
                 "header tree and no per-cell description, so reconstruction "
                 "ACCURACY is not measurable here; answer accuracy is GROQ-gated"),
        "overall": {
            **tot,
            "parse_rate": round(tot["n_parsed"] / tot["n_tables"], 3) if tot["n_tables"] else None,
            "degenerate_rate": round(tot["n_degenerate"] / tot["n_parsed"], 3) if tot["n_parsed"] else None,
            "mean_col_depth": _mean(alls["col_depth"]),
            "mean_row_depth": _mean(alls["row_depth"]),
            "col_depth_ge2_rate": round(sum(d >= 2 for d in alls["col_depth"]) / tot["n_parsed"], 3) if tot["n_parsed"] else None,
            "row_depth_ge2_rate": round(sum(d >= 2 for d in alls["row_depth"]) / tot["n_parsed"], 3) if tot["n_parsed"] else None,
        },
        "by_compstruccata": by_cata,
        "examples": examples,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    o = out["overall"]
    print(f"agg queries={len(agg)}  unique tables={len(table_cata)}  processed={len(tables)}  parse_fail={n_fail}")
    print(f"OVERALL parse_rate={o['parse_rate']}  degenerate_rate={o['degenerate_rate']}  "
          f"col_depth(mean={o['mean_col_depth']}, >=2 {o['col_depth_ge2_rate']})  "
          f"row_depth(mean={o['mean_row_depth']}, >=2 {o['row_depth_ge2_rate']})")
    print("\nby CompStrucCata:")
    hdr = f"  {'cata':<22}{'n':>4}{'parse':>7}{'degen':>7}{'colD':>6}{'col>=2':>7}{'rowD':>6}{'row>=2':>7}{'merges':>7}"
    print(hdr)
    for k, v in by_cata.items():
        print(f"  {k:<22}{v['n_tables']:>4}{v['parse_rate']!s:>7}{v['degenerate_rate']!s:>7}"
              f"{v['mean_col_depth']!s:>6}{v['col_depth_ge2_rate']!s:>7}"
              f"{v['mean_row_depth']!s:>6}{v['row_depth_ge2_rate']!s:>7}{v['mean_n_merges']!s:>7}")
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
