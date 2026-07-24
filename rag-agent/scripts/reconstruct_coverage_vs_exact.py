#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Score HiTab header-path reconstruction under BOTH metrics on the SAME paths:
exact-match (vs the gold hmt tree) and MultiHiertt's lenient segment-coverage.

Why this exists — settling "measure the row axis on other datasets with the
same yardstick":

  * The strict yardstick (exact-match vs a gold header tree) can only be applied
    to HiTab: MultiHiertt/FinQA ship no gold tree, AIT-QA ships the tree already
    (nothing to reconstruct), WikiSQL is flat. So HiTab's row .56 is HiTab-only
    by necessity, not by choice.
  * The lenient coverage proxy MultiHiertt uses (fraction of a reconstructed
    path's segments whose tokens appear in a reference) never penalises a path
    for MISSING ancestors — which is exactly HiTab's dominant row failure. So it
    saturates: scored under coverage, HiTab's row axis jumps .56 -> .998, ABOVE
    MultiHiertt's own .933. The proxy is blind to the failure mode.

Conclusion: MultiHiertt's row .93 is NOT evidence that its row reconstruction is
good; cross-dataset row-axis accuracy is simply not comparable. What transfers is
the mechanism (row collapses iff depth > stub-column count), not the number.

Run: PYTHONPATH=. python scripts/reconstruct_coverage_vs_exact.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tree_reconstruct_hitab_raw import align, norm, tree_lines  # noqa: E402

from rag_agent.bench.hitab import load_queries  # noqa: E402
from rag_agent.reconstruct import (guess_n_header_rows,  # noqa: E402
                                   reconstruct_col_paths, reconstruct_row_paths)

_TOK = re.compile(r"[a-z0-9]+")


def toks(s: str) -> set:
    return set(_TOK.findall(s.lower()))


def seg_all_found(path, gold_path) -> int:
    """MultiHiertt-style: 1 iff every segment of the reconstructed `path` has all
    its content tokens present in the reference (here the gold path)."""
    ref = set()
    for p in gold_path:
        ref |= toks(p)
    if not path:
        return 1  # empty path trivially covered, as in the proxy
    for seg in path:
        st = toks(seg)
        if st and not st.issubset(ref):
            return 0
    return 1


def main(data_dir: str = None, split: str = "dev") -> int:
    data_dir = data_dir or str(ROOT / "data/hitab")
    raw_dir = Path(data_dir) / "data/tables/raw"
    _, tables = load_queries(data_dir, split)

    col_exact = col_tot = row_exact = row_tot = col_cov = row_cov = 0
    for tid, bt in tables.items():
        p = raw_dir / f"{tid}.json"
        if not p.exists():
            continue
        try:
            raw = json.load(open(p))
        except Exception:
            continue
        texts = raw.get("texts") or []
        if not texts:
            continue
        cols_c, top_rows = tree_lines(raw.get("top_root") or {}, "top")
        rows_c, left_cols = tree_lines(raw.get("left_root") or {}, "left")
        if not cols_c or not rows_c or not top_rows or not left_cols:
            continue
        nhr_gold, nhc_gold = min(rows_c), min(cols_c)
        if nhr_gold <= 0 or nhc_gold <= 0:
            continue
        al = align(texts, rows_c, cols_c, nhr_gold, nhc_gold, bt)
        if al is None:
            continue
        rows_c, cols_c, _ = al
        nhr = guess_n_header_rows(texts, n_header_cols=nhc_gold)
        rec_cols = reconstruct_col_paths(texts, nhr, nhc_gold)
        rec_rows = reconstruct_row_paths(texts, nhr, nhc_gold)

        for j, c in enumerate(cols_c):
            i = c - nhc_gold
            rp = rec_cols[i] if 0 <= i < len(rec_cols) else []
            gp = bt.col_path(j)
            col_tot += 1
            col_exact += int(norm(gp) == norm(rp))
            col_cov += seg_all_found(rp, gp)
        for i_, r in enumerate(rows_c):
            i = r - nhr
            rp = rec_rows[i] if 0 <= i < len(rec_rows) else []
            gp = bt.row_path(i_)
            row_tot += 1
            row_exact += int(norm(gp) == norm(rp))
            row_cov += seg_all_found(rp, gp)

    print(f"HiTab real grid ({split}), guessed boundary — "
          f"n_col_paths={col_tot} n_row_paths={row_tot}")
    print(f"  COL  exact-match={col_exact / col_tot:.4f}   "
          f"seg-coverage(all-found)={col_cov / col_tot:.4f}")
    print(f"  ROW  exact-match={row_exact / row_tot:.4f}   "
          f"seg-coverage(all-found)={row_cov / row_tot:.4f}")
    print("\nMultiHiertt (its own proxy, results/tree_reconstruct_multihiertt.json): "
          "col all-found=0.9355  row all-found=0.9334")
    print("=> Under the SAME lenient proxy, HiTab row saturates to .998 (> MultiHiertt "
          ".933). The proxy cannot distinguish a good row tree from a collapsed one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
