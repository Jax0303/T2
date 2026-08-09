#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Where does the header/data boundary guess go wrong, and by how much?

Isolated on HiTab dev, the boundary guess costs the row axis 16.3pp and the
column axis 11.8pp (.7607/.9749 with the gold boundary, .5973/.8574 with the
guessed one). It is the largest remaining defect in the LLM-free half of the
pipeline, so it needs an error profile before a fix: which direction the guess
misses, by how much, and which of the two signals in `guess_n_header_rows` was
in charge when it did.

Run: PYTHONPATH=. .venv/bin/python scripts/diag_boundary.py --split dev
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_agent.reconstruct import guess_n_header_cols, guess_n_header_rows
from rag_agent.reconstruct.header_grid import (_left_region_blank,
                                               _left_region_units_note,
                                               section_label)
from tree_reconstruct_hitab_raw import align, tree_lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out_path = args.out or f"results/diag_boundary_{args.split}.json"

    from rag_agent.bench.hitab import load_queries

    raw_dir = Path(args.raw_dir) if args.raw_dir else Path(args.data_dir) / "data/tables/raw"
    _, tables = load_queries(args.data_dir, args.split)

    row_err, col_err, by_signal = Counter(), Counter(), Counter()
    examples = []
    n = 0
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
        nhr_g, nhc_g = min(rows_c), min(cols_c)
        if nhr_g <= 0 or nhc_g <= 0:
            continue
        if align(texts, rows_c, cols_c, nhr_g, nhc_g, bt) is None:
            continue
        n += 1

        nhr = guess_n_header_rows(texts, n_header_cols=nhc_g)
        nhc = guess_n_header_cols(texts, n_header_rows=nhr)
        dr, dc = nhr - nhr_g, nhc - nhc_g
        row_err[dr] += 1
        col_err[dc] += 1

        # which signal decided the row guess?
        corner = bool(nhc_g > 0 and texts[0][:nhc_g]
                      and _left_region_blank(texts, 0, nhc_g))
        sig = "corner" if corner else "numeric"
        by_signal[f"{sig}:{'ok' if dr == 0 else 'miss'}"] += 1

        if dc != 0 and len(examples) < 12:
            examples.append({
                "table_id": tid, "axis": "col",
                "gold_nhc": nhc_g, "guess_nhc": nhc, "nhr_used": nhr,
                "rows": [[str(x)[:14] for x in texts[r][:6]]
                         for r in range(min(len(texts), nhr + 3))],
            })
        elif dr != 0 and len(examples) < 12:
            examples.append({
                "table_id": tid, "gold_nhr": nhr_g, "guess_nhr": nhr,
                "signal": sig,
                "rows": [[str(x)[:14] for x in texts[r][:5]]
                         for r in range(min(len(texts), max(nhr, nhr_g) + 2))],
                "section_rows_in_head": [
                    r for r in range(min(len(texts), max(nhr, nhr_g) + 2))
                    if section_label(texts[r])],
            })

    out = {
        "population": {"split": args.split, "n_tables": n},
        "row_boundary_error_hist": {str(k): v for k, v in sorted(row_err.items())},
        "col_boundary_error_hist": {str(k): v for k, v in sorted(col_err.items())},
        "row_exact": round(row_err[0] / n, 4) if n else None,
        "col_exact": round(col_err[0] / n, 4) if n else None,
        "row_signal_breakdown": dict(by_signal),
        "examples": examples,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in
                      ("population", "row_boundary_error_hist",
                       "col_boundary_error_hist", "row_exact", "col_exact",
                       "row_signal_breakdown")}, indent=2))
    print(f"\nwrote -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
