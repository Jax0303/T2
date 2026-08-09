#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Where does the row-axis hierarchy live in the tables the reconstructor fails on?

`tree_reconstruct_hitab_raw.py` splits the row axis cleanly: gold paths whose
depth fits inside the stub block score .997, the rest score .116. The standing
explanation is that the deeper level exists only as indentation, which HiTab's
`texts` has already stripped -- i.e. the information is gone and the lever is
closed.

This tests that. For every gold row path deeper than the stub block, it asks
where the ANCESTOR label is in the raw grid, and classifies:

  section_row  -- the ancestor sits in the stub of an earlier grid row whose
                  data region is entirely blank (a group-header row). Recoverable
                  without any language cue: it is a structural property.
  stub_column  -- the ancestor sits in another stub column on the same row.
  absent       -- the ancestor string appears nowhere in the grid. Genuinely
                  unrecoverable; the annotation carries knowledge the table does
                  not show.

Run: PYTHONPATH=. .venv/bin/python scripts/diag_row_axis_signal.py --split dev
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tree_reconstruct_hitab_raw import _cell, _norm_val, align, tree_lines


def section_label(texts, r: int) -> str:
    """A section row carries exactly one text and no data — a group heading.

    The label is not necessarily in the stub: HiTab writes unit/section rows as
    ``('', 'percent', '', '')`` just as often as ``('other', '', '', '')``, so
    keying on the stub column alone misses over half of them.
    """
    if not (0 <= r < len(texts)):
        return ""
    filled = [str(x).strip() for x in texts[r] if str(x).strip()]
    return filled[0] if len(filled) == 1 else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--out", default="results/diag_row_axis_signal.json")
    args = ap.parse_args()

    from rag_agent.bench.hitab import load_queries

    raw_dir = Path(args.raw_dir) if args.raw_dir else Path(args.data_dir) / "data/tables/raw"
    _, tables = load_queries(args.data_dir, args.split)

    where = Counter()
    n_tables = n_deep_tables = 0
    per_table = []

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
        al = align(texts, rows_c, cols_c, nhr, nhc, bt)
        if al is None:
            continue
        rows_c, cols_c, _ = al
        n_tables += 1

        max_depth = max((len(bt.row_path(i)) for i in range(len(rows_c))), default=0)
        if max_depth <= nhc:
            continue                      # already expressible; scores .997
        n_deep_tables += 1

        # every grid text, for the "absent" test
        seen = {_norm_val(x) for row in texts for x in row if x}
        tally = Counter()
        for i_, r in enumerate(rows_c):
            gp = bt.row_path(i_)
            for anc in gp[:-1]:
                a = _norm_val(anc)
                if not a:
                    continue
                if any(a == _norm_val(_cell(texts, r, c)) for c in range(nhc)):
                    tally["stub_same_row"] += 1
                elif any(a == _norm_val(section_label(texts, rr))
                         for rr in range(0, r) if section_label(texts, rr)):
                    tally["section_row"] += 1
                elif any(a == _norm_val(_cell(texts, rr, c))
                         for rr in range(nhr, r) for c in range(nhc)):
                    tally["earlier_stub_with_data"] += 1
                elif any(a == _norm_val(_cell(texts, rr, c))
                         for rr in range(0, nhr) for c in range(len(texts[rr]))):
                    tally["header_block"] += 1
                elif any(a == _norm_val(_cell(texts, rr, c))
                         for rr in range(r + 1, len(texts)) for c in range(nhc)):
                    tally["later_stub"] += 1
                elif a in seen:
                    tally["data_cell_only"] += 1
                else:
                    tally["absent"] += 1
        where.update(tally)
        per_table.append({"table_id": tid, "nhc": nhc, "max_row_depth": max_depth,
                          **tally})

    # How many of the failing tables would a section-row rule fix ON ITS OWN?
    # Only tables whose ancestors are *entirely* section rows can go to 1.0.
    pure = sum(1 for t in per_table
               if t.get("section_row", 0)
               and sum(v for k, v in t.items()
                       if k not in ("table_id", "nhc", "max_row_depth",
                                    "section_row")
                       and isinstance(v, int)) == 0)
    total = sum(where.values())
    out = {
        "population": {"split": args.split, "n_tables_aligned": n_tables,
                       "n_tables_depth_not_expressible": n_deep_tables},
        "question": "for gold row paths deeper than the stub block, where is the "
                    "ancestor label in the raw grid?",
        "n_ancestor_labels": total,
        "n_tables_pure_section_row": pure,
        "where": dict(where),
        "share": {k: round(v / total, 4) for k, v in where.items()} if total else {},
        "per_table": per_table[:40],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("population", "n_ancestor_labels",
                                          "where", "share")}, indent=2))
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
