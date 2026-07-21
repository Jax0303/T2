#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Tree-reconstruction experiment C — REAL HiTab grid -> rebuild -> compare to gold.

The non-circular counterpart to ``scripts/tree_reconstruct_hitab.py``. That
script measures a round trip: it *encodes* the already-known gold paths into a
synthetic blank-after-first grid with ``flatten_to_grid()`` and *decodes* them
with ``_hierarchical_carry()``, which is the exact inverse — so its col .9991 /
row .9996 supports "the decoder inverts our encoder", not "the reconstructor
reads real 2D grids at 99.9%" (docs/RECONSTRUCTION_VALIDITY.md).

HiTab's ``data/tables/raw/*.json`` ships the genuine source grid (``texts``,
merged cells blanked everywhere but their origin, plus ``merged_regions``)
which nothing in the pipeline reads. This script feeds that real grid to the
same ``reconstruct_{col,row}_paths`` and scores exact match against gold.

Gold
----
Gold is the **hmt** parse (``data/tables/hmt``) — the canonical published form,
and the one the entire pipeline and every prior experiment treat as ground
truth (``original_store._parse_paths``). The raw file's own ``top_root`` /
``left_root`` are *not* used as gold: they are trees over header **cells**, so a
header cell merged across two data columns appears as a single node on the
first column only, leaving the second column's gold path a segment short. On
table 1017, ``merged_regions`` merges "percent" across columns 2-3, and the raw
tree gives column 3 ``(hirings, recruit)`` where hmt gives
``(hirings, recruit, percent)``. Scoring against the raw tree would charge the
reconstructor for span resolution the published gold says is correct.

Alignment
---------
The hmt data matrix is data-only, so its (row, col) indices must be mapped back
onto the raw grid. Candidate data lines come from the raw tree's occupied line
indices, minus section rows (a labelled row whose data region is entirely
blank; hmt drops these, the real grid keeps them). The mapping is then
**verified by value equality** against the hmt data matrix, and a table is
scored only if it verifies. Coverage is reported — an unverified table is
excluded rather than guessed at.

Boundary
--------
* ``known``   — header block size from the gold trees (last row the top tree
  occupies + 1; likewise columns).
* ``guessed`` — ``n_header_rows`` from ``guess_n_header_rows`` on the real grid.
  ``n_header_cols`` stays gold: no guesser exists for the row-header column
  count, so guessing it is out of scope.

The raw files' ``top_header_rows_num`` is not used: over all 3,597 tables it
exceeds the tree-derived header-row count by exactly 1 in 3,596 of them.

Run: PYTHONPATH=. python scripts/tree_reconstruct_hitab_raw.py --split dev
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.reconstruct import (guess_n_header_cols, guess_n_header_rows,
                                   reconstruct_col_paths, reconstruct_row_paths)


# ---------------------------------------------------------------------------
# raw grid helpers
# ---------------------------------------------------------------------------

def _cell(texts, r: int, c: int) -> str:
    if 0 <= r < len(texts) and 0 <= c < len(texts[r]):
        return str(texts[r][c]).strip()
    return ""


def tree_lines(root: dict, axis: str):
    """Walk a raw ``top_root``/``left_root``.

    Returns ``(line_indices, occupied)`` — the grid columns (axis='top') or rows
    (axis='left') the tree assigns a header to, and the grid rows (resp.
    columns) the header block itself occupies.
    """
    lines: set[int] = set()
    occupied: set[int] = set()

    def walk(node: dict) -> None:
        r, c = node.get("row_index", -1), node.get("column_index", -1)
        if r >= 0 and c >= 0:
            lines.add(c if axis == "top" else r)
            occupied.add(r if axis == "top" else c)
        for ch in node.get("children") or []:
            if isinstance(ch, dict):
                walk(ch)

    if root:
        walk(root)
    return sorted(lines), occupied


def _norm_val(v) -> str:
    """Loose value normalisation for the alignment check."""
    if v is None:
        return ""
    s = str(v).strip().lower().replace(",", "")
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _label_at(texts, line: int, band: range, axis: str) -> set[str]:
    """Texts in the header band for one data line, normalised."""
    if axis == "row":
        cells = (_cell(texts, line, c) for c in band)
    else:
        cells = (_cell(texts, r, line) for r in band)
    return {_norm_val(x) for x in cells if x}


def _align_axis(texts, cand, band, axis, leaf_labels):
    """Monotone match of gold data lines onto grid lines by leaf label.

    ``cand`` are the grid lines the raw tree touches — a superset of the data
    lines, because an ancestor may occupy a grid line of its own that the hmt
    data matrix does not keep. Walking both sequences in order and matching on
    the leaf label (which is visible in the grid; only the *path* is what the
    reconstructor has to infer) picks out the data lines without guessing.
    Returns the matched grid lines, or ``None`` if any gold line is unmatched.
    """
    out, k = [], 0
    for lab in leaf_labels:
        want = _norm_val(lab)
        while k < len(cand):
            line = cand[k]
            k += 1
            if want and want in _label_at(texts, line, band, axis):
                out.append(line)
                break
        else:
            return None
    return out if len(out) == len(leaf_labels) else None


def align(texts, rows_c, cols_c, nhr, nhc, bt, min_match: float = 0.90):
    """Map hmt data indices onto raw grid lines, then verify by value equality.

    Returns ``(rows, cols, rate)`` or ``None``. A table that cannot be verified
    is excluded rather than scored on a guessed correspondence.
    """
    n_r, n_c = bt.n_rows, bt.n_cols
    if not n_r or not n_c:
        return None

    row_leaves = [(bt.row_path(i) or [""])[-1] for i in range(n_r)]
    col_leaves = [(bt.col_path(j) or [""])[-1] for j in range(n_c)]
    rows = rows_c if len(rows_c) == n_r else _align_axis(
        texts, rows_c, range(0, nhc), "row", row_leaves)
    cols = cols_c if len(cols_c) == n_c else _align_axis(
        texts, cols_c, range(0, nhr), "col", col_leaves)
    if rows is None or cols is None:
        return None

    hits = tot = 0
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            g, h = _norm_val(_cell(texts, r, c)), _norm_val(bt.data[i][j])
            if not g and not h:
                continue
            tot += 1
            hits += int(g == h)
    if not tot:
        return None
    rate = hits / tot
    return (rows, cols, rate) if rate >= min_match else None


def norm(path):
    return tuple(s.strip().lower() for s in path if s and s.strip())


def size_bucket(n_rows: int, n_cols: int) -> str:
    d = max(n_rows, n_cols)
    if d <= 10:
        return "<=10x10"
    if d <= 20:
        return "10-20"
    return ">20"


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score_table(raw: dict, bt, guess_boundary: bool, guess_cols: bool = False,
                force_cols: int = 0):
    texts = raw.get("texts") or []
    if not texts:
        return None, "no_texts"
    cols_c, top_rows = tree_lines(raw.get("top_root") or {}, "top")
    rows_c, left_cols = tree_lines(raw.get("left_root") or {}, "left")
    if not cols_c or not rows_c or not top_rows or not left_cols:
        return None, "empty_tree"

    # The header block is everything above/left of the first line the data
    # occupies. Taking it from the *extent* of the header trees instead
    # (max(top_rows)+1) breaks on tables where a row-tree ancestor sits inside
    # the data-column region, which would push n_header_cols past the first
    # data column.
    nhr_gold = min(rows_c)
    nhc_gold = min(cols_c)
    if nhr_gold <= 0 or nhc_gold <= 0:
        return None, "degenerate_block"

    al = align(texts, rows_c, cols_c, nhr_gold, nhc_gold, bt)
    if al is None:
        return None, "unaligned"
    rows_c, cols_c, rate = al

    nhr = guess_n_header_rows(texts, n_header_cols=nhc_gold) if guess_boundary else nhr_gold
    boundary_ok = int(nhr == nhr_gold)
    if force_cols:
        nhc = force_cols
    elif guess_cols:
        nhc = guess_n_header_cols(texts, n_header_rows=nhr)
    else:
        nhc = nhc_gold
    cols_ok = int(nhc == nhc_gold)

    rec_cols = reconstruct_col_paths(texts, nhr, nhc)
    rec_rows = reconstruct_row_paths(texts, nhr, nhc)

    col_hit = col_tot = row_hit = row_tot = 0
    errors = []
    for j, c in enumerate(cols_c):
        i = c - nhc
        rp = rec_cols[i] if 0 <= i < len(rec_cols) else []
        gp = bt.col_path(j)
        col_tot += 1
        ok = norm(gp) == norm(rp)
        col_hit += int(ok)
        if not ok and len(errors) < 3:
            errors.append({"axis": "col", "grid_line": c, "gold": gp, "rec": rp})
    for i_, r in enumerate(rows_c):
        i = r - nhr
        rp = rec_rows[i] if 0 <= i < len(rec_rows) else []
        gp = bt.row_path(i_)
        row_tot += 1
        ok = norm(gp) == norm(rp)
        row_hit += int(ok)
        if not ok and len(errors) < 3:
            errors.append({"axis": "row", "grid_line": r, "gold": gp, "rec": rp})

    max_row_depth = max((len(bt.row_path(i)) for i in range(len(rows_c))), default=0)
    meta = {
        "n_data_rows": len(rows_c), "n_data_cols": len(cols_c),
        "nhr_gold": nhr_gold, "nhc_gold": nhc_gold, "nhr_used": nhr,
        "align_rate": round(rate, 4),
        "max_col_depth": max((len(bt.col_path(j)) for j in range(len(cols_c))), default=0),
        "max_row_depth": max_row_depth,
        # A row path deeper than the stub block has no column to live in: the
        # grid carries the level only as indentation, which `texts` drops.
        "row_depth_expressible": max_row_depth <= nhc_gold,
        "nhc_used": nhc,
    }
    return (col_hit, col_tot, row_hit, row_tot, boundary_ok, cols_ok, errors, meta), "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max-tables", type=int, default=0)
    ap.add_argument("--guess-boundary", action="store_true")
    ap.add_argument("--force-cols", type=int, default=0,
                    help="pin n_header_cols to this value (the old hardcoded baseline)")
    ap.add_argument("--guess-cols", action="store_true",
                    help="guess n_header_cols too, instead of taking it from gold")
    ap.add_argument("--out", default="results/tree_reconstruct_hitab_raw.json")
    args = ap.parse_args()

    from rag_agent.bench.hitab import load_queries

    raw_dir = Path(args.raw_dir) if args.raw_dir else Path(args.data_dir) / "data/tables/raw"
    _, tables = load_queries(args.data_dir, args.split)
    tids = list(tables.keys())
    if args.max_tables:
        tids = tids[: args.max_tables]
    print(f"[pop] HiTab tables (split={args.split}): {len(tids)}")

    bucket = defaultdict(lambda: {"col_total": 0, "col_hit": 0, "row_total": 0,
                                  "row_hit": 0, "n_tables": 0})
    depth = defaultdict(lambda: {"total": 0, "hit": 0})
    expressible = defaultdict(lambda: {"total": 0, "hit": 0, "n_tables": 0})
    col_hit = col_tot = row_hit = row_tot = 0
    b_ok = b_tot = c_ok = 0
    reasons = Counter()
    examples = []

    for tid in tids:
        p = raw_dir / f"{tid}.json"
        if not p.exists():
            reasons["no_raw_file"] += 1
            continue
        try:
            raw = json.load(open(p))
        except Exception:
            reasons["unreadable"] += 1
            continue
        res, why = score_table(raw, tables[tid], args.guess_boundary, args.guess_cols,
                               args.force_cols)
        reasons[why] += 1
        if res is None:
            continue
        ch, ct, rh, rt, bok, cok, errs, meta = res
        col_hit += ch; col_tot += ct; row_hit += rh; row_tot += rt
        b_tot += 1; b_ok += bok; c_ok += cok
        s = bucket[size_bucket(meta["n_data_rows"], meta["n_data_cols"])]
        s["n_tables"] += 1
        s["col_hit"] += ch; s["col_total"] += ct
        s["row_hit"] += rh; s["row_total"] += rt
        depth[f"col_depth{min(meta['max_col_depth'], 4)}"]["total"] += ct
        depth[f"col_depth{min(meta['max_col_depth'], 4)}"]["hit"] += ch
        depth[f"row_depth{min(meta['max_row_depth'], 4)}"]["total"] += rt
        depth[f"row_depth{min(meta['max_row_depth'], 4)}"]["hit"] += rh
        e = expressible["yes" if meta["row_depth_expressible"] else "no"]
        e["total"] += rt; e["hit"] += rh; e["n_tables"] += 1
        if errs and len(examples) < 10:
            examples.append({"table_id": tid, "errors": errs})

    by_bucket = {b: {"n_tables": s["n_tables"],
                     "col_path_exact_match": round(s["col_hit"] / s["col_total"], 4) if s["col_total"] else None,
                     "row_path_exact_match": round(s["row_hit"] / s["row_total"], 4) if s["row_total"] else None}
                 for b, s in bucket.items()}

    out = {
        "population": {"name": "hitab_real_grid", "split": args.split,
                       "n_tables_requested": len(tids), "n_tables_scored": b_tot,
                       "exclusions": dict(reasons)},
        "gold": "hmt parse (data/tables/hmt) — same ground truth as the rest of the pipeline",
        "input": "real source grid (data/tables/raw -> texts)",
        "boundary_mode": "guessed" if args.guess_boundary else "known (from gold trees)",
        "col_path_exact_match": round(col_hit / col_tot, 4) if col_tot else None,
        "row_path_exact_match": round(row_hit / row_tot, 4) if row_tot else None,
        "col_paths_scored": col_tot, "row_paths_scored": row_tot,
        "boundary_guess_accuracy": round(b_ok / b_tot, 4) if (args.guess_boundary and b_tot) else None,
        "n_header_cols_guess_accuracy": round(c_ok / b_tot, 4) if ((args.guess_cols or args.force_cols) and b_tot) else None,
        "by_size_bucket": by_bucket,
        "by_max_depth": {k: {"n": v["total"],
                             "exact_match": round(v["hit"] / v["total"], 4) if v["total"] else None}
                         for k, v in sorted(depth.items())},
        "row_axis_by_depth_expressible": {
            k: {"n_tables": v["n_tables"], "n_paths": v["total"],
                "row_path_exact_match": round(v["hit"] / v["total"], 4) if v["total"] else None}
            for k, v in sorted(expressible.items())},
        "error_examples": examples,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"scored tables        : {b_tot}/{len(tids)}   exclusions={dict(reasons)}")
    print(f"boundary mode        : {out['boundary_mode']}")
    if out["boundary_guess_accuracy"] is not None:
        print(f"boundary guess acc   : {out['boundary_guess_accuracy']}")
    if out["n_header_cols_guess_accuracy"] is not None:
        print(f"n_header_cols acc    : {out['n_header_cols_guess_accuracy']}")
    print(f"col_path_exact_match : {out['col_path_exact_match']}  ({col_hit}/{col_tot})")
    print(f"row_path_exact_match : {out['row_path_exact_match']}  ({row_hit}/{row_tot})")
    print("\nby size bucket (max(data rows, data cols)):")
    for b in ("<=10x10", "10-20", ">20"):
        if b in by_bucket:
            v = by_bucket[b]
            print(f"  {b:<10} n_tables={v['n_tables']:<5} "
                  f"col_exact={v['col_path_exact_match']}  row_exact={v['row_path_exact_match']}")
    print("\nby max header depth:")
    for k, v in out["by_max_depth"].items():
        print(f"  {k:<14} n={v['n']:<7} exact={v['exact_match']}")
    print("\nrow axis, split by whether row depth fits the stub-column block:")
    for k, v in out["row_axis_by_depth_expressible"].items():
        print(f"  expressible={k:<4} n_tables={v['n_tables']:<5} n_paths={v['n_paths']:<7} "
              f"row_exact={v['row_path_exact_match']}")
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
