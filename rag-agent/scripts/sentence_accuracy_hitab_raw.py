#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verbalized-sentence accuracy on the REAL HiTab grid.

The non-circular counterpart to ``scripts/sentence_accuracy_hitab.py``. That
script renders the *gold* header paths into a synthetic blank-after-first grid
and then decodes them back, so its .998 is a round trip through our own
encoder, not an accuracy (docs/RECONSTRUCTION_VALIDITY.md). This one feeds the
genuine source grid (``data/tables/raw/*.json`` -> ``texts``) to the same
reconstructor and scores the artifact that actually reaches the index: the
sentence.

Alignment is the one from ``tree_reconstruct_hitab_raw.py`` — hmt data indices
mapped onto raw grid lines and *verified by value equality*, tables that cannot
be verified excluded rather than guessed at. Because that mapping is 1:1 over
data cells there are no spurious/dropped sentences here, so precision = recall
and the number is a plain accuracy.

Two numbers are reported per style:

* ``sentence_exact`` — strict string equality against the sentence the gold
  structure produces. This is the one to cite.
* ``sentence_exact_value_normalized`` — the same after collapsing numeric
  formatting ("1,234" vs "1234", "5.0" vs "5"). The gap between the two is
  pure display-string difference between the raw grid and the hmt parse, not a
  header-tree error; separating them keeps the error attribution honest.

Run: PYTHONPATH=. python scripts/sentence_accuracy_hitab_raw.py --split dev
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_agent.bench.schema import BenchTable
from rag_agent.reconstruct import (guess_n_header_cols, guess_n_header_rows,
                                   reconstruct_col_paths, reconstruct_row_paths)
from rag_agent.serialize.verbalize import STYLES, _fmt, verbalize_cell
from tree_reconstruct_hitab_raw import (_cell, _norm_val, align, size_bucket,
                                        tree_lines)


def _norm_sentence(s: str) -> str:
    """Sentence with every numeric-looking token collapsed to a canonical form."""
    return " ".join(_norm_val(tok) for tok in s.split())


def score_table(raw: dict, bt: BenchTable, guess_boundary: bool, guess_cols: bool,
                force_cols: int = 0):
    """Return per-style counters for one table, or ``(None, reason)``."""
    texts = raw.get("texts") or []
    if not texts:
        return None, "no_texts"
    cols_c, _ = tree_lines(raw.get("top_root") or {}, "top")
    rows_c, _ = tree_lines(raw.get("left_root") or {}, "left")
    if not cols_c or not rows_c:
        return None, "empty_tree"

    nhr_gold, nhc_gold = min(rows_c), min(cols_c)
    if nhr_gold <= 0 or nhc_gold <= 0:
        return None, "degenerate_block"

    al = align(texts, rows_c, cols_c, nhr_gold, nhc_gold, bt)
    if al is None:
        return None, "unaligned"
    rows_c, cols_c, _rate = al

    nhr = guess_n_header_rows(texts, n_header_cols=nhc_gold) if guess_boundary else nhr_gold
    if force_cols:
        nhc = force_cols
    elif guess_cols:
        nhc = guess_n_header_cols(texts, n_header_rows=nhr)
    else:
        nhc = nhc_gold

    rec_cols = reconstruct_col_paths(texts, nhr, nhc)
    rec_rows = reconstruct_row_paths(texts, nhr, nhc)

    def _path(paths, line, start):
        i = line - start
        return paths[i] if 0 <= i < len(paths) else []

    # A BenchTable over the SAME cells as gold, but carrying the values the raw
    # grid shows and the paths the reconstructor inferred. verbalize_cell then
    # runs unmodified on both sides, so the comparison is sentence-vs-sentence.
    rec = BenchTable(
        table_id=bt.table_id, title=bt.title,
        data=[[_cell(texts, r, c) for c in cols_c] for r in rows_c],
        top_paths=[_path(rec_cols, c, nhc) for c in cols_c],
        left_paths=[_path(rec_rows, r, nhr) for r in rows_c],
        source="hitab")

    per_style = {s: Counter() for s in STYLES}
    errors = []
    for i in range(bt.n_rows):
        for j in range(bt.n_cols):
            if _fmt(bt.cell(i, j)) == "":
                continue                      # not an indexed cell
            val_ok = _norm_val(rec.cell(i, j)) == _norm_val(bt.cell(i, j))
            row_ok = rec.row_path(i) == bt.row_path(i)
            col_ok = rec.col_path(j) == bt.col_path(j)
            for style in STYLES:
                got = verbalize_cell(rec, i, j, style)
                want = verbalize_cell(bt, i, j, style)
                c = per_style[style]
                c["n"] += 1
                c["exact"] += int(got == want)
                c["exact_norm"] += int(_norm_sentence(got) == _norm_sentence(want))
                if got != want:
                    c["err_value"] += int(not val_ok)
                    c["err_row_path"] += int(not row_ok)
                    c["err_col_path"] += int(not col_ok)
                    # neither the value nor a path is wrong -> display-string only
                    c["err_format_only"] += int(val_ok and row_ok and col_ok)
                    if style == "long" and len(errors) < 3:
                        errors.append({"cell": [i, j], "generated": got, "gold": want})
    return (per_style, errors, size_bucket(bt.n_rows, bt.n_cols)), "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max-tables", type=int, default=0)
    ap.add_argument("--known-boundary", action="store_true",
                    help="take the header/data boundary from gold instead of "
                         "guessing it (isolates path reconstruction)")
    ap.add_argument("--gold-cols", action="store_true",
                    help="take n_header_cols from gold instead of guessing it")
    ap.add_argument("--force-cols", type=int, default=0,
                    help="pin n_header_cols to this value (1 = the old hardcoded baseline)")
    ap.add_argument("--out", default="results/sentence_accuracy_hitab_raw.json")
    args = ap.parse_args()

    from rag_agent.bench.hitab import load_queries

    raw_dir = Path(args.raw_dir) if args.raw_dir else Path(args.data_dir) / "data/tables/raw"
    _, tables = load_queries(args.data_dir, args.split)
    tids = list(tables.keys())
    if args.max_tables:
        tids = tids[: args.max_tables]
    print(f"[pop] HiTab tables (split={args.split}): {len(tids)}")

    totals = {s: Counter() for s in STYLES}
    buckets = {s: defaultdict(Counter) for s in STYLES}
    reasons = Counter()
    examples = []
    n_scored = 0

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
        res, why = score_table(raw, tables[tid], not args.known_boundary,
                               not args.gold_cols, args.force_cols)
        reasons[why] += 1
        if res is None:
            continue
        per_style, errs, bucket = res
        n_scored += 1
        for s in STYLES:
            totals[s].update(per_style[s])
            buckets[s][bucket].update(per_style[s])
        if errs and len(examples) < 10:
            examples.append({"table_id": tid, "errors": errs})

    def summarize(c):
        n = c["n"]
        return {
            "sentences": n,
            "sentence_exact": round(c["exact"] / n, 4) if n else None,
            "sentence_exact_value_normalized": round(c["exact_norm"] / n, 4) if n else None,
            "err_wrong_value": c["err_value"],
            "err_wrong_row_path": c["err_row_path"],
            "err_wrong_col_path": c["err_col_path"],
            "err_number_formatting_only": c["err_format_only"],
        }

    out = {
        "population": {"name": "hitab_real_grid", "split": args.split,
                       "n_tables_requested": len(tids), "n_tables_scored": n_scored,
                       "exclusions": dict(reasons)},
        "gold": "hmt parse (data/tables/hmt) verbalized by the same templates",
        "input": "real source grid (data/tables/raw -> texts)",
        "boundary_mode": "known (from gold trees)" if args.known_boundary else "guessed",
        "n_header_cols_mode": (f"forced={args.force_cols}" if args.force_cols
                               else "gold" if args.gold_cols else "guessed"),
        "by_style": {s: summarize(totals[s]) for s in STYLES},
        "by_style_by_size_bucket": {
            s: {b: summarize(c) for b, c in buckets[s].items()} for s in STYLES},
        "error_examples_long_style": examples,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"scored tables : {n_scored}/{len(tids)}   exclusions={dict(reasons)}")
    print(f"boundary      : {out['boundary_mode']}   n_header_cols: {out['n_header_cols_mode']}")
    for s in STYLES:
        v = out["by_style"][s]
        print(f"\n[{s}] sentences={v['sentences']}")
        print(f"  exact={v['sentence_exact']}   value-normalized={v['sentence_exact_value_normalized']}")
        print(f"  errors: value={v['err_wrong_value']} row_path={v['err_wrong_row_path']} "
              f"col_path={v['err_wrong_col_path']} number_format_only={v['err_number_formatting_only']}")
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
