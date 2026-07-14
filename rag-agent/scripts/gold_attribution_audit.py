#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Quantify the gold-operand attribution risk (reviewer issue M6).

Concern: if gold operands were resolved by VALUE-matching, a value occurring in
multiple cells could be attributed to the wrong cell. In the current resolver
(rag_agent.bench.hitab.resolve_gold_operands) value-matching is only the
FALLBACK; the primary path maps the annotation's own grid coordinates through
the table's header-block offset (the true annotated cell, no value lookup).

This audit reports, for the paper population (HiTab dev, arithmetic, m>=2):
  1. fraction of queries resolved via the offset (coordinate) path vs the
     value-matching fallback — the only slice where misattribution is possible;
  2. within the fallback slice, the ambiguity rate: operands whose value occurs
     in >1 data cell of the gold table (upper bound on misattribution);
  3. the same ambiguity rate over ALL gold operands, as the worst-case bound a
     reviewer would compute assuming pure value-matching.

Run: PYTHONPATH=. python scripts/gold_attribution_audit.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.bench.hitab import (_coord_offset, _coords_of, _table_offset,
                                   _to_float, build_original_table,
                                   load_samples, load_table)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default="results/gold_attribution_audit.json")
    args = ap.parse_args()

    samples = load_samples(args.data_dir, args.split, None)
    linked_by_table = {}
    for s in samples:
        linked_by_table.setdefault(s.get("table_id"), []).append(
            s.get("linked_cells") or {})

    ot_cache, offset_cache = {}, {}
    n_q = n_offset = n_fallback = n_unresolved = 0
    fb_ops = fb_ambiguous = 0
    all_ops = all_ambiguous = 0
    for s in samples:
        agg = s.get("aggregation")
        agg = agg[0] if isinstance(agg, list) and agg else agg
        if not agg or agg in ("none", ""):
            continue                                   # arithmetic only
        coords = _coords_of(s.get("linked_cells") or {})
        if len({(i, j) for i, j, _ in coords}) < 2:
            continue                                   # m >= 2
        tid = s.get("table_id")
        if tid not in ot_cache:
            raw = load_table(tid, args.data_dir)
            ot_cache[tid] = build_original_table(raw) if raw else None
        ot = ot_cache[tid]
        if ot is None:
            continue
        if tid not in offset_cache:
            offset_cache[tid] = _table_offset(ot, linked_by_table.get(tid, []))
        n_q += 1

        # which path would resolve_gold_operands take for THIS query?
        off = offset_cache[tid]
        fits = off is not None and coords and all(
            0 <= i - off[0] < ot.n_rows and 0 <= j - off[1] < ot.n_cols
            and _to_float(ot.data[i - off[0]][j - off[1]]) == fv
            for i, j, fv in coords)
        if not fits:
            off = _coord_offset(ot, coords)            # per-query re-derivation
            fits = off is not None

        # value-frequency index over the table's data cells
        freq = Counter()
        for row in ot.data:
            for v in row:
                fv = _to_float(v)
                if fv is not None:
                    freq[round(fv, 4)] += 1
        for _i, _j, fv in coords:
            all_ops += 1
            if freq.get(round(fv, 4), 0) > 1:
                all_ambiguous += 1

        if fits:
            n_offset += 1
        else:
            n_fallback += 1
            for _i, _j, fv in coords:
                fb_ops += 1
                if freq.get(round(fv, 4), 0) > 1:
                    fb_ambiguous += 1
            if not any(freq.get(round(fv, 4), 0) for _i, _j, fv in coords):
                n_unresolved += 1

    out = {
        "population": f"hitab {args.split} arithmetic m>=2",
        "n_queries": n_q,
        "resolved_by_offset_coords": n_offset,
        "resolved_by_value_fallback": n_fallback,
        "fallback_rate": round(n_fallback / n_q, 4) if n_q else None,
        "fallback_operands": fb_ops,
        "fallback_operands_value_ambiguous": fb_ambiguous,
        "worst_case_all_operands": all_ops,
        "worst_case_value_ambiguous": all_ambiguous,
        "worst_case_ambiguity_rate": round(all_ambiguous / all_ops, 4) if all_ops else None,
        "note": "offset path uses annotated coordinates directly; value ambiguity "
                "only threatens the fallback slice. worst_case_* assumes pure "
                "value-matching (the reviewer's premise).",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
