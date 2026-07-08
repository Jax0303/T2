#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Diagnostic (task 1) — structural taxonomy of the row-axis enumeration failures.

The embedding idea closed the *lexical/depth* part of the row-axis decomposition
bottleneck (E2 hybrid OSC 0.380, matching the 70b LLM), but the residual gap to
the dense baseline is **structural scope selection**: which / how-many sibling
rows to aggregate. This script tears apart every row-axis failure of the hybrid
resolver (``row_cov == 0``: gold row leaves not all enumerated) on the primary
population and classifies *why* the scope was wrong, so task-2 treatments can be
targeted at the dominant structure rather than guessed.

Taxonomy (mutually exclusive primary bucket), computed from the gold row leaves'
header paths in the original table:

  parent_expandable : all gold rows share one immediate parent AND gold == every
                      numeric-bearing child of that parent. A single "expand the
                      parent subtree" fixes it with zero precision loss.
  sibling_subset    : all gold rows share one immediate parent but gold is a
                      *strict subset* of that parent's children. Needs sibling
                      selection (which children), not blunt subtree expansion.
  cross_parent      : gold rows live under >1 immediate parent (named-set /
                      pairwise comparison / cross-cut). Hardest; not a subtree.

Orthogonal flags: total_miss (enum caught none of the gold rows) vs partial;
row_fallback (no row predicate matched -> whole axis, yet still miss = gold rows
are non-numeric-header / off by parsing); parent_recoverable (the longest common
prefix of the gold paths, enumerated via find_rows_by_header, is a superset of the
gold rows) and its precision cost (extra numeric rows the parent would add).

LLM-free. Run:
    PYTHONPATH=. python scripts/diag_row_failures.py --split dev
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import re
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.bench.hitab import load_queries
from rag_agent.data.loader import load_table
from rag_agent.query.header_embed_resolver import EmbedResolver
from rag_agent.query.operand_decomposer import Embedder
from rag_agent.retrieve.header_enum import enumerate_scope
from rag_agent.stores.original_store import build_original_table

ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}

# A "total-like" header segment: a table/section aggregate row (total, overall,
# or an "all <X>" / top-level rollup). These rows carry an empty or top-level
# header path, so a header-text resolver cannot bind them — yet share/ratio (div)
# questions need them as the denominator.
_TOTALISH = re.compile(r"\b(total|overall)\b|^all\b", re.I)


def _is_totalish_path(path) -> bool:
    """A header path that denotes a table/section total (incl. empty = unparsed)."""
    if len(path) == 0:
        return True
    return any(_TOTALISH.search(seg) for seg in path)


def _numeric_rows_with_path_prefix(ot, prefix):
    """Row indices whose header path starts with ``prefix`` and bear >=1 numeric cell."""
    pl = len(prefix)
    out = []
    for r in range(ot.n_rows):
        p = ot.row_path(r)
        if len(p) >= pl and list(p[:pl]) == list(prefix):
            if any(ot.cell_num(r, c) is not None for c in range(ot.n_cols)):
                out.append(r)
    return out


def _lcp(paths):
    """Longest common prefix across a list of paths."""
    if not paths:
        return []
    out = []
    for i in range(min(len(p) for p in paths)):
        seg = paths[0][i]
        if all(p[i] == seg for p in paths):
            out.append(seg)
        else:
            break
    return out


def classify(ot, gold_rows, enum_rows):
    gold_rows = sorted(gold_rows)
    gpaths = {r: list(ot.row_path(r)) for r in gold_rows}
    hit = sorted(set(gold_rows) & set(enum_rows))
    missed = sorted(set(gold_rows) - set(enum_rows))

    # immediate parents (path minus the leaf segment)
    parents = {tuple(p[:-1]) for p in gpaths.values() if len(p) >= 1}
    single_parent = len(parents) == 1 and all(len(p) >= 2 for p in gpaths.values())

    bucket = "cross_parent"
    parent_children = []
    if single_parent:
        parent = list(next(iter(parents)))
        # numeric children directly under that immediate parent
        parent_children = [r for r in _numeric_rows_with_path_prefix(ot, parent)
                           if len(ot.row_path(r)) == len(parent) + 1]
        if set(gold_rows) == set(parent_children):
            bucket = "parent_expandable"
        else:
            bucket = "sibling_subset"
    elif len(gpaths) == 1:
        # a single gold row leaf that wasn't matched: degenerate, treat as sibling_subset
        bucket = "sibling_subset"

    # parent-recoverability via longest common prefix (looser than immediate parent)
    lcp = _lcp(list(gpaths.values()))
    parent_recoverable = False
    parent_extra = None
    if lcp:
        cover = set(_numeric_rows_with_path_prefix(ot, lcp))
        parent_recoverable = set(gold_rows) <= cover
        parent_extra = len(cover) - len(gold_rows)  # precision cost if we expand LCP

    # Refined bucket: carve out "total_pairing" — a missed gold row is a
    # table/section total (the share-of-total / ratio structure). This dominates
    # and is *not* a sibling-selection problem, so it gets its own category.
    missed_totalish = any(_is_totalish_path(ot.row_path(r)) for r in missed)
    refined = "total_pairing" if missed_totalish else bucket
    # Oracle treatment-recovery flags (would the row axis become covered?):
    #  T_total : add every total-like row -> covers iff every MISSED row is total-like
    #  T_parent: expand the gold rows' longest-common-prefix subtree (parent_recoverable)
    all_missed_total = bool(missed) and all(_is_totalish_path(ot.row_path(r)) for r in missed)

    return {
        "bucket": bucket,
        "refined_bucket": refined,
        "missed_total_row": missed_totalish,
        "recover_by_total_aug": all_missed_total,
        "recover_by_parent_expand": parent_recoverable,
        "total_miss": len(hit) == 0,
        "n_gold_rows": len(gold_rows),
        "n_hit": len(hit),
        "n_missed": len(missed),
        "single_parent": single_parent,
        "n_parent_children": len(parent_children),
        "lcp_depth": len(lcp),
        "parent_recoverable": parent_recoverable,
        "parent_extra_rows": parent_extra,
        "gold_row_paths": [gpaths[r] for r in gold_rows],
        "missed_row_paths": [gpaths[r] for r in missed],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/diag_row_failures.json")
    ap.add_argument("--dump", default="results/diag_row_failures.jsonl")
    args = ap.parse_args()

    queries, tables = load_queries(args.data_dir, args.split, args.max)
    # match E2 population exactly: arithmetic, distinct-cell scope m>=2.
    pop = [q for q in queries
           if (q.aggregation or "none") in ARITH
           and len({(o.row, o.col) for o in q.gold_operands}) >= 2]
    print(f"[pop] arithmetic m>=2 (distinct cells): {len(pop)}")

    embedder = Embedder(args.embed_model, device=args.device)
    resolver = EmbedResolver(embedder, row_mode="embed", col_mode="lexical")  # hybrid

    needed = {q.gold_table_id for q in pop}
    ots = {}
    for tid in needed:
        ots[tid] = build_original_table(load_table(tid, args.data_dir))

    fails, n_row_fail, n_col_only = [], 0, 0
    dump = []
    for q in pop:
        ot = ots[q.gold_table_id]
        gold = q.gold_operands
        gold_rows = {o.row for o in gold}
        gold_cols = {o.col for o in gold}
        intent = resolver.resolve(q.question, ot)
        enum = enumerate_scope(ot, intent.row_paths, intent.col_paths)
        row_cov = gold_rows <= enum.rows
        col_cov = gold_cols <= enum.cols
        if row_cov:
            if not col_cov:
                n_col_only += 1
            continue
        n_row_fail += 1
        info = classify(ot, gold_rows, enum.rows)
        rec = {
            "query_id": q.query_id, "question": q.question,
            "aggregation": q.aggregation, "m": len({(o.row, o.col) for o in gold}),
            "col_cov": int(col_cov), "row_fallback": int(enum.row_fallback),
            "table_id": q.gold_table_id,
            "resolved_row_paths": intent.row_paths,
            **info,
        }
        fails.append(rec)
        dump.append(rec)

    # --- aggregate the taxonomy ---
    bk = Counter(f["bucket"] for f in fails)
    rbk = Counter(f["refined_bucket"] for f in fails)
    n = len(fails) or 1
    summary = {
        "population": {"name": "arithmetic_m>=2", "n": len(pop)},
        "row_axis_failures": len(fails),
        "row_axis_coverage": round(1 - len(fails) / len(pop), 4),
        "col_only_failures": n_col_only,
        "refined_taxonomy": {b: {"n": c, "frac_of_row_fail": round(c / n, 4)}
                             for b, c in rbk.most_common()},
        "structural_taxonomy": {b: {"n": c, "frac_of_row_fail": round(c / n, 4)}
                                for b, c in bk.most_common()},
        "total_miss_rate": round(sum(f["total_miss"] for f in fails) / n, 4),
        "partial_rate": round(sum(not f["total_miss"] for f in fails) / n, 4),
        "row_fallback_rate": round(sum(f["row_fallback"] for f in fails) / n, 4),
        "oracle_recovery_upper_bound": {
            "total_aug_only": sum(f["recover_by_total_aug"] for f in fails),
            "parent_expand_only": sum(f["recover_by_parent_expand"] for f in fails),
            "either": sum(f["recover_by_total_aug"] or f["recover_by_parent_expand"] for f in fails),
            "neither": sum(not (f["recover_by_total_aug"] or f["recover_by_parent_expand"]) for f in fails),
            "note": "row-axis recovery only; OSC also needs col_cov. Upper bound, not measured OSC.",
        },
        "parent_recoverable_rate": round(sum(f["parent_recoverable"] for f in fails) / n, 4),
        "mean_parent_extra_rows_when_recoverable": round(
            sum(f["parent_extra_rows"] for f in fails if f["parent_recoverable"])
            / max(1, sum(f["parent_recoverable"] for f in fails)), 2),
        "by_aggregation": dict(Counter(f["aggregation"] for f in fails).most_common()),
        "refined_bucket_by_aggregation": {
            b: dict(Counter(f["aggregation"] for f in fails if f["refined_bucket"] == b).most_common())
            for b in rbk},
        # cross-tab: bucket x total_miss
        "bucket_x_totalmiss": {
            b: {"total_miss": sum(1 for f in fails if f["bucket"] == b and f["total_miss"]),
                "partial": sum(1 for f in fails if f["bucket"] == b and not f["total_miss"])}
            for b in bk},
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(args.dump, "w") as fh:
        for r in dump:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote summary -> {args.out}\nwrote per-query dump -> {args.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
