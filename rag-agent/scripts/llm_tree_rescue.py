#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Gate: does TreeThinker's tree prompt rescue the rows the grid rules cannot?

STATUS-2026-08-10 §2 found 388 HiTab tables (15.7%) where a parent row carries
its own values, and **not one of them** is fully reconstructed — the parent and
its children are identical once indentation is gone, so no rule over the grid
can separate them. `rag_agent/reconstruct/llm_tree.py` borrows TreeThinker's
round-1 prompt to ask an LLM instead, at index time.

This script measures only that. It scores every table with the rule-based
reconstructor, keeps the ones it does not fully reconstruct (``--max-row-exact``,
worst first — the data-parent class sits at the bottom at .206 of row lines, so
a capped budget lands on it without re-deriving that classifier), and re-scores
just those with the LLM row reconstructor. Same alignment, same boundary, same
metric — only ``rec_rows`` differs.

Read the result as a gate, not a headline:

  row exact stays ~0 -> the information is not in the grid at all, and the LLM
                        is reading the same blanks. Drop the idea.
  row exact rises    -> rebuild the index sentences off these trees and re-run
                        the retrieval legs.

Segment F1 is reported alongside because a path can improve a lot without ever
reaching exact (STATUS-2026-08-10 §2).

Run (one LLM call per rescued table):
    PYTHONPATH=. python scripts/llm_tree_rescue.py --split dev --max-llm 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.llm.factory import build_llm  # noqa: E402
from rag_agent.reconstruct.llm_tree import reconstruct_row_paths_llm  # noqa: E402
from rag_agent.runenv import run_env  # noqa: E402

from tree_reconstruct_hitab_raw import score_table  # noqa: E402


def _rows(res):
    """(row_hit, row_tot, mean row segment F1) out of a score_table result."""
    _ch, _ct, rh, rt, _b, _c, _e, meta = res
    f1 = meta["row_segment_f1"]
    return rh, rt, (sum(f1) / len(f1) if f1 else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max-tables", type=int, default=0, help="cap tables scanned")
    ap.add_argument("--max-llm", type=int, default=50,
                    help="cap LLM calls — one per rescued table, 0 = no cap")
    ap.add_argument("--max-row-exact", type=float, default=1.0,
                    help="only rescue tables whose rule-based row exact is BELOW "
                         "this (1.0 = every table with at least one wrong row)")
    ap.add_argument("--solver", default="local:Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--guess-boundary", action="store_true")
    ap.add_argument("--guess-cols", action="store_true")
    ap.add_argument("--out", default="results/llm_tree_rescue.json")
    ap.add_argument("--dump", default="results/llm_tree_rescue.jsonl")
    args = ap.parse_args()

    from rag_agent.bench.hitab import load_queries

    raw_dir = Path(args.raw_dir) if args.raw_dir else Path(args.data_dir) / "data/tables/raw"
    _, tables = load_queries(args.data_dir, args.split)
    tids = list(tables.keys())
    if args.max_tables:
        tids = tids[: args.max_tables]

    # --- pass 1: rule-based, to find the tables with nothing to lose ---------
    targets, reasons = [], Counter()
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
        res, why = score_table(raw, tables[tid], args.guess_boundary, args.guess_cols)
        reasons[why] += 1
        if res is None:
            continue
        n_scored += 1
        rh, rt, f1 = _rows(res)
        if rt and rh / rt < args.max_row_exact:
            targets.append((tid, raw, rh, rt, f1))

    # worst first, so a capped budget is spent on the tables the rules fail
    # hardest — the data-parent class sits at the bottom (row lines .206).
    targets.sort(key=lambda t: t[2] / t[3])
    print(f"[pop] scored {n_scored} tables, "
          f"{len(targets)} with row exact < {args.max_row_exact}")
    todo = targets[: args.max_llm] if args.max_llm else targets

    # --- pass 2: same tables, LLM rows --------------------------------------
    llm = build_llm(args.solver) if todo else None
    base_hit = base_tot = llm_hit = llm_tot = 0
    base_f1_w = llm_f1_w = 0.0
    n_tables_fixed = n_err = n_covered = n_emitted = 0
    dump = []
    t0 = time.time()
    for i, (tid, raw, rh_base, rt_base, f1_base) in enumerate(todo, 1):
        got = []
        try:
            def row_fn(texts, nhr, nhc, _llm=llm, _got=got):
                paths = reconstruct_row_paths_llm(texts, nhr, nhc, _llm,
                                                  max_tokens=args.max_tokens)
                _got.append(paths)
                return paths
            res, why = score_table(raw, tables[tid], args.guess_boundary,
                                   args.guess_cols, row_fn=row_fn)
        except Exception as exc:  # a dead call must not lose the tables before it
            n_err += 1
            dump.append({"table_id": tid, "error": repr(exc)})
            print(f"  [{i}/{len(todo)}] {tid} ERROR {exc!r}")
            continue
        if res is None:
            reasons[f"llm_{why}"] += 1
            continue
        rh, rt, f1 = _rows(res)
        base_hit += rh_base; base_tot += rt_base
        llm_hit += rh; llm_tot += rt
        base_f1_w += f1_base * rt_base
        llm_f1_w += f1 * rt
        n_tables_fixed += int(rh == rt)
        # A model that numbered rows from the first DATA line instead of grid
        # line 0 also scores ~0, for a reason that has nothing to do with
        # whether it understood the hierarchy. Empty paths separate the two:
        # low coverage is an off-by-nhr, full coverage with low exact is a real
        # miss. Without this the gate cannot be read.
        emitted = got[0] if got else []
        n_empty = sum(1 for p in emitted if not p)
        n_covered += len(emitted) - n_empty
        n_emitted += len(emitted)
        dump.append({"table_id": tid, "n_rows": rt,
                     "n_empty_paths": n_empty, "n_paths": len(emitted),
                     "rule_row_exact": round(rh_base / rt_base, 4),
                     "llm_row_exact": round(rh / rt, 4) if rt else None,
                     "rule_row_f1": round(f1_base, 4), "llm_row_f1": round(f1, 4)})
        print(f"  [{i}/{len(todo)}] {tid} rows exact {rh_base}/{rt_base} -> {rh}/{rt}, "
              f"f1 {f1_base:.3f} -> {f1:.3f}")

    out = {
        "env": run_env(seed=0, embed_model=""),  # no embedder on this leg
        "population": {
            "name": "hitab_raw_grid, tables with rule-based row exact == 0",
            "split": args.split, "n_tables_scored": n_scored,
            "max_row_exact": args.max_row_exact,
            "n_below_threshold": len(targets), "n_llm_attempted": len(todo),
            "n_llm_errors": n_err, "exclusions": dict(reasons),
        },
        "method": {
            "prompt": "TreeThinker Generate_Tree (RealHiTBench, Findings of ACL 2025), verbatim",
            "solver": args.solver, "max_tokens": args.max_tokens,
            "note": "round 1 only; the reasoning round is not used. One call per table.",
        },
        "rows": {
            "n_row_paths": llm_tot,
            "rule_row_exact": round(base_hit / base_tot, 4) if base_tot else None,
            "llm_row_exact": round(llm_hit / llm_tot, 4) if llm_tot else None,
            "rule_row_segment_f1": round(base_f1_w / base_tot, 4) if base_tot else None,
            "llm_row_segment_f1": round(llm_f1_w / llm_tot, 4) if llm_tot else None,
            "n_tables_fully_reconstructed": n_tables_fixed,
            "path_coverage": round(n_covered / n_emitted, 4) if n_emitted else None,
            "coverage_note": "fraction of data rows the model gave ANY path for. "
                             "Well below 1.0 means its row numbering did not line "
                             "up with the grid — read that before reading exact.",
        },
        "wall_seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    with open(args.dump, "w") as fh:
        for r in dump:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps(out["rows"], indent=2))
    print(f"\nwrote {args.out} / {args.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
