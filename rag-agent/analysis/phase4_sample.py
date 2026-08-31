#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Phase 4 Task A2 -- the frozen 200-query sample.

Proportional (largest-remainder) allocation inside each type stratum, then a
seeded draw per pool. Written once; Task B and Task C read this file and never
redraw.

  PYTHONPATH=. .venv/bin/python analysis/phase4_sample.py
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "analysis"))

from header_path_coverage import load_corpus                         # noqa: E402
from qwen_equiv_k import POOLS, args_for, rhb_qtype                   # noqa: E402

SEED = 42
# PREREGISTER rev3: fixed per-pool n, not proportional -- a proportional draw
# leaves aitqa 31 / rhb_fact 11 / rhb_num 24, too few for the per-dataset
# comparison Phase 3 showed is the one that matters.
QUOTA = {"lookup": {"hitab_lookup": 60, "aitqa": 60, "rhb_fact": 60},
         "arith": {"hitab_arith": 60, "rhb_num": 54}}
OUT = Path("results/phase4/sample_294.csv")


def largest_remainder(sizes, total):
    """Proportional allocation; leftovers go to the largest fractional parts."""
    N = sum(sizes)
    exact = [s * total / N for s in sizes]
    base = [int(e) for e in exact]
    for i in sorted(range(len(sizes)), key=lambda i: -(exact[i] - base[i]))[
            :total - sum(base)]:
        base[i] += 1
    return base


def main() -> int:
    qt_of, pools = rhb_qtype(), {}
    corpora = {}
    for name, ds, pop, qt in POOLS:
        if (ds, pop) not in corpora:
            corpora[(ds, pop)] = load_corpus(args_for(ds, pop))
        C = corpora[(ds, pop)]
        pools[name] = (ds, [q for q in C.queries
                            if not qt or qt_of.get(str(q["query_id"])) == qt])

    rows, report = [], []
    for qtype, quota in QUOTA.items():
        total = sum(quota.values())
        for name, n_take in quota.items():
            ds, qs = pools[name]
            n_pool = len(qs)
            take = random.Random(SEED).sample(sorted(
                qs, key=lambda q: str(q["query_id"])), n_take)
            report.append([qtype, name, ds, n_pool, n_take / n_pool, n_take,
                           n_take / total])
            for q in take:
                gc = sorted(q["gold_cells"])
                rows.append({"query_id": q["query_id"], "query_type": qtype,
                             "pool": name, "dataset": ds,
                             "question": q["question"], "gold_answer": q["answer"],
                             "gold_table_id": q["gold_table"],
                             "gold_cells": json.dumps([list(c) for c in gc])})

    p = OUT
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{'type':<7} {'pool':<13} {'dataset':<13} {'n_pool':>6} {'frac':>7} "
          f"{'n_take':>6} {'stratum%':>9}")
    for t, name, ds, n_pool, fr, n_take, tr in report:
        print(f"{t:<7} {name:<13} {ds:<13} {n_pool:>6} {fr:>6.1%} "
              f"{n_take:>6} {tr:>8.1%}")
    ids = [str(r["query_id"]) for r in rows]
    assert len(ids) == 294 and len(set(ids)) == 294
    old = Path("results/phase4/sample_200.csv")
    if old.exists():
        prev = {r["query_id"] for r in csv.DictReader(open(old))}
        print(f"\noverlap with sample_200: {len(set(ids) & prev)} / 200")
    print(f"total={len(ids)}  unique query_id={len(set(ids))}\n-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
