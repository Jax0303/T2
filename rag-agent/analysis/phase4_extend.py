#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PREREGISTER rev5 step 3 -- the 129 queries that take hitab_lookup to 189.

The existing 60 are kept verbatim; the 129 are drawn from what is left of the
830-query pool with the same seed, so the union is the frozen 189.
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
from qwen_equiv_k import args_for                                    # noqa: E402

SEED, TARGET, POOL = 42, 189, "hitab_lookup"
OUT = Path("results/phase4/sample_hitab_lookup_ext129.csv")


def main() -> int:
    have = {r["query_id"] for r in
            csv.DictReader(open("results/phase4/sample_294.csv"))
            if r["pool"] == POOL}
    C = load_corpus(args_for("hitab", "hitab_dev_lookup_all"))
    qs = {str(q["query_id"]): q for q in C.queries}
    rest = sorted(set(qs) - have)
    take = random.Random(SEED).sample(rest, TARGET - len(have))
    rows = [{"query_id": q, "query_type": "lookup", "pool": POOL,
             "dataset": "hitab", "question": qs[q]["question"],
             "gold_answer": qs[q]["answer"], "gold_table_id": qs[q]["gold_table"],
             "gold_cells": json.dumps([list(c) for c in sorted(qs[q]["gold_cells"])])}
            for q in take]
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"pool={len(qs)}  기존={len(have)}  추가={len(rows)}  "
          f"합={len(have) + len(rows)}  겹침={len(have & set(take))}")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
