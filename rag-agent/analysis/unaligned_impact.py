#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""What the 116 unaligned HiTab tables cost the evaluation.

Counts only. The recomputed rates below are arithmetic on measured numbers
under one stated assumption (every excluded query counts wrong); nothing is
re-run and no reader is called.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "analysis"))

from freeze_populations import ARITH                                 # noqa: E402
from point3_reconstruction_cost import build_table_paths             # noqa: E402
from rag_agent.bench.hitab import load_queries                       # noqa: E402

OUT = Path("results/audit/unaligned_impact.json")


def main() -> int:
    queries, tables = load_queries("data/hitab", "dev")
    raw_dir = Path("data/hitab/data/tables/raw")
    ok, bad = set(), set()
    for tid, bt in tables.items():
        f = raw_dir / f"{tid}.json"
        if not f.exists():
            bad.add(tid)
            continue
        try:
            raw = json.load(open(f))
        except Exception:                                        # noqa: BLE001
            bad.add(tid)
            continue
        (ok if build_table_paths(raw, bt) is not None else bad).add(tid)

    q_bad = [q for q in queries if q.gold_table_id in bad]
    lookup_like = [q for q in q_bad if len(q.gold_operands) == 1]
    arith_like = [q for q in q_bad
                  if (q.aggregation or "none") in ARITH and q.gold_operands]

    pops = {}
    for name in ("hitab_dev_lookup_all", "hitab_dev_corpus_arith"):
        ids = [l.strip() for l in open(f"populations/{name}.txt")
               if l.strip() and not l.startswith("#")]
        pops[name] = set(ids)

    bad_ids = {str(q.query_id) for q in q_bad}
    res = {
        "n_dev_tables": len(tables), "n_aligned": len(ok), "n_unaligned": len(bad),
        "n_dev_queries": len(queries),
        "q1_queries_on_unaligned_tables": len(q_bad),
        "q1_by_aggregation": Counter(q.aggregation or "none"
                                     for q in q_bad).most_common(),
        "q1_single_operand": len(lookup_like),
        "q1_arith_eligible": len(arith_like),
        "q2_in_hitab_dev_lookup_all": len(
            bad_ids & pops["hitab_dev_lookup_all"]),
        "q2_in_hitab_dev_corpus_arith": len(
            bad_ids & pops["hitab_dev_corpus_arith"]),
        "pop_sizes": {k: len(v) for k, v in pops.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
