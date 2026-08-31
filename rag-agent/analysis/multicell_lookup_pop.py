#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""How many HiTab dev queries are multi-cell LOOKUPS -- several gold cells, no
arithmetic on them -- and how many would survive the freeze filters the
single-cell population uses. Counting only; freezes nothing.

Output: results/lookup_gap/multicell_pop.json."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "."); sys.path.insert(0, "scripts")

from rag_agent.bench.hitab import load_queries                       # noqa: E402
from point3_reconstruction_cost import build_table_paths            # noqa: E402

ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}
DATA = "data/hitab"

queries, tables = load_queries(DATA, "dev")
raw_dir = Path(DATA) / "data/tables/raw"
paths = {}
for tid, bt in tables.items():
    p = raw_dir / f"{tid}.json"
    if not p.exists():
        continue
    try:
        raw = json.load(open(p))
    except Exception:
        continue
    pt = build_table_paths(raw, bt)
    if pt is not None:
        paths[tid] = pt

agg_of = lambda q: (q.aggregation or "none")
buckets, multi_agg, answer_len = Counter(), Counter(), Counter()
eligible = []
for q in queries:
    ops = [(op.row, op.col) for op in q.gold_operands]
    m = len(ops)
    arith = agg_of(q) in ARITH
    buckets[("m=1" if m == 1 else "m>=2" if m else "m=0",
             "arith" if arith else "non-arith")] += 1
    if m < 2 or arith:
        continue
    multi_agg[agg_of(q)] += 1
    ans = q.answer if isinstance(q.answer, list) else [q.answer]
    answer_len[(m, len(ans))] += 1
    pt = paths.get(q.gold_table_id)
    in_grid = pt is not None and all(0 <= r < pt["n_r"] and 0 <= c < pt["n_c"]
                                     for r, c in ops)
    eligible.append({"query_id": q.query_id, "m": m, "n_answer": len(ans),
                     "aggregation": agg_of(q), "table": q.gold_table_id,
                     "paths_build": pt is not None, "all_operands_in_grid": in_grid,
                     "answer": q.answer, "question": q.question})

passing = [e for e in eligible if e["paths_build"] and e["all_operands_in_grid"]]
out = {
    "n_dev_queries": len(queries),
    "by_bucket": {f"{k[0]}|{k[1]}": v for k, v in sorted(buckets.items())},
    "multi_cell_lookup": {
        "n_total": len(eligible),
        "n_passing_single_cell_freeze_filters": len(passing),
        "by_aggregation": dict(multi_agg.most_common()),
        "m_distribution": dict(Counter(e["m"] for e in passing).most_common()),
        "answer_element_count": dict(Counter(e["n_answer"] for e in passing).most_common()),
        "m_equals_n_answer": sum(e["m"] == e["n_answer"] for e in passing),
    },
    "queries": passing,
}
p = Path("results/lookup_gap/multicell_pop.json")
p.write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(json.dumps({k: v for k, v in out.items() if k != "queries"},
                 ensure_ascii=False, indent=2))
print(f"-> {p}")
