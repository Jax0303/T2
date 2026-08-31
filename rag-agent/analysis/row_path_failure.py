#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Why the row axis reconstructs at .8202 while the column axis reaches .9749.

Unit = one data row's header path, on the 424 aligned HiTab dev tables
(n = 7,196). Reference = the gold row path, which cell_sentence_audit.py
measured to be identical to the raw left_root ancestors on this axis
(0 / 58,759 cell-level row-path mismatches), so "gold" and "raw tree" are the
same reference here.

Classification rules, applied in this order to (gold G, reconstructed R), both
whitespace-collapsed and lower-cased:

  equal                 G == R
  (d) disjoint          set(G) & set(R) == {}            [checked before a/b/c]
  (a) missing_ancestor  set(R) < set(G)   (proper subset)
  (b) extra_ancestor    set(G) < set(R)   (proper subset)
  (c) reordered         set(G) == set(R) and G != R
  (e) other             everything else (overlapping but neither contains
                        the other, or a multiset difference)

An empty reconstructed path against a non-empty gold is a proper subset, so it
lands in (a); an empty gold against a non-empty reconstruction lands in (b).
Both empty is `equal`.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from point3_reconstruction_cost import build_table_paths             # noqa: E402
from rag_agent.bench.hitab import load_queries                       # noqa: E402

OUT = Path("results/audit/row_path_failure.json")


def norm(p):
    return [" ".join(str(x).split()).lower() for x in p if str(x).strip()]


def classify(g, r):
    g, r = norm(g), norm(r)
    if g == r:
        return "equal"
    sg, sr = set(g), set(r)
    if not (sg & sr):
        return "d_disjoint"
    if sr < sg:
        return "a_missing_ancestor"
    if sg < sr:
        return "b_extra_ancestor"
    if sg == sr:
        return "c_reordered"
    return "e_other"


def depth(root):
    def walk(n, d):
        ch = [c for c in (n.get("children") or []) if isinstance(c, dict)]
        return max([walk(c, d + 1) for c in ch], default=d)
    return walk(root, 0) if root else 0


def main() -> int:
    _, tables = load_queries("data/hitab", "dev")
    raw_dir = Path("data/hitab/data/tables/raw")
    cls = Counter()
    ex = defaultdict(list)
    per_table = {}
    meta = {}
    for tid, bt in sorted(tables.items()):
        f = raw_dir / f"{tid}.json"
        if not f.exists():
            continue
        try:
            raw = json.load(open(f))
        except Exception:                                        # noqa: BLE001
            continue
        pt = build_table_paths(raw, bt)
        if pt is None:
            continue
        n_bad = 0
        for i in range(pt["n_r"]):
            g, r = pt["gold_rp"][i], pt["rec_rp"][i]
            c = classify(g, r)
            cls[c] += 1
            if c != "equal":
                n_bad += 1
                if len(ex[c]) < 3:
                    ex[c].append({"table_id": tid, "row": i, "col": None,
                                  "sentence_row_path": list(g),
                                  "reconstructed_row_path": list(r)})
        per_table[tid] = (n_bad, pt["n_r"])
        meta[tid] = {"left_depth": depth(raw.get("left_root") or {}),
                     "n_rows": pt["n_r"],
                     "merged": bool(raw.get("merged_regions"))}

    tot = sum(cls.values())
    res = {"n_row_paths": tot, "n_tables": len(per_table),
           "rules": __doc__.split("Classification rules")[1].strip(),
           "A_classes": {k: {"n": v, "rate_of_all": round(v / tot, 6),
                             "rate_of_mismatch": round(
                                 v / (tot - cls['equal']), 6)}
                         for k, v in cls.most_common() if k != "equal"},
           "A_equal": {"n": cls["equal"], "rate": round(cls["equal"] / tot, 6)},
           "A_examples": {k: v for k, v in ex.items()}}

    rates = {t: b / n for t, (b, n) in per_table.items() if n}
    nz = [t for t, v in rates.items() if v > 0]
    vals = sorted(rates.values())
    q = st.quantiles(vals, n=4) if len(vals) >= 2 else [None, None, None]
    res["B"] = {
        "n_tables_with_any_mismatch": len(nz),
        "n_tables_zero_mismatch": len(rates) - len(nz),
        "table_rate_min": round(min(vals), 6), "table_rate_q1": round(q[0], 6),
        "table_rate_median": round(q[1], 6), "table_rate_q3": round(q[2], 6),
        "table_rate_max": round(max(vals), 6),
        "top10": [{"table_id": t, "rate": round(rates[t], 6),
                   "n_bad": per_table[t][0], "n_rows": per_table[t][1]}
                  for t in sorted(rates, key=lambda x: (-rates[x],
                                                        -per_table[x][1], x))[:10]],
    }

    def strat(keyfn, name):
        agg = defaultdict(lambda: [0, 0, 0])
        for t, (b, n) in per_table.items():
            k = keyfn(meta[t])
            agg[k][0] += b
            agg[k][1] += n
            agg[k][2] += 1
        return {name: {str(k): {"n_tables": v[2], "n_row_paths": v[1],
                                "n_mismatch": v[0],
                                "rate": round(v[0] / v[1], 6) if v[1] else None}
                       for k, v in sorted(agg.items(), key=lambda x: str(x[0]))}}

    def rowbucket(m):
        n = m["n_rows"]
        return ("1-5" if n <= 5 else "6-10" if n <= 10 else
                "11-20" if n <= 20 else "21-50" if n <= 50 else ">50")

    res["C"] = {}
    res["C"].update(strat(lambda m: m["left_depth"], "left_root_depth"))
    res["C"].update(strat(rowbucket, "n_rows_bucket"))
    res["C"].update(strat(lambda m: m["merged"], "has_merged_regions"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("A_examples", "rules")},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
