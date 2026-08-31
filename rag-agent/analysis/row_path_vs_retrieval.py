#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Stratify Phase 4's hitab_lookup gold cells by row-path reconstruction class.

NOTE ON WHAT THE STRATIFIER IS. The deployed P4 corpus indexes the GOLD row
path (corpus_dump_vs_cell.py: cpaths.append((pt["gold_rp"][i], ...))), not the
reconstructed one, so a row's reconstruction class does not change the sentence
that was indexed. It is used here as a property OF THE ROW -- would the
reconstructor have recovered this row's ancestors -- not as a description of
the text that was retrieved.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "analysis"))

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402
from phase4_summary import em                                        # noqa: E402
from point3_reconstruction_cost import build_table_paths             # noqa: E402
from rag_agent.bench.hitab import load_queries                       # noqa: E402
from row_path_failure import classify, depth                         # noqa: E402
from stratified_recall import two_sample_bootstrap                   # noqa: E402

OUT = Path("results/audit/row_path_vs_retrieval.json")
STRATA = ["match", "a_missing_ancestor", "other_mismatch"]


def row_classes():
    """(table_id, row) -> stratum, plus the per-table metadata for the crosstab."""
    _, tables = load_queries("data/hitab", "dev")
    raw_dir = Path("data/hitab/data/tables/raw")
    cls, meta = {}, {}
    for tid, bt in tables.items():
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
        meta[tid] = {"depth": depth(raw.get("left_root") or {}),
                     "merged": bool(raw.get("merged_regions"))}
        for i in range(pt["n_r"]):
            c = classify(pt["gold_rp"][i], pt["rec_rp"][i])
            cls[(tid, i)] = ("match" if c == "equal" else
                             "a_missing_ancestor" if c == "a_missing_ancestor"
                             else "other_mismatch")
    return cls, meta


def main() -> int:
    cls, meta = row_classes()
    df = pd.DataFrame([json.loads(l) for l in
                       open("results/phase4/reader_records.jsonl")])
    df = df[(df.pool == "hitab_lookup") & (df.policy == "P4_path_cell")].copy()
    df["is_correct"] = [em(p, g) for p, g in zip(df.pred_parsed, df.gold_answer)]

    rows, unknown = [], 0
    for r in df.itertuples():
        g = json.loads(r.gold_cell)
        if len(g) != 1:
            unknown += 1
            continue
        tid, i, j = str(g[0][0]), int(g[0][1]), int(g[0][2])
        s = cls.get((tid, i))
        if s is None:
            unknown += 1
            continue
        rows.append({"query_id": r.query_id, "stratum": s,
                     "recall": int(bool(r.gold_in_topk)),
                     "em": int(r.is_correct)})
    d = pd.DataFrame(rows)

    res = {"policy": "P4_path_cell", "pool": "hitab_lookup",
           "n_queries": len(df), "n_stratified": len(d),
           "n_unclassifiable": unknown,
           "note": __doc__.split("NOTE ON WHAT THE STRATIFIER IS.")[1].strip(),
           "strata": {}}
    for s in STRATA:
        x = d[d.stratum == s]
        res["strata"][s] = {
            "n": len(x),
            "recall": round(float(x.recall.mean()), 6) if len(x) else None,
            "n_recall_hit": int(x.recall.sum()),
            "EM": round(float(x.em.mean()), 6) if len(x) else None,
            "n_em_correct": int(x.em.sum())}

    a = d[d.stratum == "match"]
    b = d[d.stratum == "a_missing_ancestor"]
    for name, col in (("recall", "recall"), ("EM", "em")):
        lo, hi = two_sample_bootstrap(list(a[col]), list(b[col]), 10000, 42)
        res[f"match_minus_missing_{name}"] = {
            "delta": round(float(a[col].mean() - b[col].mean()), 6),
            "ci_low": None if lo is None else round(lo, 6),
            "ci_high": None if hi is None else round(hi, 6),
            "n_match": len(a), "n_missing": len(b),
            "method": "two-sample percentile bootstrap, B=10000, seed=42, "
                      "groups resampled independently"}

    # ---- merged x depth crosstab, over all 7,196 row paths ----
    agg = defaultdict(lambda: [0, 0])
    for (tid, i), s in cls.items():
        m = meta[tid]
        k = (m["merged"], m["depth"])
        agg[k][1] += 1
        agg[k][0] += int(s != "match")
    res["crosstab_merged_x_depth"] = {
        f"merged={k[0]}|depth={k[1]}": {
            "n_row_paths": v[1], "n_mismatch": v[0],
            "rate": round(v[0] / v[1], 6) if v[1] else None}
        for k, v in sorted(agg.items(), key=lambda x: (str(x[0][0]), x[0][1]))}
    res["stratum_counts_all_rows"] = dict(Counter(cls.values()))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k != "note"},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
