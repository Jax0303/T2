#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Score PREREG-2026-08-27-rhb-nr-large-tables against a finished run.

The predictions were committed before the run; this only reads them off the
records. Nothing here chooses a threshold -- every bound is quoted from the
prereg, so a verdict cannot be tuned after the fact.

  PYTHONPATH=. .venv/bin/python scripts/rhbnr_verdict.py \
      results/rhbnr_s3c_512.json
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

PAPER_NR_EM = 5.32          # RealHiTBench arXiv:2506.13405 Table 2, Qwen2.5-7B
# every bound below is copied from the prereg, not chosen here
BOUNDS = {"P1": 0.05, "P2": 0.03, "P4": (0.03, 0.08), "P5": 0.03}
BIG, SMALL = 2048, 512


def mcnemar(a_vals, b_vals) -> dict:
    n01 = sum(1 for x, y in zip(a_vals, b_vals) if x and not y)
    n10 = sum(1 for x, y in zip(a_vals, b_vals) if y and not x)
    n = n01 + n10
    p = (1.0 if n == 0 else
         min(1.0, 2 * sum(comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n))
    return {"only_first": n01, "only_second": n10, "exact_p": p}


def slice_em(recs, arm, lo, hi):
    g = [r for r in recs if lo <= r.get("gold_table_tokens", 0) <= hi]
    return (sum(r[arm]["answer_em"] for r in g) / len(g), len(g)) if g else (None, 0)


def verdict(recs) -> dict:
    arms = [a for a in ("cell", "goldtable", "flat")
            if recs and a in recs[0] and recs[0][a].get("answer_em") is not None]
    out = {"n": len(recs), "arms": arms,
           "overall": {a: round(sum(r[a]["answer_em"] for r in recs) / len(recs), 4)
                       for a in arms},
           "truncated_goldtable": sum(r.get("goldtable_truncated", 0) for r in recs)}
    if "goldtable" not in arms:
        return out
    big_c, nb = slice_em(recs, "cell", BIG + 1, 10 ** 9)
    big_g, _ = slice_em(recs, "goldtable", BIG + 1, 10 ** 9)
    sm_c, ns = slice_em(recs, "cell", 0, SMALL)
    sm_g, _ = slice_em(recs, "goldtable", 0, SMALL)
    p1 = (big_c - big_g) if nb else None
    p2 = (sm_g - sm_c) if ns else None
    out["by_table_size"] = {
        f">{BIG}": {"n": nb, "cell": big_c, "goldtable": big_g, "delta": p1},
        f"<={SMALL}": {"n": ns, "cell": sm_c, "goldtable": sm_g, "delta": p2},
    }
    em_all = out["overall"]["cell"]
    out["predictions"] = {
        "P1_big_tables_cell_beats_goldtable": {
            "bound": f">= +{BOUNDS['P1']}", "measured": p1,
            "pass": bool(p1 is not None and p1 >= BOUNDS["P1"])},
        "P2_small_tables_goldtable_beats_cell": {
            "bound": f">= +{BOUNDS['P2']}", "measured": p2,
            "pass": bool(p2 is not None and p2 >= BOUNDS["P2"])},
        "P3_signs_oppose": {
            "measured": None if None in (p1, p2) else [p1, p2],
            "pass": bool(p1 is not None and p2 is not None and p1 > 0 and p2 > 0)},
        "P4_full_set_em_in_band": {
            "bound": f"{BOUNDS['P4'][0]} ~ {BOUNDS['P4'][1]}", "measured": em_all,
            "pass": bool(BOUNDS["P4"][0] <= em_all <= BOUNDS["P4"][1])},
        "P7_vs_published_5.32": {
            "note": "prereg says this is uncertain and the claim does not rest on it",
            "published": PAPER_NR_EM, "measured_pct": round(100 * em_all, 2),
            "beats": bool(100 * em_all > PAPER_NR_EM)},
    }
    if "flat" in arms:
        d = out["overall"]["cell"] - out["overall"]["flat"]
        out["predictions"]["P5_cell_beats_flat"] = {
            "bound": f">= +{BOUNDS['P5']}", "measured": round(d, 4),
            "pass": bool(d >= BOUNDS["P5"])}
    out["paired"] = {
        "answer_em:cell_vs_goldtable": mcnemar(
            [r["cell"]["answer_em"] for r in recs],
            [r["goldtable"]["answer_em"] for r in recs])}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", help="results/rhbnr_*.json (records are read beside it)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    recs_path = Path(str(Path(a.run).with_suffix("")) + "_records.jsonl")
    recs = [json.loads(l) for l in open(recs_path) if l.strip()]
    v = verdict(recs)
    out = a.out or str(Path(a.run).with_suffix("")) + "_verdict.json"
    json.dump(v, open(out, "w"), indent=2)
    print(json.dumps(v, indent=2, ensure_ascii=False))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
