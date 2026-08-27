#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""cell vs ROW CHUNKING across every committed run that holds both arms.

Why this table replaces the one built on ``dump``: a RAG system does not put a
whole table in the context, so an advantage over ``dump`` does not transfer to
practice. Row chunking is the configuration actually deployed, and the record
already contains it -- every ``corpus_dump_vs_cell`` run scored a ``row`` arm
alongside ``cell``. No new inference is needed to re-seat the baseline.

Selection rule, stated so the table cannot be accused of picking its cases:
EVERY ``*_records.jsonl`` under ``results/`` that scores both arms is included,
losses reported beside wins. Nothing is dropped for being unflattering; runs are
dropped only for being unusable, and each exclusion is printed with its reason.

Rows are never merged across reader, retriever, cell scheme or budget -- a
paired test is only meaningful when the two arms saw the same everything else.
The split by operand count ``m`` is not decoration: MultiHiertt at 512 has cell
ahead by .075 at m=1 and behind by .070 at m>=2, so a single pooled number hides
the sign flip.

  PYTHONPATH=. python3 scripts/row_baseline_table.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.stats import binomtest

CORPUS_OF = {"hitab": "HiTab", "aitqa": "AIT-QA", "multihiertt": "MultiHiertt",
             "rhb": "RealHiTBench", "realhitbench": "RealHiTBench"}


def corpus_of(population: str) -> str:
    for key, name in CORPUS_OF.items():
        if population.startswith(key):
            return name
    return population or "?"


def paired(recs: list[dict], a: str, b: str) -> dict:
    """Sign test on per-query EM flips a->b, the convention the verdict files use."""
    a_only = sum(1 for r in recs if r[a]["answer_em"] and not r[b]["answer_em"])
    b_only = sum(1 for r in recs if r[b]["answer_em"] and not r[a]["answer_em"])
    n = len(recs)
    p = (float(binomtest(b_only, a_only + b_only, 0.5).pvalue)
         if a_only + b_only else 1.0)
    return {"n": n,
            "em_a": round(sum(r[a]["answer_em"] for r in recs) / n, 4),
            "em_b": round(sum(r[b]["answer_em"] for r in recs) / n, 4),
            "delta": round((sum(r[b]["answer_em"] for r in recs)
                            - sum(r[a]["answer_em"] for r in recs)) / n, 4),
            "b_only": b_only, "a_only": a_only, "p": round(p, 4)}


def meta_of(stem: Path) -> dict:
    """Run metadata from the summary JSON, falling back to the sidecar."""
    m = {}
    for path in (stem.with_suffix(".json"),
                 Path(str(stem) + "_records.jsonl.run.json")):
        if path.exists():
            try:
                m = {**json.loads(path.read_text()), **m}
            except json.JSONDecodeError:
                pass
    pop = m.get("population")
    pop = pop.get("name", "") if isinstance(pop, dict) else str(pop or "")
    return {"population": pop, "budget": m.get("budget_tokens") or m.get("budget"),
            "scheme": m.get("cell_scheme", "?"), "retriever": m.get("retriever", "?"),
            "reader": str(m.get("reader_spec") or m.get("reader") or "?")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--min-n", type=int, default=50,
                    help="runs smaller than this are listed as excluded, not scored")
    ap.add_argument("--out", default="results/row_baseline_table.json")
    args = ap.parse_args()

    rows, excluded = [], []
    for f in sorted(Path(args.results).glob("*_records.jsonl")):
        recs = []
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                break
            if all(isinstance(r.get(a), dict) and "answer_em" in r[a]
                   for a in ("cell", "row")):
                recs.append(r)
        if not recs:
            continue
        stem = Path(str(f)[: -len("_records.jsonl")])
        meta = meta_of(stem)
        row = {"run": stem.name, **meta, **paired(recs, "row", "cell")}
        if len(recs) < args.min_n:
            excluded.append({**row, "reason": f"n={len(recs)} < {args.min_n}"})
            continue
        for label, keep in (("m1", lambda r: r.get("m", 1) == 1),
                            ("m2+", lambda r: r.get("m", 1) >= 2)):
            sub = [r for r in recs if keep(r)]
            row[label] = paired(sub, "row", "cell") if sub else None
        rows.append(row)

    def tally(sel) -> dict:
        w = sum(1 for r in rows if sel(r) and sel(r)["delta"] > 0 and sel(r)["p"] < .05)
        l = sum(1 for r in rows if sel(r) and sel(r)["delta"] < 0 and sel(r)["p"] < .05)
        return {"cell_wins": w, "row_wins": l,
                "not_significant": sum(1 for r in rows if sel(r)) - w - l}

    out = {"experiment": "cell vs row chunking, every committed run with both arms",
           "note": "no new inference; re-read of committed records",
           "runs": rows, "excluded": excluded,
           "tally_overall": tally(lambda r: r),
           "tally_m1": tally(lambda r: r.get("m1")),
           "tally_m2plus": tally(lambda r: r.get("m2+"))}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))

    def cell_str(d):
        return f"{d['delta']:+.3f}({d['n']:>3d}){'*' if d['p'] < .05 else ' '}" if d else "      —    "

    print(f"{'run':38s} {'corpus':12s} {'ret':6s} {'bud':>5s} {'sch':4s} {'n':>4s} "
          f"{'row':>6s} {'cell':>6s} {'Δ전체':>8s} {'p':>7s}  {'Δ m=1':>12s} {'Δ m>=2':>12s}")
    for r in sorted(rows, key=lambda r: (corpus_of(r["population"]), r["run"])):
        print(f"{r['run'][:38]:38s} {corpus_of(r['population']):12s} "
              f"{r['retriever'][:6]:6s} {str(r['budget'] or '?'):>5s} "
              f"{r['scheme'][:4]:4s} {r['n']:4d} "
              f"{r['em_a']:6.3f} {r['em_b']:6.3f} {r['delta']:+8.4f} {r['p']:7.4f}  "
              f"{cell_str(r.get('m1')):>12s} {cell_str(r.get('m2+')):>12s}")
    for name, key in (("전체", "tally_overall"), ("m=1", "tally_m1"),
                      ("m>=2", "tally_m2plus")):
        t = out[key]
        print(f"[{name:5s}] cell 유의 승 {t['cell_wins']} · row 유의 승 "
              f"{t['row_wins']} · 무의미 {t['not_significant']}")
    for e in excluded:
        print(f"제외: {e['run']} ({e['reason']})")
    print(f"wrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
