#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Characterise RealHiTBench's question types as a retrieval-difficulty ladder.

The answer-accuracy leg has only ever run on ``Calculation`` +
``Multi-hop Numerical Reasoning`` — both of which need arithmetic over several
cells. That leaves the easy end of the ladder unmeasured, so nothing in the repo
distinguishes "retrieval found the cell" from "the solver did the arithmetic",
and a retrieval claim cannot be made at the level where retrieval is the whole
task.

Three strata, cheapest first:

  L1 lookup   — the answer IS a cell. Retrieval alone decides it.
  L2 compute  — the answer is derived from several cells by one operation.
  L3 derived  — multi-step, and the answer is not in the table at all.

The stratum labels come from the benchmark's own ``SubQType`` rather than
anything invented here. This script checks whether they *behave* that way, by
asking the LLM-free question the labels imply: does the gold answer literally
appear as a cell of the reconstructed table? L1 should mostly say yes and L3
mostly no. Where the label and the mechanical test disagree, the count is
reported rather than silently reconciled — a "lookup" question whose answer is
not in the table is either a reconstruction failure or a mislabel, and both are
worth seeing before any accuracy number is quoted per stratum.

No LLM, no API key. Run:
  PYTHONPATH=. python3 scripts/rhb_difficulty_strata.py --limit 400
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.realhitbench_answer_accuracy import HF_REPO, build_table, gold_is_numeric

# The benchmark's SubQType -> our stratum. Types that are not a table-QA
# retrieval task at all (chart generation, free-text summary, structure probes)
# are left out rather than forced onto the ladder.
TIERS = {
    "L1_lookup": {"Value-Matching"},
    "L2_compute": {"Calculation", "Counting", "Comparison", "Ranking"},
    "L3_derived": {"Multi-hop Numerical Reasoning", "Multi-hop Fact Checking"},
}
TIER_OF = {sq: tier for tier, sqs in TIERS.items() for sq in sqs}


def _norm_num(s):
    """Numeric value of a cell/answer string, or None."""
    t = re.sub(r"[,\s$]", "", str(s)).rstrip("%")
    try:
        return float(t)
    except (TypeError, ValueError):
        return None


def split_gold(gold) -> list:
    """Gold answer as its list of values — one element for a single answer.

    RealHiTBench writes multi-part answers as one comma-joined string
    ("128154, 21538", "Men, 16 years and over, Married, 43.30"). Matched whole
    they are in no table, which is how 44 of 119 Value-Matching questions first
    scored as "answer absent from the table" — every one of which turned out to
    be present in the grid, just in two cells instead of one. They are a
    different task (return a SET of cells), so they get their own bucket rather
    than being counted as a lookup the pipeline failed.
    """
    parts = [p.strip() for p in re.split(r",(?![^(]*\))", str(gold))]
    return [p for p in parts if p]


def locate_gold(gold, table) -> str:
    """Where the gold answer sits in the table: ``value`` / ``header`` / ``absent``.

    Both locations are reachable by cell retrieval, but by different halves of
    the index unit, so they are not the same result: a ``value`` gold is carried
    by the sentence's predicate, a ``header`` gold by its path. Checking only
    ``table.data`` misses every header gold, because :func:`build_table` moves the
    stub column and the header band OUT of ``data`` and into ``left_paths`` /
    ``top_paths`` — and RealHiTBench's Value-Matching answers are often a year or
    a category label, i.e. exactly those. Reporting them merged hides which half
    of the sentence is doing the work.

    Numbers compare numerically ("1,234" and "1234" are one value written twice),
    text case-insensitively on the stripped string. Deliberately generous: this
    measures whether retrieval COULD reach the answer, so a false positive is
    safer than a missed one.
    """
    parts = split_gold(gold)
    if not parts:
        return "absent"
    if len(parts) > 1:
        return "multi"

    g_num = _norm_num(parts[0])
    g_txt = re.sub(r"\s+", " ", parts[0].lower())

    def same(v) -> bool:
        if g_num is not None:
            c = _norm_num(v)
            return c is not None and abs(c - g_num) < 1e-9
        return re.sub(r"\s+", " ", str(v).strip().lower()) == g_txt

    if any(same(v) for row in table.data for v in row):
        return "value"
    if any(same(s) for p in list(table.left_paths) + list(table.top_paths) for s in p):
        return "header"
    return "absent"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-repo", default=HF_REPO)
    ap.add_argument("--limit", type=int, default=0,
                    help="max queries per stratum (tables are the slow part)")
    ap.add_argument("--out", default="results/rhb_difficulty_strata.json")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    qa = json.load(open(hf_hub_download(args.hf_repo, "QA_final.json",
                                        repo_type="dataset")))["queries"]

    by_tier = defaultdict(list)
    for q in sorted(qa, key=lambda x: x["id"]):
        tier = TIER_OF.get(q.get("SubQType"))
        if tier:
            by_tier[tier].append(q)
    if args.limit:
        by_tier = {t: qs[: args.limit] for t, qs in by_tier.items()}

    print("[pop] " + "  ".join(f"{t}={len(qs)}" for t, qs in sorted(by_tier.items())),
          flush=True)

    tables: dict = {}
    out = {}
    for tier, qs in sorted(by_tier.items()):
        where = Counter()
        n_num = n_scored = skipped = 0
        subq = Counter()
        for i, q in enumerate(qs):
            fname = q["FileName"]
            if fname not in tables:
                tables[fname] = build_table(fname, args.hf_repo)
            t = tables[fname]
            if t is None:
                skipped += 1
                continue
            gold = q["ProcessedAnswer"]
            n_scored += 1
            n_num += int(gold_is_numeric(gold))
            where[locate_gold(gold, t)] += 1
            subq[q["SubQType"]] += 1
            if (i + 1) % 50 == 0:
                print(f"  {tier} {i+1}/{len(qs)}", flush=True)
        frac = (lambda k: round(where[k] / n_scored, 4) if n_scored else None)
        out[tier] = {
            "n_queries": len(qs),
            "n_scored": n_scored,
            "n_tables_unusable": skipped,
            "gold_is_a_value_cell": frac("value"),
            "gold_is_a_header_label": frac("header"),
            "gold_is_multi_value": frac("multi"),
            "gold_absent_from_table": frac("absent"),
            "gold_is_numeric": round(n_num / n_scored, 4) if n_scored else None,
            "subqtypes": dict(subq),
        }
        print(f"[{tier}] n={n_scored}  value={out[tier]['gold_is_a_value_cell']}"
              f"  header={out[tier]['gold_is_a_header_label']}"
              f"  multi={out[tier]['gold_is_multi_value']}"
              f"  absent={out[tier]['gold_absent_from_table']}"
              f"  numeric={out[tier]['gold_is_numeric']}  (skipped {skipped})",
              flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({
            "population": "RealHiTBench QA_final.json, SubQType mapped to 3 strata",
            "tier_map": {k: sorted(v) for k, v in TIERS.items()},
            "gold_location": "where the gold answer sits in the RECONSTRUCTED table. "
                             "value=a data cell (the sentence's predicate carries it); "
                             "header=a row/col header label (the sentence's PATH carries "
                             "it); absent=neither, so it is derived or reconstruction "
                             "lost it. Reachable-by-retrieval, not solved-by-solver.",
            "limit_per_tier": args.limit or None,
            "by_tier": out,
        }, fh, indent=2)
    print(f"wrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
