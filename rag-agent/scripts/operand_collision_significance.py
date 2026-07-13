"""Significance tests over operand_collision_multihiertt per-cell rank records.

Input: *_records.jsonl rows {scheme, retriever, query, cell, rank, colliding, total_like}
(global schemes flat/S2/S3; cascade rows, if present, may carry null ranks and are
excluded from rank-based tests but kept for coverage).

Tests
  (1) flat: colliding-label vs unique-label operand ranks, Mann-Whitney U (per retriever)
  (2) colliding operands: flat -> S2/S3 paired rank change, Wilcoxon signed-rank
  (3) all_covered@k query-level flips flat -> S2/S3, exact two-sided binomial (sign test)
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest, mannwhitneyu, wilcoxon

KS = (10, 20, 50)
FAIL_RANK = None  # sentinel in records for "not retrieved"


def load(path: str):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("--out", default=None)
    ap.add_argument("--schemes", default="S2,S3")
    args = ap.parse_args()

    rows = load(args.records)
    retrievers = sorted({r["retriever"] for r in rows})
    schemes = args.schemes.split(",")
    # (scheme, retriever) -> {(query, cell): rank}
    ranks = defaultdict(dict)
    meta = {}  # (query, cell) -> (colliding, total_like)
    for r in rows:
        ranks[(r["scheme"], r["retriever"])][(r["query"], r["cell"])] = r["rank"]
        meta[(r["query"], r["cell"])] = (r["colliding"], r["total_like"])

    report = {"records": args.records, "n_rows": len(rows), "retrievers": retrievers}

    # (1) flat colliding vs unique, Mann-Whitney U
    t1 = {}
    for ret in retrievers:
        flat = ranks.get(("flat", ret), {})
        coll = [v for k, v in flat.items() if v is not None and meta[k][0]]
        uniq = [v for k, v in flat.items() if v is not None and not meta[k][0]]
        if not coll or not uniq:
            continue
        u, p = mannwhitneyu(coll, uniq, alternative="greater")
        t1[ret] = {
            "n_colliding": len(coll), "n_unique": len(uniq),
            "median_colliding": statistics.median(coll),
            "median_unique": statistics.median(uniq),
            "U": float(u), "p_one_sided_greater": float(p),
        }
    report["flat_colliding_vs_unique_mannwhitney"] = t1

    # (2) colliding operands, flat -> scheme, Wilcoxon signed-rank
    t2 = {}
    for ret in retrievers:
        flat = ranks.get(("flat", ret), {})
        for sch in schemes:
            alt = ranks.get((sch, ret), {})
            pairs = [(flat[k], alt[k]) for k in flat
                     if meta[k][0] and k in alt
                     and flat[k] is not None and alt[k] is not None]
            if len(pairs) < 5:
                continue
            a = [p[0] for p in pairs]
            b = [p[1] for p in pairs]
            diffs = [x - y for x, y in pairs if x != y]
            if not diffs:
                continue
            w, p = wilcoxon(a, b, alternative="greater")  # flat ranks worse (larger)
            t2[f"{ret}/flat->{sch}"] = {
                "n_pairs": len(pairs),
                "median_flat": statistics.median(a),
                "median_alt": statistics.median(b),
                "improved": sum(1 for x, y in pairs if y < x),
                "worsened": sum(1 for x, y in pairs if y > x),
                "W": float(w), "p_one_sided": float(p),
            }
    report["colliding_flat_to_scheme_wilcoxon"] = t2

    # (3) all_covered@k query flips, exact binomial sign test (two-sided)
    t3 = {}
    for ret in retrievers:
        flat = ranks.get(("flat", ret), {})
        by_q_flat = defaultdict(list)
        for (q, c), v in flat.items():
            by_q_flat[q].append(v)
        for sch in schemes:
            alt = ranks.get((sch, ret), {})
            by_q_alt = defaultdict(list)
            for (q, c), v in alt.items():
                by_q_alt[q].append(v)
            for k in KS:
                cov = lambda vs: all(v is not None and v <= k for v in vs)
                gains = losses = 0
                n_q = 0
                for q in by_q_flat:
                    if q not in by_q_alt:
                        continue
                    n_q += 1
                    f_ok, a_ok = cov(by_q_flat[q]), cov(by_q_alt[q])
                    if a_ok and not f_ok:
                        gains += 1
                    elif f_ok and not a_ok:
                        losses += 1
                if gains + losses == 0:
                    continue
                bt = binomtest(gains, gains + losses, 0.5, alternative="two-sided")
                bt1 = binomtest(gains, gains + losses, 0.5, alternative="greater")
                t3[f"{ret}/flat->{sch}@{k}"] = {
                    "n_queries": n_q,
                    "flat_covered": sum(cov(by_q_flat[q]) for q in by_q_flat),
                    "alt_covered": sum(cov(by_q_alt[q]) for q in by_q_alt),
                    "gain": gains, "loss": losses,
                    "p_two_sided": float(bt.pvalue),
                    "p_one_sided_gain": float(bt1.pvalue),
                }
    report["all_covered_flip_binomial"] = t3

    out = args.out or str(Path(args.records).with_suffix("")).replace(
        "_records", "") + "_significance.json"
    Path(out).write_text(json.dumps(report, indent=2))
    print(f"[out] {out}")

    for name, block in (("(1) flat colliding vs unique (MWU)", t1),
                        ("(2) colliding flat->scheme (Wilcoxon)", t2),
                        ("(3) all_covered flips (binomial)", t3)):
        print(f"\n{name}")
        for key, st in block.items():
            sig = " *" if [v for k2, v in st.items() if k2.startswith("p_")][0] < 0.05 else ""
            print(f"  {key:<24} {json.dumps(st)}{sig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
