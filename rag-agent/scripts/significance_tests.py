#!/usr/bin/env python3
"""Honest paired significance tests on the consolidated per-question results.

Uses McNemar's EXACT test (two-sided binomial on discordant pairs) — the correct
test for paired binary outcomes — instead of the percentile-bootstrap CI used in
the eval harnesses, which is anti-conservative at small n and over-stated
significance. Reads docs/review_data/results_long.csv.
"""
import csv
from collections import defaultdict
from math import comb
from pathlib import Path

CSV = Path(__file__).resolve().parents[1] / "docs/review_data/results_long.csv"


def load():
    cell = defaultdict(dict)
    for r in csv.DictReader(open(CSV)):
        cell[(r["experiment"], r["model"], r["split"], r["qid"])][r["condition"]] = int(r["correct"])
    return cell


def mcnemar_exact(pairs):
    """pairs = list of (a_correct, b_correct); two-sided exact p for b vs a."""
    n01 = sum(1 for x, y in pairs if x == 0 and y == 1)   # b better
    n10 = sum(1 for x, y in pairs if x == 1 and y == 0)   # a better
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = min(1.0, sum(comb(n, i) for i in range(k + 1)) / (2 ** n) * 2)
    return n01, n10, p


def contrast(cell, exp, model, split, a, b):
    pa, pb = [], []
    for k, d in cell.items():
        if k[:3] == (exp, model, split) and a in d and b in d:
            pa.append(d[a]); pb.append(d[b])
    n = len(pa)
    if n == 0:
        return
    aa, ab = sum(pa) / n, sum(pb) / n
    n01, n10, p = mcnemar_exact(list(zip(pa, pb)))
    flag = "**SIG" if p < 0.05 else "ns  "
    print(f"  {exp:18} {model:12} {split:4} {b}-{a}: d={ab-aa:+.3f} "
          f"(n={n}, {aa:.2f}->{ab:.2f}) discord b+:{n01} a+:{n10} McNemar p={p:.3f} {flag}")


def main():
    cell = load()
    print("== interaction: header_path - flat_leaf (S2-S1) ==")
    for m in ["llama-8b", "gpt-oss-120b"]:
        for sp in ["flat", "hier"]:
            contrast(cell, "interaction", m, sp, "flat_leaf", "header_path")
    print("== token control: header_path - header_shuffle (structure vs matched tokens) ==")
    contrast(cell, "token_control", "llama-8b", "hier", "header_shuffle", "header_path")
    print("== retrieval pipeline: header_path - flat_leaf ==")
    contrast(cell, "retrieval_pipeline", "llama-8b", "hier", "flat_leaf", "header_path")


if __name__ == "__main__":
    main()
