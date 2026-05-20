#!/usr/bin/env python3
"""Paired-bootstrap 95% CI for the headline metrics.

Given a result JSON from run_eval.py, computes (mean, 2.5%, 97.5%) for:
  - R@1 (vector only)
  - R@1 (after verifier)
  - delta = R@1(final) - R@1(vec)   ← significance of the verifier
  - Numeric Match
  - Exact Match

Bootstrap is RESAMPLED OVER QUERIES (n=40), 10,000 iterations.

Multiple files can be passed; reports each plus a "delta vs first"
contrast for comparing seeds / configs / ablations.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def per_query_metrics(rows):
    """Returns list of dicts with the four per-query booleans/floats."""
    out = []
    for r in rows:
        gold = r["gold_table"]
        vec = r.get("vector_top") or []
        fin = r.get("final_top") or []
        out.append({
            "r1_vec": int(gold in vec[:1]),
            "r1_fin": int(gold in fin[:1]),
            "r5_fin": int(gold in fin[:5]),
            "nm": int(bool(r.get("answer_numeric_match"))),
            "em": int(bool(r.get("answer_em"))),
        })
    return out


def bootstrap(per_q, key, n_iters=10000, seed=0):
    rng = random.Random(seed)
    n = len(per_q)
    means = []
    for _ in range(n_iters):
        s = 0
        for _ in range(n):
            s += per_q[rng.randrange(n)][key]
        means.append(s / n)
    means.sort()
    return (means[n_iters // 2],
            means[int(0.025 * n_iters)],
            means[int(0.975 * n_iters)])


def paired_delta_bootstrap(per_q, key_a, key_b, n_iters=10000, seed=0):
    """Paired bootstrap on (key_a - key_b)."""
    rng = random.Random(seed)
    n = len(per_q)
    deltas = []
    for _ in range(n_iters):
        s_a = s_b = 0
        for _ in range(n):
            idx = rng.randrange(n)
            s_a += per_q[idx][key_a]
            s_b += per_q[idx][key_b]
        deltas.append((s_a - s_b) / n)
    deltas.sort()
    return (deltas[n_iters // 2],
            deltas[int(0.025 * n_iters)],
            deltas[int(0.975 * n_iters)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--iters", type=int, default=10000)
    args = p.parse_args()

    summaries = []
    for path in args.paths:
        data = json.load(open(path))
        per_q = per_query_metrics(data["rows"])
        n = len(per_q)
        summary = {"path": path, "n": n, "metrics": {}}
        for key, label in [("r1_vec", "R@1 (vector)"),
                           ("r1_fin", "R@1 (final)"),
                           ("r5_fin", "R@5 (final)"),
                           ("em",     "Exact Match"),
                           ("nm",     "Numeric Match")]:
            mean = sum(q[key] for q in per_q) / n
            med, lo, hi = bootstrap(per_q, key, args.iters)
            summary["metrics"][label] = (mean, lo, hi)
        # paired delta
        d_mean = sum(q["r1_fin"] - q["r1_vec"] for q in per_q) / n
        d_med, d_lo, d_hi = paired_delta_bootstrap(per_q, "r1_fin", "r1_vec", args.iters)
        summary["metrics"]["Δ R@1 (verifier)"] = (d_mean, d_lo, d_hi)
        summaries.append(summary)

    for s in summaries:
        print(f"\n=== {s['path']}  (n={s['n']}, {args.iters} iters)")
        print(f"{'metric':<22s} {'mean':>7s}   {'95% CI':>16s}")
        for k, (m, lo, hi) in s["metrics"].items():
            sign = " *" if (lo > 0 or hi < 0) and "Δ" in k else ""
            print(f"{k:<22s} {m:>7.3f}   [{lo:.3f}, {hi:.3f}]{sign}")


if __name__ == "__main__":
    main()
