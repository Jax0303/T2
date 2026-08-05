#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""E-A significance: does the reranker's OSC effect flip sign with pool size?

``operand_collision_significance.py`` pairs *schemes* (flat -> S2/S3) inside one
records file. E-A's axis is different: the same scheme, baseline retriever vs
``cross``, swept over the per-size records ``ea_pool_size_sweep.py`` writes. Same
test as that script's (3): query-level all-covered flips, exact binomial sign
test, which is McNemar's exact form.

Holm correction is applied over the whole family reported here
(RESEARCH_STRUCTURE.md §6: only uncorrected p-values were being printed).

Run:
    .venv/bin/python scripts/ea_crossover_stats.py results/ea_pool_size_sweep_{flat,S3}.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scipy.stats import binomtest

from operand_collision_significance import load


def covered_by_query(rows, scheme: str, retriever: str, k: int) -> dict:
    per_q = defaultdict(list)
    for r in rows:
        if r["scheme"] == scheme and r["retriever"] == retriever:
            per_q[r["query"]].append(r["rank"])
    return {q: all(v is not None and v <= k for v in vs) for q, vs in per_q.items()}


def holm(tests: list[dict]) -> None:
    """In-place Holm-Bonferroni over ``p_two_sided``; sets ``p_holm``/``sig_05``."""
    order = sorted(range(len(tests)), key=lambda i: tests[i]["p_two_sided"])
    m, running = len(order), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * tests[i]["p_two_sided"]))
        tests[i]["p_holm"] = running
        tests[i]["sig_05"] = running < 0.05


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweeps", nargs="+", help="ea_pool_size_sweep*.json (records read alongside)")
    ap.add_argument("--baseline", default="hybrid")
    ap.add_argument("--treatment", default="cross")
    ap.add_argument("--ks", type=int, nargs="+", default=[10, 50])
    ap.add_argument("--out", default="results/ea_crossover_stats.json")
    args = ap.parse_args()

    tests = []
    for sweep_path in args.sweeps:
        stem = str(Path(sweep_path).with_suffix(""))
        sweep = json.loads(Path(sweep_path).read_text())
        for size in sweep["pool_sizes"]:
            rec = Path(f"{stem}_records_p{size}.jsonl")
            if not rec.exists():
                print(f"[skip] {rec} missing", file=sys.stderr)
                continue
            rows = load(str(rec))
            for scheme in sorted({r["scheme"] for r in rows}):
                for k in args.ks:
                    base = covered_by_query(rows, scheme, args.baseline, k)
                    treat = covered_by_query(rows, scheme, args.treatment, k)
                    qs = sorted(set(base) & set(treat))
                    gain = sum(1 for q in qs if treat[q] and not base[q])
                    loss = sum(1 for q in qs if base[q] and not treat[q])
                    if gain + loss == 0:
                        continue
                    tests.append({
                        "scheme": scheme, "pool_size": size, "k": k,
                        "n_queries": len(qs),
                        f"{args.baseline}_covered": sum(base[q] for q in qs),
                        f"{args.treatment}_covered": sum(treat[q] for q in qs),
                        "gain": gain, "loss": loss,
                        "delta_osc": round((gain - loss) / len(qs), 4),
                        "p_two_sided": float(
                            binomtest(gain, gain + loss, 0.5, alternative="two-sided").pvalue),
                    })

    holm(tests)
    out = {
        "experiment": "E-A crossover — {} vs {} OSC flips by pool size".format(
            args.baseline, args.treatment),
        "test": "exact binomial sign test on query-level all-covered flips "
                "(McNemar exact), Holm-corrected over the whole family below",
        "family_size": len(tests),
        "sources": args.sweeps,
        "tests": sorted(tests, key=lambda t: (t["scheme"], t["k"], t["pool_size"])),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"{'scheme':<6} {'pool':>5} {'k':>3} {'base':>5} {'cross':>6} "
          f"{'gain':>5} {'loss':>5} {'dOSC':>7} {'p':>10} {'p_holm':>9}")
    for t in out["tests"]:
        print(f"{t['scheme']:<6} {t['pool_size']:>5} {t['k']:>3} "
              f"{t[args.baseline + '_covered']:>5} {t[args.treatment + '_covered']:>6} "
              f"{t['gain']:>5} {t['loss']:>5} {t['delta_osc']:>+7.3f} "
              f"{t['p_two_sided']:>10.2e} {t['p_holm']:>9.2e}"
              + ("  *" if t["sig_05"] else ""))
    print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
