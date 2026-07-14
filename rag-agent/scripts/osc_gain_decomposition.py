#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""P5: decompose the S3-over-flat completeness gain into POOL EXPANSION vs
WITHIN-POOL RANKING, over the reranker-experiment pools (top-100 per scheme).

Question this answers (closes the §5.1c candidate-generation argument): of the
queries S3 covers at k but flat does not, how many were UNREACHABLE for flat by
any reordering of its own pool (>=1 gold operand cell absent from flat's
top-100 candidate pool -> the gain REQUIRES new cells entering the pool), and
how many had the full gold set inside flat's pool but ranked too low (a perfect
reranker over flat's candidates could in principle have recovered them)?

Input: *_records.jsonl from operand_collision_rerank.py — rows {scheme,
retriever, query, cell, rank, ...} where rank is the gold cell's position in
that scheme's ordering of its own top-100 pool and null means the cell is NOT
in the pool at all. Both flat orderings (hybrid_pool100, rerank_pool100) share
one flat pool, so pool-limited classification is identical across them; only
the rank-limited share moves.

Per contrast (flat baseline ordering -> S3 hybrid) and k:
  * gains / losses: queries whose all-gold-<=k coverage flips.
  * gains split into pool_limited (>=1 baseline rank null) vs rank_limited
    (all in pool, some rank > k), with shares of the gross gain.
  * oracle_flat@k: coverage if flat's pool were PERFECTLY reranked (all gold in
    pool and scope <= k) — the ceiling any flat reranker obeys.
Sanity: recomputed set_recall@k and pool ceilings must match the run summary
(operand_collision_rerank_n300.json) before the numbers are citable.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

KS = (10, 20, 50)


def load_ranks(path: str):
    """(scheme, retriever) -> {query: {cell: rank-or-None}}."""
    ranks = defaultdict(lambda: defaultdict(dict))
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            ranks[(r["scheme"], r["retriever"])][r["query"]][r["cell"]] = r["rank"]
    return ranks


def covered(cell_ranks: dict, k: int) -> bool:
    return all(r is not None and r <= k for r in cell_ranks.values())


def in_pool(cell_ranks: dict) -> bool:
    return all(r is not None for r in cell_ranks.values())


def decompose(base: dict, treat: dict, k: int) -> dict:
    """base/treat: {query: {cell: rank-or-None}} on the SAME query set."""
    qids = sorted(base)
    assert sorted(treat) == qids, "query sets differ between conditions"
    gains, losses = [], []
    for q in qids:
        b, t = covered(base[q], k), covered(treat[q], k)
        if t and not b:
            gains.append(q)
        elif b and not t:
            losses.append(q)
    pool_limited = [q for q in gains if not in_pool(base[q])]
    rank_limited = [q for q in gains if in_pool(base[q])]
    # operand-level view of the same gains: which specific cells were missing?
    cells_missing_from_pool = sum(
        1 for q in gains for r in base[q].values() if r is None)
    cells_ranked_out = sum(
        1 for q in gains for r in base[q].values() if r is not None and r > k)
    n = len(qids)
    oracle_base = sum(1 for q in qids
                      if in_pool(base[q]) and len(base[q]) <= k) / n
    return {
        "n_queries": n,
        "base_covered": sum(covered(base[q], k) for q in qids),
        "treat_covered": sum(covered(treat[q], k) for q in qids),
        "gain": len(gains), "loss": len(losses),
        "gain_pool_limited": len(pool_limited),
        "gain_rank_limited": len(rank_limited),
        "share_pool_limited": round(len(pool_limited) / len(gains), 4) if gains else None,
        "share_rank_limited": round(len(rank_limited) / len(gains), 4) if gains else None,
        "gain_cells_missing_from_base_pool": cells_missing_from_pool,
        "gain_cells_in_pool_ranked_below_k": cells_ranked_out,
        "oracle_base_set_recall@k": round(oracle_base, 4),
        "treat_set_recall@k": round(sum(covered(treat[q], k) for q in qids) / n, 4),
        "treat_beats_oracle_base": sum(covered(treat[q], k) for q in qids) / n > oracle_base,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("records",
                    default="results/operand_collision_rerank_n300_records.jsonl",
                    nargs="?")
    ap.add_argument("--treat", default="S3,hybrid_pool100")
    ap.add_argument("--baselines",
                    default="flat,hybrid_pool100;flat,rerank_pool100")
    ap.add_argument("--summary", default="results/operand_collision_rerank_n300.json",
                    help="run summary to cross-check recomputed aggregates against")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ranks = load_ranks(args.records)
    t_scheme, t_ret = args.treat.split(",")
    treat = ranks[(t_scheme, t_ret)]

    report = {"records": args.records, "treat": args.treat, "contrasts": {}}

    # sanity: pool membership rate must reproduce the run's pool_ceiling@100
    if args.summary and Path(args.summary).exists():
        summ = json.load(open(args.summary))
        for sch in {b.split(",")[0] for b in args.baselines.split(";")} | {t_scheme}:
            per_q = ranks[(sch, t_ret)]
            rate = sum(in_pool(per_q[q]) for q in per_q) / len(per_q)
            want = summ["by_scheme"][sch]["pool_ceiling@100"]
            assert abs(rate - want) < 5e-4, (sch, rate, want)
            report.setdefault("pool_ceiling@100", {})[sch] = round(rate, 4)

    for spec in args.baselines.split(";"):
        b_scheme, b_ret = spec.split(",")
        base = ranks[(b_scheme, b_ret)]
        key = f"{b_scheme}_{b_ret}->{t_scheme}_{t_ret}"
        report["contrasts"][key] = {f"@{k}": decompose(base, treat, k) for k in KS}

    out = args.out or str(Path(args.records).with_suffix("")).replace(
        "_records", "") + "_gain_decomposition.json"
    Path(out).write_text(json.dumps(report, indent=2))
    print(f"[out] {out}  pool_ceiling@100={report.get('pool_ceiling@100')}")
    for key, per_k in report["contrasts"].items():
        for k, d in per_k.items():
            print(f"  {key:44s} {k:4s} gain={d['gain']:>3} "
                  f"(pool-limited {d['gain_pool_limited']:>3} "
                  f"[{d['share_pool_limited']}] / rank-limited {d['gain_rank_limited']:>3}"
                  f" [{d['share_rank_limited']}])  loss={d['loss']:>3}  "
                  f"oracle_flat={d['oracle_base_set_recall@k']}  "
                  f"S3={d['treat_set_recall@k']}"
                  f"{'  ** S3 > flat-oracle' if d['treat_beats_oracle_base'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
