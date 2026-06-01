#!/usr/bin/env python3
"""두 codegen_eval 결과 JSON을 *쌍대(paired)*로 비교한다.

같은 SEED로 돌린 런은 동일 쿼리셋이므로, 독립 CI 비교보다 쌍대 검정이
훨씬 검정력이 높다. 쿼리별로 짝지어 다음을 계산한다.

  - ΔR@1, ΔR@5, ΔMRR, ΔnDCG@10, ΔNM (B - A)  + 95% paired bootstrap CI
  - R@1 / NM 에 대한 McNemar 검정 (불일치쌍 exact binomial)

사용:
  python scripts/compare_runs.py A.json B.json
  # A=baseline(VDB), B=제안(structural). 첫 파일이 기준.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional


def _load_rows(path: str) -> Dict[str, dict]:
    """query → row. retrieval 미시도(oracle) 행은 제외."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = {}
    for r in data.get("rows", []):
        q = r.get("query")
        if q is None:
            continue
        rows[q] = r
    return rows


def _rank(r: dict) -> Optional[int]:
    gr = r.get("gold_rank")
    return gr if isinstance(gr, int) else None


def _r_at_1(r):  return 1.0 if _rank(r) == 1 else 0.0
def _r_at_5(r):  rk = _rank(r); return 1.0 if rk and rk <= 5 else 0.0
def _rr(r):      rk = _rank(r); return (1.0 / rk) if rk else 0.0
def _ndcg(r):    rk = _rank(r); return (1.0 / math.log2(rk + 1)) if rk and rk <= 10 else 0.0
def _nm(r):      return 1.0 if r.get("correct") else 0.0


_METRICS = {
    "R@1": _r_at_1, "R@5": _r_at_5, "MRR": _rr, "nDCG@10": _ndcg, "NM": _nm,
}


def _paired_bootstrap(diffs: List[float], B: int = 5000, alpha: float = 0.05):
    n = len(diffs)
    if n == 0:
        return (0.0, 0.0, 0.0)
    rng = random.Random(0)
    means = []
    for _ in range(B):
        s = sum(diffs[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int(B * alpha / 2)]
    hi = means[int(B * (1 - alpha / 2))]
    return (sum(diffs) / n, lo, hi)


def _mcnemar(a_hits: List[float], b_hits: List[float]) -> dict:
    """B가 맞고 A가 틀린(b01) vs A가 맞고 B가 틀린(b10) 불일치쌍 exact binomial."""
    b01 = sum(1 for a, b in zip(a_hits, b_hits) if b > a)   # B win
    b10 = sum(1 for a, b in zip(a_hits, b_hits) if a > b)   # A win
    n = b01 + b10
    # two-sided exact binomial p (p=0.5)
    if n == 0:
        p = 1.0
    else:
        k = min(b01, b10)
        tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
        p = min(1.0, 2 * tail)
    return {"B_wins": b01, "A_wins": b10, "p_value": p}


def main():
    ap = argparse.ArgumentParser(description="두 결과 JSON 쌍대 비교")
    ap.add_argument("baseline", help="기준 런 JSON (예: VDB)")
    ap.add_argument("proposed", help="비교 런 JSON (예: structural)")
    ap.add_argument("-B", type=int, default=5000, help="bootstrap 반복수")
    args = ap.parse_args()

    A = _load_rows(args.baseline)
    Brows = _load_rows(args.proposed)
    shared = [q for q in A if q in Brows]

    print(f"A (baseline): {args.baseline}  ({len(A)} rows)")
    print(f"B (proposed): {args.proposed}  ({len(Brows)} rows)")
    print(f"공유 쿼리: {len(shared)}\n")
    if not shared:
        print("⚠ 공유 쿼리가 없음 — 같은 SEED/per-class로 돌렸는지 확인.")
        return

    print(f"{'Metric':9s}  {'A':>6s}  {'B':>6s}  {'Δ(B-A)':>8s}  {'95% CI':>18s}")
    print(f"{'─'*9}  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*18}")
    for name, fn in _METRICS.items():
        a_vals = [fn(A[q]) for q in shared]
        b_vals = [fn(Brows[q]) for q in shared]
        a_mean = sum(a_vals) / len(a_vals)
        b_mean = sum(b_vals) / len(b_vals)
        diffs = [b - a for a, b in zip(a_vals, b_vals)]
        mean_d, lo, hi = _paired_bootstrap(diffs, B=args.B)
        sig = "" if (lo <= 0 <= hi) else "  *"   # CI가 0을 안 포함하면 유의
        print(f"{name:9s}  {a_mean:6.3f}  {b_mean:6.3f}  {mean_d:+8.3f}  [{lo:+.3f}, {hi:+.3f}]{sig}")

    # McNemar (R@1, NM) — 이진 정오 쌍대 검정
    print()
    for name in ("R@1", "NM"):
        fn = _METRICS[name]
        a_hits = [fn(A[q]) for q in shared]
        b_hits = [fn(Brows[q]) for q in shared]
        m = _mcnemar(a_hits, b_hits)
        verdict = "유의(B≠A)" if m["p_value"] < 0.05 else "유의차 없음"
        print(f"McNemar {name:6s}: B_wins={m['B_wins']:3d}  A_wins={m['A_wins']:3d}  "
              f"p={m['p_value']:.4f}  → {verdict}")

    print("\n* = 95% paired-bootstrap CI가 0을 포함하지 않음 (차이 유의).")


if __name__ == "__main__":
    main()
