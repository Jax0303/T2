#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PREREG-2026-09-02-totto-page-title-stage2.md 판정.

sel(756)에서 α를 고르고, matched α로 p0 vs a0 페어드 McNemar를 돌린다.
dev/test는 별도 확정 실행의 덤프를 읽는다. 사전등록이 고정한 것만 찍는다.

  python analysis/p0_verdict.py [sel|confirm]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from scipy.stats import binomtest

GRID = ["0.0", "0.3", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
A0 = Path("results/devbias")
P0 = Path("results/p0_sel")


def ranks(path):
    return {r["query_id"]: r for r in map(json.loads, open(path))}


def setem(recs, k=10):
    return sum(1 for r in recs.values() if max(r["ranks"]) < k) / len(recs)


def sel_file(d, model, alpha, page):
    # 두 덤프의 파일명 규약이 다르다: p0 는 title-mode 를 낀 새 규약,
    # a0 는 devbias 때의 모델명+알파 규약.
    if page:
        return d / (f"hitab_train_sel_lookup_all_S3c_page_"
                    f"hybrid{alpha}_tp0.0_ranks.jsonl")
    return d / (f"hitab_train_sel_lookup_all_S3c_"
                f"bge-base-cell-ft-{model}_a{alpha}_ranks.jsonl")


def mcnemar(a, b, k=10):
    """a가 맞고 b가 틀린 건 / 그 반대."""
    qs = sorted(set(a) & set(b))
    hit = lambda r: max(r["ranks"]) < k
    x = sum(1 for q in qs if hit(a[q]) and not hit(b[q]))
    y = sum(1 for q in qs if not hit(a[q]) and hit(b[q]))
    p = binomtest(min(x, y), x + y, 0.5).pvalue if x + y else 1.0
    return x, y, p, len(qs)


def main(mode="sel"):
    print(f"{'α':>6}{'a0 sel':>10}{'p0 sel':>10}")
    best, bs = None, -1.0
    for al in GRID:
        fa, fp = sel_file(A0, "a0", al, False), sel_file(P0, "p0", al, True)
        a = setem(ranks(fa)) if fa.exists() else float("nan")
        if not fp.exists():
            print(f"{al:>6}{a:>10.4f}{'--':>10}")
            continue
        p = setem(ranks(fp))
        print(f"{al:>6}{a:>10.4f}{p:>10.4f}")
        if p > bs:                       # 동률이면 먼저 본 것 = 작은 α (GRID 순)
            best, bs = al, p
    if best is None:
        print("\n아직 p0 덤프가 없다."); return 0
    print(f"\n선택 α = {best}  (sel setEM@10 = {bs:.4f})")

    # 주검정: matched α. a0 의 sel 최적은 1.0 (2차 핸드오프 §2).
    for al in sorted({best, "1.0"}):
        fa, fp = sel_file(A0, "a0", al, False), sel_file(P0, "p0", al, True)
        if not (fa.exists() and fp.exists()):
            continue
        x, y, p, n = mcnemar(ranks(fp), ranks(fa))
        tag = "선택 α" if al == best else "a0 최적 α"
        print(f"  [{tag}={al}] n={n}  p0만 맞음 {x}  a0만 맞음 {y}  p={p:.4g}"
              f"  {'유의' if p < .05 else '유의하지 않음'}")
    print("\n주검정이 유의하지 않으면 arm 을 바꾸지 않는다 (사전등록 기각 조건).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "sel"))
