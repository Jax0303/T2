#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Does the gold operand set survive corpus construction intact?

Written from the raw dataset, not from any committed result. OSC is subset
containment over a gold cell set, so every number built on it inherits whatever
that set actually is. Two things in `corpus_dump_vs_cell.multihiertt_corpus`
can change it, and both are invisible once the run has finished:

  1. `gold_table = next(iter(gold))[0]` picks ONE table out of a set. When the
     gold cells span several tables that pick is arbitrary, so `gold_table_any`
     asks about a table chosen by set iteration order.
  2. `if t_idx in local` drops gold cells whose table failed to parse. OSC is
     then measured against a smaller set than the dataset annotated, and m is
     under-counted, without either being recorded.

Neither can bite at m=1. Both can at m>=2, which is exactly the regime where
OSC falls. This counts how often each happens.

  PYTHONPATH=.:scripts .venv/bin/python scripts/gold_set_integrity.py
"""
from __future__ import annotations

import sys
from collections import Counter

N_QUERIES, SEED = 400, 42


def main() -> int:
    from baseline_comparison_multihiertt import load_population, parse_doc

    pop, _ = load_population(N_QUERIES, SEED)
    print(f"MultiHiertt 원본 모집단: {len(pop)}건 (n={N_QUERIES}, seed={SEED})\n")

    unparseable_doc = 0
    span = Counter()          # 주석된 gold 셀이 걸친 표 수
    dropped_any = 0           # 파싱 실패로 gold 셀을 하나라도 잃은 질의
    dropped_all = 0
    m_before, m_after = [], []

    for q in pop:
        tabs = parse_doc(q)
        if tabs is None:
            unparseable_doc += 1
            continue
        ann = list(q["cells"])                       # 데이터셋이 준 gold
        kept = [c for c in ann if c[0] in tabs]      # 코퍼스가 실제로 쓰는 gold
        m_before.append(len({(t, r, c) for t, r, c in ann}))
        m_after.append(len({(t, r, c) for t, r, c in kept}))
        span[len({t for t, _, _ in ann})] += 1
        if len(kept) < len(ann):
            dropped_any += 1
            if not kept:
                dropped_all += 1

    n = len(m_before)
    print(f"문서 자체가 파싱 안 됨 (전 arm에서 제외): {unparseable_doc}건\n")

    print("=== 1. gold 셀이 몇 개의 표에 걸쳐 있나 ===")
    for k in sorted(span):
        flag = "  <-- gold_table 이 임의로 하나 선택됨" if k > 1 else ""
        print(f"  표 {k}개: {span[k]:>4}건 ({span[k] / n:.1%}){flag}")
    multi = sum(v for k, v in span.items() if k > 1)
    print(f"  => 여러 표에 걸친 질의 {multi}건 ({multi / n:.1%})"
          f" 에서 gold_table 은 의미 없는 값이다\n")

    print("=== 2. 파싱 실패로 버려진 gold 셀 ===")
    print(f"  gold 셀을 하나라도 잃은 질의: {dropped_any}건 ({dropped_any / n:.1%})")
    print(f"  gold 셀을 전부 잃은 질의:     {dropped_all}건 ({dropped_all / n:.1%})")
    lost = sum(a - b for a, b in zip(m_before, m_after))
    print(f"  주석된 gold 셀 총 {sum(m_before)}개 중 {lost}개 소실"
          f" ({lost / sum(m_before):.1%})\n")

    print("=== 3. m 이 얼마나 줄었나 (OSC 난이도가 낮아진다) ===")
    print(f"{'m(주석)':>8} {'질의수':>6} {'m(실제사용) 평균':>16}")
    by_m: dict[int, list[int]] = {}
    for a, b in zip(m_before, m_after):
        by_m.setdefault(a, []).append(b)
    for k in sorted(by_m):
        v = by_m[k]
        print(f"{k:>8} {len(v):>6} {sum(v) / len(v):>16.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
