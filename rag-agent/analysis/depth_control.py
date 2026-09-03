#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""실험 2 (싼 판) — 표 내부 셀 오인이 **계층 때문인가**.

실험 1은 원인이 "한 축의 헤더 경로 정확일치"임을 보였으나 계층 특이성은 보이지 않았다.
평면 표 코퍼스를 새로 붙이는 대신, HiTab 안에서 **헤더 깊이로 층을 갈라** 본다.
행 경로 깊이 1 · 열 경로 깊이 1인 칸은 구조적으로 평범한 표의 칸과 같다.

깊이가 깊을수록 표 내부 오인이 늘면 원인은 계층이다. 평평하면 계층이 아니다.

**이것은 교차 데이터셋 대조가 아니다.** HiTab 안의 얕은 칸은 여전히 계층형 표 안에
있고, 코퍼스 전체가 그 표들로 채워져 있다. 진짜 평면 코퍼스(WTQ 등) 대조는 별도로
남는다. GAPCHECK §5 실험 2 참조.

  python analysis/depth_control.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict

sys.path[:0] = ["scripts", "analysis", "."]
from header_path_coverage import load_corpus                        # noqa: E402

RANKS = "results/sibling/hitab_dev_lookup_all_S3c_hybrid1.0_tp0.0_ranks.jsonl"
ABOVE = "results/sibling/hitab_dev_lookup_all_S3c_hybrid1.0_above.jsonl"


def main():
    a = argparse.Namespace(dataset="hitab", split="dev", data_dir="data/hitab",
                           population="hitab_dev_lookup_all", cell_scheme="S3c",
                           seed=42, max_queries=0, mh_queries=400,
                           rhb_question_types=[], rhb_em_only=False,
                           cache_dir=".cache/corpus_dump_vs_cell")
    C = load_corpus(a)
    pos_of = {c: n for n, c in enumerate(C.cell_owner)}
    depth = {}
    for q in C.queries:
        g = sorted(q["gold_cells"])[0]
        if g in pos_of:
            rp, cp, _v = C.cell_paths[pos_of[g]]
            depth[q["query_id"]] = (len(rp), len(cp))

    hit1 = {json.loads(l)["query_id"]: min(json.loads(l)["ranks"]) == 0
            for l in open(RANKS)}
    intab = {json.loads(l)["query_id"] for l in open(ABOVE)
             if json.loads(l)["top_above"]
             and json.loads(l)["top_above"][0]["table_id"]
             == json.loads(l)["gold_table"]}

    # 층: 평면 = 두 축 다 깊이 1. 나머지는 최대 깊이로 묶는다.
    def stratum(d):
        return "평면 (1,1)" if d == (1, 1) else f"깊이 {max(d)}"

    by = defaultdict(list)
    for qid, d in depth.items():
        if qid in hit1:
            by[stratum(d)].append(qid)

    print(f"{'층':<12}{'n':>6}{'R@1':>9}{'표내부오인율':>13}{'표오인율':>11}")
    rows = []
    for k in sorted(by, key=lambda s: (s != "평면 (1,1)", s)):
        qs = by[k]
        n = len(qs)
        r1 = sum(hit1[q] for q in qs) / n
        it = sum(1 for q in qs if q in intab) / n
        ot = sum(1 for q in qs if not hit1[q] and q not in intab) / n
        rows.append((k, n, r1, it, ot))
        print(f"{k:<12}{n:>6}{r1:>9.4f}{it:>13.4f}{ot:>11.4f}")
    tot = sum(r[1] for r in rows)
    assert tot == 830, tot
    print(f"{'합계':<12}{tot:>6}")
    print("\n표내부오인율 = 1등이 정답 표 안의 다른 칸. 표오인율 = 1등이 다른 표의 칸.")
    print("깊이별로 표내부오인율이 오르면 원인이 계층이다.")
    print(f"\n깊이 조합 분포 (전체 830): "
          f"{dict(Counter(depth[q] for q in hit1).most_common(6))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
