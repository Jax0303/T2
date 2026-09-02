#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""실험 1 (GAPCHECK-2026-09-02-cell-rerank.md §5) — 표 내부 셀 오인의 원인이
계층 구조인가.

정답 셀을 밀어낸 1등 셀이 정답 셀과 헤더 경로 상 어떤 관계인지 센다. 관계 이름은
`cell_rank_dump.above_detail` 이 매긴 것을 그대로 쓴다.

**기저율 없이는 아무 말도 못 한다.** 계층형 표에서는 표 전체 칸의 70%가 넓은 의미의
형제(경로 한 마디 차)라서, "1등이 형제였다 92%"는 그것만으로는 우연과 구별되지 않는다.
그래서 관계마다 **그 질의의 정답 표 안에서의 기저율**을 같이 재고 농축 배수를 낸다.

입력: `cell_rank_dump.py --dump-above N` 이 쓴 `*_above.jsonl`.

  python analysis/sibling_distance.py [above.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path[:0] = ["scripts", "analysis", "."]
from header_path_coverage import load_corpus                        # noqa: E402

DEF = "results/sibling/hitab_dev_lookup_all_S3c_hybrid1.0_above.jsonl"
ORDER = ["same_row", "same_col", "same_address", "sibling_elsewhere",
         "same_table_far"]


def relation(rp, cp, grp, gcp):
    """above_detail 과 같은 판정. 표가 같다는 것은 호출부가 보장한다."""
    rp, cp = tuple(rp), tuple(cp)
    if rp == grp and cp == gcp:
        return "same_address"
    if rp == grp:
        return "same_row"
    if cp == gcp:
        return "same_col"
    if list(rp)[:-1] == list(grp)[:-1] or list(cp)[:-1] == list(gcp)[:-1]:
        return "sibling_elsewhere"
    return "same_table_far"


def main(path=DEF):
    a = argparse.Namespace(dataset="hitab", split="dev", data_dir="data/hitab",
                           population="hitab_dev_lookup_all", cell_scheme="S3c",
                           seed=42, max_queries=0, mh_queries=400,
                           rhb_question_types=[], rhb_em_only=False,
                           cache_dir=".cache/corpus_dump_vs_cell")
    C = load_corpus(a)
    by_t = {}
    for n, (t, _i, _j) in enumerate(C.cell_owner):
        by_t.setdefault(t, []).append(n)

    recs = [json.loads(l) for l in open(path)]
    it = [r for r in recs if r["top_above"]
          and r["top_above"][0]["table_id"] == r["gold_table"]]
    n = len(it)
    # 산술 통제: 이 분해가 랭크 덤프의 실패 수와 맞아야 한다.
    assert len(recs) == 264 and n == 156, (len(recs), n)
    print(f"정답 셀이 1등이 아닌 질의 {len(recs)}건 "
          f"= 표 오인 {len(recs) - n} + 표 내부 셀 오인 {n}\n")

    top = Counter(r["top_above"][0]["relation"] for r in it)
    base = Counter()
    for r in it:
        grp, gcp = tuple(r["gold_row_path"]), tuple(r["gold_col_path"])
        cells = by_t[r["gold_table"]]
        c = Counter(relation(*C.cell_paths[p][:2], grp, gcp) for p in cells)
        for k, v in c.items():
            base[k] += v / len(cells)

    print(f"{'관계':<20}{'1등에서':>10}{'표 기저율':>11}{'농축':>9}")
    for k in ORDER:
        t, b = top[k] / n, base[k] / n
        print(f"{k:<20}{t:>10.4f}{b:>11.4f}"
              f"{(f'{t / b:.2f}x' if b else '-'):>9}")
    t = (top["same_row"] + top["same_col"]) / n
    b = (base["same_row"] + base["same_col"]) / n
    print(f"{'행/열 경로 정확일치':<17}{t:>10.4f}{b:>11.4f}{t / b:>8.2f}x")
    med = sorted(len(by_t[r["gold_table"]]) for r in it)[n // 2]
    print(f"\n정답 표 중앙 칸수 {med}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else DEF))
