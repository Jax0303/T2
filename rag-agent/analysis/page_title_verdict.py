#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PREREG-2026-09-02-totto-page-title.md 1단계 판정.

사전등록이 고정한 것만 찍는다: totto 질의(86)와 그 외(744)로 갈라 R@1과 표 오인율,
그리고 주검정(page vs raw, totto 86건 페어드 exact McNemar).

  python analysis/page_title_verdict.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from scipy.stats import binomtest

D = Path("results/titlemode")
MAP = Path("data/table_meta/totto_page_titles.json")
MODES = ("raw", "page")


def load(mode, kind):
    tm = "" if mode == "raw" else f"_{mode}"
    suf = "tp0.0_ranks" if kind == "ranks" else "above"
    f = D / f"hitab_dev_lookup_all_S3c{tm}_hybrid1.0_{suf}.jsonl"
    return {r["query_id"]: r for r in map(json.loads, open(f))}


def main():
    tot_t = set(json.load(open(MAP)))
    R = {m: load(m, "ranks") for m in MODES}
    A = {m: load(m, "above") for m in MODES}
    qs = sorted(R["raw"])

    def wrong_table(m, q):
        r = A[m].get(q)
        return bool(r and r["top_above"]
                    and r["top_above"][0]["table_id"] != r["gold_table"])

    grp = {q: ("totto" if R["raw"][q]["gold_table"] in tot_t else "그 외")
           for q in qs}
    print(f"{'':<7}{'층':<8}{'n':>5}{'R@1':>9}{'표오인':>9}")
    for m in MODES:
        for g in ("totto", "그 외"):
            sel = [q for q in qs if grp[q] == g]
            r1 = sum(min(R[m][q]["ranks"]) == 0 for q in sel) / len(sel)
            wt = sum(wrong_table(m, q) for q in sel) / len(sel)
            print(f"{m:<7}{g:<8}{len(sel):>5}{r1:>9.4f}{wt:>9.4f}")
        allr1 = sum(min(R[m][q]["ranks"]) == 0 for q in qs) / len(qs)
        print(f"{m:<7}{'전체':<8}{len(qs):>5}{allr1:>9.4f}\n")

    print("주검정 — totto 질의 86건, 표 오인 여부 페어드 exact McNemar")
    sel = [q for q in qs if grp[q] == "totto"]
    b = sum(1 for q in sel if wrong_table("raw", q) and not wrong_table("page", q))
    c = sum(1 for q in sel if not wrong_table("raw", q) and wrong_table("page", q))
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    print(f"  고쳐짐 {b}  망가짐 {c}  p={p:.4g}  "
          f"{'유의 -> 통과' if p < .05 and b > c else '유의하지 않음 -> 기각'}")

    print("\n부검정 — totto 질의 R@1 페어드")
    b = sum(1 for q in sel if min(R["raw"][q]["ranks"]) > 0
            and min(R["page"][q]["ranks"]) == 0)
    c = sum(1 for q in sel if min(R["raw"][q]["ranks"]) == 0
            and min(R["page"][q]["ranks"]) > 0)
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    print(f"  고쳐짐 {b}  망가짐 {c}  p={p:.4g}")
    print("\n버그 점검: '그 외' 744건이 ±.02 안이어야 한다 (±.01 예측).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
