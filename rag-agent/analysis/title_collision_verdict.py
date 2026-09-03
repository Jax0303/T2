#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PREREG-2026-09-02-title-collision.md 1단계 판정.

세 모드(raw/drop/sig)의 랭크 덤프를 읽어, 사전등록이 고정한 것만 찍는다:
R@1, 제목 겹침/고유 질의의 표 오인율, 그리고 주검정(sig vs raw, 겹침 75건 페어드).

  python analysis/title_collision_verdict.py [모델태그]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from scipy.stats import binomtest

sys.path[:0] = ["scripts", "analysis", "."]
from header_path_coverage import load_corpus                        # noqa: E402

D = Path("results/titlemode")
MODES = ("raw", "drop", "sig")


def load(mode, alpha=1.0):
    tm = "" if mode == "raw" else f"_{mode}"
    f = D / f"hitab_dev_lookup_all_S3c{tm}_hybrid{alpha}_tp0.0_ranks.jsonl"
    return {r["query_id"]: r for r in map(json.loads, open(f))}


def load_above(mode, alpha=1.0):
    tm = "" if mode == "raw" else f"_{mode}"
    f = D / f"hitab_dev_lookup_all_S3c{tm}_hybrid{alpha}_above.jsonl"
    return {r["query_id"]: r for r in map(json.loads, open(f))}


def main():
    a = argparse.Namespace(dataset="hitab", split="dev", data_dir="data/hitab",
                           population="hitab_dev_lookup_all", cell_scheme="S3c",
                           seed=42, max_queries=0, mh_queries=400,
                           rhb_question_types=[], rhb_em_only=False,
                           cache_dir=".cache/corpus_dump_vs_cell")
    C = load_corpus(a)
    tc = Counter(C.title.get(t, "").strip().lower() for t in C.tids)
    dup = {t for t in C.tids if tc[C.title.get(t, "").strip().lower()] > 1}

    R = {m: load(m) for m in MODES}
    AB = {m: load_above(m) for m in MODES}
    qs = sorted(R["raw"])
    grp = {q: ("겹침" if R["raw"][q]["gold_table"] in dup else "고유") for q in qs}

    # 표 오인 = 1등 칸이 정답 표 밖. results/tableconf/VERDICT.md 의 진단과 같은
    # 정의여야 한다. ranks 파일의 above_same_table==0 은 "정답 셀 위 모든 칸이
    # 다른 표"라 더 엄격하고 과소 계수한다 -- 쓰지 말 것.
    def wrong_table(m, q):
        r = AB[m].get(q)
        return bool(r and r["top_above"]
                    and r["top_above"][0]["table_id"] != r["gold_table"])

    print(f"{'':<8}{'R@1':>9}{'표오인 겹침(75)':>16}{'표오인 고유(755)':>17}")
    for m in MODES:
        r1 = sum(min(R[m][q]["ranks"]) == 0 for q in qs) / len(qs)
        w = {g: [wrong_table(m, q) for q in qs if grp[q] == g]
             for g in ("겹침", "고유")}
        print(f"{m:<8}{r1:>9.4f}"
              f"{sum(w['겹침']) / len(w['겹침']):>16.4f}"
              f"{sum(w['고유']) / len(w['고유']):>17.4f}")

    print("\n주검정 — 제목 겹침 질의, 표 오인 여부 페어드 exact McNemar")
    dupq = [q for q in qs if grp[q] == "겹침"]
    for m in ("drop", "sig"):
        b = sum(1 for q in dupq if wrong_table("raw", q)
                and not wrong_table(m, q))
        c = sum(1 for q in dupq if not wrong_table("raw", q)
                and wrong_table(m, q))
        p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
        print(f"  raw -> {m:<5} 고쳐짐 {b:>3}  망가짐 {c:>3}  p={p:.4g}"
              f"  {'유의' if p < .05 else '유의하지 않음'}")
    b = sum(1 for q in dupq if wrong_table("drop", q)
            and not wrong_table("sig", q))
    c = sum(1 for q in dupq if not wrong_table("drop", q)
            and wrong_table("sig", q))
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    print(f"  drop -> sig  고쳐짐 {b:>3}  망가짐 {c:>3}  p={p:.4g}")
    print("\n버그 점검: 고유 제목(755) 표 오인율이 세 모드에서 ±.01 안이어야 한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
