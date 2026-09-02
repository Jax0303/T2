#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""dev와 test가 어디서 갈라지는지 — 표 검색인가 표 안 셀 선택인가.

`RESULTS.md` §13 을 만든 계측기. 리더를 부르지 않고 랭크 덤프만 읽는다.
**dev 와 test 는 독립 표본이다** (질의도 표도 겹치지 않는다). 짝지은 McNemar 를
쓸 수 없으므로 2x2 chi-square(연속성 보정)로 잰다 -- 이 저장소의 다른 판정들이
전부 페어드라 여기서 습관적으로 McNemar 를 쓰면 틀린다.

지표 이름은 `CLAUDE.md` §4.5 를 따른다 (`R@1` = 정답셀 1등률, `setEM@k` =
all-covered@k). 둘 다 리더를 안 부르는 검색 지표다.

  PYTHONPATH=. .venv/bin/python analysis/devtest_split.py
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from scipy.stats import chi2_contingency

TPL = ("results/p0_confirm/hitab_{}_lookup_all_S3c_page_hybrid0.8_tp0.0_ranks.jsonl")
TITLES = Path("results/tableconf/totto_page_titles.json")


def load(split, tpl):
    return [json.loads(l) for l in open(tpl.format(split))]


def rates(recs):
    n = len(recs)
    return {
        "정답셀 1등률": sum(1 for q in recs if max(q["ranks"]) < 1) / n,
        "표 1등률": sum(1 for q in recs if q["table_rank_cellvote"] == 1) / n,
        "오라클 표 게이팅@1": sum(1 for q in recs
                                 if max(q["ranks_in_table"]) < 1) / n,
        "all-covered@10": sum(1 for q in recs if max(q["ranks"]) < 10) / n,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tpl", default=TPL)
    a = ap.parse_args()

    d, t = load("dev", a.tpl), load("test", a.tpl)
    rd, rt = rates(d), rates(t)
    print(f"\n{'지표':>20}{'dev':>10}{'test':>10}{'Δ':>10}{'chi2 p':>11}"
          f"   (독립 표본)")
    for k in rd:
        na, nb = len(d), len(t)
        x, y = round(rd[k] * na), round(rt[k] * nb)
        _, p, _, _ = chi2_contingency([[x, na - x], [y, nb - y]], correction=True)
        print(f"{k:>20}{rd[k]:>10.4f}{rt[k]:>10.4f}{rt[k] - rd[k]:>+10.4f}"
              f"{p:>11.4g}")

    print("\n[표 안으로 내려온 실패] 표는 1등인데 셀은 1등이 아닌 질의 비율")
    for tag, recs in (("dev", d), ("test", t)):
        n = len(recs)
        v = sum(1 for q in recs
                if q["table_rank_cellvote"] == 1 and min(q["ranks"]) > 0) / n
        print(f"  {tag:>4} {v:.4f}")

    print("\n[배제된 후보 원인]")
    for tag, recs in (("dev", d), ("test", t)):
        g = [q["gold_table_cells"] for q in recs]
        print(f"  {tag:>4} gold 표의 셀 수 중앙 {st.median(g):.0f} "
              f"평균 {st.mean(g):.0f} 최대 {max(g)}")
    if TITLES.exists():
        pt = json.load(open(TITLES))
        for tag, recs in (("dev", d), ("test", t)):
            gt = [q["gold_table"] for q in recs]
            print(f"  {tag:>4} ToTTo 제목 보유 — 표 기준 "
                  f"{sum(1 for x in set(gt) if x in pt) / len(set(gt)):.4f} "
                  f"질의 기준 {sum(1 for x in gt if x in pt) / len(gt):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
