#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Phase 4d Task C -- final tally under the R1 scorer, hitab_lookup at n=189."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402
from phase4_summary import em                                        # noqa: E402
from scipy.stats import binomtest, norm as znorm                     # noqa: E402

POOLS = ["hitab_lookup", "hitab_arith", "aitqa", "rhb_fact", "rhb_num"]
POLS = ["P1_fixed_512", "P4_path_cell", "gold_cell"]
ZA, ZB = znorm.ppf(0.975), znorm.ppf(0.80)


def paired_boot(a, b, B=10000, seed=42):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    i = rng.integers(0, len(a), size=(B, len(a)))
    d = a[i].mean(1) - b[i].mean(1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> int:
    df = pd.DataFrame([json.loads(l) for l in
                       open("results/phase4/reader_records.jsonl")])
    df["is_correct"] = [em(p, g) for p, g in zip(df.pred_parsed, df.gold_answer)]

    L = ["# Phase 4 최종 집계", "",
         f"행 {len(df)}. `Qwen/Qwen2.5-7B-Instruct` 4-bit, temperature=0, seed=42, "
         "max_new_tokens=32, B_reader=4096 (greedy fill).", "",
         "채점: PREREGISTER 개정 5의 R1 포함 — 통화/퍼센트 기호 제거 → 천단위 콤마 "
         "제거 → 공백 축약·소문자화 → 순수 소수의 후행 0 제거 → **수치 정답에 한해 "
         "상대오차 <0.01 허용**. R2/R3/R4 미채택. 부호 정규화 없음. 다중 정답(원소 "
         "2개)은 전부 일치 요구.", "",
         "`hitab_lookup`만 n=189로 확장했다(개정 5). 나머지 pool은 필요 n이 pool "
         "크기를 넘어 확장하지 않았다.", "",
         "## pool x 조건별 EM", "",
         "| pool | dataset | " + " | ".join(POLS) + " |",
         "|---|---|" + "---|" * len(POLS)]
    for p in POOLS:
        d = df[df.pool == p]
        L.append(f"| {p} | {d.dataset.iloc[0]} | " + " | ".join(
            f"{d[d.policy == pol].is_correct.mean():.4f} (n={len(d[d.policy == pol])})"
            for pol in POLS) + " |")

    L += ["", "## McNemar 정확검정 (P4_path_cell vs P1_fixed_512, paired)", "",
          "| pool | n | P1 EM | P4 EM | b (P4만) | c (P1만) | delta | 95% CI | p | "
          "필요 n (power .80) | 검정력 |", "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|"]
    for p in POOLS:
        d = df[df.pool == p]
        a = d[d.policy == "P1_fixed_512"].set_index("query_id").is_correct
        b_ = d[d.policy == "P4_path_cell"].set_index("query_id").is_correct.reindex(a.index)
        b = int(((b_ == 1) & (a == 0)).sum())
        c = int(((b_ == 0) & (a == 1)).sum())
        n = len(a)
        lo, hi = paired_boot(b_.values, a.values)
        pv = binomtest(b, b + c, 0.5).pvalue if b + c else None
        pd_, pdiff = (b + c) / n, (b - c) / n
        if pdiff == 0:
            need, note = "정의 안 됨", "**부족** (b=c)"
        else:
            need = math.ceil((ZA * math.sqrt(pd_) + ZB
                              * math.sqrt(pd_ - pdiff ** 2)) ** 2 / pdiff ** 2)
            note = "충족" if n >= need else "**부족**"
            if b + c < 10:
                note += " (b+c<10, 추정 불안정)"
            need = str(need)
        L.append(f"| {p} | {n} | {a.mean():.4f} | {b_.mean():.4f} | {b} | {c} | "
                 f"{b_.mean()-a.mean():+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
                 f"{'n/a' if pv is None else (f'{pv:.2e}' if pv < 1e-4 else f'{pv:.4f}')} | {need} | {note} |")
    L += ["", "p는 다중비교 보정 전 값이다. 보정하지 않았다. paired bootstrap "
          "B=10000, seed=42. 필요 n은 Connor(1987) 정규근사, 관측 pi_d·pi_diff 고정, "
          "alpha=0.05 양측, power=0.80.", ""]

    L += ["## gold_cell 조건 = 리더 상한", "", "| pool | EM | n |", "|---|---:|---:|"]
    g = df[df.policy == "gold_cell"]
    for p in POOLS:
        d = g[g.pool == p]
        L.append(f"| {p} | {d.is_correct.mean():.4f} | {len(d)} |")
    L += [f"| **전체** | {g.is_correct.mean():.4f} | {len(g)} |", "",
          "`gold_cell`은 개정 4로 pool당 30건으로 축소했고(`hitab_lookup`은 개정 5로 "
          "189건 전량), `hitab_arith`의 31은 재배치 이전 실행분 1건이 포함된 결과다.", ""]

    L += ["## 검정력 부족 pool", "",
          "`hitab_lookup`을 뺀 네 pool 전부 현재 n이 필요 n에 못 미친다.", "",
          "| pool | 현재 n | 필요 n | pool 전체 크기 | 불일치쌍 b+c |",
          "|---|---:|---:|---:|---:|"]
    for p, size in (("hitab_arith", 175), ("aitqa", 451),
                    ("rhb_fact", 164), ("rhb_num", 54)):
        d = df[df.pool == p]
        a = d[d.policy == "P1_fixed_512"].set_index("query_id").is_correct
        b_ = d[d.policy == "P4_path_cell"].set_index("query_id").is_correct.reindex(a.index)
        b = int(((b_ == 1) & (a == 0)).sum())
        c = int(((b_ == 0) & (a == 1)).sum())
        n = len(a)
        pd_, pdiff = (b + c) / n, (b - c) / n
        need = math.ceil((ZA * math.sqrt(pd_) + ZB
                          * math.sqrt(pd_ - pdiff ** 2)) ** 2 / pdiff ** 2)
        L.append(f"| {p} | {n} | {need} | {size} | {b + c} |")
    L += ["", "네 pool 모두 필요 n이 pool 전체 크기를 넘어 현재 설계로는 충족할 수 "
          "없다. `hitab_arith`(b+c=3)와 `rhb_num`(b+c=8)은 불일치쌍이 10 미만이라 "
          "pi_d·pi_diff 추정 자체가 불안정하며, 필요 n 값도 그만큼 신뢰할 수 없다.", ""]
    Path("results/phase4/final_summary.md").write_text("\n".join(L))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
