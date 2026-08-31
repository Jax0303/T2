#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Phase 4c -- would a looser arithmetic scorer change anything, and how many
pairs would McNemar need. Diagnostic only: no rule is applied to the records."""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "analysis"))

import pandas as pd                                                  # noqa: E402
from phase4_summary import em, gold_parts, norm_em                   # noqa: E402
from scipy.stats import norm as znorm                                # noqa: E402

POOLS = ["hitab_lookup", "hitab_arith", "aitqa", "rhb_fact", "rhb_num"]


def nums(s):
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", str(s))]


def rule_ok(pred, gold, rel=False, pct=False, absv=False):
    g, p = nums(gold), nums(pred)
    if len(g) != 1 or len(p) != 1:
        return False
    g, p = g[0], p[0]
    cands = [p]
    if pct and "%" in str(pred):
        cands.append(p / 100)
    for c in cands:
        for a, b in ([(abs(c), abs(g))] if absv else []) + [(c, g)]:
            if rel:
                if b != 0 and abs(a - b) / abs(b) < 0.01:
                    return True
            elif a == b:
                return True
    return False


def main() -> int:
    df = pd.DataFrame([json.loads(l) for l in
                       open("results/phase4/reader_records.jsonl")])
    df["is_correct"] = [em(p, g) for p, g in zip(df.pred_parsed, df.gold_answer)]
    d = df[(df.pool == "hitab_arith") & (df.policy == "gold_cell")]
    bad = d[d.is_correct == 0]

    print("## Task A -- 산술 채점 규칙 진단 (적용하지 않음, 진단만)\n")
    print(f"대상: hitab_arith / gold_cell 오답 {len(bad)}건 (전체 {len(d)}건)\n")
    rules = {"R1 상대오차<0.01": dict(rel=True),
             "R2 퍼센트 통일": dict(pct=True),
             "R3 R1+R2": dict(rel=True, pct=True),
             "R4 R3+절댓값": dict(rel=True, pct=True, absv=True)}
    print("| 규칙 | 정답으로 바뀌는 건수 | 바뀐 뒤 gold_cell EM |")
    print("|---|---:|---:|")
    flips = {}
    for name, kw in rules.items():
        f = [r.query_id for r in bad.itertuples()
             if rule_ok(r.pred_parsed, r.gold_answer, **kw)]
        flips[name] = f
        print(f"| {name} | {len(f)} | "
              f"{(len(d) - len(bad) + len(f)) / len(d):.4f} |")
    print()
    for name, f in flips.items():
        print(f"- **{name}** ({len(f)}건): {', '.join(x[:8] for x in f) or '없음'}")

    print("\n### HiTab 원본 answer 필드 (오답 24건 중 5건, 원문 그대로)\n")
    raw = {}
    for line in open("data/hitab/data/dev_samples.jsonl"):
        j = json.loads(line)
        raw[j["id"]] = j
    print("| query_id | 원본 answer | aggregation | answer_formulas | 리더 출력 |")
    print("|---|---|---|---|---|")
    for r in list(bad.itertuples())[:5]:
        j = raw.get(r.query_id, {})
        print(f"| {r.query_id[:8]} | `{j.get('answer')}` | `{j.get('aggregation')}` "
              f"| `{j.get('answer_formulas')}` | `{r.pred_answer_raw}` |")

    print("\n\n## Task B -- McNemar 검정력 분석\n")
    print("Connor (1987) 정규근사 표본수 공식. 불일치 비율 pi_d=(b+c)/n 과 "
          "차이 pi_diff=(b-c)/n 을 현재 관측값으로 고정하고\n")
    print("    n = [ z_(a/2)*sqrt(pi_d) + z_b*sqrt(pi_d - pi_diff^2) ]^2 / pi_diff^2\n")
    print("alpha=0.05 (양측), power=0.80. z_(a/2)=1.959964, z_b=0.841621.\n")
    za, zb = znorm.ppf(0.975), znorm.ppf(0.80)
    print("| pool | 현재 n | b | c | pi_d | pi_diff | 필요 n | 필요 불일치쌍 | 비고 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for pool in POOLS:
        s = df[df.pool == pool]
        p1 = s[s.policy == "P1_fixed_512"].set_index("query_id").is_correct
        p4 = s[s.policy == "P4_path_cell"].set_index("query_id").is_correct
        p4 = p4.reindex(p1.index)
        b = int(((p4 == 1) & (p1 == 0)).sum())
        c = int(((p4 == 0) & (p1 == 1)).sum())
        n = len(p1)
        pd_, pdiff = (b + c) / n, (b - c) / n
        if pdiff == 0:
            print(f"| {pool} | {n} | {b} | {c} | {pd_:.4f} | {pdiff:+.4f} | - | - | "
                  f"b=c 이므로 pi_diff=0, 공식이 정의되지 않는다 |")
            continue
        need = (za * math.sqrt(pd_) + zb * math.sqrt(pd_ - pdiff ** 2)) ** 2 / pdiff ** 2
        note = ("b+c<10 이라 관측 비율이 불안정하다" if b + c < 10 else "")
        print(f"| {pool} | {n} | {b} | {c} | {pd_:.4f} | {pdiff:+.4f} | "
              f"{math.ceil(need)} | {math.ceil(need * pd_)} | {note} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
