#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Phase 4g -- rescore everything with the fixed gold_parts(), redo the tests,
and write the final artefact."""
from __future__ import annotations

import ast
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402
import phase4_summary as S                                           # noqa: E402
from scipy.stats import binomtest, norm as znorm                     # noqa: E402

POOLS = ["hitab_lookup", "hitab_arith", "aitqa", "rhb_fact", "rhb_num"]
POLS = ["P1_fixed_512", "P4_path_cell", "gold_cell"]
SIZE = {"hitab_lookup": 830, "hitab_arith": 175, "aitqa": 451,
        "rhb_fact": 164, "rhb_num": 54}
ZA, ZB, ALPHA = znorm.ppf(0.975), znorm.ppf(0.80), 0.05


def em_buggy(pred, gold, rel=True):
    """The pre-fix scorer, kept only to report the before/after delta."""
    try:
        v = ast.literal_eval(gold)
        gs = [str(x) for x in v] if isinstance(v, (list, tuple)) else [gold]
    except (ValueError, SyntaxError):
        gs = [gold]
    if len(gs) == 1:
        return int(S._same(pred, gs[0], rel))
    ps = list(str(pred).split(","))
    if len(ps) != len(gs):
        return 0
    left = list(ps)
    for g in gs:
        hit = next((x for x in left if S._same(x, g, rel)), None)
        if hit is None:
            return 0
        left.remove(hit)
    return int(not left)


def paired_boot(a, b, B=10000, seed=42):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    i = rng.integers(0, len(a), size=(B, len(a)))
    d = a[i].mean(1) - b[i].mean(1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def pair(d):
    p1 = d[d.policy == "P1_fixed_512"].set_index("query_id").is_correct
    p4 = d[d.policy == "P4_path_cell"].set_index("query_id").is_correct.reindex(p1.index)
    return p1, p4, int(((p4 == 1) & (p1 == 0)).sum()), int(((p4 == 0) & (p1 == 1)).sum())


def need_n(b, c, n):
    pd_, pdiff = (b + c) / n, (b - c) / n
    if pdiff == 0:
        return None
    return math.ceil((ZA * math.sqrt(pd_) + ZB
                      * math.sqrt(pd_ - pdiff ** 2)) ** 2 / pdiff ** 2)


def main() -> int:
    df = pd.DataFrame([json.loads(l) for l in
                       open("results/phase4/reader_records.jsonl")])
    df["old"] = [em_buggy(p, g) for p, g in zip(df.pred_parsed, df.gold_answer)]
    df["is_correct"] = [S.em(p, g) for p, g in zip(df.pred_parsed, df.gold_answer)]
    up = df[(df.old == 0) & (df.is_correct == 1)]
    dn = df[(df.old == 1) & (df.is_correct == 0)]

    L = ["# Phase 4 FINAL", "",
         f"행 {len(df)}. `Qwen/Qwen2.5-7B-Instruct` "
         "(rev a09a35458c702b33eeacc393d103063234e8bc28) 4-bit NF4, "
         "temperature=0, seed=42, max_new_tokens=32, B_reader=4096 Qwen 토큰 "
         "(greedy fill).", "",
         "채점 코드는 `BUGFIX_LOG.md`의 `gold_parts()` 수정을 반영한 상태다. "
         "규칙 변경이 아니라 사전등록 규칙(천단위 콤마 제거)이 실행되지 않던 "
         "구현 버그의 수정이다.", "",
         "## 0. 수정 전후 EM (pool x 조건)", "",
         "| pool | 조건 | n | 수정 전 EM | 수정 후 EM | 변경 |",
         "|---|---|---:|---:|---:|---:|"]
    for p in POOLS:
        for pol in POLS:
            d = df[(df.pool == p) & (df.policy == pol)]
            ch = int(((d.old == 0) & (d.is_correct == 1)).sum())
            L.append(f"| {p} | {pol} | {len(d)} | {d.old.mean():.4f} | "
                     f"{d.is_correct.mean():.4f} | {'+' + str(ch) if ch else '0'} |")
    L += ["", f"변경 행 수: **0→1 {len(up)}행, 1→0 {len(dn)}행** (전 {len(df)}행 중).", "",
          "| pool | 조건 | query_id | pred_parsed | gold_answer |", "|---|---|---|---|---|"]
    for r in up.sort_values(["pool", "policy", "query_id"]).itertuples():
        L.append(f"| {r.pool} | {r.policy} | {r.query_id} | `{r.pred_parsed}` "
                 f"| `{r.gold_answer}` |")

    L += ["", "## 1. pool x 조건별 EM", "",
          "| pool | dataset | " + " | ".join(POLS) + " |",
          "|---|---|" + "---|" * len(POLS)]
    for p in POOLS:
        d = df[df.pool == p]
        L.append(f"| {p} | {d.dataset.iloc[0]} | " + " | ".join(
            f"{d[d.policy == pol].is_correct.mean():.4f} "
            f"(n={len(d[d.policy == pol])})" for pol in POLS) + " |")
    L += ["", "`hitab_lookup`만 n=189로 확장했다(PREREGISTER 개정 5). 나머지 pool은 "
          "필요 n이 pool 크기를 넘어 확장하지 않았다. `gold_cell`은 개정 4로 pool당 "
          "30건으로 축소했고, `hitab_arith`의 31은 재배치 이전 실행분 1건이 포함된 "
          "결과다.", ""]

    # --- McNemar + Holm ---
    st = {}
    for p in POOLS:
        p1, p4, b, c = pair(df[df.pool == p])
        lo, hi = paired_boot(p4.values, p1.values)
        pv = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
        st[p] = dict(n=len(p1), e1=p1.mean(), e4=p4.mean(), b=b, c=c,
                     lo=lo, hi=hi, p=pv, need=need_n(b, c, len(p1)))
    order = sorted(POOLS, key=lambda p: st[p]["p"])
    prev = True
    for k, p in enumerate(order):
        thr = ALPHA / (len(POOLS) - k)
        prev = prev and (st[p]["p"] < thr)
        st[p]["rank"], st[p]["thr"], st[p]["rej"] = k + 1, thr, prev

    L += ["## 2. McNemar 정확검정 (P4_path_cell vs P1_fixed_512, paired) + "
          "Holm-Bonferroni", "",
          "| pool | n | P1 EM | P4 EM | b (P4만) | c (P1만) | delta | 95% CI | "
          "원 p | Holm 순위 | 임계값 | Holm 판정 | 필요 n | 검정력 |",
          "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---|"]
    for p in POOLS:
        s = st[p]
        need = "정의 안 됨" if s["need"] is None else str(s["need"])
        if s["need"] is None:
            pw = "**부족** (b=c)"
        else:
            pw = "충족" if s["n"] >= s["need"] else "**부족**"
            if s["b"] + s["c"] < 10:
                pw += " (b+c<10, 추정 불안정)"
        ps = f"{s['p']:.2e}" if s["p"] < 1e-4 else f"{s['p']:.4f}"
        L.append(f"| {p} | {s['n']} | {s['e1']:.4f} | {s['e4']:.4f} | {s['b']} | "
                 f"{s['c']} | {s['e4']-s['e1']:+.4f} | [{s['lo']:+.4f}, {s['hi']:+.4f}] "
                 f"| {ps} | {s['rank']} | {s['thr']:.6f} | "
                 f"{'기각 (유의)' if s['rej'] else '기각 실패'} | {need} | {pw} |")
    L += ["", "McNemar는 정확검정 `binomtest(b, b+c, 0.5)`, 연속성 보정 없음. "
          "delta의 CI는 paired bootstrap B=10000, seed=42. Holm-Bonferroni는 m=5, "
          "alpha=0.05, 순위 k의 임계값 alpha/(m-k+1); 순차 절차이므로 한 번 기각에 "
          "실패하면 이후 순위는 모두 기각 실패다. 필요 n은 Connor(1987) 정규근사, "
          "관측 pi_d·pi_diff 고정, alpha=0.05 양측, power=0.80.", ""]

    # --- subset stability ---
    base = {r["query_id"] for r in csv.DictReader(
        open("results/phase4/sample_294.csv")) if r["pool"] == "hitab_lookup"}
    ext = {r["query_id"] for r in csv.DictReader(
        open("results/phase4/sample_hitab_lookup_ext129.csv"))}
    L += ["## 3. hitab_lookup 부분집합 안정성", "",
          "| 부분집합 | n | P1 EM | P4 EM | delta | b | c | Recall P1 | Recall P4 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    keep = {}
    for name, ids, f in (("기존 60", base, "results/phase4/taskB.json"),
                         ("추가 129", ext,
                          "results/phase4/taskB_retrieval_ext129.json")):
        p1, p4, b, c = pair(df[(df.pool == "hitab_lookup") & (df.query_id.isin(ids))])
        keep[name] = p4.values - p1.values
        r = {e["policy"]: e["recall"] for e in json.load(open(f))["b2"]
             if e["pool"] == "hitab_lookup"}
        L.append(f"| {name} | {len(p1)} | {p1.mean():.4f} | {p4.mean():.4f} | "
                 f"{p4.mean()-p1.mean():+.4f} | {b} | {c} | {r['P1_fixed_512']:.4f} "
                 f"| {r['P4_path_cell']:.4f} |")
    da, db = keep["기존 60"], keep["추가 129"]
    diff = db.mean() - da.mean()
    se = math.sqrt(da.var(ddof=1) / len(da) + db.var(ddof=1) / len(db))
    L += ["", f"두 부분집합 delta 차이 (추가 129 − 기존 60): **{diff:+.4f}**, "
          f"SE={se:.4f}, 95% CI [{diff - ZA*se:+.4f}, {diff + ZA*se:+.4f}] "
          "(독립 두 표본, 쿼리별 차이값 d=P4−P1의 정규근사). Recall은 gold 셀 수를 "
          "분모로 한 greedy fill 컨텍스트 내 포함률이다.", ""]

    # --- ceiling / power ---
    g = df[df.policy == "gold_cell"]
    L += ["## 4. gold_cell 조건 = 리더 상한", "", "| pool | EM | n |", "|---|---:|---:|"]
    for p in POOLS:
        d = g[g.pool == p]
        L.append(f"| {p} | {d.is_correct.mean():.4f} | {len(d)} |")
    L += [f"| **전체** | {g.is_correct.mean():.4f} | {len(g)} |", "",
          "## 5. 검정력 부족 pool", "",
          "| pool | 현재 n | 필요 n | pool 전체 크기 | 불일치쌍 b+c |",
          "|---|---:|---:|---:|---:|"]
    short = [p for p in POOLS if st[p]["need"] is None or st[p]["n"] < st[p]["need"]]
    for p in short:
        s = st[p]
        L.append(f"| {p} | {s['n']} | "
                 f"{'정의 안 됨' if s['need'] is None else s['need']} | {SIZE[p]} | "
                 f"{s['b'] + s['c']} |")
    L += ["", "위 pool은 현재 n이 필요 n에 못 미친다. 필요 n이 pool 전체 크기를 "
          "넘는 경우 현재 설계로는 충족할 수 없다. b+c<10인 pool은 pi_d·pi_diff "
          "추정 자체가 불안정하며 필요 n 값도 그만큼 신뢰할 수 없다.", ""]

    # --- scoring rule verbatim ---
    src = Path("analysis/phase4_summary.py").read_text().splitlines()
    i0 = next(i for i, l in enumerate(src) if l.startswith("_CUR ="))
    i1 = next(i for i, l in enumerate(src) if l.startswith("def em("))
    i2 = next(i for i, l in enumerate(src[i1:], i1)
              if l.strip() == "return int(not left)")
    L += ["## 6. 채점 규칙 전문", "",
          "Exact Match. 정규화는 통화/퍼센트 기호 제거 → 천단위 콤마 제거 → 공백 "
          "축약·소문자화 → 순수 소수의 후행 0 제거. 그 위에 PREREGISTER 개정 5의 "
          "R1(수치 정답에 한해 상대오차 <0.01 허용)만 적용한다. R2/R3/R4 미채택, "
          "부호 정규화 없음. 다중 gold(원소 2개 이상)는 전부 일치를 요구한다.", "",
          "```python", *src[i0:i2 + 1], "```", "",
          "`gold_parts()`의 괄호 검사는 `BUGFIX_LOG.md`의 수정 사항이다.", "",
          "## 7. 참조", "",
          "- `BUGFIX_LOG.md` — `gold_parts()` 콤마 오파싱 수정",
          "- `PREREGISTER.md` — 개정 1~5",
          "- `results/phase4/reader_records.jsonl` (1156행), `reader_records.xlsx`",
          "- 수정 전 수치: `results/phase4/final_summary.md`, `phase4b.md`, "
          "`phase4c.md`, `phase4d_taskA.md`, `phase4e.md`", ""]

    Path("results/phase4/FINAL.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:200]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
