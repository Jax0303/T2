#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""R① 판정 — 기성품 크로스인코더 재정렬이 오라클 상한 중 얼마를 가져오는가.

사전등록 `PREREG-2026-09-03-ce-rerank-r1.md`. 새 실행(`rr_dev.jsonl`)과 기존
오라클 실행(`p0_dev_all.jsonl`)을 query_id 로 짝지어 읽는다. 두 파일은 인코더·α·
title-mode·리더·디코딩·채점기가 전부 같고, 새 파일의 `top1` 은 기존 `top1` 과
프롬프트가 글자 단위로 같으므로 **결정성 통제**가 된다.

  PYTHONPATH=. .venv/bin/python analysis/rerank_verdict.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from scipy.stats import binomtest                                    # noqa: E402
from phase4_summary import em, em_lenient                            # noqa: E402


def load(path):
    """(cond, query_id) -> record. 같은 키가 두 번 나오면 마지막이 이긴다."""
    out = {}
    for line in open(path):
        r = json.loads(line)
        r["ok"] = int(em(r["pred_parsed"], r["gold_answer"]))
        r["ok_lenient"] = int(em_lenient(r["pred_parsed"], r["gold_answer"]))
        out[(r["cond"], r["query_id"])] = r
    return out


def mcnemar(recs, a, b, qids):
    """a 만 맞음 : b 만 맞음, 양측 exact McNemar."""
    x = sum(1 for q in qids if recs[(a, q)]["ok"] and not recs[(b, q)]["ok"])
    y = sum(1 for q in qids if recs[(b, q)]["ok"] and not recs[(a, q)]["ok"])
    p = binomtest(min(x, y), x + y, 0.5).pvalue if x + y else 1.0
    return x, y, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rr", default="results/rerank/rr_dev.jsonl")
    ap.add_argument("--orc", default="results/answer_ret/p0_dev_all.jsonl")
    a = ap.parse_args()

    rr, orc = load(a.rr), load(a.orc)
    recs = orc | rr                      # 겹치는 top1 은 새 파일 쪽을 쓴다
    qids = sorted({q for _c, q in orc})
    conds = [c for c in ("gold", "orc50", "orc10", "rr10_3", "top10", "top3",
                         "rr10", "top1") if all((c, q) in recs for q in qids)]

    # --- 통제 1: 결정성. 같은 프롬프트, 같은 seed, 두 번 실행.
    both = [q for q in qids if ("top1", q) in rr and ("top1", q) in orc]
    dis = [q for q in both if rr[("top1", q)]["pred_parsed"]
           != orc[("top1", q)]["pred_parsed"]]
    fx = sum(1 for q in both if rr[("top1", q)]["ok"] > orc[("top1", q)]["ok"])
    fy = sum(1 for q in both if rr[("top1", q)]["ok"] < orc[("top1", q)]["ok"])
    pf = binomtest(min(fx, fy), fx + fy, 0.5).pvalue if fx + fy else 1.0
    print(f"\n[통제 1] 결정성 — top1 재실행 vs 원본, n={len(both)}")
    print(f"  예측 문자열 불일치 {len(dis)}/{len(both)}   "
          f"채점 뒤집힘 {fx}:{fy}, p={pf:.4g}   "
          f"{'OK' if len(dis) <= 5 and pf > .05 else '❌ 교란'}")

    # --- 통제 2: 산술. 고정 풀을 다시 정렬해도 멤버십은 안 바뀐다.
    K = 10
    key = f"rr{K}_gold_rank"
    have = [q for q in qids if key in rr.get(("rr10", q), {})]
    pre = {q: min(orc[("top1", q)]["gold_ranks"]) for q in have}
    post = {q: rr[("rr10", q)][key] for q in have}
    n = len(have)
    print(f"\n[통제 2] 고정 풀 — 재정렬 전후 hit@{K}, n={n}")
    b4 = sum(1 for q in have if pre[q] < K) / n
    af = sum(1 for q in have if post[q] < K) / n
    print(f"  전 {b4:.4f}  후 {af:.4f}   "
          f"{'OK' if abs(b4 - af) < 1e-9 else '❌ 멤버십이 바뀌었다'}")

    print(f"\n[검색] 재정렬 전 → 후 hit@k  (풀 = 검색 top{K}, n={n})")
    print(f"{'k':>4}{'전':>9}{'후':>9}{'Δ':>9}")
    for k in (1, 2, 3, 5, 10):
        x = sum(1 for q in have if pre[q] < k) / n
        y = sum(1 for q in have if post[q] < k) / n
        print(f"{k:>4}{x:>9.4f}{y:>9.4f}{y - x:>+9.4f}")

    # --- 재정렬기가 무엇을 고치고 무엇을 부수는가 (리더 없이 읽힌다)
    up = [q for q in have if pre[q] > 0 and post[q] == 0]
    dn = [q for q in have if pre[q] == 0 and post[q] > 0]
    pm = binomtest(min(len(up), len(dn)), len(up) + len(dn), 0.5).pvalue \
        if up or dn else 1.0
    mrr = lambda d: sum(1 / (d[q] + 1) for q in have) / n
    print(f"\n[순위] 1등 승격 {len(up)} : 강등 {len(dn)}, exact McNemar p={pm:.4g}")
    print(f"  MRR(풀 안) 전 {mrr(pre):.4f} → 후 {mrr(post):.4f}   "
          f"무작위 1등 기준선 {b4 / K:.4f}")

    cs = Path("results/colsig/col_signal_a0.8.json")
    if cs.exists():
        kind = {r["query_id"]: r["kind"] for r in json.loads(cs.read_text())["rows"]}
        print(f"\n[왜] 강등된 {len(dn)}건이 원래 어떤 질의였나 / colsig 분류는 "
              f"표 안 실패 {len(kind)}건에만 있다")
        seen = Counter(kind.get(q, "표 안 실패 아님 (1등이었다)") for q in dn)
        for k, v in seen.most_common():
            print(f"  {v:>4}  {k}")
        fix = [q for q in have if kind.get(q, "").startswith("질문이 정답")]
        if fix:
            print(f"  참고: colsig 가 '고칠 수 있다'고 분류한 {len(fix)}건 중 "
                  f"재정렬이 1등으로 올린 것 "
                  f"{sum(1 for q in fix if post[q] == 0)}건 "
                  f"(재정렬 전 {sum(1 for q in fix if pre[q] == 0)}건)")

    print(f"\n[답변] n={len(qids)}")
    print(f"{'cond':>8}{'주입':>6}{'EM':>9}{'EMlenient':>11}"
          f"{'gold주입률':>11}{'EM|주입됨':>11}{'ptok중앙':>10}")
    for c in conds:
        rs = [recs[(c, q)] for q in qids]
        f = [r for r in rs if r["gold_in_ctx"] == r["m"]]
        g = lambda s, k="ok": (sum(r[k] for r in s) / len(s)) if s else float("nan")
        pt = sorted(r["prompt_tokens"] for r in rs)[len(rs) // 2]
        ni = sorted(r["n_injected"] for r in rs)[len(rs) // 2]
        print(f"{c:>8}{ni:>6}{g(rs):>9.4f}{g(rs, 'ok_lenient'):>11.4f}"
              f"{len(f)/len(rs):>11.4f}{g(f):>11.4f}{pt:>10}")

    pairs = [("rr10_3", "top10", "**주검정**"), ("rr10", "top1", "순수 순위"),
             ("rr10", "top10", ""), ("top3", "top10", "자르기 단독"),
             ("rr10_3", "top3", "재정렬 단독"), ("rr10_3", "orc10", "상한까지")]
    print("\n[페어드 exact McNemar]")
    print(f"{'비교':>18}{'앞만':>7}{'뒤만':>7}{'p':>12}   비고")
    for x, y, note in pairs:
        if x not in conds or y not in conds:
            continue
        bb, cc, p = mcnemar(recs, x, y, qids)
        print(f"{x + ' vs ' + y:>18}{bb:>7}{cc:>7}{p:>12.4g}   {note}")

    if all(c in conds for c in ("rr10_3", "top10", "orc10")):
        e = lambda c: sum(recs[(c, q)]["ok"] for q in qids) / len(qids)
        gap = e("orc10") - e("top10")
        for c in ("rr10", "rr10_3", "top3"):
            if c in conds:
                print(f"  {c:>7}: top10 대비 {e(c) - e('top10'):+.4f}  "
                      f"= 오라클 격차({gap:+.4f})의 {(e(c) - e('top10')) / gap:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
