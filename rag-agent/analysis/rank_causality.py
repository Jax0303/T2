#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PREREG-2026-09-09-rank-causality.md 판정.

셀 집합을 고정하고 gold 의 자리만 옮겼을 때 리더가 움직이는가.
판정 규칙은 사전등록에 박혀 있다 — 여기서는 그것을 계산만 한다.

  PYTHONPATH=. .venv/bin/python analysis/rank_causality.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.answer_stats import (bootstrap_ci, holm,                 # noqa: E402
                                   mcnemar_exact, paired_bootstrap)

D = ROOT / "results/rankcause"
ARMS = ["asis", "promote", "shuffle"]
BASELINE = 0.6375          # 사전등록에 박은 표 2b 상의 이 331건 EM
DRIFT_TOL = 0.02


def main() -> int:
    got = {a: {json.loads(l)["query_id"]: json.loads(l)["answer_correct"]
               for l in (D / f"rank2plus_{a}_answer_retrieved.jsonl").open()}
           for a in ARMS if (D / f"rank2plus_{a}_answer_retrieved.jsonl").exists()}
    missing = [a for a in ARMS if a not in got]
    if missing:
        raise SystemExit(f"아직 안 돌았다: {missing}")
    ids = sorted(set.intersection(*(set(v) for v in got.values())))
    for a in ARMS:
        assert len(got[a]) == len(ids), f"{a}: 짝지음이 깨졌다 {len(got[a])} != {len(ids)}"
    v = {a: [got[a][q] for q in ids] for a in ARMS}
    pos = {json.loads(l)["query_id"]: json.loads(l)["gold_pos_orig"]
           for l in (D / "rank2plus_asis_records.jsonl").open()}

    print(f"n = {len(ids)}  (주지표·검색성공·gold 가 2번 자리 이후)\n")
    for a in ARMS:
        lo, hi = bootstrap_ci(v[a])
        print(f"  {a:8s} EM = {sum(v[a])/len(ids):.4f}   [{lo:.4f}, {hi:.4f}]  "
              f"({sum(v[a])}/{len(ids)})")

    drift = abs(sum(v["asis"]) / len(ids) - BASELINE)
    n01, n10, p_drift = mcnemar_exact(
        [1] * round(BASELINE * len(ids)) + [0] * (len(ids) - round(BASELINE * len(ids))),
        v["asis"])
    print(f"\n1. 드리프트 검사 — asis {sum(v['asis'])/len(ids):.4f} 대 사전등록 기준 "
          f"{BASELINE:.4f}, 차이 {drift:.4f} (허용 {DRIFT_TOL})")
    ok = drift <= DRIFT_TOL
    print(f"   {'통과' if ok else '실패 — 나머지 arm 을 해석하지 않는다'}")

    print("\n2. 주 검정 — McNemar exact, Holm 보정")
    raw = {}
    for a in ("promote", "shuffle"):
        n01, n10, p = mcnemar_exact(v["asis"], v[a])
        raw[a] = p
        d = (sum(v[a]) - sum(v["asis"])) / len(ids)
        bs = paired_bootstrap(v["asis"], v[a])
        print(f"   {a:8s} vs asis: Δ={d:+.4f}  [{bs['ci_lo']:+.4f}, {bs['ci_hi']:+.4f}]  "
              f"바뀐 판정 {n01}:{n10}  p={p:.4g}")
    for a, p in holm(raw).items():
        print(f"   {a:8s} Holm p = {p:.4g}")

    print("\n3. 원래 자리별 (promote 가 자리를 실제로 옮긴 폭)")
    b = defaultdict(list)
    for q in ids:
        k = "2" if pos[q] == 2 else ("3-5" if pos[q] <= 5 else "6-20")
        b[k].append(q)
    for k in ("2", "3-5", "6-20"):
        qs = b[k]
        if not qs:
            continue
        A = [got["asis"][q] for q in qs]
        P = [got["promote"][q] for q in qs]
        _n01, _n10, p = mcnemar_exact(A, P)
        print(f"   자리 {k:4s} n={len(qs):3d}  asis {sum(A)/len(qs):.4f} -> "
              f"promote {sum(P)/len(qs):.4f}  Δ={(sum(P)-sum(A))/len(qs):+.4f}  p={p:.4g}")

    d = (sum(v["promote"]) - sum(v["asis"])) / len(ids)
    print(f"\n4. 판정 — promote − asis = {d:+.4f}")
    if not ok:
        print("   드리프트 검사 실패. 해석 없음.")
    elif d >= 0.05 and holm(raw)["promote"] < 0.05:
        print(f"   자리는 인과다. 주지표 기준 상금 = {d*len(ids)/991:+.4f} "
              f"({d*len(ids):.0f}건 / 991)")
    else:
        print("   .8609 대 .6375 는 인과가 아니라 교란이다 — 재정렬 계열로는 못 얻는다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
