#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PREREG-2026-09-24-mh-answer-cap300.md 의 보고 — 그룹별 EM, 모집단 비율 가중 전체 EM,
그룹 층화 부트스트랩 95% CI(10,000회, seed 0), 본 방법 대 각 조건 정확 McNemar(그룹별) 와 가중 차이 CI.

  .venv/bin/python scripts/mh_cap300_report.py            # results/mh_arms/cap300_20260924/*.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "results" / "mh_arms" / "cap300_20260924"
POP = {"lookup_m1": 211, "lookup_m2+": 365, "arith_m1": 71, "arith_m2+": 2224}   # 채점 2,871 의 그룹 크기
W = {g: n / sum(POP.values()) for g, n in POP.items()}
REF = "cell_uniq"   # 본 방법 주 조건(v3.3u, 2026-09-25 사용자 결정). v1 은 "cell"
ORDER = ["cell_uniq", "cell", "fulltable", "chunk", "rowcol", "trag_hetero", "tablerag_path", "tablerag_leaf", "randrow"]
REPS, SEED = 10_000, 0


def load(name):
    rows = [json.loads(l) for l in open(D / f"{name}.jsonl", encoding="utf-8")]
    return {r["query_id"]: r for r in rows}


def weighted(ok_by_group):
    return sum(W[g] * np.mean(v) for g, v in ok_by_group.items())


def boot(ids_by_group, f, rng):
    """그룹마다 질의를 복원 추출해 f(표본 id 묶음) 를 REPS 번."""
    out = np.empty(REPS)
    for b in range(REPS):
        out[b] = f({g: rng.choice(ids, len(ids)) for g, ids in ids_by_group.items()})
    return np.percentile(out, [2.5, 97.5])


def main() -> int:
    names = [n for n in ORDER if (D / f"{n}.jsonl").exists()]
    data = {n: load(n) for n in names}
    ref = data[REF]
    groups = {g: sorted(q for q, r in ref.items() if r["layer"] == g) for g in POP}
    for n, rows in data.items():
        if set(rows) != set(ref):
            raise SystemExit(f"{n}: 질의 집합이 본 방법과 다르다")
    rng = np.random.default_rng(SEED)
    res = {}
    for n, rows in data.items():
        ok = {g: np.array([rows[q]["answer_correct"] for q in ids]) for g, ids in groups.items()}
        est = weighted(ok)
        ci = boot({g: np.arange(len(ids)) for g, ids in groups.items()},
                  lambda s: sum(W[g] * ok[g][s[g]].mean() for g in s), rng)
        res[n] = {"groups": {g: {"n": len(v), "em": round(float(v.mean()), 4)} for g, v in ok.items()},
                  "weighted_em": round(float(est), 4), "ci95": [round(float(x), 4) for x in ci],
                  "retrieval": {g: round(float(np.mean([rows[q]["retrieval_correct"] for q in ids])), 4)
                                for g, ids in groups.items()}}
        if n != REF:
            d = {g: np.array([ref[q]["answer_correct"] - rows[q]["answer_correct"] for q in ids])
                 for g, ids in groups.items()}
            dci = boot({g: np.arange(len(ids)) for g, ids in groups.items()},
                       lambda s: sum(W[g] * d[g][s[g]].mean() for g in s), rng)
            res[n]["vs_cell"] = {"weighted_diff_cell_minus_this": round(float(sum(W[g] * d[g].mean() for g in d)), 4),
                                 "ci95": [round(float(x), 4) for x in dci], "mcnemar": {}}
            for g, ids in groups.items():
                b = int(sum(ref[q]["answer_correct"] and not rows[q]["answer_correct"] for q in ids))
                c = int(sum(rows[q]["answer_correct"] and not ref[q]["answer_correct"] for q in ids))
                res[n]["vs_cell"]["mcnemar"][g] = {"cell_only": b, "this_only": c,
                                                  "p": float(binomtest(b, b + c).pvalue) if b + c else 1.0}
    (D / "report.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["| 조건 | 조회 1 | 조회 2+ | 산술 1 | 산술 2+ | 가중 전체 [95% CI] | 본 방법 − 이 조건 [95% CI] |",
             "|---|---:|---:|---:|---:|---|---|"]
    for n, r in res.items():
        cells = []
        for g in POP:
            m = r.get("vs_cell", {}).get("mcnemar", {}).get(g)
            cells.append(f"{r['groups'][g]['em']:.4f}" + (f" ({m['cell_only']}:{m['this_only']}, p={m['p']:.2g})" if m else ""))
        v = r.get("vs_cell")
        lines.append(f"| {n} | " + " | ".join(cells) + f" | {r['weighted_em']:.4f} [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}] | "
                     + (f"{v['weighted_diff_cell_minus_this']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]" if v else "—") + " |")
    (D / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
