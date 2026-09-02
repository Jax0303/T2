#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PREREG-2026-09-01-encoder-ft.md 판정: base vs 파인튜닝 인코더, 페어드.

두 `cell_rank_dump` ranks 덤프를 query_id로 짝지어 setEM@k / R@k / MRR / 표 recall을
같은 질의 위에서 비교하고, 주지표 setEM@10에 exact McNemar 양측을 건다.

  PYTHONPATH=. .venv/bin/python analysis/rank_ft_verdict.py
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

import sys as _sys
BASE = Path("results/rank_ft/baseline")
# 기본 개입은 x2(`models/bge-cell-ft`). 다른 arm을 판정하려면 디렉터리를 인자로 준다:
#   PYTHONPATH=. .venv/bin/python analysis/rank_ft_verdict.py results/rank_ft/x0
FT = Path(_sys.argv[1]) if len(_sys.argv) > 1 else Path("results/rank_ft/ft")
TAG = "_S3c_hybrid0.7_tp0.0_ranks.jsonl"
OUT_MD = FT / "VERDICT.md"
OUT_JSON = FT / "verdict.json"
KS = (1, 5, 10, 20, 50)
THRESHOLD, NOISE = 0.90, 0.025      # 사전등록 판정 규칙 3, 2


def load(d, pop):
    f = d / f"{pop}{TAG}"
    return {r["query_id"]: r for r in map(json.loads, open(f))} if f.exists() else None


def setem(r, k):
    return int(max(r["ranks"]) < k)


def recall(r, k):
    return sum(1 for x in r["ranks"] if x < k) / len(r["ranks"])


def mcnemar_exact(b, c):
    """Two-sided exact McNemar on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    tail = sum(comb(n, i) for i in range(lo + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> int:
    out, L = {}, ["# 인코더 파인튜닝 판정 — PREREG-2026-09-01-encoder-ft.md", "",
                  "통제 `BAAI/bge-small-en-v1.5` vs 개입 `models/bge-cell-ft` "
                  "(형제 2 + 채굴 타표 2). 같은 코퍼스·색인 단위 S3c·hybrid α=0.7, "
                  "같은 질의 위 페어드. 리더 호출 없음.", ""]
    for pop in ("hitab_dev_lookup_all", "hitab_dev_lookup_multi",
                "hitab_test_lookup_all", "hitab_test_lookup_multi"):
        A, B = load(BASE, pop), load(FT, pop)
        if A is None or B is None:
            L += [f"## `{pop}`", "", "**미실행** — "
                  f"{'통제' if A is None else '개입'} 덤프 없음.", ""]
            out[pop] = {"status": "not run"}
            continue
        ids = sorted(set(A) & set(B))
        assert len(ids) == len(A) == len(B), f"{pop}: 질의 집합이 다르다"
        n = len(ids)
        row = {"n": n}
        L += [f"## `{pop}`  n={n}", "",
              "| 지표 | 통제 | 파인튜닝 | 델타 |", "|---|---:|---:|---:|"]
        for k in KS:
            for name, fn in (("setEM", setem), ("R", recall)):
                a = sum(fn(A[i], k) for i in ids) / n
                b = sum(fn(B[i], k) for i in ids) / n
                row[f"{name}@{k}"] = {"base": round(a, 4), "ft": round(b, 4),
                                      "delta": round(b - a, 4)}
                L.append(f"| {name}@{k} | {a:.4f} | {b:.4f} | {b - a:+.4f} |")
        for name, fn in (("MRR", lambda r: 1.0 / (min(r["ranks"]) + 1)),
                         ("표 recall@1", lambda r: int(r["table_rank_cellvote"] == 1)),
                         ("표 recall@10", lambda r: int(r["table_rank_cellvote"] <= 10))):
            a = sum(fn(A[i]) for i in ids) / n
            b = sum(fn(B[i]) for i in ids) / n
            row[name] = {"base": round(a, 4), "ft": round(b, 4), "delta": round(b - a, 4)}
            L.append(f"| {name} | {a:.4f} | {b:.4f} | {b - a:+.4f} |")

        # 주검정: setEM@10
        bb = sum(1 for i in ids if setem(B[i], 10) and not setem(A[i], 10))
        cc = sum(1 for i in ids if setem(A[i], 10) and not setem(B[i], 10))
        p = mcnemar_exact(bb, cc)
        d = row["setEM@10"]["delta"]
        verdict = ("유의하지 않음" if p >= 0.05 else
                   "노이즈 안 (규칙 2)" if abs(d) < NOISE else
                   "유의, 목표 도달" if row["setEM@10"]["ft"] >= THRESHOLD else
                   "유의, 목표 미달")
        row["mcnemar"] = {"ft_only": bb, "base_only": cc, "p_exact_two_sided": p,
                          "verdict": verdict}
        L += ["", f"주검정 setEM@10 exact McNemar: 파인튜닝만 맞음 **{bb}** / "
              f"통제만 맞음 **{cc}**, p={p:.3e} → **{verdict}** "
              f"(목표 {THRESHOLD}, 노이즈 임계 {NOISE})", ""]
        out[pop] = row

    # 사전등록 예측 대조 (PREREG-2026-09-01-encoder-ft.md §예측). 그 예측은 x2에
    # 대한 것이므로 다른 arm을 판정할 때는 붙이지 않는다.
    PRED = [] if FT.name != "ft" else [("hitab_dev_lookup_all", "setEM@10", 0.89, (0.86, 0.92)),
            ("hitab_dev_lookup_all", "R@1", 0.62, (0.56, 0.68)),
            ("hitab_dev_lookup_all", "setEM@50", 0.94, (0.92, 0.96)),
            ("hitab_dev_lookup_all", "MRR", 0.70, (0.65, 0.75)),
            ("hitab_dev_lookup_all", "표 recall@1", 0.87, (0.84, 0.90))]
    if PRED:
        L += ["## 사전등록 예측 대조", "",
              "| 모집단 | 지표 | 예측 (점) | 예측 구간 | 실측 | 구간 안 |",
              "|---|---|---:|---|---:|---|"]
    hits = []
    for pop, met, pt, (lo, hi) in PRED:
        got = out.get(pop, {}).get(met, {}).get("ft")
        ok = got is not None and lo <= got <= hi
        hits.append(ok)
        L.append(f"| `{pop}` | {met} | {pt:.4f} | {lo:.2f} – {hi:.2f} | "
                 f"{got if got is None else f'{got:.4f}'} | {'예' if ok else '**아니오**'} |")
    d = out.get("hitab_dev_lookup_multi", {}).get("setEM@10", {}).get("delta")
    PRED and L.append(f"| `hitab_dev_lookup_multi` | setEM@10 델타 | +0.0600 | −0.03 – +0.15 | "
             f"{d:+.4f} | {'예' if d is not None and -0.03 <= d <= 0.15 else '**아니오**'} |")
    out["prereg_hits"] = f"{sum(hits)}/{len(hits)} (dev 주·부지표)"
    L += ["", "test 예측은 \"dev 델타가 test에서 절반 이상 재현\"이었다:", ""]
    for a, b in (("hitab_dev_lookup_all", "hitab_test_lookup_all"),
                 ("hitab_dev_lookup_multi", "hitab_test_lookup_multi")):
        da = out.get(a, {}).get("setEM@10", {}).get("delta")
        db = out.get(b, {}).get("setEM@10", {}).get("delta")
        if da and db is not None:
            L.append(f"- `{b}` setEM@10 델타 {db:+.4f} / dev {da:+.4f} = "
                     f"**{db / da:.0%}** → {'재현' if db / da >= 0.5 else '**미재현**'}")
            out.setdefault("test_reproduction", {})[b] = round(db / da, 4)
    L.append("")

    OUT_MD.write_text("\n".join(L) + "\n")
    json.dump(out, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
