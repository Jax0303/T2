#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PREREG-2026-09-01-neg-cross-sweep.md 판정.

arm x0..x4는 `--neg-cross` 하나만 다르다 (형제 : 타표 비율, 합은 4로 고정). 선택은
`hitab_dev_lookup_all` setEM@10 하나로만 하고, test는 선택된 arm 1회다 -- 그 규칙은
사전등록에 박혀 있고 여기서는 dev 표를 만들고 선택만 출력한다.

  PYTHONPATH=. .venv/bin/python analysis/neg_cross_sweep.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from rank_ft_verdict import load, mcnemar_exact, recall, setem   # noqa: E402

# x2는 선행 실행분을 재사용한다 (같은 코드·시드·사양)
DIRS = {"x0": Path("results/rank_ft/x0"), "x1": Path("results/rank_ft/x1"),
        "x2": Path("results/rank_ft/ft"), "x3": Path("results/rank_ft/x3"),
        "x4": Path("results/rank_ft/x4")}
BASE = Path("results/rank_ft/baseline")
POPS = ("hitab_dev_lookup_all", "hitab_dev_lookup_multi")
KS = (1, 5, 10, 20, 50)
PRED = {"x0": (0.870, 0.84, 0.90), "x1": (0.895, 0.87, 0.92),
        "x3": (0.900, 0.87, 0.93), "x4": (0.890, 0.85, 0.92)}
OUT_MD, OUT_JSON = Path("results/rank_ft/SWEEP.md"), Path("results/rank_ft/sweep.json")


def main() -> int:
    out = {"arms": {}, "missing": []}
    L = ["# `--neg-cross` 스윕 — PREREG-2026-09-01-neg-cross-sweep.md", "",
         "`--neg-per-query 4` 고정, 형제 : 타표 비율만 이동. 나머지 사양·시드 전부 동일.",
         "선택은 `hitab_dev_lookup_all` setEM@10 하나로만 한다.", ""]

    for pop in POPS:
        ctrl = load(BASE, pop)
        arms = {a: load(d, pop) for a, d in DIRS.items()}
        have = {a: v for a, v in arms.items() if v is not None}
        out["missing"] += [f"{a}/{pop}" for a, v in arms.items() if v is None]
        if not have:
            L += [f"## `{pop}`", "", "**전 arm 미실행**", ""]
            continue
        ids = sorted(set(ctrl) & set(next(iter(have.values()))))
        n = len(ids)
        cols = sorted(have)
        L += [f"## `{pop}`  n={n}", "",
              "| 지표 | 통제(base) | " + " | ".join(cols) + " |",
              "|---|---:|" + "---:|" * len(cols)]
        for k in KS:
            for name, fn in (("setEM", setem), ("R", recall)):
                c = sum(fn(ctrl[i], k) for i in ids) / n
                vals = {a: sum(fn(have[a][i], k) for i in ids) / n for a in cols}
                for a, v in vals.items():
                    out["arms"].setdefault(a, {}).setdefault(pop, {})[f"{name}@{k}"] = round(v, 4)
                L.append(f"| {name}@{k} | {c:.4f} | "
                         + " | ".join(f"{vals[a]:.4f}" for a in cols) + " |")
        for name, fn in (("MRR", lambda r: 1.0 / (min(r["ranks"]) + 1)),
                         ("표 recall@1", lambda r: int(r["table_rank_cellvote"] == 1)),
                         ("표 recall@10", lambda r: int(r["table_rank_cellvote"] <= 10))):
            c = sum(fn(ctrl[i]) for i in ids) / n
            vals = {a: sum(fn(have[a][i]) for i in ids) / n for a in cols}
            for a, v in vals.items():
                out["arms"].setdefault(a, {}).setdefault(pop, {})[name] = round(v, 4)
            L.append(f"| {name} | {c:.4f} | "
                     + " | ".join(f"{vals[a]:.4f}" for a in cols) + " |")
        L.append("")

    # 선택 (규칙 1·2)
    pop = "hitab_dev_lookup_all"
    scores = {a: out["arms"][a][pop]["setEM@10"] for a in out["arms"]
              if pop in out["arms"][a]}
    if scores:
        best = max(sorted(scores), key=lambda a: (scores[a], -int(a[1:])))
        best = min([a for a in scores if scores[a] == scores[best]],
                   key=lambda a: int(a[1:]))          # 동점이면 neg-cross 작은 쪽
        out["selected"] = {"arm": best, "dev_setEM@10": scores[best],
                           "rule": "dev setEM@10 argmax, 동점이면 neg-cross 작은 쪽"}
        L += ["## 선택", "",
              "| arm | dev setEM@10 | 예측 (점) | 예측 구간 | 구간 안 |",
              "|---|---:|---:|---|---|"]
        for a in sorted(scores):
            p = PRED.get(a)
            ok = "—" if p is None else ("예" if p[1] <= scores[a] <= p[2] else "**아니오**")
            L.append(f"| {a}{' **←선택**' if a == best else ''} | {scores[a]:.4f} | "
                     + (f"{p[0]:.4f} | {p[1]:.2f} – {p[2]:.2f} | {ok} |" if p
                        else "(실측 재사용) | — | — |"))
        L.append("")

        # 규칙 5의 분해. x0 자신이 이기면 "선택 arm vs x0"는 공허하므로, 타표
        # negative의 몫은 x0가 아닌 arm 중 최고와 x0의 차로 잰다 -- 그 값이 음수면
        # 음수 그대로가 답이다 (규칙 7).
        def paired(a_dir, b_dir, k=10):
            A, B = load(a_dir, pop), load(b_dir, pop)
            if not A or not B:
                return None
            ids = sorted(set(A) & set(B))
            bb = sum(1 for i in ids if setem(B[i], k) and not setem(A[i], k))
            cc = sum(1 for i in ids if setem(A[i], k) and not setem(B[i], k))
            return bb, cc, mcnemar_exact(bb, cc)

        r = paired(BASE, DIRS["x0"])
        if r:
            bb, cc, pv = r
            ctrl_pop = load(BASE, pop)
            base_rate = sum(setem(r, 10) for r in ctrl_pop.values()) / len(ctrl_pop)
            d = scores["x0"] - base_rate
            out["finetuning_share"] = {"vs": "base", "delta": round(d, 4),
                                       "x0_only": bb, "base_only": cc,
                                       "p_exact_two_sided": pv}
            L += [f"**파인튜닝 자체의 몫** — x0(형제만) vs 통제 base, 페어드 setEM@10 "
                  f"exact McNemar: x0만 맞음 **{bb}** / base만 맞음 **{cc}**, "
                  f"p={pv:.3e}, 델타 **{d:+.4f}**", ""]

        alt = [a for a in scores if a != "x0"]
        if alt:
            top = max(sorted(alt), key=lambda a: scores[a])
            r = paired(DIRS["x0"], DIRS[top])
            bb, cc, pv = r
            d = scores[top] - scores["x0"]
            out["cross_negative_share"] = {"best_non_x0": top, "vs": "x0",
                                           "delta": round(d, 4), "top_only": bb,
                                           "x0_only": cc, "p_exact_two_sided": pv}
            L += [f"**타표 negative의 몫** (규칙 5) — x0가 아닌 최고 arm {top} vs "
                  f"x0(형제만), 페어드 setEM@10 exact McNemar: {top}만 맞음 **{bb}** / "
                  f"x0만 맞음 **{cc}**, p={pv:.3e}, 델타 **{d:+.4f}** "
                  f"(예측 +.032, 구간 +.005 – +.06 → "
                  f"{'구간 안' if 0.005 <= d <= 0.06 else '**구간 밖**'})", "",
                  f"예측한 arm은 x3, 실제 dev 최고는 **{best}**.", ""]
    if out["missing"]:
        L += ["## 미실행", "", "- " + "\n- ".join(out["missing"]), ""]

    OUT_MD.write_text("\n".join(L) + "\n")
    json.dump(out, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
