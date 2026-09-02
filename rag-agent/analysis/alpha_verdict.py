#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PREREG-2026-09-02-alpha-retune.md 판정 표.

`alpha_retune.py`의 dev 스윕 + α=0.9 test 확정을 모아 사전등록 예측과 대조하고,
주검정(선택 α vs 현행 0.7, 830건 페어드 exact McNemar)을 건다.

  PYTHONPATH=. .venv/bin/python analysis/alpha_verdict.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from rank_ft_verdict import mcnemar_exact, setem                    # noqa: E402

D = Path("results/alpha_ft")
MODEL = "bge-cell-ft-x0"
SEL, CUR, TARGET = 0.9, 0.7, 0.90
ALPHAS = (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
PRED = {0.0: (.700, .62, .78), 0.3: (.880, .85, .905), 0.5: (.898, .88, .915),
        0.6: (.905, .89, .918), 0.8: (.912, .900, .925),
        0.9: (.908, .895, .922), 1.0: (.900, .88, .920)}
OUT_MD, OUT_JSON = Path("results/alpha_ft/VERDICT.md"), Path("results/alpha_ft/verdict.json")


def summary(pop):
    f = D / f"{pop}_S3c_{MODEL}_summary.json"
    return json.load(open(f)) if f.exists() else None


def ranks(pop, al):
    f = D / f"{pop}_S3c_{MODEL}_a{al}_ranks.jsonl"
    return {r["query_id"]: r for r in map(json.loads, open(f))} if f.exists() else None


def main() -> int:
    out, L = {}, ["# hybrid α 재조정 판정 — PREREG-2026-09-02-alpha-retune.md", "",
                  f"인코더 `models/{MODEL}` 고정. 색인 1회, α만 이동. 리더 호출 없음.", ""]
    for pop in ("hitab_dev_lookup_all", "hitab_dev_lookup_multi"):
        s = summary(pop)
        if not s:
            L += [f"## `{pop}`", "", "**미실행**", ""]
            continue
        ctl = s.get("_control", {})
        n = s[str(ALPHAS[0])]["n"]
        out[pop] = s
        L += [f"## `{pop}`  n={n}", "",
              f"통제: α=0.7이 `cell_rank_dump` 덤프를 재현하는가 → "
              f"mine {ctl.get('mine')} / ref {ctl.get('ref')} → "
              f"**{'통과' if ctl.get('ok') else '실패'}**", "",
              "| α | setEM@10 | R@1 | setEM@50 | MRR | 표 R@1 |",
              "|---:|---:|---:|---:|---:|---:|"]
        for al in ALPHAS:
            r = s[str(al)]
            mark = " **←선택**" if al == SEL and pop.endswith("lookup_all") else ""
            L.append(f"| {al}{mark} | {r['setEM@10']:.4f} | {r['R@1']:.4f} | "
                     f"{r['setEM@50']:.4f} | {r['MRR']:.4f} | {r['표 recall@1']:.4f} |")
        L.append("")

    s = summary("hitab_dev_lookup_all")
    if s:
        L += ["## 사전등록 예측 대조 (dev setEM@10)", "",
              "| α | 예측 (점) | 예측 구간 | 실측 | 구간 안 |", "|---:|---:|---|---:|---|"]
        for al in ALPHAS:
            got = s[str(al)]["setEM@10"]
            p = PRED.get(al)
            L.append(f"| {al} | " + (f"{p[0]:.4f} | {p[1]:.2f} – {p[2]:.2f} | {got:.4f} | "
                                     f"{'예' if p[1] <= got <= p[2] else '**아니오**'} |"
                                     if p else f"(실측 재사용) | — | {got:.4f} | — |"))
        best = max(ALPHAS, key=lambda al: (s[str(al)]["setEM@10"], -abs(al - CUR)))
        d = s[str(SEL)]["setEM@10"] - s[str(CUR)]["setEM@10"]
        out["selected"] = {"alpha": SEL, "dev_setEM@10": s[str(SEL)]["setEM@10"],
                           "delta_vs_0.7": round(d, 4)}
        L += ["", f"- 예측한 최적 α는 **0.8**, 실제 dev 최적은 **{best}**. "
              f"방향(0.7보다 위로 올라간다)은 맞았다.",
              f"- 0.7 대비 델타 **{d:+.4f}** (예측 +.005, 구간 −.002 – +.015 → "
              f"{'구간 안' if -0.002 <= d <= 0.015 else '**구간 밖**'})", ""]

        A, B = ranks("hitab_dev_lookup_all", CUR), ranks("hitab_dev_lookup_all", SEL)
        if A and B:
            ids = sorted(set(A) & set(B))
            bb = sum(1 for i in ids if setem(B[i], 10) and not setem(A[i], 10))
            cc = sum(1 for i in ids if setem(A[i], 10) and not setem(B[i], 10))
            pv = mcnemar_exact(bb, cc)
            out["mcnemar"] = {"sel_only": bb, "cur_only": cc, "p_exact_two_sided": pv}
            L += [f"주검정 (규칙 5) — α={SEL} vs α={CUR}, {len(ids)}건 페어드 setEM@10 "
                  f"exact McNemar: α={SEL}만 맞음 **{bb}** / α={CUR}만 맞음 **{cc}**, "
                  f"p={pv:.3e} → **{'유의' if pv < 0.05 else '유의하지 않음'}**", ""]

    L += ["## test 확정 (α=0.9만, 사전등록 규칙 3)", "",
          "| 모집단 | n | α=0.7 (앞선 실행) | α=0.9 | 델타 |", "|---|---:|---:|---:|---:|"]
    PREV = {"hitab_test_lookup_all": 0.8674, "hitab_test_lookup_multi": 0.8065}
    for pop, prev in PREV.items():
        s = summary(pop)
        if not s:
            L.append(f"| `{pop}` | — | {prev:.4f} | **미실행** | — |")
            continue
        r = s[str(SEL)]
        out[pop] = r
        L.append(f"| `{pop}` | {r['n']} | {prev:.4f} | **{r['setEM@10']:.4f}** | "
                 f"{r['setEM@10'] - prev:+.4f} |")
    ta = summary("hitab_test_lookup_all")
    if ta:
        got = ta[str(SEL)]["setEM@10"]
        out["target"] = {"test_setEM@10": got, "threshold": TARGET,
                         "reached": got >= TARGET}
        L += ["", f"목표 (규칙 4) test setEM@10 ≥ {TARGET}: 실측 **{got:.4f}** → "
              f"**{'도달' if got >= TARGET else '미달'}**. "
              f"사전등록 예측 .875 (구간 .855 – .895) → "
              f"{'구간 안' if .855 <= got <= .895 else '**구간 밖**'}, "
              f"'미달로 예측한다'도 적중.", ""]
    OUT_MD.write_text("\n".join(L) + "\n")
    json.dump(out, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
