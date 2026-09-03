#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""선택 전용 split(`hitab_train_sel_lookup_all`) 위의 판정 표.

`alpha_retune.py`가 뱉은 요약/랭크 덤프를 모아 두 arm을 sel에서 페어드로 붙이고,
선택된 α 하나로 잰 dev/test 수치를 같은 표에 붙인다. dev와 test는 **보고만** 한다 --
선택은 sel에서 끝났다 (PREREG-2026-09-02-devbias.md 규칙 1~2).

  PYTHONPATH=. .venv/bin/python analysis/sel_verdict.py \
      --ref bge-small-en-v1.5 --arm bge-cell-ft-b0 --alpha 0.9 \
      --dir results/devbias --title "B — dev 편향 정리"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from rank_ft_verdict import mcnemar_exact, setem                    # noqa: E402

ALPHAS = (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
POPS = ("hitab_train_sel_lookup_all", "hitab_dev_lookup_all",
        "hitab_dev_lookup_multi", "hitab_test_lookup_all", "hitab_test_lookup_multi")
COLS = ("setEM@10", "R@1", "setEM@50", "MRR", "표 recall@1")


def load(d, pop, model, suffix="summary.json"):
    f = Path(d) / f"{pop}_S3c_{model}_{suffix}"
    return json.load(open(f)) if f.exists() else None


def ranks(d, pop, model, al):
    f = Path(d) / f"{pop}_S3c_{model}_a{al}_ranks.jsonl"
    return {r["query_id"]: r for r in map(json.loads, open(f))} if f.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/devbias")
    ap.add_argument("--ref", default="bge-small-en-v1.5", help="대조 arm 모델 이름")
    ap.add_argument("--arm", default="bge-cell-ft-b0", help="개입 arm 모델 이름")
    ap.add_argument("--alpha", type=float, required=True, help="sel에서 고른 α")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = Path(a.out or (Path(a.dir) / "VERDICT.md"))

    L = [f"# {a.title or 'sel 판정'}", "",
         f"선택 α = **{a.alpha}** (`hitab_train_sel_lookup_all`에서 고름). "
         f"대조 `{a.ref}` / 개입 `{a.arm}`. 리더 호출 없음.", ""]

    for pop in POPS:
        sa, sr = load(a.dir, pop, a.arm), load(a.dir, pop, a.ref)
        if not sa and not sr:
            continue
        L += [f"## `{pop}`", "",
              "| arm | α | " + " | ".join(COLS) + " |",
              "|---|---:|" + "---:|" * len(COLS)]
        for lab, s in (("대조 " + a.ref, sr), ("개입 " + a.arm, sa)):
            if not s:
                L.append(f"| {lab} | — | " + " | ".join(["미실행"] * len(COLS)) + " |")
                continue
            for al in ALPHAS:
                if str(al) not in s:
                    continue
                r = s[str(al)]
                mark = " **←선택**" if al == a.alpha else ""
                L.append(f"| {lab}{mark} (n={r['n']}) | {al} | "
                         + " | ".join(f"{r[c]:.4f}" for c in COLS) + " |")
        L.append("")

    # 주검정: sel에서 개입 vs 대조, setEM@10 페어드 exact McNemar
    ra = ranks(a.dir, "hitab_train_sel_lookup_all", a.arm, a.alpha)
    rr = ranks(a.dir, "hitab_train_sel_lookup_all", a.ref, a.alpha)
    res = {}
    if ra and rr:
        ids = sorted(set(ra) & set(rr))
        b = sum(1 for i in ids if setem(ra[i], 10) and not setem(rr[i], 10))
        c = sum(1 for i in ids if setem(rr[i], 10) and not setem(ra[i], 10))
        p = mcnemar_exact(b, c)
        res = {"n": len(ids), "arm_only": b, "ref_only": c, "p": p,
               "significant": p < 0.05}
        L += ["## 주검정 — sel setEM@10, 페어드 exact McNemar (양측 α=.05)", "",
              f"n={len(ids)} | 개입만 맞음 **{b}** / 대조만 맞음 **{c}** | "
              f"p={p:.4g} → **{'유의' if p < 0.05 else '유의하지 않음'}**", ""]
    else:
        L += ["## 주검정", "", "**미실행** (양쪽 랭크 덤프가 같은 α에 없다)", ""]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    json.dump(res, open(out.with_suffix(".json"), "w"), indent=1, ensure_ascii=False)
    print("\n".join(L))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
