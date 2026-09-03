#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""조회 축 현재 위치 — 지금까지 측정된 모든 설정 × 모든 모집단, 한 표.

세 사전등록(`PREREG-2026-09-01-encoder-ft`, `-neg-cross-sweep`,
`PREREG-2026-09-02-alpha-retune`, `-train-expand`)의 덤프를 모아 setEM@10을 한 자리에
놓고, dev 최고와 test 최고가 갈리는 지점을 그대로 드러낸다. 리더 호출 없음.

  PYTHONPATH=. .venv/bin/python analysis/lookup_scoreboard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from rank_ft_verdict import mcnemar_exact, setem                    # noqa: E402

POPS = ("hitab_train_sel_lookup_all", "hitab_dev_lookup_all",
        "hitab_dev_lookup_multi", "hitab_test_lookup_all", "hitab_test_lookup_multi")
# (라벨, 설명, ranks 파일을 찾는 함수)
RANKDIR = Path("results/rank_ft")
ALPHADIR = Path("results/alpha_ft")
DEVBIAS = Path("results/devbias")     # 선택 전용 split 도입 이후의 arm


def f_rank(d, pop):
    return RANKDIR / d / f"{pop}_S3c_hybrid0.7_tp0.0_ranks.jsonl"


def f_alpha(model, al, pop):
    return ALPHADIR / f"{pop}_S3c_{model}_a{al}_ranks.jsonl"


def f_devbias(model, al, pop):
    return DEVBIAS / f"{pop}_S3c_{model}_a{al}_ranks.jsonl"


CONFIGS = [
    ("base α=.7", "기성품 bge-small, 개입 전", lambda p: f_rank("baseline", p)),
    ("x0 α=.7", "FT, 형제만 (dev 최적 arm)", lambda p: f_rank("x0", p)),
    ("x2 α=.7", "FT, 형제2+타표2", lambda p: f_rank("ft", p)),
    ("x0 α=.9", "위 + α 재조정", lambda p: f_alpha("bge-cell-ft-x0", 0.9, p)),
    ("e1 α=.9", "위 + 학습 데이터 확대 (질문당 1행)",
     lambda p: f_alpha("bge-cell-ft-e1", 0.9, p)),
    ("e2 α=.9", "위 + 확대 (gold셀당 1행)",
     lambda p: f_alpha("bge-cell-ft-e2", 0.9, p)),
    # 아래 두 줄부터는 dev가 아니라 `hitab_train_sel_lookup_all`에서 골랐다
    ("base α=.8 (sel)", "기성품 bge-small, sel 최적 α",
     lambda p: f_devbias("bge-small-en-v1.5", 0.8, p)),
    ("b0 α=.9", "B: 선택 전용 split 도입, fit(2,911)로 재학습",
     lambda p: f_devbias("bge-cell-ft-b0", 0.9, p)),
    ("a0 α=1.0", "A: 베이스를 bge-base(110M)로",
     lambda p: f_devbias("bge-base-cell-ft-a0", 1.0, p)),
]
OUT_MD, OUT_JSON = Path("results/LOOKUP_BOARD.md"), Path("results/lookup_board.json")


def load(f):
    return {r["query_id"]: r for r in map(json.loads, open(f))} if f.exists() else None


def main() -> int:
    cells, out = {}, {}
    for label, _d, fn in CONFIGS:
        for pop in POPS:
            r = load(fn(pop))
            if r:
                cells[(label, pop)] = sum(setem(x, 10) for x in r.values()) / len(r)
    L = ["# 조회 축 보드 — setEM@10 (상위 10개 안에 gold 셀이 **전부** 들어온 비율)", "",
         "색인 단위 S3c, hybrid. 리더 호출 없음. 빈 칸은 그 조합을 안 돌린 것이다 "
         "(사전등록이 test를 선택된 arm 1회로 제한한다).", "",
         "| 설정 | " + " | ".join(p.replace("hitab_", "") for p in POPS) + " |",
         "|---|" + "---:|" * len(POPS)]
    for label, desc, _fn in CONFIGS:
        row = [f"{cells[(label, p)]:.4f}" if (label, p) in cells else "—" for p in POPS]
        L.append(f"| **{label}** — {desc} | " + " | ".join(row) + " |")
        out[label] = {p: round(cells[(label, p)], 4) for p in POPS if (label, p) in cells}
    L.append("")

    for pop in ("hitab_dev_lookup_all", "hitab_test_lookup_all"):
        have = {lab: cells[(lab, pop)] for lab, _d, _f in CONFIGS if (lab, pop) in cells}
        best = max(have, key=lambda k: have[k])
        out.setdefault("best", {})[pop] = {"config": best, "setEM@10": round(have[best], 4)}
        L.append(f"- `{pop}` 최고: **{best}** = {have[best]:.4f}")
    L += ["", "dev 최고와 test 최고가 **같은 설정이 아니다.** dev에서 고른 것이 test에서 "
          "그대로 1등이 아니라는 뜻이고, 세 사전등록 전부 이 편향을 미리 적어두었다.", "",
          "**표를 위아래로 갈라 읽을 것.** `e2` 줄까지는 dev에서 고른 arm이고 그 test "
          "수치는 3중 dev 선택 위에 얹혀 있다. `base α=.8 (sel)` 줄부터는 "
          "`hitab_train_sel_lookup_all`(n=756)에서 골랐고 dev는 아무것도 고르지 않은 "
          "평가 수치다 (PREREG-2026-09-02-devbias.md). 두 구간의 dev 값을 같은 의미로 "
          "비교하지 말 것.", ""]

    # 확대 arm 주검정 (PREREG-2026-09-02-train-expand 규칙 4)
    pop = "hitab_dev_lookup_all"
    A = load(f_alpha("bge-cell-ft-x0", 0.9, pop))
    B = load(f_alpha("bge-cell-ft-e1", 0.9, pop))
    if A and B:
        ids = sorted(set(A) & set(B))
        bb = sum(1 for i in ids if setem(B[i], 10) and not setem(A[i], 10))
        cc = sum(1 for i in ids if setem(A[i], 10) and not setem(B[i], 10))
        pv = mcnemar_exact(bb, cc)
        out["expand_mcnemar"] = {"e1_only": bb, "x0_only": cc, "p": pv}
        L += [f"주검정 (train-expand 규칙 4) — e1 vs x0, {len(ids)}건 페어드 setEM@10 "
              f"exact McNemar: e1만 맞음 **{bb}** / x0만 맞음 **{cc}**, p={pv:.3e} → "
              f"**{'유의' if pv < 0.05 else '유의하지 않음'}**", ""]

    base, best = cells.get(("base α=.7", "hitab_test_lookup_all")), \
        out["best"]["hitab_test_lookup_all"]["setEM@10"]
    L += ["## 목표까지", "",
          f"| | test setEM@10 |", "|---|---:|",
          f"| 개입 전 | {base:.4f} |", f"| 현재 최고 | {best:.4f} |",
          f"| 목표 | 0.9000 |", f"| **남은 거리** | **{0.90 - best:+.4f}** |", ""]
    OUT_MD.write_text("\n".join(L) + "\n")
    json.dump(out, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
