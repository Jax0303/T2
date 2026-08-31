#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Phase 4d Task A -- re-score every row with R1, side by side with the old rule."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

import pandas as pd                                                  # noqa: E402
from phase4_summary import em                                        # noqa: E402

POOLS = ["hitab_lookup", "hitab_arith", "aitqa", "rhb_fact", "rhb_num"]
POLS = ["P1_fixed_512", "P4_path_cell", "gold_cell"]


def main() -> int:
    df = pd.DataFrame([json.loads(l) for l in
                       open("results/phase4/reader_records.jsonl")])
    df["old"] = [em(p, g, rel=False) for p, g in zip(df.pred_parsed, df.gold_answer)]
    df["new"] = [em(p, g, rel=True) for p, g in zip(df.pred_parsed, df.gold_answer)]

    print(f"## Task A -- R1 전/후 EM (전체 {len(df)}행)\n")
    print("| pool | " + " | ".join(f"{p} 전 → 후 (n)" for p in POLS) + " |")
    print("|---|" + "---|" * len(POLS))
    for pool in POOLS:
        cells = []
        for pol in POLS:
            d = df[(df.pool == pool) & (df.policy == pol)]
            cells.append(f"{d.old.mean():.4f} → **{d.new.mean():.4f}** (n={len(d)})")
        print(f"| {pool} | " + " | ".join(cells) + " |")

    ch = df[df.old != df.new]
    print(f"\n총 변경 행 **{len(ch)}** (전부 오답→정답 방향: "
          f"{int((ch.new > ch.old).sum())}건, 반대 방향 {int((ch.new < ch.old).sum())}건)\n")
    print("| pool | policy | 변경 건수 | query_id |")
    print("|---|---|---:|---|")
    for pool in POOLS:
        for pol in POLS:
            c = ch[(ch.pool == pool) & (ch.policy == pol)]
            if len(c):
                print(f"| {pool} | {pol} | {len(c)} | "
                      f"{', '.join(x[:8] for x in c.query_id)} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
