# SPDX-License-Identifier: MIT
"""v3.1 에서 남는 오류 중 기계로 셀 수 있는 두 유형 — 모델 없음.

  S1n  끊어서 열 경로가 통째로 빈 셀(v2 에는 경로가 있었다)
  S5n  붙인 구획 표제가 숫자('544,823')거나 단위 표기('(Stated in millions)')인 행

  HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python analysis/header_v3_1_residual_counts.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from analysis.header_v3_impact import cells                        # noqa: E402
from mh_arms import build_tables, load_population                  # noqa: E402
from rag_agent.reconstruct.header_grid import _number_like         # noqa: E402

OUT = ROOT / "results/mh_header_v3_1/residual_counts.json"
UNITS = re.compile(r"^\(.*\)$|\bin\s+(thousands|millions|billions)\b", re.I)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--rule", default="v3.1", choices=["v3.1", "v3.2"], help="v3.2 는 S5n 대신 S5n2 를 잰다")
    a = ap.parse_args()
    out_path, s5 = Path(a.out), {"v3.1": "S5n", "v3.2": "S5n2"}[a.rule]
    if out_path.exists():
        raise SystemExit(f"{out_path} exists — 덮어쓰지 않는다")
    _, docs, _ = load_population("train")
    v2 = build_tables(docs, "v2")[0]
    c2 = cells(v2)
    out = {"rule": a.rule}
    c1 = cells(build_tables(docs, "v2", rules={"S1n"})[0])
    emptied = [k for k in c2 if c2[k][1] and not c1[k][1]]
    out["S1n_col_path_emptied"] = {"cells": len(emptied), "tables": len({k[0] for k in emptied}),
                                   "examples": [{"v2": c2[k][3], "new": c1[k][3]} for k in emptied[:3]]}
    del c1
    c5 = cells(build_tables(docs, "v2", rules={s5})[0])
    rows, numeric, units = set(), set(), set()
    for k in c2:
        added = [s for s in c5[k][0] if s not in c2[k][0]]
        if not added:
            continue
        row = (k[0], k[1])
        rows.add(row)
        if any(_number_like(s) for s in added):
            numeric.add(row)
        if any(UNITS.search(s) for s in added):
            units.add(row)
    out["S5n_added_heading"] = {"rows": len(rows), "rows_numeric_heading": len(numeric),
                                "rows_units_note_heading": len(units)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("x", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
