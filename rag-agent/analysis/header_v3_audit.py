# SPDX-License-Identifier: MIT
"""헤더 규칙 v3 / v3.1 의 오작동 유형 집계 — 모델을 돌리지 않는다.

`header_v3_impact.py` 의 무작위 예시에서 사전등록 지표(M2·M3·gold_in_header)가 잡지 못한 잘못된 변경이
보였다. 그 유형을 규칙마다 셀 수로 센다. 판정은 원표 격자·병합 표기·경로 문자열에서 기계적으로 하며,
어느 쪽이 맞는지 사람이 가려야 하는 유형은 '의심'으로만 센다.

  HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python analysis/header_v3_audit.py              # v3
  HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python analysis/header_v3_audit.py --rule v3.1
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from analysis.header_v3_impact import cells, population                  # noqa: E402
from mh_arms import build_tables, load_population                        # noqa: E402
from rag_agent.reconstruct.header_grid import parse_html_table_layout   # noqa: E402

OUT = {"v3": ROOT / "results/mh_header_v3/audit.json", "v3.1": ROOT / "results/mh_header_v3_1/audit.json",
       "v3.2": ROOT / "results/mh_header_v3_2/audit.json", "v3.3": ROOT / "results/mh_header_v3_3/audit.json"}
RULE = {"v3": {"S1": "S1", "S2": "S2", "S3": "S3"}, "v3.1": {"S1": "S1n", "S2": "S2n", "S3": "S3n"},
        "v3.2": {"S1": "S1n", "S2": "S2n", "S3": "S3n"}, "v3.3": {"S1": "S1n", "S2": "S2n", "S3": "S3n"}}
YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
UNITS = re.compile(r"\bin\s+(thousands|millions|billions)\b", re.I)
SEED = 20260914


def years(segs):
    return set(y for s in segs for y in YEAR.findall(s))


def loose_number(s):
    """숫자인데 `looks_numeric` 이 못 읽는 표기('$-9.7', '(1.2)%'). 연도 하나는 제외."""
    t = re.sub(r"[\s$,()%\-–—]", "", s)
    return bool(re.fullmatch(r"\d*\.?\d+", t)) and not re.fullmatch(r"(?:19|20)\d{2}", t)


def sample(xs, k=4):
    return random.Random(SEED).sample(xs, min(k, len(xs)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rule", default="v3", choices=sorted(OUT))
    ap.add_argument("--out", default="", help="기본은 규칙별 경로")
    a = ap.parse_args()
    out_path, rn = (Path(a.out) if a.out else OUT[a.rule]), RULE[a.rule]
    if out_path.exists():
        raise SystemExit(f"{out_path} exists — 덮어쓰지 않는다")
    queries, docs, _ = load_population("train")
    v2 = build_tables(docs, "v2")[0]
    c2 = cells(v2)
    out = {"rule": a.rule}

    # S1: v2 열 경로에 있던 연도가 사라진 셀 / 붙은 단계가 전부 단위 표기인 셀
    t1 = build_tables(docs, "v2", rules={rn["S1"]})[0]
    c1 = cells(t1)
    lost = [k for k in c2 if years(c2[k][1]) - years(c1[k][1])]
    added = [k for k in c2 if c1[k][1] != c2[k][1]]
    units_only = [k for k in added if (plus := [s for s in c1[k][1] if s not in c2[k][1]])
                  and all(UNITS.search(s) for s in plus)]
    out["S1"] = {"rule": rn["S1"], "cells_changed": len(added), "cells_year_lost": len(lost),
                 "tables_year_lost": len({k[0] for k in lost}),
                 "cells_only_units_note_added": len(units_only),
                 "examples_year_lost": [{"cell": list(k), "v2": c2[k][3], "S1": c1[k][3]} for k in sample(lost)]}
    del t1, c1

    # S2: 표제를 뗀 행이 빈 표제 데이터 행의 연속(2행 이상)에 속하는가 — 시각적 묶음일 수 있다
    t2 = build_tables(docs, "v2", rules={rn["S2"]})[0]
    c2s = cells(t2)
    rows = sorted({(k[0], k[1]) for k in c2 if c2s[k][0] != c2[k][0]})
    runs, ex = Counter(), {"isolated": [], "run": []}
    for tid, r in rows:
        uid, t_idx = tid.split("::")
        g = t2[tid].table.grid
        _, cover = parse_html_table_layout(docs[uid][0][int(t_idx)])

        def blank_data(x):
            return (x < len(g) and not g[x][0].strip() and cover[x][0] == (x, 0)
                    and any(v.strip() for v in g[x][1:]))
        lo = r
        while lo - 1 >= t2[tid].table.nhr and blank_data(lo - 1):
            lo -= 1
        hi = r
        while blank_data(hi + 1):
            hi += 1
        kind = "isolated" if lo == hi else "run"
        runs[kind] += 1
        if len(ex[kind]) < 3 and (not ex[kind] or ex[kind][-1]["table"] != tid):
            ex[kind].append({"table": tid, "rows": [" | ".join(g[x][:4]) for x in range(max(0, lo - 1), hi + 1)]})
    out["S2"] = {"rule": rn["S2"], "rows_changed": len(rows), "isolated_blank_row": runs["isolated"],
                 "blank_row_in_run_of_2plus": runs["run"], "examples": ex}
    del t2, c2s

    # S3: 헤더로 옮긴 행에 숫자 표기가 있는가
    t3 = build_tables(docs, "v2", rules={rn["S3"]})[0]
    moved = [(tid, r) for tid in v2 for r in range(v2[tid].table.nhr, t3[tid].table.nhr)]
    sus = [(tid, r) for tid, r in moved
           if any(loose_number(v) for v in v2[tid].table.grid[r][v2[tid].table.nhc:] if v.strip())]
    out["S3"] = {"rule": rn["S3"], "tables_nhr_changed": len({t for t, _ in moved}), "rows_moved_to_header": len(moved),
                 "rows_with_number_like_cell": len(sus), "tables_with_number_like_row": len({t for t, _ in sus}),
                 "examples": [" | ".join(v2[t].table.grid[r][:5]) for t, r in sample(sus, 6)]}
    del t3

    # S5·S6: 전체 규칙에서 붙은 단계가 경로의 연도와 어긋나는가
    v3 = build_tables(docs, a.rule)[0]
    c3 = cells(v3)
    s5_bad, s6_rowyear, s6_long = [], [], []
    for k in c2.keys() & c3.keys():
        b, x = c2[k], c3[k]
        plus_r = [s for s in x[0] if s not in b[0]]
        if plus_r and years(plus_r) and years(b[0] + b[1]) and years(plus_r) - years(b[0] + b[1]):
            s5_bad.append(k)
        plus_c = [s for s in x[1] if s not in b[1]]
        if plus_c and x[1][:1] != b[1][:1] and years(x[1][:1]):
            if years(x[0]):
                s6_rowyear.append(k)
            if len(x[1][0]) > 60:
                s6_long.append(k)
    out["S5"] = {"cells_heading_with_other_year_added": len(s5_bad),
                 "examples": [{"v2": c2[k][3], "new": c3[k][3]} for k in sample(s5_bad)]}
    out["S6"] = {"cells_scope_added_but_row_path_has_year": len(s6_rowyear),
                 "tables": len({k[0] for k in s6_rowyear}), "cells_scope_longer_than_60_chars": len(s6_long),
                 "examples": [{"v2": c2[k][3], "new": c3[k][3]} for k in sample(s6_rowyear)]}

    # 모집단: 제외 여부가 바뀐 질의
    p2, g2 = population(queries, v2, {t: (d.table.nhr, d.table.nhc) for t, d in v2.items()})
    p3, g3 = population(queries, v3, {t: (d.table.nhr, d.table.nhc) for t, d in v3.items()})
    ids = set(json.loads((ROOT / "results/mh_interim200/ids_200.json").read_text(encoding="utf-8"))["ids"])
    out["population"] = {"v2": p2, "new": p3,
                         "newly_excluded": sorted(set(g2) - set(g3)), "newly_included": sorted(set(g3) - set(g2)),
                         "newly_excluded_in_interim200": sorted((set(g2) - set(g3)) & ids)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("x", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
