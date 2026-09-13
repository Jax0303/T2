# SPDX-License-Identifier: MIT
"""v3 / v3.1 부작용의 원인 확인 — 각 부작용이 어떤 원표 모양에서 나는지 센다. 모델 없음.

  S1  v2 열 경로에서 단계를 뗀 셀: 끊긴 라벨이 병합 범위를 선언한 칸(colspan/rowspan)인가, 1×1 칸인가
  S3  헤더로 옮긴 행: 숫자 표기('$-87,991')를 읽는 판정이면 거절되는가, 새로 빠진 질의의 gold 행인가
  S4  구획을 닫은 뒤 그 구획 이름의 합계 행('Total <표제>')이 아래에 또 나오는가 (너무 일찍 닫음)
  S5  전체 규칙에서 S5 가 붙인 표제가 S4 가 이미 뗀 표제인가

  HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python analysis/header_v3_side_effects.py              # v3
  HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python analysis/header_v3_side_effects.py --rule v3.1
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

from analysis.header_v3_impact import cells, population                         # noqa: E402
from mh_arms import build_tables, load_population                               # noqa: E402
from rag_agent.reconstruct.header_grid import (_TOTAL_RE, _marked_rows,         # noqa: E402
                                               _number_like, parse_html_table_layout, section_label)

OUT = {"v3": ROOT / "results/mh_header_v3_1/v3_side_effects.json",
       "v3.1": ROOT / "results/mh_header_v3_1/v3_1_side_effects.json",
       "v3.2": ROOT / "results/mh_header_v3_2/v3_2_side_effects.json"}
RULE = {"v3": {"S1": "S1", "S3": "S3", "S4": "S4"}, "v3.1": {"S1": "S1n", "S3": "S3n", "S4": "S4n"},
        "v3.2": {"S1": "S1n", "S3": "S3n", "S4": "S4n2"}}
YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
SEED = 20260914


def years(segs):
    return {y for s in segs for y in YEAR.findall(s)}


def words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rule", default="v3", choices=sorted(OUT))
    ap.add_argument("--out", default="", help="기본은 규칙별 경로")
    a = ap.parse_args()
    out_path, rn = (Path(a.out) if a.out else OUT[a.rule]), RULE[a.rule]
    if out_path.exists():
        raise SystemExit(f"{out_path} exists — 덮어쓰지 않는다")
    queries, docs, _ = load_population("train")
    layout = {}

    def lay(tid):
        if tid not in layout:
            uid, t = tid.split("::")
            layout[tid] = parse_html_table_layout(docs[uid][0][int(t)])
        return layout[tid]

    v2 = build_tables(docs, "v2")[0]
    c2 = cells(v2)
    out = {"rule": a.rule}

    # S1 ------------------------------------------------------------------
    t1 = build_tables(docs, "v2", rules={rn["S1"]})[0]
    c1 = cells(t1)
    cls, cls_year, ex = Counter(), Counter(), {"declared_span": [], "one_by_one": []}
    for k in c2:
        removed = [s for s in c2[k][1] if s not in c1[k][1]]
        if not removed:
            continue
        tid, _r, c = k
        t = v2[tid].table
        grid, cover = lay(tid)
        marked = _marked_rows(cover)
        kinds = []
        for r in range(t.nhr):
            if grid[r][c].strip():
                continue
            o = cover[r][c]
            hard = o is not None and ((o != (r, c) and not grid[o[0]][o[1]].strip()) or (o == (r, c) and r in marked))
            if not hard:
                continue
            left = next((cc for cc in range(c - 1, t.nhc - 1, -1) if grid[r][cc].strip()), None)
            if left is None:
                continue
            extent = sum(1 for line in cover for p in line if p == (r, left))
            kinds.append("declared_span" if extent > 1 else "one_by_one")
        kind = "one_by_one" if "one_by_one" in kinds else ("declared_span" if kinds else "other")
        cls[kind] += 1
        if years(c2[k][1]) - years(c1[k][1]):
            cls_year[kind] += 1
        if kind in ex and len(ex[kind]) < 3:
            ex[kind].append({"v2": c2[k][3], "S1": c1[k][3]})
    out["S1_removed_segment_cells"] = {"rule": rn["S1"], "by_left_label": dict(cls),
                                       "year_lost_by_left_label": dict(cls_year), "examples": ex}
    del t1, c1

    # S3 ------------------------------------------------------------------
    t3 = build_tables(docs, "v2", rules={rn["S3"]})[0]
    moved, refused, first_refused, ex3 = 0, 0, 0, []
    for tid in v2:
        rows = list(range(v2[tid].table.nhr, t3[tid].table.nhr))
        if not rows:
            continue
        g = v2[tid].table.grid
        flags = [any(_number_like(v) for v in g[r][1:] if v.strip()) for r in rows]
        moved += len(rows)
        refused += sum(flags)
        first_refused += flags[0]
        ex3 += [" | ".join(g[r][:5]) for r, f in zip(rows, flags) if f][:1]
    hdr = lambda ts: {t: (d.table.nhr, d.table.nhc) for t, d in ts.items()}  # noqa: E731
    _, g2 = population(queries, v2, hdr(v2))
    _, g3 = population(queries, t3, hdr(t3))
    lost = []
    for q in sorted(set(g2) - set(g3)):
        for tid, r, _c in sorted(g2[q]):
            if v2[tid].table.nhr <= r < t3[tid].table.nhr:
                lost.append({"query_id": q, "row": " | ".join(v2[tid].table.grid[r][:5]),
                             "refused_by_number_test": any(_number_like(v) for v in v2[tid].table.grid[r][1:] if v.strip())})
    out["S3_moved_rows"] = {"rule": rn["S3"], "rows_moved": moved, "rows_refused_by_number_test": refused,
                            "tables_where_first_moved_row_refused": first_refused,
                            "examples_number_like": random.Random(SEED).sample(ex3, min(4, len(ex3))),
                            "newly_excluded_gold_rows": lost}
    del t3

    # S4 ------------------------------------------------------------------
    t4 = build_tables(docs, "v2", rules={rn["S4"]})[0]
    c4 = cells(t4)
    early_rows, closer_kind, ex4 = set(), Counter(), []
    seen_rows = set()
    for k in c2:
        gone = [s for s in c2[k][0] if s not in c4[k][0]]
        if not gone or (k[0], k[1]) in seen_rows:
            continue
        seen_rows.add((k[0], k[1]))
        tid, r, _c = k
        g = v2[tid].table.grid
        h = gone[-1]
        head = max((x for x in range(v2[tid].table.nhr, r) if section_label(g[x]) == h), default=None)
        closer = max((x for x in range(head + 1 if head is not None else 0, r)
                      if _TOTAL_RE.match(g[x][0]) and not section_label(g[x])), default=None)
        if closer is not None:
            rest = words(g[closer][0])
            rest = rest[1:] if rest and rest[0] in ("total", "totals") else rest[2:] if rest[:2] == ["sub", "total"] else rest[1:]
            closer_kind["bare" if not rest else "names_heading" if rest == words(h) else "names_other"] += 1
        own_total = ["total", *words(h)]
        if any(words(g[x][0]) == own_total for x in range(r + 1, len(g))):
            early_rows.add((tid, r))
            if len(ex4) < 3:
                ex4.append({"heading": h, "closer": g[closer][0] if closer is not None else None,
                            "v2": c2[k][3], "S4": c4[k][3]})
    out["S4_rows_losing_heading"] = {"rule": rn["S4"], "rows": len(seen_rows), "closer_label": dict(closer_kind),
                                     "rows_before_their_sections_own_total": len(early_rows),
                                     "tables": len({t for t, _ in early_rows}), "examples": ex4}

    # S5 on top of S4 ----------------------------------------------------
    v3 = build_tables(docs, a.rule)[0]
    c3 = cells(v3)
    rows5 = set()
    for k in c3.keys() & c4.keys():
        if (k[0], k[1]) not in rows5 and [s for s in c3[k][0] if s not in c4[k][0] and s in c2[k][0]]:
            rows5.add((k[0], k[1]))
    out["S5_readds_heading_S4_removed"] = {"rows": len(rows5)}

    ids1047 = {json.loads(line)["query_id"] for line in open(ROOT / "results/mh_arms/mh_cell_hv2_answer_doc.jsonl",
                                                                  encoding="utf-8")}
    ids200 = set(json.loads((ROOT / "results/mh_interim200/ids_200.json").read_text(encoding="utf-8"))["ids"])
    _, gv3 = population(queries, v3, hdr(v3))
    out["evaluation_sets"] = {label: {"excluded": sorted(ids - set(gv3)),
                                      "gold_set_changed": sum(1 for q in ids if q in gv3 and g2.get(q) != gv3[q])}
                              for label, ids in (("ids_1047", ids1047), ("interim200", ids200))}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("x", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
