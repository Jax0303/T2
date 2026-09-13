# SPDX-License-Identifier: MIT
"""v3.1 에서 남는 오류 — 모델 없음.

  S2n  표제를 뗀 빈 표제 행이 위 행들의 합계인가(기계 판정). 합계가 아니면 앞 행의 둘째 줄일 수 있다.
  S1n  끊은 열의 헤더 병합 표기 표본.
  S5n  붙인 구획 표제와 그 행 사이의 표제 열 표본.
S1n·S5n 표본은 판정하지 않는다 — 사람 판정이 없고 자동 판정 신호도 없으므로 오류율로 쓰지 않는다.
두 규칙의 남는 오류는 바뀐 셀 수를 상한으로만 보고한다.

  HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python analysis/header_v3_1_residual.py
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from analysis.header_v3_impact import cells                                   # noqa: E402
from mh_arms import build_tables, load_population                            # noqa: E402
from rag_agent.reconstruct.header_grid import parse_html_table_layout, section_label  # noqa: E402

OUT = ROOT / "results/mh_header_v3_1/residual.json"
SEED = 20260914


def num(s):
    t = s.strip()
    if t in ("—", "–", "-", "$—", "$–", "$-"):
        return 0.0
    neg = t.startswith("(") and t.endswith(")") or "-" in t
    t = re.sub(r"[\s$,()%\-–—]", "", t)
    try:
        return -float(t) if neg else float(t)
    except ValueError:
        return None


def header_view(grid, cover, nhr):
    ext = {}
    for line in cover:
        for p in line:
            if p is not None:
                ext.setdefault(p, []).append(p)
    pos = {}
    for rr, line in enumerate(cover):
        for cc, p in enumerate(line):
            if p is not None:
                pos.setdefault(p, []).append((rr, cc))
    rows = []
    for r in range(nhr):
        out = []
        for c, text in enumerate(grid[r]):
            o = cover[r][c]
            if o is None:
                out.append("∅")
            elif o != (r, c):
                out.append("^")
            else:
                cs = pos[(r, c)]
                w = len({x[1] for x in cs})
                h = len({x[0] for x in cs})
                out.append((text.strip() or "·") + (f"[{w}x{h}]" if w > 1 or h > 1 else ""))
        rows.append(" | ".join(out))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT))
    out_path = Path(ap.parse_args().out)
    if out_path.exists():
        raise SystemExit(f"{out_path} exists — 덮어쓰지 않는다")
    _, docs, _ = load_population("train")
    v2 = build_tables(docs, "v2")[0]
    c2 = cells(v2)
    out = {}

    def cover_of(tid):
        uid, t = tid.split("::")
        return parse_html_table_layout(docs[uid][0][int(t)])[1]

    # S2n: 합계 검사 -------------------------------------------------------
    t2 = build_tables(docs, "v2", rules={"S2n"})[0]
    c2n = cells(t2)
    rows = sorted({(k[0], k[1]) for k in c2 if c2n[k][0] != c2[k][0]})
    total_like, not_total = [], []
    for tid, r in rows:
        t = v2[tid].table
        g = t.grid
        above, x = [], r - 1
        while x >= t.nhr and g[x][0].strip() and not section_label(g[x]) and any(v.strip() for v in g[x][t.nhc:]):
            above.append(x)
            x -= 1
        hit = False
        for c in range(t.nhc, len(g[0])):
            tv = num(g[r][c])
            if tv is None or not g[r][c].strip():
                continue
            vals = [num(g[y][c]) for y in above]
            run = 0.0
            for k, v in enumerate(vals, 1):
                if v is None:
                    break
                run += v
                if k >= 2 and abs(run - tv) <= max(1.0, abs(tv) * 0.005):
                    hit = True
                    break
            if hit:
                break
        (total_like if hit else not_total).append((tid, r))
    out["S2n"] = {"rows_changed": len(rows), "total_of_rows_above": len(total_like), "not_a_total": len(not_total),
                  "examples_not_a_total": [
                      {"table": tid, "rows": [" | ".join(v2[tid].table.grid[y][:4]) for y in range(max(v2[tid].table.nhr, r - 2), r + 1)]}
                      for tid, r in random.Random(SEED).sample(not_total, min(8, len(not_total)))]}
    del t2, c2n

    # S1n: 표본 -----------------------------------------------------------
    t1 = build_tables(docs, "v2", rules={"S1n"})[0]
    c1 = cells(t1)
    changed = sorted(k for k in c2 if c1[k][1] != c2[k][1])
    by_table = {}
    for k in changed:
        by_table.setdefault(k[0], k)
    picks = random.Random(SEED).sample(sorted(by_table), min(10, len(by_table)))
    out["S1n"] = {"cells_changed": len(changed), "tables": len(by_table), "sample": [
        {"table": tid, "column": by_table[tid][2],
         "header": header_view(v2[tid].table.grid, cover_of(tid), v2[tid].table.nhr),
         "v2_col_path": list(c2[by_table[tid]][1]), "new_col_path": list(c1[by_table[tid]][1])} for tid in picks]}
    del t1, c1

    # S5n: 표본 -----------------------------------------------------------
    t5 = build_tables(docs, "v2", rules={"S5n"})[0]
    c5 = cells(t5)
    changed = sorted(k for k in c2 if c5[k][0] != c2[k][0])
    rows5 = {}
    for k in changed:
        rows5.setdefault((k[0], k[1]), k)
    picks = random.Random(SEED).sample(sorted(rows5), min(10, len(rows5)))
    sample = []
    for tid, r in picks:
        k = rows5[(tid, r)]
        added = [s for s in c5[k][0] if s not in c2[k][0]]
        g, nhr = v2[tid].table.grid, v2[tid].table.nhr
        start = max((y for y in range(nhr, r) if section_label(g[y]) in added), default=nhr)
        sample.append({"table": tid, "added": added, "v2_row_path": list(c2[k][0]), "new_row_path": list(c5[k][0]),
                       "stub_from_added_heading": [("§ " if section_label(g[y]) else "  ") + g[y][0].strip()
                                                   for y in range(start, r + 1)][:25]})
    out["S5n"] = {"rows_changed": len(rows5), "sample": sample}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("x", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
