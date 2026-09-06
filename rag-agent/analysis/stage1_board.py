#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""1단계 보드 — 쿼리 종류 × (표 검색, 표 조건부 셀 검색). 리더 없음.

`CLAUDE.md` §3 1단계가 요구하는 보고 형태. 검색은 코퍼스 전체 셀을 한 번에 정렬하고
(표 검색 단계가 따로 없다, `results/tabsig/VERDICT.md` §0), 여기서 두 축을 갈라 읽는다:

  표@k          gold 표가 셀 투표 순위 k 안        (`table_rank_cellvote`)
  셀|표@k       gold 표 **안에서만** 셀을 정렬했을 때 gold 셀 전부가 k 안
                (`ranks_in_table` = 오라클 표 게이팅; "표를 맞혔을 때 셀을 맞히는가")
  전체@k        코퍼스 전체 정렬에서 gold 셀 전부가 k 안 (= all-covered@k, 주지표)
  표@1×셀|표@k  표 1등을 먼저 고르고 그 안에서 셀을 고르는 엄격한 2단계의 값.
                전체@k 보다 낮다 — 다른 표의 셀이 1등이어도 gold 가 k 안이면 전체@k 는 맞음.

입력은 `analysis/cell_rank_dump.py` 의 `*_ranks.jsonl`. 파일명에서 모집단을 읽는다.

  PYTHONPATH=. .venv/bin/python analysis/stage1_board.py results/p0_confirm/*_ranks.jsonl \
      results/stage1/*_ranks.jsonl --out results/stage1/BOARD.md
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

KS = (1, 3, 5, 10, 20, 50)
POP = re.compile(r"^(hitab_(?:dev|test|train(?:_sel|_fit)?)_[a-z_]+?)_S3c(?:_page|_sig|_drop)?(_clean)?_")


def load(path):
    return [json.loads(l) for l in open(path)]


def row(name, R):
    n = len(R)
    t1 = [r for r in R if r["table_rank_cellvote"] == 1]
    tab = {k: sum(r["table_rank_cellvote"] <= k for r in R) / n for k in (1, 3, 10)}
    cell = {k: sum(max(r["ranks_in_table"]) < k for r in R) / n for k in KS}
    full = {k: sum(max(r["ranks"]) < k for r in R) / n for k in KS}
    casc = {k: sum(max(r["ranks_in_table"]) < k for r in t1) / n for k in KS}
    m = sorted(r["m"] for r in R)
    miss = [r for r in R if max(r["ranks"]) >= 10]
    tw = sum(r["table_rank_cellvote"] != 1 for r in miss)
    return {"pop": name, "n": n, "m_median": m[n // 2], "m_max": m[-1],
            "table": tab, "cell_given_table": cell, "full": full, "cascade": casc,
            "miss10": len(miss), "miss10_table_wrong": tw,
            "miss10_table_right": len(miss) - tw}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="results/stage1/BOARD.md")
    a = ap.parse_args()
    rows = []
    for f in a.files:
        m = POP.match(Path(f).name)
        name = (m.group(1) + (m.group(2) or "")) if m else Path(f).stem
        rows.append(row(name, load(f)) | {"file": f})

    L = ["# 1단계 보드 — 쿼리 종류 × (표 검색, 표 조건부 셀 검색)", "",
         "계측기 `analysis/stage1_board.py`. 리더 없음. 각 행의 출처 파일은 맨 아래.", "",
         "| 모집단 | n | m 중앙/최대 | 표@1 | 표@3 | 셀\\|표@1 | 셀\\|표@3 | 셀\\|표@5 | "
         "셀\\|표@10 | 셀\\|표@20 | 전체@10 | 표@1×셀\\|표@10 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        c, t, f, s = r["cell_given_table"], r["table"], r["full"], r["cascade"]
        L.append(f"| `{r['pop']}` | {r['n']} | {r['m_median']}/{r['m_max']} | {t[1]:.4f} | "
                 f"{t[3]:.4f} | {c[1]:.4f} | {c[3]:.4f} | {c[5]:.4f} | {c[10]:.4f} | "
                 f"{c[20]:.4f} | {f[10]:.4f} | {s[10]:.4f} |")
    L += ["", "## 전체@10 실패의 위치", "",
          "| 모집단 | 실패 | 표를 틀림 | 표는 맞고 셀을 틀림 |", "|---|---:|---:|---:|"]
    for r in rows:
        L.append(f"| `{r['pop']}` | {r['miss10']} | {r['miss10_table_wrong']} | "
                 f"{r['miss10_table_right']} |")
    L += ["", "## 출처", ""] + [f"- `{r['pop']}` ← `{r['file']}`" for r in rows]
    L += ["", "셀|표@k 는 **오라클** 표 게이팅이다 — 표를 맞힌다고 가정하고 그 표 안에서만 "
          "정렬한 값. 표@1×셀|표@k 는 실제 2단계 캐스케이드가 내는 값이고, 전체@k 는 "
          "현 파이프라인(코퍼스 전체 한 번 정렬)의 값이다. m>k 인 질의는 정의상 @k 에서 0이다."]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    out.with_suffix(".json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
