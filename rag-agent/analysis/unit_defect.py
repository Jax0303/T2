#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""단위 불일치 라벨 결함 — 질문은 퍼센트를 묻는데 정답은 비율로 실려 있다.

HiTab 의 일부 질의는 ``answer_formulas`` 가 ``=C4/B4`` 처럼 순수 나눗셈이고,
그 몫을 그대로 정답으로 싣는다. 그런데 질문은 "how many percent ...", "what was
the percentage ..." 이다. 정답 ``0.691446`` 과 질문이 요구하는 ``69.1%`` 는 같은
양의 다른 단위이므로, 어느 쪽을 답해도 한쪽 기준에서는 오답이 된다.

**판정은 예측을 보지 않는다.** 질의와 데이터셋 주석만으로 결정되므로, 이 규칙에
걸리는 질의는 우리가 맞힌 것까지 전부 제외된다 — 결과를 보고 고르면 그것은
모집단 고르기이지 결함 제외가 아니다.

  PYTHONPATH=. .venv/bin/python analysis/unit_defect.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_agent.data.loader import load_samples                        # noqa: E402

ASKS_PCT = re.compile(r"\b(percent|percentage|proportion|share)\b", re.I)
BARE_DIV = re.compile(r"^=\s*\$?[A-Z]{1,2}\$?\d+\s*/\s*\$?[A-Z]{1,2}\$?\d+\s*$")
IS_PCT_HDR = re.compile(r"percent|%|rate|proportion|share", re.I)


def is_unit_defect(sample: dict) -> bool:
    agg = sample.get("aggregation")
    agg = agg[0] if isinstance(agg, list) and agg else agg
    if agg != "div":
        return False
    if not ASKS_PCT.search(sample.get("question", "")):
        return False
    fs = sample.get("answer_formulas") or []
    if not (len(fs) == 1 and BARE_DIV.match(str(fs[0]).strip())):
        return False
    ans = sample.get("answer") or []
    if len(ans) != 1:
        return False
    try:
        v = float(str(ans[0]).replace(",", ""))
    except (TypeError, ValueError):
        return False
    return 0.0 < v < 1.0            # 몫을 그대로 실었다는 표시


def is_wrong_column(sample: dict, table) -> bool:
    """질문은 퍼센트를 묻는데 gold 셀이 그 표의 *다른 단위 열*에 달려 있다.

    예: "what was the percentage of alberta reporting the most amount of natural
    pasture area" 의 정답이 ``6435825.0`` (hectares 열). 같은 표에 percent 열이
    나란히 있는데 주석이 hectares 셀을 가리킨다. 리더가 퍼센트를 답하면 오답이
    되지만, 질문에 답한 쪽은 리더다.

    표에 퍼센트 열이 실제로 있을 때만 인정한다 — 없으면 질문 표현이 느슨한 것이지
    주석이 틀린 것이 아니다.
    """
    if not ASKS_PCT.search(sample.get("question", "")):
        return False
    if table is None:
        return False
    cells, _mode, why = _gold(sample, table)
    if why or not cells:
        return False
    t = table.table
    paths = []
    for _tid, i, j in cells:
        paths += list(t.row_path(i)) + list(t.col_path(j))
    if IS_PCT_HDR.search(" ".join(paths)):
        return False                       # gold 가 이미 퍼센트 열이면 정상
    return any(IS_PCT_HDR.search(" ".join(t.col_path(c))) for c in range(t.n_cols))


def _gold(sample, table):
    from rag_agent.bench.hitab_grid import gold_target
    return gold_target(sample, table)


def defect_ids(data_dir="data/hitab", split="test") -> set:
    """두 부류를 합친 결함 질의 id.

    (1) 몫을 그대로 실은 비율/퍼센트 불일치 — :func:`is_unit_defect`
    (2) 다른 단위 열을 가리킨 주석      — :func:`is_wrong_column`

    둘 다 질의와 주석만으로 판정하며 예측을 보지 않는다.
    """
    from rag_agent.bench.hitab_grid import load_table
    out, tabs = set(), {}
    for s in load_samples(data_dir, split):
        if is_unit_defect(s):
            out.add(s["id"]); continue
        tid = s["table_id"]
        if tid not in tabs:
            tabs[tid] = load_table(tid, data_dir)
        if is_wrong_column(s, tabs[tid]):
            out.add(s["id"])
    return out


if __name__ == "__main__":
    ids = defect_ids()
    D = ROOT / "results/retrieval_accuracy"
    recs = [json.loads(l) for l in (D / "t_s3c_hybrid_records.jsonl").open()
            if '"correct"' in l]
    pop = [r for r in recs if r["mode"] == "all"]
    hit = [r for r in pop if r["query_id"] in ids]
    print(f"단위 결함 질의 (test 전체): {len(ids)}건")
    print(f"  그중 주지표 모집단 안: {len(hit)}건 / {len(pop)}건")
    for leg in ("retrieved", "gold"):
        f = D / f"t_s3c_hybrid_answer_{leg}.jsonl"
        if not f.exists():
            continue
        a = [json.loads(l) for l in f.open() if json.loads(l)["mode"] == "all"]
        inn = [x for x in a if x["query_id"] in ids]
        out = [x for x in a if x["query_id"] not in ids]
        ok = lambda v: sum(x["answer_correct"] for x in v) / len(v) if v else 0
        print(f"  [{leg}] 결함 질의 {len(inn)}건의 현재 정답률 {ok(inn):.4f} "
              f"| 제외 후 {len(out)}건 정답률 {ok(out):.4f} "
              f"(제외 전 {ok(a):.4f})")
    r_ok = lambda v: sum(x["correct"] for x in v) / len(v) if v else 0
    print(f"  [검색] 결함 질의 {len(hit)}건의 검색 정확도 {r_ok(hit):.4f} "
          f"| 제외 후 {len(pop)-len(hit)}건 {r_ok([r for r in pop if r['query_id'] not in ids]):.4f} "
          f"(제외 전 {r_ok(pop):.4f})")
