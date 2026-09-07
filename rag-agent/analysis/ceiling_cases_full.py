#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""리더 천장의 오답 전수 분석 — 정답 셀만 줬는데 왜 틀렸나.

`gold` 조건은 리더에게 그 질의의 정답 셀만, 그 셀의 헤더 경로와 함께 넣는다.
검색이 완벽해도(1.0) 답변은 이 조건을 넘지 못하므로, 여기서 틀린 몫은
**검색으로 회수 불가능한 상한**이다. 그 몫을 한 건도 남기지 않고 분류한다.

분류는 상호배타적이고 전수를 덮는다. 각 분류마다 "무엇을 바꾸면 회수되는가"를
같이 적는다 — 회수 경로가 다르면 같은 오답이라도 다른 문제이기 때문이다.

산출: ``results/retrieval_accuracy/CEILING_CASES.json`` (건별 근거 포함)

  PYTHONPATH=. .venv/bin/python analysis/ceiling_cases_full.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations, permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_agent.bench import hitab_grid as hg                          # noqa: E402
from rag_agent.data.loader import load_samples                        # noqa: E402

D = ROOT / "results/retrieval_accuracy"
NUM = re.compile(r"-?\d[\d,]*\.?\d*")

# 분류 -> (회수 경로, 설명)
ROUTES = {
    "산술 정밀도":      ("리더 교체", "계산은 맞았는데 자릿수가 데이터셋 표기에 못 미침"),
    "단위 스케일":      ("데이터셋/채점 규약", "비율과 퍼센트 — 같은 양의 다른 단위"),
    "집계 미수행":      ("리더 교체", "피연산자를 나열하고 연산을 하지 않음"),
    "연산 오선택":      ("리더 교체", "계산은 했으나 다른 연산을 함"),
    "다중값 순서":      ("채점 규약", "값 집합은 같고 순서만 다름"),
    "다중값 일부 누락": ("리더 교체", "요구된 값 중 일부만 답함"),
    "셀 오독":          ("리더 교체", "문맥에 있는 다른 정답 셀의 값을 답함"),
    "환각":             ("리더 교체", "문맥에 없는 값을 답함"),
    "비수치 응답":      ("리더 교체", "숫자를 내놓지 않음"),
    "gold 비수치":      ("측정 불가", "정답 자체가 숫자가 아니라 이 축으로 못 잼"),
}


def floats(x):
    out = []
    for m in NUM.finditer(str(x)):
        try:
            out.append(float(m.group(0).replace(",", "")))
        except ValueError:
            pass
    return out


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(abs(b), 1.0) or abs(a - b) < 1e-9


def rounded(pred, gold):
    """pred 가 gold 를 자릿수만 줄여 쓴 것인가 (소수 N자리 / 유효숫자 N자리, N<=5)."""
    for n in range(6):
        if abs(round(gold, n) - pred) < 1e-9:
            return True
    for n in range(1, 6):
        if gold == 0:
            break
        try:
            if abs(float(f"%.{n}g" % gold) - pred) < 1e-9:
                return True
        except (ValueError, OverflowError):
            break
    return False


def classify(pred, gold, cell_vals, agg, m):
    p, g = floats(pred), floats(gold if isinstance(gold, list) else [gold])
    if not g:
        return "gold 비수치"
    if not str(pred).strip() or not p:
        return "비수치 응답"
    computed = (agg or "none") != "none"

    if len(g) == 1:
        b = g[0]
        if len(p) == 1:
            a = p[0]
            if rounded(a, b):
                return "산술 정밀도" if computed else "셀 오독"
            for k in (100.0, 0.01):
                if rounded(a, b * k) or close(a, b * k, 1e-3):
                    return "단위 스케일"
            if any(close(a, v, 1e-9) or rounded(a, v) for v in cell_vals):
                # 계산 질의에서 피연산자를 그대로 답한 것은 오독이 아니라
                # 연산을 하지 않은 것이다. 처방이 다르므로 먼저 가른다.
                return "집계 미수행" if computed else "셀 오독"
            # 계산은 했는데 다른 연산인가
            if computed and len(cell_vals) >= 2:
                for x, y in permutations(cell_vals, 2):
                    for cand in (x + y, x - y, x * y, (x / y if y else None)):
                        if cand is not None and rounded(a, cand):
                            return "연산 오선택"
            return "환각"
        # 예측이 여러 값, gold 는 하나
        if close(sum(p), b, 1e-3) or rounded(sum(p), b):
            return "집계 미수행"
        for x, y in combinations(p, 2):
            for cand in (abs(x - y), (x / y if y else None), (y / x if x else None)):
                if cand is not None and rounded(cand, b):
                    return "집계 미수행"
        if all(any(close(a, v, 1e-9) for v in cell_vals) for a in p):
            return "집계 미수행"
        return "환각"

    # gold 가 여러 값
    if len(p) == len(g):
        if sorted(p) != p and all(rounded(a, b) for a, b in zip(sorted(p), sorted(g))):
            return "다중값 순서"
        if all(rounded(a, b) for a, b in zip(p, g)):
            return "산술 정밀도" if computed else "셀 오독"
        if computed and all(any(close(x, v, 1e-9) for v in cell_vals) for x in p):
            return "집계 미수행"
        return "셀 오독"
    if len(p) < len(g) and all(any(rounded(a, b) for b in g) for a in p):
        return "다중값 일부 누락"
    return "셀 오독"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--exclude-defect", action="store_true",
                    help="analysis/unit_defect.py 가 잡는 라벨 결함 질의를 모집단에서 뺀다")
    ap.add_argument("--out", default=str(D / "CEILING_CASES.json"))
    a = ap.parse_args()

    ans = {r["query_id"]: r for r in
           map(json.loads, (D / "t_s3c_hybrid_answer_gold.jsonl").open())
           if r["mode"] == "all"}
    recs = {json.loads(l)["query_id"]: json.loads(l)
            for l in (D / "t_s3c_hybrid_records.jsonl").open() if '"correct"' in l}
    samples = {s["id"]: s for s in load_samples(a.data_dir, "test")}
    from analysis.unit_defect import defect_ids
    defect = defect_ids(a.data_dir, "test")

    if a.exclude_defect:
        ans = {q: r for q, r in ans.items() if q not in defect}
    tabs, cases = {}, []
    for qid, r in ans.items():
        if r["answer_correct"]:
            continue
        rec, s = recs.get(qid, {}), samples.get(qid, {})
        tid = rec.get("table_id")
        if tid and tid not in tabs:
            tabs[tid] = hg.load_table(tid, a.data_dir)
        tab = tabs.get(tid)
        cells, vals = [], []
        for _t, i, j in rec.get("gold_cells", []):
            if tab is None:
                continue
            v = tab.table.data[i][j]
            vals.extend(floats(v))
            cells.append({"rc": [i, j], "value": str(v),
                          "row_path": list(tab.table.row_path(i)),
                          "col_path": list(tab.table.col_path(j))})
        agg = s.get("aggregation")
        agg = agg[0] if isinstance(agg, list) and agg else agg
        cls = ("단위 스케일" if qid in defect else
               classify(r["pred"], r["answer"], vals, agg, rec.get("m", 1)))
        cases.append({
            "query_id": qid, "question": s.get("question"), "gold": r["answer"],
            "pred": r["pred"], "aggregation": agg, "m": rec.get("m"),
            "formula": s.get("answer_formulas"), "gold_cells": cells,
            "class": cls, "route": ROUTES[cls][0],
            "dataset_defect": qid in defect,
        })

    Path(a.out).write_text(json.dumps(cases, ensure_ascii=False, indent=1))
    n_all = len(ans)
    cls = Counter(c["class"] for c in cases)
    route = Counter(c["route"] for c in cases)
    print(f"gold 조건 {n_all}건 중 오답 {len(cases)}건 (EM {1-len(cases)/n_all:.4f})")
    print(f"→ 검색을 1.0 으로 만들어도 남는 몫 = {len(cases)/n_all:.4f}\n")
    print(f"{'분류':16}{'건수':>5}{'오답중':>8}{'전체중':>8}  회수 경로")
    print("-" * 72)
    for k, v in cls.most_common():
        print(f"{k:16}{v:5}{v/len(cases):>8.1%}{v/n_all:>8.2%}  {ROUTES[k][0]}")
    print("-" * 72)
    for k, v in route.most_common():
        print(f"{'':16}{v:5}{v/len(cases):>8.1%}{v/n_all:>8.2%}  <= {k}")
    print(f"\n건별 근거: {a.out}")
    ex = defaultdict(list)
    for c in cases:
        if len(ex[c["class"]]) < 2:
            ex[c["class"]].append(c)
    for k, _v in cls.most_common():
        print(f"\n[{k}] {ROUTES[k][1]}")
        for c in ex[k]:
            gv = " / ".join(f"{x['value']}" for x in c["gold_cells"][:4])
            print(f"  Q {(c['question'] or '')[:66]}")
            print(f"    gold={c['gold']} pred={c['pred'][:34]!r} "
                  f"agg={c['aggregation']} 수식={c['formula']}")
            print(f"    준 셀 값: {gv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
