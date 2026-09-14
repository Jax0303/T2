#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""세 유형 검색 정확도를 이미 저장된 검색 레코드에서 다시 낸다 — 비교군까지 같은 자로.

`scripts/retrieval_accuracy.py` 는 한 실행에서 검색과 채점을 같이 한다. 비교군을
같은 자로 채점하려면 그 실행을 다시 해야 하는데, 각 arm 의 `_records.jsonl` 에는
질의별 gold 셀과 배달된 셀 좌표가 이미 들어 있다. 이 스크립트는 그 좌표만 읽어
**채점 함수 자체는 `scripts/retrieval_accuracy.py` 에서 그대로 가져다 쓴다** —
규칙이 두 벌 생기지 않게, `query_type` / `type_row` / `type_accuracy` 를 import 한다.

판정은 하나다: gold 셀 집합 G 가 배달된 셀 집합 R 에 **전부** 들어 있으면 1,
하나라도 빠지면 0 (G ⊆ R). 셀 예산은 각 실행의 검색 설정으로만 적고 지표 이름에
붙이지 않는다. 답이 헤더인 질의(``mode="any"``)는 G 가 "이 중 하나"의 범위라
G ⊆ R 판정이 성립하지 않으므로 세 유형 밖에서 따로 센다.

두 가지 증거 등급을 구분해서 적는다:

``recomputed_from_cells``  레코드가 ``context_cells`` 를 들고 있다(context_version=2).
    G ⊆ R 을 좌표에서 처음부터 다시 계산하고, 저장된 ``correct`` 와 일치하는지
    질의마다 확인한다. 한 건이라도 어긋나면 실행을 멈춘다. 부분 회수율
    (evidence_recall) 진단값도 이때만 나온다.
``stored_pass_fail``  구형 레코드에 셀 좌표가 없다. 저장된 ``correct`` (같은 코드가
    같은 G ⊆ R 규칙으로 매긴 값)를 그대로 쓰고, 진단값은 ``null`` 로 비운다.
    정확도는 유효하고 회수율만 없다.

  PYTHONPATH=. python3 analysis/type_accuracy_offline.py \
      --arm s3c=results/evaluation_v2/s3c_v2_records.jsonl \
      --verify results/retrieval_accuracy/t_s3c_hybrid_qtype_type_accuracy.jsonl \
      --out results/type_accuracy/hitab_test_arms.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rag_agent.eval.artifacts import digest, file_digest, provenance, read_records
from retrieval_accuracy import TYPES, query_type, type_accuracy, type_row

Z95 = 1.959963984540054


def as_cells(coordinates) -> set:
    """JSON 리스트 좌표를 좌표 집합으로. 중복 좌표는 데이터 결함이라 멈춘다."""
    cells = [tuple(c) for c in coordinates]
    if len(cells) != len(set(cells)):
        raise ValueError("duplicate cell coordinate in record")
    return set(cells)


def rows_from_records(records: dict, budget: int) -> tuple[list, str]:
    """레코드 → 질의별 판정 행. 증거 등급도 같이 돌려준다."""
    rows, grades = [], set()
    for qid, r in records.items():
        if "correct" not in r:
            if not r.get("excluded"):
                raise ValueError(f"{qid}: unscored record without an exclusion reason")
            rows.append(type_row({"query_id": qid, "gold": set(), "mode": r.get("mode", "all"),
                                  "aggregation": r.get("aggregation"),
                                  "excluded": r["excluded"]}, None, budget))
            continue
        if r.get("context_cells") is not None:
            grades.add("recomputed_from_cells")
            gold, got = as_cells(r["gold_cells"]), as_cells(r["context_cells"])
            if len(gold) != r["m"]:
                raise ValueError(f"{qid}: gold cell count differs from m")
            if r["cells_in_context"] != len(got):
                raise ValueError(f"{qid}: cells_in_context differs from delivered coordinates")
            row = type_row({"query_id": qid, "gold": gold, "mode": r["mode"],
                            "aggregation": r.get("aggregation"), "excluded": None}, got, budget)
            hit = (gold <= got) if r["mode"] == "all" else bool(gold & got)
            if int(hit) != r["correct"]:
                raise ValueError(f"{qid}: recomputed G subset R disagrees with stored correct")
            if row["query_type"] in TYPES and row["retrieval_success"] != r["correct"]:
                raise ValueError(f"{qid}: type row disagrees with stored correct")
        else:
            # 배달된 셀 좌표가 없는 구형 레코드. 판정은 저장된 ``correct`` 를 쓴다 —
            # 같은 코드가 같은 G ⊆ R 규칙으로 매긴 값이다. 회수율은 계산할 수 없으니
            # 지어내지 않고 ``null`` 로 둔다. gold 목록도 믿지 않는다: 구형 실행은
            # 이 필드를 64개에서 잘랐고(m>=64 인 mode="any" 29건, 세 유형 밖),
            # 아예 저장하지 않은 실행도 있다. 개수는 자르기 전 값인 ``m`` 을 쓴다.
            grades.add("stored_pass_fail")
            probe = {"query_id": qid, "gold": [None] * r["m"], "mode": r["mode"],
                     "aggregation": r.get("aggregation"), "excluded": None}
            row = {"query_id": qid, "query_type": query_type(probe),
                   "retrieval_budget_cells": budget, "gold_cell_ids": None,
                   "retrieved_cell_ids": None, "num_gold_cells": r["m"],
                   "num_retrieved_cells": r.get("cells_in_context"),
                   "num_gold_retrieved": None, "evidence_recall": None,
                   "retrieval_success": r["correct"], "over_budget": None,
                   "unresolved_gold_reason": None}
        rows.append(row)
    if len(grades) > 1:
        raise ValueError(f"one arm mixes evidence grades: {sorted(grades)}")
    return rows, grades.pop() if grades else "empty"


def aggregate(rows) -> dict:
    """유형별·overall·macro 정확도. 진단값이 없는 arm 은 회수율만 ``null`` 이다.

    좌표가 다 있는 arm 에서는 이 함수의 결과가 `scripts/retrieval_accuracy.py` 의
    ``type_accuracy`` 와 **글자 단위로 같아야** 한다 (`check_against_committed`).
    """
    def rate(num, n):
        return round(num / n, 4) if n else None

    def block(v):
        recalls = [r["num_gold_retrieved"] / r["num_gold_cells"] for r in v
                   if r["num_gold_retrieved"] is not None]
        over = [r["over_budget"] for r in v if r["over_budget"] is not None]
        return {"n": len(v), "success": sum(r["retrieval_success"] for r in v),
                "accuracy": rate(sum(r["retrieval_success"] for r in v), len(v)),
                "mean_evidence_recall_DIAGNOSTIC":
                    rate(sum(recalls), len(v)) if len(recalls) == len(v) else None,
                "over_budget": sum(over) if len(over) == len(v) else None}

    scored = [r for r in rows if r["retrieval_success"] is not None]
    out = {t: block([r for r in scored if r["query_type"] == t]) for t in TYPES}
    out["overall"] = block([r for r in scored if r["query_type"] in TYPES])
    out["overall"]["macro_accuracy"] = (
        rate(sum(out[t]["success"] / out[t]["n"] for t in TYPES), len(TYPES))
        if all(out[t]["n"] for t in TYPES) else None)
    out["n_excluded"] = sum(1 for r in rows if r["unresolved_gold_reason"])
    return out


def wilson(success: int, n: int, z: float = Z95) -> list | None:
    """이항 비율의 95% Wilson 구간. 다중 셀은 n=38 이라 점추정만 적으면 오독된다."""
    if not n:
        return None
    p = success / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def pairwise_tests(table) -> dict:
    """유형 사이의 정확도 차가 표본으로 갈리는지 — Fisher 정확검정(양측).

    다중 셀은 n=38 이다. 단일 셀보다 낮은 점추정이 나와도 그 차이가 표본 크기 안에서
    구별되지 않을 수 있으므로, 가설을 확인했다고 쓰기 전에 이 p 값을 본다.
    """
    from scipy.stats import fisher_exact
    out = {}
    for a, b in (("single_cell", "multi_cell"), ("single_cell", "arithmetic"),
                 ("multi_cell", "arithmetic")):
        x, y = table[a], table[b]
        if not x["n"] or not y["n"]:
            continue
        odds, p = fisher_exact([[x["success"], x["n"] - x["success"]],
                                [y["success"], y["n"] - y["success"]]])
        out[f"{a}_vs_{b}"] = {
            "accuracy": [x["accuracy"], y["accuracy"]],
            "difference": round(x["accuracy"] - y["accuracy"], 4),
            "fisher_exact_two_sided_p": float(f"{float(p):.3g}"),
            "separated_at_05": bool(p < 0.05)}
    return out


def check_against_committed(rows, table) -> None:
    """좌표가 다 있으면 커밋된 채점 함수와 결과가 같은지 확인한다."""
    if any(r["num_gold_retrieved"] is None for r in rows
           if r["retrieval_success"] is not None):
        return
    if table != type_accuracy(rows):
        raise ValueError("offline aggregation differs from scripts/retrieval_accuracy.py")


def header_answer_count(rows) -> int:
    return sum(1 for r in rows if r["query_type"] == "header_answer"
               and r["retrieval_success"] is not None)


def score_arm(name: str, records_path: Path, budget: int) -> dict:
    records = read_records(records_path)
    summary_path = records_path.with_suffix(".json")
    meta = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    if meta.get("budget_cells") not in (None, budget):
        raise ValueError(f"{name}: run used budget {meta['budget_cells']}, expected {budget}")
    rows, grade = rows_from_records(records, budget)
    table = aggregate(rows)
    check_against_committed(rows, table)
    for key in (*TYPES, "overall"):
        table[key]["accuracy_ci95"] = wilson(table[key]["success"], table[key]["n"])
    if meta.get("accuracy_all_mode") is not None:
        pooled = [r for r in rows if r["retrieval_success"] is not None
                  and r["query_type"] in TYPES]
        got = round(sum(r["retrieval_success"] for r in pooled) / len(pooled), 4)
        if got != meta["accuracy_all_mode"]:
            raise ValueError(f"{name}: pooled accuracy {got} differs from the run's "
                             f"accuracy_all_mode {meta['accuracy_all_mode']}")
    return {"arm": name, "records": str(records_path).replace("\\", "/"),
            "records_sha256": file_digest(records_path), "evidence": grade,
            "retrieval_setting": {"split": meta.get("split"), "corpus": meta.get("corpus"),
                                  "unit": meta.get("unit"), "template": meta.get("template"),
                                  "encoder": meta.get("encoder"), "alpha": meta.get("alpha"),
                                  "budget_cells": meta.get("budget_cells", budget),
                                  "n_units": meta.get("n_units"),
                                  "context_version": meta.get("context_version")},
            "type_accuracy": table,
            "pairwise_tests": pairwise_tests(table),
            "n_header_answer_not_typed": header_answer_count(rows),
            "rows": rows}


def verify_rows(rows, reference_path: Path) -> dict:
    """커밋된 질의별 판정 파일과 한 줄씩 대조한다 — 규칙이 같은지 증명한다.

    대조 모집단은 **채점된 질의**다. 제외 질의는 두 파일 모두 판정이 ``null`` 이지만
    유형 이름이 다를 수 있다 — 검색 실행은 데이터셋의 ``aggregation`` 을 그대로
    들고 있고, 레코드에는 그 필드가 남지 않는다. 판정에 쓰이지 않는 이름이므로
    제외 질의는 id 집합만 같은지 본다.
    """
    want = read_records(reference_path)
    # 커밋된 파일은 세 유형 + 제외만 담는다. 헤더답 질의는 애초에 행이 없다.
    mine = {r["query_id"]: r for r in rows
            if r["retrieval_success"] is not None and r["query_type"] in TYPES}
    theirs = {q: r for q, r in want.items()
              if r["retrieval_success"] is not None and r["query_type"] in TYPES}
    if set(mine) != set(theirs):
        raise ValueError(f"scored population differs (mine={len(mine)}, reference={len(theirs)})")
    mine_out = {r["query_id"] for r in rows if r["retrieval_success"] is None}
    theirs_out = {q for q, r in want.items() if r["retrieval_success"] is None}
    if mine_out != theirs_out:
        raise ValueError("excluded query ids differ")
    bad = [q for q, r in theirs.items()
           if (mine[q]["query_type"], mine[q]["retrieval_success"], mine[q]["num_gold_cells"],
               mine[q]["num_gold_retrieved"])
           != (r["query_type"], r["retrieval_success"], r["num_gold_cells"],
               r["num_gold_retrieved"])]
    if bad:
        raise ValueError(f"{len(bad)} per-query verdicts differ, e.g. {bad[:3]}")
    return {"reference": str(reference_path).replace("\\", "/"),
            "reference_sha256": file_digest(reference_path),
            "scored_queries_compared": len(theirs), "excluded_ids_matched": len(theirs_out),
            "identical": True}


def table_markdown(arms) -> str:
    def pct(b):
        return "n/a" if b["accuracy"] is None else f"{b['accuracy']:.4f}"
    head = ("| arm | 단일 셀 | 다중 셀 | 산술 | overall | macro | 헤더답(유형 밖) | 증거 |\n"
            "|---|---|---|---|---|---|---|---|\n")
    body = "".join(
        f"| {a['arm']} | {pct(a['type_accuracy']['single_cell'])} "
        f"| {pct(a['type_accuracy']['multi_cell'])} | {pct(a['type_accuracy']['arithmetic'])} "
        f"| {pct(a['type_accuracy']['overall'])} "
        f"| {a['type_accuracy']['overall']['macro_accuracy']:.4f} "
        f"| {a['n_header_answer_not_typed']} | {a['evidence']} |\n" for a in arms)
    counts = arms[0]["type_accuracy"]
    tests = ("\n유형 차의 Fisher 정확검정 (양측 p):\n\n"
             "| arm | 단일 vs 다중 | 단일 vs 산술 | 다중 vs 산술 |\n|---|---|---|---|\n")
    tests += "".join(
        f"| {a['arm']} | {a['pairwise_tests']['single_cell_vs_multi_cell']['fisher_exact_two_sided_p']:.3g} "
        f"| {a['pairwise_tests']['single_cell_vs_arithmetic']['fisher_exact_two_sided_p']:.3g} "
        f"| {a['pairwise_tests']['multi_cell_vs_arithmetic']['fisher_exact_two_sided_p']:.3g} |\n"
        for a in arms)
    return (head + body + f"\n질의 수: 단일 셀 {counts['single_cell']['n']} · "
            f"다중 셀 {counts['multi_cell']['n']} · 산술 {counts['arithmetic']['n']} "
            f"· 합 {counts['overall']['n']}\n" + tests)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=RECORDS.jsonl",
                    help="비교군 하나. 여러 번 쓴다.")
    ap.add_argument("--budget", type=int, default=20,
                    help="각 실행의 셀 예산 — 검색 설정이지 지표 이름이 아니다")
    ap.add_argument("--verify", type=Path,
                    help="커밋된 {tag}_type_accuracy.jsonl 과 질의별로 대조할 arm 의 파일")
    ap.add_argument("--verify-arm", default=None, help="--verify 를 적용할 arm 이름")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        ap.error("output exists; use a new path so existing evidence is preserved")

    arms = []
    for spec in a.arm:
        if "=" not in spec:
            ap.error(f"--arm needs NAME=PATH, got {spec}")
        name, _, path = spec.partition("=")
        arms.append(score_arm(name, Path(path), a.budget))

    verification = None
    if a.verify:
        target = a.verify_arm or arms[0]["arm"]
        picked = next((x for x in arms if x["arm"] == target), None)
        if picked is None:
            ap.error(f"--verify-arm {target} is not among the scored arms")
        verification = {"arm": target, **verify_rows(picked["rows"], a.verify)}

    payload = {"metric": "retrieval_accuracy_G_subset_R",
               "scored_by": "scripts/retrieval_accuracy.py:type_row/query_type (imported)",
               "budget_note": "셀 예산은 검색 설정이며 지표 이름에 붙이지 않는다",
               "retrieval_budget_cells": a.budget,
               "provenance": provenance(ROOT), "verification": verification,
               "arms": [{k: v for k, v in x.items() if k != "rows"} for x in arms],
               "rows_sha256": {x["arm"]: digest(x["rows"]) for x in arms}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for x in arms:
        rows_path = a.out.with_name(f"{a.out.stem}_{x['arm']}_rows.jsonl")
        with rows_path.open("x", encoding="utf-8", newline="\n") as stream:
            for row in x["rows"]:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(table_markdown(arms))
    if verification:
        print(f"verified against {verification['reference']}: "
              f"{verification['scored_queries_compared']} scored verdicts identical, "
              f"{verification['excluded_ids_matched']} excluded ids matched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
