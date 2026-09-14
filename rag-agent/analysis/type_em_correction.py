#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""세 유형별 답변 EM — 검색 문맥 그대로일 때와, 맞다고 판정된 셀 문장만 넣었을 때.

검색 정확도는 G ⊆ R 판정 하나다(`analysis/type_accuracy_offline.py`). 그 판정이
1인 질의는 배달된 문맥 안에 gold 셀이 **전부** 들어 있다는 뜻이므로, 그 문맥에서
gold 셀 문장만 남기면 "맞다고 판정된 셀 문장만" 리더에게 주는 조건이 된다. 이
조건의 답변은 이미 생성돼 있다 — gold 문맥 레그가 그것이다. 판정이 0인 질의는
남길 수 있는 완전한 근거가 없으므로 파이프라인 관점에서 0점이다.

유형마다 네 수를 적는다:

``retrieval_accuracy``      G ⊆ R 비율.
``em_as_retrieved``         검색 문맥을 그대로 넣었을 때의 EM (HiTab 공식 채점기).
``em_corrected_pipeline``   판정 1이면 gold 셀 문장만 넣은 답의 EM, 판정 0이면 0.
                            검색과 보정을 합친 파이프라인 수치다.
``em_corrected_conditional``  판정 1인 질의에서만 잰 EM. 리더가 완전한 근거를
                            받았을 때 남는 오류의 몫이라 분모가 arm 마다 다르다.
``em_gold_all_queries``     판정과 무관하게 모든 질의에 gold 셀 문장을 넣은 EM.
                            검색을 완전히 고친 상한이며 arm 과 무관한 상수다.

gold 문맥 레그는 한 번만 생성됐고 셀 문장 표현은 s3c 다. s3c arm 에서는 그 arm 이
실제로 배달한 문장을 거른 것과 같다. 다른 표현의 arm 에서는 **표현을 s3c 로 고정한
채 판정 집합만 그 arm 의 것**을 쓰는 조건이다 — 표현 차이가 아니라 어떤 질의가
통과했는가의 차이만 남는다. 출력의 ``gold_context_rendering`` 에 그렇게 적는다.

  PYTHONPATH=. python3 analysis/type_em_correction.py \
      --gold results/evaluation_v2/s3c_v2_answer_gold_modeall.jsonl \
      --arm s3c=results/evaluation_v2/s3c_v2 --out results/type_accuracy/hitab_test_em.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analysis.type_accuracy_offline import rows_from_records, wilson
from rag_agent.eval.artifacts import file_digest, provenance, read_records
from rag_agent.eval.metrics import hitab_exact_match_text
from retrieval_accuracy import TYPES

# 리더 조건이 이 중 하나라도 다르면 두 레그의 EM 을 나란히 두지 않는다.
READER_KEYS = ("prompt", "prompt_sha256", "seed", "max_new_tokens",
               "scorer", "context_limit", "excluded_unit_defect")
# ``revision_requested`` 는 실행이 revision 을 적어 넣었는지만 말한다. 실제로 어떤
# 가중치가 올라갔는지는 ``revision_resolved`` 이고, 그 값이 같으면 같은 모델이다.
READER_DETAIL_KEYS = ("name", "revision_resolved", "quantization", "dtype",
                      "context_limit", "chat_template")


def read_leg(path: Path) -> tuple[dict, dict]:
    rows = read_records(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if meta.get("context_version") != 2 or meta.get("records_sha256") != file_digest(path):
        raise ValueError(f"{path}: missing or stale v2 answer metadata")
    for qid, r in rows.items():
        if r["answer_correct"] != int(hitab_exact_match_text(r["pred"], r["answer"])):
            raise ValueError(f"{path}/{qid}: stored EM differs from rescoring")
    return rows, meta


def require_same_reader(a: dict, b: dict, label: str) -> None:
    for key in READER_KEYS:
        if key not in a or key not in b or a[key] != b[key]:
            raise ValueError(f"{label}: answer legs disagree on {key}")
    da, db = a.get("reader_details") or {}, b.get("reader_details") or {}
    for key in READER_DETAIL_KEYS:
        if key not in da or key not in db or da[key] != db[key]:
            raise ValueError(f"{label}: answer legs disagree on reader_details.{key}")


def rate(success: int, n: int):
    return round(success / n, 4) if n else None


def arm_table(name: str, base: Path, gold: dict, gold_meta: dict, budget: int) -> dict:
    records = read_records(base.with_name(base.name + "_records.jsonl"))
    rows, _ = rows_from_records(records, budget)
    answers, meta = read_leg(base.with_name(base.name + "_answer_retrieved.jsonl"))
    if meta.get("condition") != "retrieved":
        raise ValueError(f"{name}: expected the retrieved-context answer leg")
    require_same_reader(meta, gold_meta, name)
    verdict = {r["query_id"]: r for r in rows if r["query_type"] in TYPES
               and r["retrieval_success"] is not None}
    if not set(verdict) <= set(gold):
        raise ValueError(f"{name}: typed queries missing from the gold-context leg")
    for qid, v in verdict.items():
        for leg, tag in ((answers, "retrieved"), (gold, "gold")):
            if qid not in leg:
                raise ValueError(f"{name}/{qid}: absent from the {tag} answer leg")
            # 질의도 정답도 같은 질의여야 두 레그의 EM 을 같은 칸에 놓을 수 있다.
            for key in ("question", "answer", "mode", "aggregation"):
                if leg[qid].get(key) != records[qid].get(key):
                    raise ValueError(f"{name}/{qid}: {tag} leg disagrees on {key}")
        # ``retrieval_correct`` 는 그 레그를 만든 실행의 판정이다. 검색 문맥 레그는
        # 이 arm 의 실행이므로 레코드와 같아야 한다. gold 레그는 s3c 실행에서 한 번만
        # 만들어졌으므로 다른 arm 에서는 그 필드가 이 arm 의 판정이 아니다 — 쓰지 않는다.
        if answers[qid]["retrieval_correct"] != records[qid]["correct"]:
            raise ValueError(f"{name}/{qid}: retrieved leg disagrees on the retrieval verdict")
        if v["retrieval_success"] != records[qid]["correct"]:
            raise ValueError(f"{name}/{qid}: type row disagrees with the record")
        # gold 레그가 정말 gold 셀만 받았는지 — 문맥 셀 수가 gold 셀 수와 같아야 한다.
        if gold[qid]["n_ctx"] != records[qid]["m"]:
            raise ValueError(f"{name}/{qid}: gold leg context is not exactly the gold cells")

    def block(ids):
        n = len(ids)
        passed = [q for q in ids if verdict[q]["retrieval_success"]]
        as_ret = sum(answers[q]["answer_correct"] for q in ids)
        as_ret_passed = sum(answers[q]["answer_correct"] for q in passed)
        corrected = sum(gold[q]["answer_correct"] for q in passed)
        ceiling = sum(gold[q]["answer_correct"] for q in ids)
        return {"n": n, "retrieval_success": len(passed),
                "retrieval_accuracy": rate(len(passed), n),
                "retrieval_accuracy_ci95": wilson(len(passed), n),
                "em_as_retrieved": rate(as_ret, n),
                "em_as_retrieved_ci95": wilson(as_ret, n),
                "em_corrected_pipeline": rate(corrected, n),
                "em_corrected_pipeline_ci95": wilson(corrected, n),
                "em_as_retrieved_conditional": rate(as_ret_passed, len(passed)),
                "em_corrected_conditional": rate(corrected, len(passed)),
                "em_corrected_conditional_ci95": wilson(corrected, len(passed)),
                # 같은 질의(판정 1)에서 여분 셀만 걷어냈을 때의 차. 분모가 같아서
                # 검색 차이가 아니라 문맥에 섞인 여분 셀의 몫만 남는다.
                "delta_conditional_paired":
                    round((corrected - as_ret_passed) / len(passed), 4) if passed else None,
                "em_gold_all_queries": rate(ceiling, n),
                "delta_em_corrected_minus_as_retrieved":
                    round(corrected / n - as_ret / n, 4) if n else None}

    by_type = {t: [q for q, v in verdict.items() if v["query_type"] == t] for t in TYPES}
    out = {t: block(v) for t, v in by_type.items()}
    out["overall"] = block(list(verdict))
    return {"arm": name, "records_sha256": file_digest(base.with_name(base.name + "_records.jsonl")),
            "answers_sha256": file_digest(base.with_name(base.name + "_answer_retrieved.jsonl")),
            "reader": meta["reader_details"]["name"], "prompt": meta["prompt"],
            "by_type": out}


def markdown(arms) -> str:
    def f(x):
        return "n/a" if x is None else f"{x:.4f}"
    out = []
    for key, title in (("single_cell", "단일 셀"), ("multi_cell", "다중 셀"),
                       ("arithmetic", "산술"), ("overall", "세 유형 합산")):
        n = arms[0]["by_type"][key]["n"]
        out.append(f"\n### {title} (n={n})\n\n"
                   "| arm | 검색 정확도 | EM(검색 문맥) | EM(보정, 파이프라인) | 차이 "
                   "| 통과분 EM: 검색 문맥 → 보정 | 쌍대 차 |\n"
                   "|---|---|---|---|---|---|---|")
        for a in arms:
            b = a["by_type"][key]
            out.append(f"| {a['arm']} | {f(b['retrieval_accuracy'])} | {f(b['em_as_retrieved'])} "
                       f"| {f(b['em_corrected_pipeline'])} "
                       f"| {b['delta_em_corrected_minus_as_retrieved']:+.4f} "
                       f"| {f(b['em_as_retrieved_conditional'])} → {f(b['em_corrected_conditional'])} "
                       f"| {b['delta_conditional_paired']:+.4f} |")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=PREFIX",
                    help="PREFIX_records.jsonl 과 PREFIX_answer_retrieved.jsonl 을 읽는다")
    ap.add_argument("--gold", type=Path, required=True,
                    help="gold 셀 문장만 넣은 답변 레그 (condition=gold)")
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        ap.error("output exists; use a new path so existing evidence is preserved")

    gold, gold_meta = read_leg(a.gold)
    if gold_meta.get("condition") != "gold":
        ap.error("--gold must point at a gold-context answer leg")
    arms = []
    for spec in a.arm:
        if "=" not in spec:
            ap.error(f"--arm needs NAME=PREFIX, got {spec}")
        name, _, prefix = spec.partition("=")
        arms.append(arm_table(name, Path(prefix), gold, gold_meta, a.budget))

    payload = {"metric": "answer_exact_match (HiTab official scorer)",
               "retrieval_metric": "retrieval_accuracy_G_subset_R",
               "retrieval_budget_cells": a.budget,
               "budget_note": "셀 예산은 검색 설정이며 지표 이름에 붙이지 않는다",
               "gold_leg": str(a.gold).replace("\\", "/"),
               "gold_leg_sha256": file_digest(a.gold),
               "gold_context_rendering": "s3c cell sentence, generated once in the s3c_v2 run",
               "gold_context_note": ("s3c arm 에서는 그 arm 이 배달한 문장을 거른 것과 같다. "
                                     "다른 표현의 arm 에서는 표현을 s3c 로 고정하고 판정 집합만 "
                                     "그 arm 의 것을 쓴 조건이다."),
               "reader": gold_meta["reader_details"]["name"], "prompt": gold_meta["prompt"],
               "seed": gold_meta["seed"], "scorer": gold_meta["scorer"],
               "provenance": provenance(ROOT), "arms": arms}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(markdown(arms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
