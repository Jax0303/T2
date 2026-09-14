# SPDX-License-Identifier: MIT
"""저장된 레코드에서 세 유형 정확도를 다시 낼 때 지켜야 하는 것.

판정은 하나다 — gold 셀이 **전부** 배달됐으면 1, 하나라도 빠지면 0. 이 파일은
(1) 그 판정을 좌표에서 다시 계산한 값이 저장된 값과 어긋나면 멈추는지,
(2) 좌표가 없는 구형 레코드에서 회수율을 지어내지 않는지,
(3) 오프라인 집계가 `scripts/retrieval_accuracy.py` 의 집계와 같은지,
(4) 보정 EM 이 "판정 1이면 gold 문맥의 답, 0이면 0점"으로 합성되는지 본다.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analysis.type_accuracy_offline import aggregate, rows_from_records, wilson
from rag_agent.eval.artifacts import file_digest
from retrieval_accuracy import type_accuracy

T = "t"


def cell(j):
    return [T, 0, j]


def record(qid, gold, got, correct, *, agg="none", mode="all", v2=True):
    row = {"query_id": qid, "table_id": T, "mode": mode, "m": len(gold),
           "correct": correct, "aggregation": agg, "cells_in_context": len(got),
           "gold_cells": [cell(j) for j in gold]}
    if v2:
        row["context_cells"] = [cell(j) for j in got]
    return row


def test_recomputed_verdict_must_match_the_stored_one():
    good = {"q": record("q", [0], [0, 1], 1)}
    rows, grade = rows_from_records(good, 20)
    assert grade == "recomputed_from_cells"
    assert (rows[0]["retrieval_success"], rows[0]["evidence_recall"]) == (1, 1.0)

    lying = {"q": record("q", [0, 5], [0, 1], 1)}      # gold 5 가 없는데 correct=1
    with pytest.raises(ValueError, match="disagrees with stored correct"):
        rows_from_records(lying, 20)


def test_partial_gold_is_a_failure_not_partial_credit():
    rows, _ = rows_from_records({"q": record("q", [0, 1, 2], [0, 1, 9], 0)}, 20)
    assert rows[0]["query_type"] == "multi_cell"
    assert rows[0]["retrieval_success"] == 0          # 2/3 을 찾아도 0 이다
    assert rows[0]["evidence_recall"] == 0.6667       # 진단값에만 남는다


def test_legacy_records_keep_the_verdict_and_refuse_to_invent_recall():
    legacy = {"q": record("q", [0, 1], [0], 0, v2=False)}
    legacy["q"].pop("gold_cells")                      # 아예 저장하지 않은 실행도 있다
    rows, grade = rows_from_records(legacy, 20)
    assert grade == "stored_pass_fail"
    assert rows[0]["retrieval_success"] == 0
    assert rows[0]["num_gold_cells"] == 2              # 잘리기 전 값인 m 을 쓴다
    assert rows[0]["evidence_recall"] is None and rows[0]["num_gold_retrieved"] is None
    assert aggregate(rows)["multi_cell"]["mean_evidence_recall_DIAGNOSTIC"] is None


def test_types_come_from_the_dataset_fields():
    records = {"s": record("s", [0], [0], 1),
               "m": record("m", [0, 1], [0, 1], 1),
               "a": record("a", [0, 1], [0, 1], 1, agg="sum"),
               "h": record("h", [0], [0], 1, mode="any")}
    rows, _ = rows_from_records(records, 20)
    assert {r["query_id"]: r["query_type"] for r in rows} == {
        "s": "single_cell", "m": "multi_cell", "a": "arithmetic", "h": "header_answer"}
    table = aggregate(rows)
    assert table["overall"]["n"] == 3                  # 헤더답은 세 유형 밖이다
    assert table["overall"]["macro_accuracy"] == 1.0


def test_offline_aggregate_matches_the_committed_scorer():
    records = {f"q{i}": record(f"q{i}", [0], [0] if i % 3 else [7], int(bool(i % 3)))
               for i in range(12)}
    records["mm"] = record("mm", [0, 1], [0], 0)
    records["aa"] = record("aa", [0, 1], [0, 1], 1, agg="average")
    rows, _ = rows_from_records(records, 20)
    assert aggregate(rows) == type_accuracy(rows)


def test_excluded_queries_stay_out_of_every_denominator():
    records = {"ok": record("ok", [0], [0], 1),
               "no": {"query_id": "no", "table_id": T, "mode": "all",
                      "excluded": "gold_unmappable"}}
    rows, _ = rows_from_records(records, 20)
    table = aggregate(rows)
    assert (table["single_cell"]["n"], table["n_excluded"]) == (1, 1)


def test_wilson_interval_brackets_the_point_estimate():
    low, high = wilson(32, 38)
    assert low < 32 / 38 < high
    assert (round(low, 4), round(high, 4)) == (0.6958, 0.9256)
    assert wilson(0, 0) is None


# --- 보정 EM ---------------------------------------------------------------

def write_pair(path, rows, summary):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")
    path.with_suffix(".json").write_text(
        json.dumps({**summary, "records_sha256": file_digest(path)}), encoding="utf-8")


READER = {"reader_details": {"name": "r", "revision_resolved": "abc", "quantization": "4bit",
                             "dtype": "bf16", "context_limit": 8, "chat_template": "x"},
          "prompt": "neutral", "prompt_sha256": "p", "seed": 42, "max_new_tokens": 64,
          "scorer": "hitab_exact_match_text", "context_limit": 8,
          "excluded_unit_defect": False, "context_version": 2}


def answer_row(qid, correct_flag, pred, gold_answer, n_ctx, m_agg="none"):
    return {"query_id": qid, "question": f"q{qid}", "answer": gold_answer, "mode": "all",
            "aggregation": m_agg, "retrieval_correct": correct_flag, "pred": pred,
            "answer_correct": 1 if pred == str(gold_answer[0]) else 0, "n_ctx": n_ctx}


def test_correction_takes_the_gold_answer_only_where_retrieval_passed(tmp_path):
    from analysis.type_em_correction import arm_table, read_leg

    base = tmp_path / "arm"
    # pass 는 gold 를 다 찾았고, fail 은 gold 하나를 놓쳤다.
    records = [record("pass", [0], [0, 1], 1), record("fail", [1], [0, 2], 0)]
    for r in records:
        r["question"], r["answer"] = f"q{r['query_id']}", [1.0]
    write_pair(base.with_name("arm_records.jsonl"), records, {"budget_cells": 20})
    # 검색 문맥에서는 둘 다 틀렸다.
    write_pair(base.with_name("arm_answer_retrieved.jsonl"),
               [answer_row("pass", 1, "9", [1.0], 2), answer_row("fail", 0, "9", [1.0], 2)],
               {**READER, "condition": "retrieved"})
    # gold 문맥에서는 둘 다 맞았다 — 보정은 pass 한 쪽만 가져와야 한다.
    gold_path = tmp_path / "gold.jsonl"
    write_pair(gold_path, [answer_row("pass", 1, "1.0", [1.0], 1),
                           answer_row("fail", 1, "1.0", [1.0], 1)],
               {**READER, "condition": "gold"})
    gold, gold_meta = read_leg(gold_path)

    out = arm_table("arm", base, gold, gold_meta, 20)["by_type"]["single_cell"]
    assert (out["n"], out["retrieval_success"]) == (2, 1)
    assert out["em_as_retrieved"] == 0.0
    assert out["em_corrected_pipeline"] == 0.5        # fail 은 0점으로 남는다
    assert out["em_corrected_conditional"] == 1.0     # 통과분만 보면 1/1
    assert out["em_gold_all_queries"] == 1.0          # 검색을 다 고쳤을 때의 상한


def test_correction_refuses_a_gold_leg_from_a_different_reader(tmp_path):
    from analysis.type_em_correction import arm_table, read_leg

    base = tmp_path / "arm"
    records = [record("pass", [0], [0], 1)]
    records[0]["question"], records[0]["answer"] = "qpass", [1.0]
    write_pair(base.with_name("arm_records.jsonl"), records, {"budget_cells": 20})
    write_pair(base.with_name("arm_answer_retrieved.jsonl"),
               [answer_row("pass", 1, "1.0", [1.0], 1)], {**READER, "condition": "retrieved"})
    gold_path = tmp_path / "gold.jsonl"
    write_pair(gold_path, [answer_row("pass", 1, "1.0", [1.0], 1)],
               {**READER, "condition": "gold", "seed": 7})
    gold, gold_meta = read_leg(gold_path)
    with pytest.raises(ValueError, match="disagree on seed"):
        arm_table("arm", base, gold, gold_meta, 20)
