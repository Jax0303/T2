#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""phase4_summary.em_lenient 은 표기 차이만 통과시키고 값 오독은 통과시키지 않는다.

이름 주의: tests/test_em_norm.py 는 별개다 -- 그쪽은
scripts/realhitbench_answer_accuracy.em_norm(RealHiTBench 자릿수 맞춤 비교)을 잰다.
"""
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "analysis")]
from phase4_summary import em, em_lenient  # noqa: E402

# results/answer_ret/a0_dev_all.jsonl 의 cond=gold 실패에서 그대로 가져온 것.
FORMAT = [("[1.2]", "-1.2"), ("[56]", "-56.0%"), ("[9848]", "-9848"),
          ("[55269.5]", "55.2695"), ("[0.165]", "16.5%"), ("[0.95]", "95%"),
          ("[4205.3]", "42.053"), ("[9682692.0]", "9.682692"),
          ("[76.7]", "0.767")]
VALUE = [("[80.9]", "19.1"),           # 여집합
         ("[145408.0]", "147934.68"),  # 2% 를 계산해버림
         ("[3.9]", "1.0"),             # 문장에 있는데 딴 숫자
         ("[7.5]", "92.5%"),           # 여집합 + 퍼센트 표기
         ("[0.16]", "1.6%")]           # 10배, 어떤 표기 관행도 아니다


def test_format_slips_recovered():
    for g, p in FORMAT:
        assert not em(p, g), (g, p)
        assert em_lenient(p, g), (g, p)


def test_value_errors_still_wrong():
    for g, p in VALUE:
        assert not em_lenient(p, g), (g, p)


def test_is_a_superset_of_em_on_the_830():
    import json
    src = Path(__file__).resolve().parents[1] / "results/answer_ret/a0_dev_all.jsonl"
    if not src.exists():
        return
    for line in open(src):
        r = json.loads(line)
        if em(r["pred_parsed"], r["gold_answer"]):
            assert em_lenient(r["pred_parsed"], r["gold_answer"]), r["query_id"]


if __name__ == "__main__":
    test_format_slips_recovered()
    test_value_errors_still_wrong()
    test_is_a_superset_of_em_on_the_830()
    print("ok")
