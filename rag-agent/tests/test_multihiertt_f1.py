"""공식 F1 포팅 검사 — 원본 규칙(DROP bag F1, 숫자 일치 문, 프로그램 문항 허용오차)이 그대로인지.
실행: .venv/bin/python -m pytest tests/test_multihiertt_f1.py -q
"""
from rag_agent.eval.multihiertt_em import mh_exact_match
from rag_agent.eval.multihiertt_f1 import mh_f1, span_f1


def test_span_exact_is_one():
    assert span_f1("net income", "net income") == 1.0


def test_span_partial_overlap():
    # 예측 {income} 대 정답 {net, income}: precision 1, recall .5 → 2/3 → round 2
    assert span_f1("income", "net income") == 0.67


def test_multi_span_order_free():
    assert span_f1(["c d", "a b"], ["a b", "c d"]) == 1.0


def test_multi_span_one_missing():
    # 정답 2 span 중 1개만 맞힘: bag 점수 [1, 0] 평균
    assert span_f1(["a b"], ["a b", "c d"]) == 0.5


def test_number_gate_blocks_credit():
    # gold bag 에 숫자가 있으면 예측 bag 에 같은 숫자가 있어야 점수
    assert span_f1("2018 revenue", "2019 revenue") == 0.0
    assert span_f1("2019 revenue", "2019 revenue") == 1.0


def test_program_numeric_tolerance_matches_em():
    for pred, gold in [("12.5%", 12.5), ("$1,234", 1234.0), ("13", 12.5), ("12.49", 12.5)]:
        assert mh_f1(pred, gold, True) == (1.0 if mh_exact_match(pred, gold, True) else 0.0)


def test_program_non_numeric_pred_falls_back_to_span():
    assert mh_f1("twelve", 12.5, True) == 0.0
    assert mh_f1("12.5", 12.5, True) == 1.0


def test_none_pred():
    assert mh_f1(None, "x", False) == 0.0


def test_f1_never_below_em():
    cases = [("net income", "net income", False), ("income", "net income", False), ("13", 12.5, True), (["a b"], ["a b", "c d"], False)]
    for p, g, prog in cases:
        assert mh_f1(p, g, prog) >= float(mh_exact_match(p, g, prog))
