# SPDX-License-Identifier: MIT
"""MultiHiertt 의 공식 F1 — 원본에서 그대로 옮긴다 (2026-09-29, EM 모듈 `multihiertt_em.py` 와 짝).

출처 (psunlpgroup/MultiHiertt, 45bd9cc):
  ``utils/span_selection_utils.py: get_span_selection_metrics`` — DROP 식 bag F1:
      gold·예측을 span 마다 토큰 집합(bag)으로 만들고, 헝가리안 정렬(``linear_sum_assignment``)로
      1:1 짝을 지어 bag 별 F1 의 평균. 숫자가 든 gold bag 은 예측 bag 에 같은 숫자가 있어야 점수를 준다
      (``_match_numbers_if_present``). 결과는 소수 둘째 자리로 반올림(원본 ``round(f1, 2)``).
  ``evaluate.py: evaluate_span_program_result`` — 프로그램 문항(정답 숫자)에 텍스트 답: 예측이
      숫자로 읽히면 EM 판정과 같은 허용오차로 F1 = 1/0, 아니면 위 span F1.

정규화·bag 만들기·``str_to_num``·허용오차는 EM 모듈의 것을 그대로 쓴다(같은 원본).
"""
from __future__ import annotations

import math
from typing import List, Set

import numpy as np
from scipy.optimize import linear_sum_assignment

from .multihiertt_em import _bags, _is_number, _spans, str_to_num


def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    precision = 1.0 if not predicted_bag else intersection / float(len(predicted_bag))
    recall = 1.0 if not gold_bag else intersection / float(len(gold_bag))
    return (2 * precision * recall) / (precision + recall) if not (precision == 0.0 and recall == 0.0) else 0.0


def _match_numbers_if_present(gold_bag: Set[str], predicted_bag: Set[str]) -> bool:
    gold_numbers = {w for w in gold_bag if _is_number(w)}
    predicted_numbers = {w for w in predicted_bag if _is_number(w)}
    return (not gold_numbers) or bool(gold_numbers.intersection(predicted_numbers))


def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> np.ndarray:
    scores = np.zeros([len(gold), len(predicted)])
    for gi, g in enumerate(gold):
        for pi, p in enumerate(predicted):
            if _match_numbers_if_present(g, p):
                scores[gi, pi] = _compute_f1(p, g)
    row_ind, col_ind = linear_sum_assignment(-scores)
    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def span_f1(pred, gold) -> float:
    """``get_span_selection_metrics`` 의 F1 부분 (EM 부분은 ``multihiertt_em.span_exact_match``)."""
    return round(float(np.mean(_align_bags(_bags(pred), _bags(gold)))), 2)


def mh_f1(pred, gold, program: bool) -> float:
    """공식 F1. ``program`` = 정답 문항에 프로그램이 있는가 — ``mh_exact_match`` 와 같은 갈래."""
    if pred is None:
        return 0.0
    if not program:
        return span_f1(pred, gold)
    p = str_to_num(pred)
    if p != "n/a":
        g = float(gold)
        return 1.0 if math.isclose(g, p, abs_tol=min(abs(min(g, p) / 1000), 0.1)) else 0.0
    return span_f1(str(pred), str(gold))


__all__ = ["mh_f1", "span_f1", "_spans"]
