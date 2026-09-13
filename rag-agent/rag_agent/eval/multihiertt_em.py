# SPDX-License-Identifier: MIT
"""MultiHiertt 의 공식 답변 채점기 — 원본에서 그대로 옮긴다.

출처 (psunlpgroup/MultiHiertt, 45bd9cc):

  ``evaluate.py: evaluate_span_program_result``  예측이 숫자로 읽히면 숫자 비교,
      아니면 DROP 방식 bag-of-tokens EM.
  ``utils/utils.py: str_to_num``                 ``$ , - %`` 를 지우고 float.
  ``utils/span_selection_utils.py``              DROP 정규화와 bag 정렬.

숫자 비교의 허용오차는 그들의 것이다:

    math.isclose(gold, pred, abs_tol=min(abs(min(gold, pred)) / 1000, 0.1))

⚠️ ``str_to_num`` 은 ``-`` 까지 지운다. 부호가 사라지므로 ``-5`` 와 ``5`` 가 같아진다.
원본의 성질이고 예측·정답 양쪽에 똑같이 걸리므로 arm 사이 비교는 공정하지만,
**발표된 MultiHiertt 수치와 비교할 때만 쓰는 값**이라는 것을 적어 둔다. HiTab 쪽
``hitab_exact_match_text`` 와 섞어 평균 내지 않는다 — 채점기가 다르다.
"""
from __future__ import annotations

import math
import re
import string
from typing import List, Set

_EXCLUDE = set(string.punctuation)


def str_to_num(text) -> object:
    """``utils/utils.py: str_to_num`` (프로그램 상수 분기는 답변 채점에 안 쓴다)."""
    text = str(text).replace("$", "").replace(",", "").replace("-", "").replace("%", "")
    try:
        return float(text)
    except ValueError:
        return "n/a"


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _normalize_number(text: str) -> str:
    return str(float(text)) if _is_number(text) else text


def _remove_punc(text: str) -> str:
    return text if _is_number(text) else "".join(c for c in text if c not in _EXCLUDE)


def _normalize_answer(text: str) -> str:
    parts = []
    for token in re.split(" |-", str(text)):
        t = _remove_punc(token.lower())
        t = _normalize_number(t)
        t = re.sub(re.compile(r"\b(a|an|the)\b", re.UNICODE), " ", t)
        t = " ".join(t.split())
        if t.strip():
            parts.append(t)
    return " ".join(parts).strip()


def _bags(answer) -> List[Set[str]]:
    spans = answer if isinstance(answer, (list, tuple)) else [answer]
    return [set(_normalize_answer(s).split()) for s in spans]


def _spans(answer) -> List[str]:
    spans = answer if isinstance(answer, (list, tuple)) else [answer]
    return [_normalize_answer(s) for s in spans]


def span_exact_match(pred, gold) -> bool:
    """``get_span_selection_metrics`` 의 EM 부분. F1 은 판정에 쓰지 않는다(§0.1)."""
    p, g = _spans(pred), _spans(gold)
    return set(p) == set(g) and len(p) == len(g)


def mh_exact_match(pred, gold) -> bool:
    """공식 EM. 양쪽이 숫자로 읽히면 그들의 허용오차, 아니면 DROP EM."""
    if pred is None:
        return False
    g, p = str_to_num(gold), str_to_num(pred)
    if g != "n/a" and p != "n/a":
        return math.isclose(g, p, abs_tol=min(abs(min(g, p)) / 1000, 0.1))
    return span_exact_match(pred, gold)


def self_check(golds) -> dict:
    """gold 를 예측으로 넣으면 EM 이 1.0 인가. 아니면 그 채점기의 EM 은 하한이다."""
    ok = sum(mh_exact_match(g, g) for g in golds)
    return {"em": ok / len(golds) if golds else None, "n": len(golds),
            "fail_examples": [g for g in golds if not mh_exact_match(g, g)][:5]}
