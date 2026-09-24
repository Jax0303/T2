# SPDX-License-Identifier: MIT
"""MultiHiertt 의 공식 답변 채점기 — 원본에서 그대로 옮긴다.

출처 (psunlpgroup/MultiHiertt, 45bd9cc):

  ``evaluate.py: evaluation_prediction_result``  예측에 프로그램이 없는(텍스트 답) 경우의 두 갈래:
      정답이 span 문항이면 ``get_span_selection_metrics``, 정답이 프로그램 문항이면
      ``evaluate_span_program_result`` — 예측이 숫자로 읽히면 숫자 비교, 아니면 DROP EM.
  ``utils/utils.py: str_to_num``                 ``$ , - %`` 를 지우고 float.
  ``utils/span_selection_utils.py``              DROP 정규화와 bag 정렬.

숫자 비교의 허용오차는 그들의 것이다:

    math.isclose(gold, pred, abs_tol=min(abs(min(gold, pred) / 1000), 0.1))

⚠️ ``str_to_num`` 은 **예측에만** 걸린다 — ``-`` 까지 지우므로 예측은 늘 0 이상이고, 정답(float)은
부호를 유지한다. 그래서 정답이 음수인 프로그램 문항은 텍스트 답으로 **공식 채점에서 맞힐 수 없다**
(정답을 그대로 예측으로 넣어도 틀린다). 원본의 성질이라 고치지 않는다 — ``self_check`` 가 그 상한을
잰다. 2026-09-23 전까지 이 파일은 양쪽에 ``str_to_num`` 을 걸고 문항 유형 대신 "둘 다 숫자로
읽히는가"로 갈래를 나눴다 — 원본과 다른 채점이었다.

``docmath_match`` 는 보조 지표다: 같은 저자 그룹이 LLM 을 평가한 DocMath-Eval(ACL 2024,
yale-nlp/DocMath-Eval ``utils/evaluation_utils.py``)의 ``compare_two_numbers`` 를 그대로 옮긴 것.
×100/×1000/×100000 척도 차이와 10 의 거듭제곱 비를 정답으로 받고 상대오차 0.15% 를 준다.
원본의 숫자 추출(``normalize``)은 모델 출력을 ``eval`` 하므로 옮기지 않고 앞부분 문자열 정리만 옮겼다.
HiTab 쪽 ``hitab_exact_match_text`` 와 섞어 평균 내지 않는다 — 채점기가 다르다.
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


def mh_exact_match(pred, gold, program: bool) -> bool:
    """공식 EM. ``program`` = 정답 문항에 프로그램이 있는가(데이터셋 ``question_type == arithmetic``)."""
    if pred is None:
        return False
    if not program:                                   # both span selection
        return span_exact_match(pred, gold)
    p = str_to_num(pred)                              # gold is program generation, pred is span
    if p != "n/a":
        g = float(gold)
        return math.isclose(g, p, abs_tol=min(abs(min(g, p) / 1000), 0.1))
    return span_exact_match(str(pred), str(gold))


def self_check(golds, programs) -> dict:
    """gold 를 예측으로 넣었을 때의 EM — 이 채점기로 텍스트 답이 받을 수 있는 상한."""
    fail = [g for g, pr in zip(golds, programs) if not mh_exact_match(str(g), g, pr)]
    return {"em": 1 - len(fail) / len(golds) if golds else None, "n": len(golds),
            "fail_examples": fail[:5], "n_fail": len(fail)}


def _round_up_to_decimal(number, decimals):
    factor = 10 ** decimals
    return math.ceil(number * factor) / factor


def _within_eps(pred: float, gt: float):
    eps = abs(gt) * 0.0015
    return gt - eps <= pred <= gt + eps


def compare_two_numbers(p, gt):
    """DocMath-Eval ``compare_two_numbers`` 그대로 (float/int 만 받는다)."""
    v1, v2 = max(abs(gt), abs(p)), min(abs(gt), abs(p))
    if (v1 != 0 and v2 != 0) and int(math.log10(v1 / v2)) == math.log10(v1 / v2):
        return True
    if v2 <= v1 / 50 and _within_eps(pred=v2 * 100, gt=v1):
        return True
    elif v2 <= v1 / 500 and _within_eps(pred=v2 * 1000, gt=v1):
        return True
    elif v2 <= v1 / 50000 and _within_eps(pred=v2 * 100000, gt=v1):
        return True
    if _round_up_to_decimal(v1, 3) == _round_up_to_decimal(v2, 3):
        return True
    return _within_eps(pred=p, gt=gt)


def docmath_number(pred):
    """DocMath-Eval ``normalize`` 의 문자열 정리 단계만(eval 없음). 숫자가 아니면 None."""
    s = str(pred).strip().rstrip(".")
    for money in ["£", "€", "¥", "million", "billion", "thousand", "US", "USD", "RMB"]:
        s = s.replace(money, "")
    if "=" in s:
        s = s.split("=")[-1].strip()
    if "≈" in s:
        s = s.split("≈")[-1].strip()
    for ch in "`%$°":
        s = s.replace(ch, "")
    if "approximately" in s:
        s = s.replace("approximately", "").strip()
    if " or " in s:
        s = s.split(" or ")[0]
    if re.match(r"[-+]?(?:[\d,]*\.*\d+) [^0-9 ]+$", s):
        s = re.search(r"([-+]?(?:[\d,]*\.*\d+)) [^0-9 ]+$", s).group(1)
    if re.match(r"[^0-9 ]+ [-+]?(?:[\d,]*\.*\d+)$", s):
        s = re.search(r"[^0-9 ]+ ([-+]?(?:[\d,]*\.*\d+))$", s).group(1)
    if re.match(r"[-+]?(?:[\d,]*\.*\d+)[^\d]{1,2}$", s):
        s = re.search(r"([-+]?(?:[\d,]*\.*\d+))[^\d]{1,2}$", s).group(1)
    if re.match(r"[^-+\d]{1,2}(?:[\d,]*\.*\d+)$", s):
        s = re.search(r"[^-+\d]{1,2}((?:[\d,]*\.*\d+))$", s).group(1)
    if re.match(r"^[-+]?(\d{1,3}(,\d{3})*|(\d+))(\.\d+)?$", s):
        s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def docmath_match(pred, gold, program: bool) -> bool:
    """보조 지표. 프로그램 문항만 DocMath 비교로 채점하고, span 문항은 공식 EM 그대로다
    (DocMath-Eval 은 숫자 답만 남긴 벤치마크라 span 규칙이 없다)."""
    if pred is None:
        return False
    if not program:
        return span_exact_match(pred, gold)
    p = docmath_number(pred)
    # 자릿수가 너무 긴 예측·'inf'·'nan' 은 float 가 inf/nan 이 되어 compare_two_numbers 가 죽는다 — 오답으로 둔다
    return p is not None and math.isfinite(p) and compare_two_numbers(p, float(gold))
