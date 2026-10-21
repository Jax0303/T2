"""정답 포함 채점 v2 — PREREG-2026-10-14-score-fix.md. 10-12 규칙(score_inclusion.included) 위에 세 가지만 더한다:
(1) 출력의 숫자 낱말 zero~twenty → 숫자("one of"·"no one"·"one-time"·"one another" 제외)
(2) 조회 문항 정답 |g|<1 이면 출력의 X% 를 X/100 으로도 대조
(3) 정답 "2008,2009" 는 쉼표를 공백으로 → 낱개 숫자 모두 있어야 맞힘
"""
import math, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "results/cellform_20261012")]
from score_inclusion import gold_kind, included, nums, NUM            # noqa: E402

WORDS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
_W = re.compile(r"\b(" + "|".join(WORDS) + r")\b(?![- ](?:of|one|time|another)\b)", re.I)
_KEEP = re.compile(r"\bno one\b", re.I)


def words_to_digits(raw):
    raw = _KEEP.sub("no_one", raw or "")
    return _W.sub(lambda m: str(WORDS.index(m.group(1).lower())), raw)


def included_v2(raw, gold, arith):
    g = str(gold).strip()
    raw2 = raw or ""
    if re.fullmatch(r"\d{4},\d{4}", g):                           # (3) 출력의 "2008,2009" 도 같이 가른다(자기 점검용)
        g = g.replace(",", " ")
        raw2 = re.sub(r"(\d{4}),(\d{4})", r"\1 \2", raw2)
    if gold_kind(g) == "num":                                     # (1) 숫자 정답일 때만(글자 정답 "Twelve months…" 보호 — 이탈 기록)
        raw2 = words_to_digits(raw2)
    if included(raw2, g, arith):
        return True
    if not arith and gold_kind(g) == "num":                       # (2)
        try:
            gv = float(g.replace(",", "").replace("$", "").replace("%", ""))
        except ValueError:
            return False
        if 0 < abs(gv) < 1:
            for t in NUM.findall(raw2):
                if t.endswith("%"):
                    try:
                        if math.isclose(float(t.strip("()%$").replace(",", "")) / 100, gv, rel_tol=1e-6):
                            return True
                    except ValueError:
                        pass
    return False


def included_sens(raw, gold, arith):
    """민감도: 산술 정답을 출력 숫자의 소수 자릿수로 반올림해 같으면 맞힘(×100 도 허용). 주 지표 아님."""
    if included_v2(raw, gold, arith):
        return True
    if not arith:
        return False
    try:
        gv = float(str(gold).replace(",", ""))
    except ValueError:
        return False
    for t in NUM.findall(words_to_digits(raw)):
        s = t.strip("()%$").replace(",", "")
        try:
            tv = float(s)
        except ValueError:
            continue
        d = len(s.split(".")[1]) if "." in s else 0
        if any(round(gv * k, d) == round(tv, d) and (tv != 0 or gv == 0) for k in (1, 100)):
            return True
    return False
