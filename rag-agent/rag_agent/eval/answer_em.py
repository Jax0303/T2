# SPDX-License-Identifier: MIT
"""질의 유형별 답변 채점 — 공식 EM 하나만 정확도로 쓴다.

채점기는 `metrics.py` 것을 그대로 쓰고, 여기서는 **유형을 나누는 일**만 한다.

  official      `hitab_exact_match_text` — HiTab 공식 채점기. 허용오차 없음.
                발표된 HiTab 수치와 비교 가능한 유일한 값이라 **주지표**다.
  format_defect 공식이 기각했는데 **숫자 집합이 정확히 같은** 경우. 표기만
                다른 것이므로 "채점기 몫"의 크기를 재는 진단값이고, 정확도가
                아니다.

2026-09-09 정정 — 처음에는 `numeric_match(rel_tol=0.01)` 을 `lenient` 로 실었는데,
그 함수는 상대오차만 보는 것이 아니라 ①×100/÷100 재척도 ②abs() 부호 뒤집힘
③**예측 문자열에서 뽑은 아무 숫자 하나만 맞아도 통과** ④문자열 부분일치까지
받아 준다. P4 산술에서 그것이 추가로 통과시킨 92건을 전수로 뜯어 보니 76건이
오답이었다(gold `['africa']` 에 `'african'`, gold `['weekend']` 에
`'sedentary time > weekend'`, `opposite` 질의의 부호 반대 등). 관대 채점을
고치는 대신 **없앴다** — 남길 값은 "숫자는 맞는데 형식이 다르다" 하나뿐이고,
그것만 `format_defect` 로 좁혀 잰다. `numeric_match` 자체는 다른 실험이 쓰므로
건드리지 않는다.

유형은 우리가 나눈 것이 아니라 HiTab 이 질의마다 붙여 둔 `aggregation` 이다:
`none` 이면 조회, 그 밖은 산술.
"""
from __future__ import annotations


from .metrics import _to_nums, hitab_exact_match_text


def query_type(aggregation) -> str:
    """`lookup` / `arithmetic` — HiTab 의 `aggregation` 필드가 정한다."""
    a = aggregation
    if isinstance(a, list):
        a = a[0] if a else None
    return "lookup" if (a or "none") == "none" else "arithmetic"


def em_official(pred, gold) -> bool:
    """주지표. 공식 채점기 그대로 — 여기에 관대함을 더하지 않는다."""
    return bool(hitab_exact_match_text(pred, gold))


def format_defect(pred, gold, tol: float = 1e-9) -> bool:
    """공식이 기각했는데 **숫자 집합이 정확히 같다** — 표기만 다른 경우.

    개수까지 같기를 요구하는 것이 핵심이다. "52.1 에서 31.2 를 빼면 20.9" 처럼
    피연산자를 읊은 답은 숫자가 셋이라 gold 하나와 같아질 수 없다. 개수를 보지
    않으면 문맥의 값을 나열하기만 해도 통과하고, 그것이 이 리더의 가장 큰
    오류 유형이다.
    """
    if em_official(pred, gold):
        return False
    g, p = _to_nums(gold), _to_nums(pred)
    if not g or len(g) != len(p):
        return False
    return all(any(abs(x - gv) <= tol for x in p) for gv in g)


def score(pred, gold, aggregation) -> dict:
    """한 질의의 채점 결과. `type` 은 계층 분리 보고에 그대로 쓴다."""
    return {"type": query_type(aggregation),
            "official": int(em_official(pred, gold)),
            "format_defect": int(format_defect(pred, gold))}


def self_check(records) -> dict:
    """gold 를 예측으로 넣었을 때 EM 이 1.0 인가.

    1.0 이 아니면 정규화가 gold 자신을 통과시키지 못한다는 뜻이고, 그 채점기로
    잰 모든 EM 은 하한이지 성능이 아니다. 유형별로 나누어 돌려준다 —
    한쪽만 깨지는 것이 흔하고, 뭉치면 그것이 보이지 않는다.
    """
    out: dict = {}
    for r in records:
        gold = r["answer"]
        s = score(gold, gold, r.get("aggregation"))
        for key in ("official",):
            b = out.setdefault(f"{s['type']}/{key}", [0, 0, []])
            b[1] += 1
            b[0] += s[key]
            if not s[key] and len(b[2]) < 5:
                b[2].append({"query_id": r.get("query_id"), "gold": gold})
    return {k: {"em": v[0] / v[1], "n": v[1], "fail_examples": v[2]}
            for k, v in sorted(out.items())}


if __name__ == "__main__":                                   # 자체 검증 실행부
    import json
    import sys
    from pathlib import Path

    src = Path(sys.argv[1] if len(sys.argv) > 1 else
               "results/retrieval_accuracy/t_s3c_hybrid_records.jsonl")
    recs = [j for j in map(json.loads, src.open()) if "answer" in j]
    res = self_check(recs)
    print(f"자체 검증 — gold 를 예측으로 입력 ({src}, n={len(recs)})\n")
    bad = 0
    for k, v in res.items():
        mark = "OK" if v["em"] == 1.0 else "FAIL"
        print(f"  {mark:4} {k:22} EM={v['em']:.4f}  n={v['n']}")
        if v["em"] != 1.0:
            bad += 1
            for e in v["fail_examples"]:
                print(f"         통과 못한 gold: {e}")
    raise SystemExit(1 if bad else 0)
