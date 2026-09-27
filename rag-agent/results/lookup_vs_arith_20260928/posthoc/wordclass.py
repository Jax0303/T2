"""GATE 0: 비교 단어 분류 재계산과 2026-09-28 채팅 보고 숫자 대조(최종 버전). 불일치면 AssertionError.
실행: HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=. .venv/bin/python results/lookup_vs_arith_20260928/posthoc/wordclass.py
"""
import json
from collections import Counter

from common import HERE, rows, CATS, NONE

EXPECT = {"조회": {"가장 큰": (78, 20), "가장 작은": (11, 2), "두 번째": (6, 0), "기준값 비교": (46, 10), NONE: (71, 44)},
          "산술": {"비교": (4, 2), NONE: (67, 43)}}

if __name__ == "__main__":
    rs = rows("v3.3u")
    got = {"조회": {}, "산술": {}}
    for tp in got:
        xs = [x for x in rs if x["type"] == tp]
        for name in [c for c, _ in CATS] + [NONE]:
            ys = [x for x in xs if x["comparison"] == name]
            got[tp][name] = (len(ys), sum(x["rank1"] for x in ys))
    arith_cmp = [x for x in rs if x["type"] == "산술" and x["comparison"] != NONE]
    check = {"조회": got["조회"], "산술": {"비교": (len(arith_cmp), sum(x["rank1"] for x in arith_cmp)),
                                         NONE: got["산술"][NONE]}}
    mismatch = [(tp, k, check[tp][k], v) for tp in EXPECT for k, v in EXPECT[tp].items() if check[tp][k] != v]
    out = {"counts_n_rank1": got, "check_against_2026-09-28_report": check, "mismatch": mismatch}
    (HERE / "wordclass.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))
    assert not mismatch, mismatch
