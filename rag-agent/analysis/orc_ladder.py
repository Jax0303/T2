#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""`orcK` 를 임의의 K 에서 재구성하고 `topK` 와 나란히 놓는다.

리더를 다시 부르지 않는다. `orcK` 의 프롬프트는 gold 가 top-K 안이면 `gold` 조건과,
아니면 `top1` 조건과 **글자 단위로 같다** (VERDICT_ORC.md 무결성 §결정성 검증에서
784/784 일치로 확인됨). 그래서 그 두 조건의 예측만 있으면 K 를 바꿔가며 정확히
재구성할 수 있다 -- 근사가 아니다.

왜 필요한가. `orcK` 와 `topK` 가 K 에 대해 **반대로** 움직이는데 K=10 한 점만 재놨어서
"K 를 키우면 distractor 때문에 EM 이 떨어진다"로 읽히는 자리가 있었다. distractor 비용은
`topK` 축에서만 나고 `orcK` 는 주입 셀이 늘 1개다.

  PYTHONPATH=analysis:. python3 analysis/orc_ladder.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase4_summary import em                                        # noqa: E402

SRC = "results/answer_ret/p0_dev_all.jsonl"
KS = (1, 3, 5, 10, 20, 50)
# 기록값 (VERDICT_ORC.md 결과표). 재구성이 이것을 재현해야 한다.
RECORDED = {"gold": .9024, "top1": .6807, "top10": .7205,
            "orc10": .8699, "orc50": .8904}


def main() -> int:
    R = {}
    for line in open(SRC):
        r = json.loads(line)
        R[(r["cond"], r["query_id"])] = r
    qids = sorted({q for _c, q in R})
    n = len(qids)

    def rate(cond):
        return sum(em(R[(cond, q)]["pred_parsed"],
                      R[(cond, q)]["gold_answer"]) for q in qids) / n

    for cond, want in RECORDED.items():          # 산술 통제
        got = rate(cond)
        assert abs(got - want) < 5e-5, f"{cond}: {got:.4f} != 기록 {want}"

    print(f"n={n}  (dev, p0 a=0.8, Qwen2.5-7B 4-bit)")
    print(f"{'K':>3}{'orcK EM':>10}{'gold 주입률':>13}{'topK EM':>10}")
    top = {1: rate("top1"), 10: rate("top10")}   # 잰 것만. top3 은 rerank/VERDICT.md
    for k in KS:
        hit = inj = 0
        for q in qids:
            g, t = R[("gold", q)], R[("top1", q)]
            ok = max(g["gold_ranks"]) < k        # gold 가 전부 top-k 안인가
            src = g if ok else t
            hit += em(src["pred_parsed"], src["gold_answer"])
            inj += ok
        tk = f"{top[k]:>10.4f}" if k in top else f"{'-':>10}"
        print(f"{k:>3}{hit / n:>10.4f}{inj / n:>13.4f}{tk}")
    # orc1 은 정의상 top1 이다 (top-1 안에 있으면 그것이 1위 -- 프롬프트가 같다).
    assert abs(sum(em(*(lambda s: (s["pred_parsed"], s["gold_answer"]))(
        R[("gold", q)] if max(R[("gold", q)]["gold_ranks"]) < 1
        else R[("top1", q)])) for q in qids) / n - top[1]) < 5e-5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
