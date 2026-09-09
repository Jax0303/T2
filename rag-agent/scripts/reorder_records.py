#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""검색 순위가 답변을 만드는가, 아니면 같은 원인의 결과인가.

HANDOFF-2026-09-09 §3.1 은 "gold 셀이 1등이면 정답률이 높다"에서 "1등으로 올리면
정답률이 오른다"로 넘어간다. 그 사이에 교란이 있다 — 질의가 형제 셀 중 어느 것인지
말해주지 않으면 순위도 밀리고 리더도 틀린다. 순위는 원인이 아니라 같은 난이도의 표시일
수 있다.

가르는 방법: **셀 집합은 그대로 두고 순서만 바꾼다.** 리더가 받는 20셀이 한 칸도
바뀌지 않으므로 검색 난이도는 상수이고, 움직이는 것은 gold 의 자리 하나다.

  asis      원본 순서 (동일 조건 재실행 = 환경 드리프트·재현성 검사)
  promote   gold 를 1번 자리로, 나머지는 상대 순서 유지 (§3.1 이 가정하는 상한)
  shuffle   seed 고정 무작위 순열 (순서를 흔든 것 자체의 효과, promote 의 대조군)

  PYTHONPATH=. .venv/bin/python scripts/reorder_records.py \
      --records results/retrieval_accuracy/t_s3c_hybrid_records.jsonl \
      --mode promote --out /tmp/x.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.answer_accuracy import gold_context                      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True)
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--mode", required=True, choices=["asis", "promote", "shuffle"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists — pass a new --out")

    tabs: dict = {}
    kept, skipped = [], {"not_main_pop": 0, "retrieval_miss": 0,
                         "gold_not_in_context": 0, "gold_already_first": 0}
    for line in open(a.records):
        r = json.loads(line)
        if not (r.get("mode") == "all" and r.get("aggregation") == "none"
                and r.get("m") == 1 and "correct" in r):
            skipped["not_main_pop"] += 1
            continue
        if not r["correct"]:
            skipped["retrieval_miss"] += 1
            continue
        ctx = list(r.get("context") or [])
        gold = gold_context(r, tabs, a.data_dir)
        # m==1 이므로 gold 문장은 하나다. 문맥 안의 자리를 문장 일치로 찾는다 —
        # gold_rank 를 믿지 않고 리더가 실제로 받는 리스트에서 직접 센다.
        if len(gold) != 1 or gold[0] not in ctx:
            skipped["gold_not_in_context"] += 1
            continue
        i = ctx.index(gold[0])
        if i == 0:
            skipped["gold_already_first"] += 1
            continue
        if a.mode == "promote":
            ctx = [ctx[i]] + ctx[:i] + ctx[i + 1:]
        elif a.mode == "shuffle":
            random.Random(a.seed + int(r["query_id"][:8], 16)).shuffle(ctx)
        r["context"] = ctx
        r["gold_pos_orig"] = i + 1
        r["gold_pos_now"] = ctx.index(gold[0]) + 1
        kept.append(r)

    with out.open("w") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[{a.mode}] {len(kept)}건 -> {out}")
    print("  제외:", json.dumps(skipped, ensure_ascii=False))
    assert all(len(r["context"]) == r["cells_in_context"] for r in kept), \
        "순서만 바꿔야 하는데 셀 수가 달라졌다"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
