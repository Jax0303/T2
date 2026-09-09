#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The `oracle` leg without a second GPU run.

`oracle` (scripts/answer_accuracy.py) hands the reader the gold cells when
retrieval found them and the retrieved context when it did not. Every query is
its own prompt at temperature 0, so that leg is not a new measurement -- it is
the `gold` rows on retrieval hits plus the `retrieved` rows on misses. Compose
it instead of paying for 1,581 more generations.

Reads what it needs from the leg outputs only, so it is exact as long as the two
legs ran on the same records with the same reader and prompt (asserted below).

  PYTHONPATH=. .venv/bin/python analysis/compose_oracle.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.answer_accuracy import summarize                        # noqa: E402

D = ROOT / "results/retrieval_accuracy"
TAG = "t_s3c_hybrid"


def leg(name):
    rows = {json.loads(l)["query_id"]: json.loads(l)
            for l in (D / f"{TAG}_answer_{name}.jsonl").open()}
    return rows, json.loads((D / f"{TAG}_answer_{name}.json").read_text())


def main() -> int:
    out = D / f"{TAG}_answer_oracle.jsonl"
    if out.exists():
        raise SystemExit(f"{out} exists — answer legs never overwrite")
    gold, gmeta = leg("gold")
    retr, rmeta = leg("retrieved")

    # `.get` 이지 `[]` 가 아닌 이유: 두 기준선 레그는 prompt/seed 키가 생기기 전
    # 실행이라 둘 다 없다. 없는 것끼리도 같아야 한다는 뜻은 그대로 성립한다.
    for k in ("records", "reader", "prompt", "seed", "excluded_unit_defect"):
        assert gmeta.get(k) == rmeta.get(k), \
            f"legs disagree on {k}: {gmeta.get(k)} vs {rmeta.get(k)}"
    assert gold.keys() == retr.keys(), "legs cover different queries"

    rows = []
    for q, r in retr.items():
        assert r["retrieval_correct"] == gold[q]["retrieval_correct"]
        rows.append(gold[q] if r["retrieval_correct"] else r)

    # 합성이 맞았는지: 성공분은 gold 와, 실패분은 retrieved 와 한 건도 어긋나면 안 된다.
    hit = [x for x in rows if x["retrieval_correct"]]
    miss = [x for x in rows if not x["retrieval_correct"]]
    assert sum(x["answer_correct"] for x in hit) == sum(
        gold[q]["answer_correct"] for q in gold if gold[q]["retrieval_correct"])
    assert sum(x["answer_correct"] for x in miss) == sum(
        retr[q]["answer_correct"] for q in retr if not retr[q]["retrieval_correct"])
    assert len(rows) == len(retr)

    with out.open("w") as fh:
        for x in rows:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    summary = {"records": gmeta["records"], "condition": "oracle",
               "reader": gmeta["reader"], "prompt": gmeta.get("prompt", "base"),
               "seed": gmeta.get("seed"),
               "max_new_tokens": gmeta.get("max_new_tokens"),
               "batch_size": 1,
               "excluded_unit_defect": gmeta.get("excluded_unit_defect", False),
               # 새 생성이 아니라 두 레그의 합성이다. 표에 그렇게 적힌다.
               "composed": True,
               "composed_from": {"retrieval_hit": f"{TAG}_answer_gold.jsonl",
                                 "retrieval_miss": f"{TAG}_answer_retrieved.jsonl"},
               **summarize(rows, gmeta.get("context_limit") or 0)}
    (D / f"{TAG}_answer_oracle.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
