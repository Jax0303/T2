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

Which `gold` leg to compose from is an argument, because there is more than one
on disk: `gold` predates the 2026-09-09 title bug fix and `gold_v2` is the rerun
after it. Composing from the stale leg is how a superseded ceiling stays in the
tables (`analysis/accuracy_tables.py: LEG`).

  PYTHONPATH=. .venv/bin/python analysis/compose_oracle.py [gold_v2]
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
    gname = sys.argv[1] if len(sys.argv) > 1 else "gold_v2"
    oname = "oracle" + gname.removeprefix("gold")      # gold_v2 -> oracle_v2
    out = D / f"{TAG}_answer_{oname}.jsonl"
    if out.exists():
        raise SystemExit(f"{out} exists — answer legs never overwrite")
    gold, gmeta = leg(gname)
    retr, rmeta = leg("retrieved")

    # 두 레그가 같은 조건인가. `records`·`reader` 는 둘 다 항상 적는다. prompt·seed·
    # max_new_tokens 는 나중에 생긴 키라 옛 레그에는 없다 -- **한쪽만** 적은 키는
    # 같다고 우길 수 없으므로 검사에서 빼고, 뺐다는 사실을 요약에 적는다.
    KEYS = ("records", "reader", "prompt", "seed", "excluded_unit_defect")
    unverified = [k for k in KEYS if (k in gmeta) != (k in rmeta)]
    for k in KEYS:
        if k in unverified:
            continue
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
               "composed_from": {"retrieval_hit": f"{TAG}_answer_{gname}.jsonl",
                                 "retrieval_miss": f"{TAG}_answer_retrieved.jsonl"},
               # 한쪽 레그에만 있어서 같은지 확인하지 못한 메타 키.
               "meta_unverified": unverified,
               **summarize(rows, gmeta.get("context_limit") or 0)}
    (D / f"{TAG}_answer_{oname}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
