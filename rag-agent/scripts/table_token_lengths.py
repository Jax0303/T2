#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""표 전체를 리더에 넣으면 몇 토큰인가 — 문제 정의 ①의 근거와 "표 전체" 비교군의 가능 여부.

직렬화는 청킹 arm 과 같은 markdown(`chunks.markdown_source`: "Table name: …" + 파이프 행).
토큰은 각 데이터셋 리더의 토크나이저(HiTab Qwen2.5-7B-Instruct, MultiHiertt Qwen3-8B)로 센다.
  HiTab       질의가 속한 표 1개 — test 단일 셀 조회 991건(주 모집단 레코드의 질의 id)
  MultiHiertt 질의가 속한 문서의 표 전부 — train 채점 2,871건(본 방법 검색 레코드의 질의 id)

  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python scripts/table_token_lengths.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from transformers import AutoTokenizer                                   # noqa: E402

from mh_arms import build_tables, load_population                       # noqa: E402
from rag_agent.bench import hitab_grid as hg                            # noqa: E402
from rag_agent.serialization.chunks import markdown_source              # noqa: E402

LIMITS = (512, 1024, 2048, 4096, 8192)


def stats(v):
    v = sorted(v)
    q = lambda p: v[min(len(v) - 1, int(p * len(v)))]                   # noqa: E731
    return {"n": len(v), "mean": round(sum(v) / len(v), 1), "median": q(.5), "p90": q(.9),
            "max": v[-1], **{f"share_over_{k}": round(sum(x > k for x in v) / len(v), 4) for k in LIMITS}}


def main() -> int:
    out = {}
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    recs = [r for r in map(json.loads, open(ROOT / "results/retrieval_accuracy/t_sleaf_gold_records.jsonl"))
            if r.get("aggregation") == "none" and r.get("m") == 1 and r.get("mode") == "all"
            and not r.get("excluded")]
    if len(recs) != 991:
        raise SystemExit(f"HiTab 주 모집단이 991 이 아니다: {len(recs)}")
    lens, ctx = [], []
    for r in recs:
        tab = hg.load_table(r["table_id"])
        text, _ = markdown_source(tab, tab.table, tab.title)
        lens.append(len(tok(text)["input_ids"]))
        ctx.append(len(tok("\n".join(r["context"]))["input_ids"]))
    out["hitab_table"] = stats(lens)
    out["hitab_top20_context"] = stats(ctx)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    want = {json.loads(l)["query_id"]: json.loads(l)
            for l in open(ROOT / "results/mh_arms/mh_train_cell_hv1_none_doc_records.jsonl")}
    want = {q: r for q, r in want.items() if "doc" in r}
    queries, docs, _ = load_population("train")
    tables, _ = build_tables(docs, "v1", "none")
    by_doc = {}
    for tid, tab in tables.items():
        by_doc.setdefault(tid.split("::")[0], []).append((tid, tab))
    lens, ctx = [], []
    for q in queries:
        if q["uid"] not in want:
            continue
        parts = [markdown_source(tab, tab.table, tab.title)[0]
                 for _, tab in sorted(by_doc[q["uid"]], key=lambda x: int(x[0].split("::")[1]))]
        lens.append(len(tok("\n\n".join(parts))["input_ids"]))
        ctx.append(len(tok("\n".join(want[q["uid"]]["doc"].get("context") or []))["input_ids"]))
    if len(lens) != len(want):
        raise SystemExit(f"MultiHiertt 질의 수 불일치: {len(lens)} != {len(want)}")
    out["multihiertt_doc_tables"] = stats(lens)
    out["multihiertt_top20_context"] = stats(ctx)
    dst = ROOT / "results/table_token_lengths.json"
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
