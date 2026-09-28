#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""MultiHiertt LLM 필터 단계 (PREREG-2026-09-28-fair-mh.md ④).

입력 = mh_arms.py 검색 기록(doc 범위). 질문마다 검색이 배달한 줄(셀 문장 arm 은 줄 = 셀 1개, 청크 arm 은 줄 = 표의 행)에
번호를 붙여 필터 LLM 에 주고, 답에 필요한 줄 번호만 받는다. 선택된 줄의 **원문**을 그대로 모아 answer_accuracy_mh.py 가 읽는
검색 기록 형식(records.jsonl + 요약 .json)으로 다시 쓴다. 필터 호출과 답변 호출은 분리된다(같은 모델, 다른 실행).

규칙(실행 전 고정):
- 입력에 없는 번호는 버리고 `n_invalid_ids` 에 센다. 숫자를 하나도 못 뽑으면(빈 선택·파싱 실패) 문맥은 **빈 목록**이 되고
  `filter_empty=1` 로 센다. 전체 줄로 되돌리는 자동 대체는 하지 않는다.
- 필터는 새 텍스트를 만들지 않는다. 답변 모델에는 선택된 줄의 원문만 간다.
- 정답 정보는 필터 입력에 넣지 않는다. 채점(doc.correct = 선택된 줄이 정답 셀 전부를 담는가)은 저장할 때만 한다.
  셀 arm 은 셀 좌표(ranked_units)로 정확히 세고, 청크 arm 은 줄↔셀 대응이 없어 정답 셀 값 문자열이 남았는지(대리 지표)만 센다.

  PYTHONPATH=. .venv/bin/python scripts/mh_filter.py --records <검색 기록> --out <filtered_records.jsonl> \
      [--same-queries-as results/mh_arms/cap300_20260924/cell.jsonl] [--batch-size 128]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from answer_accuracy import check_context_limit                          # noqa: E402
from mh_arms import build_tables, load_population, resolve_gold          # noqa: E402
from rag_agent.eval.artifacts import digest, file_digest, provenance, read_records  # noqa: E402
from rag_agent.llm.factory import build_llm                             # noqa: E402

FILTER_SYS = (
    "You select which context lines are needed to answer a question about tables from a financial report. "
    "Reply with only the line numbers, comma-separated, in increasing order. "
    "If the question needs a calculation or a comparison over several values, include EVERY line that holds a value "
    "the calculation or comparison needs — not just one. Also include any header or label lines needed to interpret "
    "those values. Do not explain, do not repeat the lines — output numbers only."
)


def split_lines(context):
    return [ln for entry in (context or []) for ln in str(entry).split("\n") if ln.strip()]


def parse_keep(raw: str, n: int):
    nums = [int(x) for x in re.findall(r"\d+", raw or "")]
    keep = sorted({x for x in nums if 1 <= x <= n})
    return keep, sum(1 for x in nums if not 1 <= x <= n)


def gold_values(split, header_rule, label_rule):
    """청크 arm 대리 지표용: 질의 id -> 정답 셀 값 문자열 목록 (표 원문 값)."""
    queries, docs, _ = load_population(split)
    tables, hdr = build_tables(docs, header_rule, label_rule)
    live = {(tid, i, j) for tid, tab in tables.items()
            for i, row in enumerate(tab.table.data) for j, v in enumerate(row) if str(v).strip()}
    return {q["uid"]: [str(tables[t].table.data[i][j]).strip() for t, i, j in q["gold"]]
            for q in resolve_gold(queries, tables, hdr, live) if q["gold"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scope", default="doc")
    ap.add_argument("--same-queries-as", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="local:Qwen/Qwen3-8B?quantization=4bit")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--split", default="train")
    ap.add_argument("--header-rule", default="v3.3u")
    ap.add_argument("--label-rule", default="none")
    a = ap.parse_args()

    recs = [r for r in read_records(a.records).values() if a.scope in r]
    meta_path = Path(a.records).with_name(Path(a.records).stem.removesuffix("_records") + ".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("records_sha256") != file_digest(a.records) or meta.get("context_version") != 2:
        raise SystemExit("retrieval summary is missing, stale, or incompatible")
    if a.same_queries_as:
        keep = {json.loads(l)["query_id"] for l in open(a.same_queries_as, encoding="utf-8")}
        recs = [r for r in recs if r["query_id"] in keep]
        if len(recs) != len(keep):
            raise SystemExit(f"--same-queries-as {len(keep)}건 중 {len(recs)}건만 기록에 있다")
    if a.limit:
        recs = recs[:a.limit]
    cell_arm = meta.get("unit") == "cell"
    gv = None if cell_arm else gold_values(a.split, a.header_rule, a.label_rule)
    out = Path(a.out)
    meta_out = out.with_name(out.stem.removesuffix("_records") + ".json")   # answer_accuracy_mh 가 찾는 이름
    if out.exists() or meta_out.exists():
        raise SystemExit(f"{out} 있음 — 덮어쓰지 않는다")
    out.parent.mkdir(parents=True, exist_ok=True)

    llm = build_llm(a.model)
    limit = getattr(llm, "context_limit", 0)
    details = llm.metadata() if hasattr(llm, "metadata") else {"name": llm.name}
    t0, rows = time.time(), []
    with out.open("x", encoding="utf-8", newline="\n") as stream:
        for s in range(0, len(recs), a.batch_size):
            batch = recs[s:s + a.batch_size]
            lines_b = [split_lines(r[a.scope]["context"]) for r in batch]
            users = [f"Question: {r['question']}\n\nLines:\n" + "\n".join(f"{i}. {ln}" for i, ln in enumerate(lines, 1))
                     + "\n\nLine numbers needed:" for r, lines in zip(batch, lines_b)]
            n_toks = [llm.n_prompt_tokens(FILTER_SYS, u) for u in users]
            for n in n_toks:
                check_context_limit(n, a.max_tokens, limit)
            raws = llm.complete_batch(FILTER_SYS, users, max_tokens=a.max_tokens)
            for r, lines, u, n_tok, raw in zip(batch, lines_b, users, n_toks, raws):
                keep, n_invalid = parse_keep(raw, len(lines))
                kept = [lines[i - 1] for i in keep]
                d = r[a.scope]
                if cell_arm:
                    # 셀 arm: 단위 하나 = 셀 하나 = 줄 하나. ranked_units 는 순위 순서, context 도 같은 순서.
                    units = d["ranked_units"][:len(d["context"])]
                    assert len(units) == len(lines) == len(d["context"]), r["query_id"]
                    kept_cells = sorted({c for i in keep for c in units[i - 1]})
                    gold = set(r["gold_ids"])
                    correct, n_cells = int(gold <= set(kept_cells)), len(kept_cells)
                    gold_kept, proxy = len(gold & set(kept_cells)), None
                else:
                    kept_cells, n_cells = None, None
                    vals = gv[r["query_id"]]
                    text = "\n".join(kept)
                    gold_kept = sum(v in text for v in vals)
                    correct, proxy = int(gold_kept == len(vals)), "gold value string present"
                new = {k: v for k, v in r.items() if k not in ("corpus", "table")}
                new[a.scope] = {"correct": correct, "any_DIAGNOSTIC": int(gold_kept > 0),
                                "cells_in_context": n_cells, "gold_doc_in_context": d["gold_doc_in_context"],
                                "context": kept, "context_sha256": digest(kept),
                                "kept_cells": kept_cells, "n_gold_kept": gold_kept, "correct_is_proxy": proxy,
                                "filter": {"n_lines": len(lines), "n_kept": len(keep), "kept_idx": keep,
                                           "raw": raw, "n_invalid_ids": n_invalid, "filter_empty": int(not keep),
                                           "input_tokens": n_tok, "input_sha256": digest([FILTER_SYS, u])},
                                "pre_filter": {"correct": d["correct"], "cells_in_context": d["cells_in_context"],
                                               "context_sha256": d["context_sha256"]}}
                stream.write(json.dumps(new, ensure_ascii=False) + "\n")
                rows.append(new)
            stream.flush(); os.fsync(stream.fileno())
            print(f"  {len(rows)}/{len(recs)}  {time.time() - t0:.0f}s", flush=True)
    f = [x[a.scope]["filter"] for x in rows]
    summary = {**{k: meta[k] for k in ("unit", "template", "arguments", "split", "corpus", "header_rule") if k in meta},
               "context_version": 2, "stage": "llm_filter", "source_records": a.records,
               "source_records_sha256": file_digest(a.records), "provenance": provenance(ROOT),
               "filter_model": llm.name, "filter_details": details, "filter_prompt": FILTER_SYS,
               "filter_prompt_sha256": digest(FILTER_SYS), "max_new_tokens": a.max_tokens, "batch_size": a.batch_size,
               "n": len(rows), "n_filter_empty": sum(x["filter_empty"] for x in f),
               "n_queries_with_invalid_ids": sum(x["n_invalid_ids"] > 0 for x in f),
               "n_invalid_ids_total": sum(x["n_invalid_ids"] for x in f),
               "lines_mean": round(sum(x["n_lines"] for x in f) / len(f), 2),
               "kept_mean": round(sum(x["n_kept"] for x in f) / len(f), 2),
               "correct_is_proxy": None if cell_arm else "gold value string present in kept lines",
               "seconds": round(time.time() - t0, 1)}
    summary["records_sha256"] = file_digest(out)
    meta_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("provenance", "filter_details", "arguments")},
                     ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
