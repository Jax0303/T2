#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""MultiHiertt 답변 정확도 — 검색이 배달한 그 문맥 그대로.

`scripts/mh_arms.py` 의 표 1 에 대응하는 표 2 다. 같은 분할·같은 질의·같은 셀:
리더는 검색이 고른 단위 텍스트를 그대로 받는다. 그래야 "검색이 .84 인데 답이 왜
.84 가 아닌가"가 추측이 아니라 산수가 된다 (`CLAUDE.md` §0.2).

채점기는 MultiHiertt 공식 것을 옮긴 `rag_agent/eval/multihiertt_em.py` 하나뿐이다.
HiTab 쪽 EM 과 **섞어 평균 내지 않는다** — 채점기가 다르다.

조건 둘:
  retrieved  검색이 고른 셀. 운영 수치.
  gold       주석된 근거 셀만. 리더 천장이지 수학적 상한이 아니다.

  PYTHONPATH=. .venv/bin/python scripts/answer_accuracy_mh.py \
      --records results/mh_arms/mh_s3c_records.jsonl --scope doc
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from answer_accuracy import PROMPTS, check_context_limit                # noqa: E402
from mh_arms import (NO_TITLE, build_tables, load_population,           # noqa: E402
                     resolve_gold)
from rag_agent.eval.artifacts import (digest, file_digest, provenance,  # noqa: E402
                                     read_records, write_pair)
from rag_agent.eval.multihiertt_em import mh_exact_match, self_check    # noqa: E402
from rag_agent.llm.factory import build_llm                             # noqa: E402
from rag_agent.serialization.caption import caption_sentence            # noqa: E402
from rag_agent.serialization.templates import STRUCTURAL_COMPACT        # noqa: E402

# 풀이 허용 프롬프트 — scripts/reader_cot_pilot.py 에서 문구 그대로 옮겼다. 마지막
# 'Final answer:' 뒤만 채점하고, 표시가 없으면 출력 전체로 채점한다(추출을 관대하게 하지 않는다).
COT = ("Answer the question using only the provided table context. Read the row and "
       "column labels to identify the relevant values. Think step by step and show the "
       "calculation. On the last line write 'Final answer: ' followed by only the final "
       "value, with no units or explanation.")
_FINAL = re.compile(r"final answer:\s*(.+)", re.I)
PROMPTS = {**PROMPTS, "cot": COT}


def extract(text: str):
    hits = _FINAL.findall(text or "")
    return (hits[-1].strip().rstrip(".").strip(), True) if hits else ((text or "").strip(), False)


def gold_cells_by_query(split, header_rule="v1", label_rule="none"):
    """(질의 id -> gold 셀, 표) — 검색 레코드는 gold 좌표를 싣지 않으므로 다시 푼다.

    `mh_arms.py` 와 같은 함수·같은 arm 무관 규칙(표의 비어 있지 않은 데이터 셀)으로
    푼다. 채점 레코드의 `m` 과 개수가 어긋나면 멈춘다.
    """
    queries, docs, _ = load_population(split)
    tables, hdr = build_tables(docs, header_rule, label_rule)
    live = {(tid, i, j) for tid, tab in tables.items()
            for i, row in enumerate(tab.table.data)
            for j, v in enumerate(row) if str(v).strip()}
    queries = resolve_gold(queries, tables, hdr, live)
    return {q["uid"]: sorted(q["gold"]) for q in queries if q["gold"]}, tables


def gold_context(cells, tables):
    """주석된 gold 셀만, 색인과 같은 문장으로."""
    out = []
    for tid, i, j in cells:
        t = tables[tid].table
        # 색인과 같은 라벨. HiTab 에서 gold 문맥만 제목을 빠뜨렸던 결함의 전례.
        out.append(caption_sentence(tables[tid].title, t.row_path(i), t.col_path(j),
                                    value=t.data[i][j], template=STRUCTURAL_COMPACT))
    return out


def summarize(rows, limit=0):
    def acc(v):
        return round(sum(v) / len(v), 4) if v else None

    by = defaultdict(list)
    for x in rows:
        by[x["layer"]].append(x)
        by["ALL"].append(x)

    def block(rs):
        hit = [x["answer_correct"] for x in rs if x["retrieval_correct"]]
        miss = [x["answer_correct"] for x in rs if not x["retrieval_correct"]]
        return {"n": len(rs),
                "retrieval_accuracy_here": acc([x["retrieval_correct"] for x in rs]),
                "answer_em": acc([x["answer_correct"] for x in rs]),
                "answer_given_retrieval_hit": acc(hit), "n_retrieval_hit": len(hit),
                "answer_given_retrieval_miss": acc(miss), "n_retrieval_miss": len(miss),
                "context_lines_mean": round(sum(x["n_ctx"] for x in rs) / len(rs), 1) if rs else None,
                "input_tokens_mean": (round(sum(x["n_tok"] for x in rs) / len(rs), 1)
                                      if rs and rs[0]["n_tok"] is not None else None)}

    over = [x for x in rows if x.get("n_tok") and limit and x["n_tok"] > limit]
    return {"context_limit": limit, "n_over_context_limit": len(over),
            "by_layer": {k: block(v) for k, v in sorted(by.items())}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True)
    ap.add_argument("--scope", default="doc", choices=["doc", "corpus"])
    ap.add_argument("--condition", default="retrieved", choices=["retrieved", "gold"])
    ap.add_argument("--reader", default="local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit")
    ap.add_argument("--prompt", default="neutral", choices=list(PROMPTS))
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stratum-cap", type=int, default=0,
                    help="층마다 최대 이 개수까지만 리더에 태운다 (0 = 전부). "
                         "질의 id 를 정렬한 뒤 --sample-seed 로 뽑으므로 arm 이 "
                         "달라도 **같은 질의 집합**이 나온다. 리더가 병목이라 "
                         "arith_m2+ 만 2,224건이어서 두는 장치이고, 층별 n 은 "
                         "요약에 그대로 적힌다.")
    ap.add_argument("--sample-seed", type=int, default=20260913)
    ap.add_argument("--same-queries-as", default="",
                    help="이 답변 결과(jsonl)와 **같은 질의 id** 만 태운다. 헤더 규칙이 바뀌면 "
                         "모집단이 몇 건 달라져 층별 표본이 다시 뽑히므로, 짝지음을 지키려고 "
                         "표본 대신 id 목록을 고정한다. 새 레코드에 없는 id 는 세어서 요약에 적는다.")
    ap.add_argument("--same-contexts-as", default="",
                    help="--same-queries-as 에 더해 질의마다 context_sha256 까지 이 답변 결과와 "
                         "같아야 한다. 리더·프롬프트만 바꾼 재실행용이며, 하나라도 다르면 리더를 "
                         "싣기 전에 멈춘다.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="train")
    ap.add_argument("--header-rule", default="v1", choices=["v1", "v2"])
    ap.add_argument("--label-rule", default="none", choices=["none", "L1", "L2"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.same_contexts_as:
        if a.same_queries_as:
            raise SystemExit("--same-queries-as 와 --same-contexts-as 는 함께 주지 않는다")
        a.same_queries_as = a.same_contexts_as

    recs = list(read_records(a.records).values())
    scored = [r for r in recs if a.scope in r]
    if not scored:
        raise SystemExit("no scored records for that scope")
    meta_path = Path(a.records).with_name(
        Path(a.records).stem.removesuffix("_records") + ".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("records_sha256") != file_digest(a.records) or meta.get("context_version") != 2:
        raise SystemExit("retrieval summary is missing, stale, or incompatible")
    n_missing_same = None
    if a.same_queries_as:
        want = [json.loads(l)["query_id"] for l in open(a.same_queries_as, encoding="utf-8")]
        have = {r["query_id"] for r in scored}
        n_missing_same = sum(1 for q in want if q not in have)
        keep = set(want)
        scored = sorted((r for r in scored if r["query_id"] in keep), key=lambda x: x["query_id"])
        print(f"[same-queries] {len(want)} 요청 -> {len(scored)} (새 레코드에 없음 {n_missing_same})", flush=True)
    if a.stratum_cap and not a.same_queries_as:
        import random
        by = defaultdict(list)
        for r in scored:
            by[r["layer"]].append(r)
        keep = []
        for name in sorted(by):
            ids = sorted(x["query_id"] for x in by[name])
            if len(ids) > a.stratum_cap:
                ids = sorted(random.Random(a.sample_seed).sample(ids, a.stratum_cap))
            chosen = set(ids)
            keep += [x for x in by[name] if x["query_id"] in chosen]
            print(f"[sample] {name}: {len(by[name])} -> {len(chosen)}", flush=True)
        scored = sorted(keep, key=lambda x: x["query_id"])
    if a.limit:
        scored = scored[:a.limit]
    out = Path(a.out or meta_path.with_name(
        meta_path.stem + f"_answer_{a.scope}_{a.condition}.jsonl"))
    if out.suffix != ".jsonl":
        raise SystemExit("--out must end in .jsonl")
    if out.exists() or out.with_suffix(".json").exists():
        raise SystemExit(f"{out} exists — answer legs never overwrite; pass a new --out")

    tables = gold = None
    if a.condition == "gold":
        gold, tables = gold_cells_by_query(a.split, a.header_rule, a.label_rule)
        bad = [r["query_id"] for r in scored if len(gold.get(r["query_id"], [])) != r["m"]]
        if bad:
            raise SystemExit(f"gold 셀 수가 검색 레코드와 다르다: {len(bad)}건 (예: {bad[:3]})")

    check = self_check([r["answer"] for r in scored])
    print(f"[scorer] gold 를 예측으로 넣었을 때 EM={check['em']} (n={check['n']})", flush=True)

    ctxs = {r["query_id"]: (gold_context(gold[r["query_id"]], tables) if a.condition == "gold"
                            else r[a.scope].get("context") or []) for r in scored}
    if a.same_contexts_as:
        ref = {x["query_id"]: x["context_sha256"]
               for x in map(json.loads, open(a.same_contexts_as, encoding="utf-8"))}
        bad = [q for q, c in ctxs.items() if digest(c) != ref.get(q)]
        if bad:
            raise SystemExit(f"문맥이 기준 실행과 다르다: {len(bad)}건 (예: {bad[:3]})")
        print(f"[same-contexts] {len(ctxs)}건 context_sha256 일치", flush=True)

    llm = build_llm(a.reader)
    import torch                                                       # noqa: E402
    torch.manual_seed(a.seed)
    limit = getattr(llm, "context_limit", 0)
    rows, t0 = [], time.time()
    for k, r in enumerate(scored, 1):
        ctx = ctxs[r["query_id"]]
        user = "Context:\n" + "\n".join(ctx) + f"\n\nQuestion: {r['question']}\nAnswer:"
        n_tok = (llm.n_prompt_tokens(PROMPTS[a.prompt], user)
                 if hasattr(llm, "n_prompt_tokens") else None)
        check_context_limit(n_tok, a.max_tokens, limit)
        raw = llm.complete(PROMPTS[a.prompt], user, max_tokens=a.max_tokens,
                           temperature=0.0)
        pred, marked = extract(raw) if a.prompt == "cot" else (raw, True)
        rows.append({"query_id": r["query_id"], "layer": r["layer"], "kind": r["kind"],
                     "m": r["m"], "question": r["question"],
                     "retrieval_correct": r[a.scope]["correct"],
                     "answer_correct": int(mh_exact_match(pred, r["answer"])),
                     "n_ctx": len(ctx), "n_tok": n_tok,
                     "cells_in_context": r[a.scope]["cells_in_context"],
                     "context_sha256": digest(ctx),
                     "source_context_sha256": r[a.scope].get("context_sha256"),
                     "context_condition": a.condition,
                     "pred": pred, "answer": r["answer"]})
        if a.prompt == "cot":
            rows[-1].update(raw=raw, marker_found=marked)
        if k % 100 == 0:
            print(f"  {k}/{len(scored)}  {time.time() - t0:.0f}s  "
                  f"em={sum(x['answer_correct'] for x in rows) / k:.4f}", flush=True)

    summary = {"records": a.records, "scope": a.scope, "condition": a.condition,
               "context_version": 2, "provenance": provenance(ROOT),
               "retrieval_records_sha256": file_digest(a.records),
               "retrieval_unit": meta.get("unit"), "retrieval_template": meta.get("template"),
               "query_ids_sha256": digest(sorted(r["query_id"] for r in scored)),
               "reader_details": llm.metadata() if hasattr(llm, "metadata") else {"name": llm.name},
               "prompt_sha256": digest(PROMPTS[a.prompt]), "prompt_text": PROMPTS[a.prompt],
               "scorer": "multihiertt_em.mh_exact_match (공식 포팅)",
               "scorer_self_check": check, "reader": llm.name, "prompt": a.prompt,
               "seed": a.seed, "max_new_tokens": a.max_tokens, "batch_size": 1,
               "stratum_cap": a.stratum_cap, "sample_seed": a.sample_seed,
               "header_rule": a.header_rule, "label_rule": a.label_rule,
               "same_queries_as": a.same_queries_as or None,
               "same_contexts_as": a.same_contexts_as or None,
               "n_missing_from_same_queries": n_missing_same,
               "generation_seconds": round(time.time() - t0, 1),
               "marker_missing": (sum(not x["marker_found"] for x in rows)
                                  if a.prompt == "cot" else None),
               **summarize(rows, limit)}
    write_pair(out, rows, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "provenance"},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
