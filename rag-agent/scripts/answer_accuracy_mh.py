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

저장과 재개: 문항마다 결과 행을 즉시 쓰고, 실행 조건(질의 id·순서, 문맥 해시, 모델 revision, 프롬프트,
생성 설정, 코드 해시)을 `<out>.run.json` 에 남긴다. 중단되면 같은 명령에 --resume 을 붙여 남은 문항만
생성한다. 조건이 하나라도 다르거나 저장된 행이 어긋나면 이어 쓰지 않는다.

  PYTHONPATH=. .venv/bin/python scripts/answer_accuracy_mh.py \
      --records results/mh_arms/mh_s3c_records.jsonl --scope doc
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from answer_accuracy import PROMPTS, check_context_limit                # noqa: E402
from mh_arms import (NO_TITLE, OFFICIAL_REPO, OFFICIAL_REV,              # noqa: E402
                     build_tables, load_population, official_answers, resolve_gold)
from rag_agent.eval.artifacts import (digest, file_digest, provenance,  # noqa: E402
                                     read_records)
from rag_agent.eval.multihiertt_em import (docmath_match, mh_exact_match,  # noqa: E402
                                           self_check)
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
                "answer_em_docmath": acc([x["answer_correct_docmath"] for x in rs
                                          if "answer_correct_docmath" in x]),
                "answer_given_retrieval_hit": acc(hit), "n_retrieval_hit": len(hit),
                "answer_given_retrieval_miss": acc(miss), "n_retrieval_miss": len(miss),
                "context_lines_mean": round(sum(x["n_ctx"] for x in rs) / len(rs), 1) if rs else None,
                "input_tokens_mean": (round(sum(x["n_tok"] for x in rs) / len(rs), 1)
                                      if rs and rs[0]["n_tok"] is not None else None)}

    over = [x for x in rows if x.get("n_tok") and limit and x["n_tok"] > limit]
    return {"context_limit": limit, "n_over_context_limit": len(over),
            "by_layer": {k: block(v) for k, v in sorted(by.items())}}


def run_config(a, llm, details, ctxs):
    """재개할 때 저장된 실행과 같아야 하는 조건. 코드는 이 시점에 실제로 불러온 저장소 파일의 해시다."""
    root = ROOT.resolve()
    code = {}
    for module in list(sys.modules.values()):
        f = getattr(module, "__file__", None)
        # 상대 경로 __file__(torch 가 등록하는 '_ops.py' 등)은 저장소 파일이 아니다 — cwd 로 풀지 않는다
        p = Path(f).resolve() if f and os.path.isabs(f) else None
        if p and p.suffix == ".py" and p.is_file() and p.is_relative_to(root) and ".venv" not in p.parts:
            code[str(p.relative_to(root))] = file_digest(p)
    return {"records_sha256": file_digest(a.records), "scope": a.scope, "condition": a.condition,
            "split": a.split, "header_rule": a.header_rule, "label_rule": a.label_rule,
            "query_ids_sha256": digest(list(ctxs)),
            "contexts_sha256": digest({q: digest(c) for q, c in ctxs.items()}),
            "reader": llm.name,
            "reader_details": {k: v for k, v in details.items() if k != "chat_template"},
            "chat_template_sha256": digest(details.get("chat_template")),
            "prompt": a.prompt, "prompt_sha256": digest(PROMPTS[a.prompt]),
            "max_new_tokens": a.max_tokens, "temperature": 0.0, "seed": a.seed, "batch_size": 1,
            "packages": provenance(ROOT)["packages"],
            "code_sha256": digest(code), "code_files": code}


def load_saved_rows(path, order, ctxs):
    """이미 저장된 행. 중복·문맥 불일치·중간 누락이면 파일을 건드리지 않고 멈추고,
    통과하면 쓰다 끊긴 마지막 줄만 잘라 낸다."""
    raw = path.read_bytes()
    whole = raw[:raw.rfind(b"\n") + 1]
    done = {}
    for line in whole.decode("utf-8").splitlines():
        row = json.loads(line)
        q = row["query_id"]
        if q in done:
            raise SystemExit(f"같은 질의가 두 번 저장돼 있다: {q}")
        if q not in ctxs or row["context_sha256"] != digest(ctxs[q]):
            raise SystemExit(f"저장된 행의 문맥이 이번 실행과 다르다: {q}")
        done[q] = row
    if list(done) != order[:len(done)]:
        raise SystemExit(f"저장된 {len(done)}행이 실행 순서의 앞부분이 아니다 — 중간 누락")
    if len(whole) != len(raw):
        with path.open("r+b") as stream:
            stream.truncate(len(whole))
        print(f"[resume] 쓰다 끊긴 마지막 줄 {len(raw) - len(whole)}바이트를 잘라 냈다", flush=True)
    print(f"[resume] 저장된 {len(done)}행을 이어 쓴다", flush=True)
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True)
    ap.add_argument("--scope", default="doc", choices=["doc", "corpus", "table"])
    ap.add_argument("--condition", default="retrieved", choices=["retrieved", "gold"])
    # 기본값 = 채택 설정 (PREREG-2026-09-13-reader-qwen3.md, 재확인 PREREG-2026-09-23-reader-thinking-pilot.md).
    # 2026-09-21 실행 15개가 옛 기본값(neutral, 64)으로 돌아 채택 설정이 아닌 수치를 낸 전례가 있다.
    ap.add_argument("--reader", default="local:Qwen/Qwen3-8B?quantization=4bit")
    ap.add_argument("--prompt", default="cot", choices=list(PROMPTS))
    ap.add_argument("--max-tokens", type=int, default=384)
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
    ap.add_argument("--resume", action="store_true",
                    help="--out 에 저장된 행이 있으면 이어 쓴다. 실행 조건(<out>.run.json)이 하나라도 "
                         "다르면 멈춘다. 저장된 행이 없으면 새로 시작한다.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="train")
    ap.add_argument("--header-rule", default="v1", choices=["v1", "v2", "v3", "v3.1", "v3.2", "v3.3"])
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
    run_path = out.with_suffix(".run.json")
    if out.with_suffix(".json").exists():
        raise SystemExit(f"{out.with_suffix('.json')} exists — 완료된 답변 레그는 다시 돌리지 않는다; pass a new --out")
    if (out.exists() or run_path.exists()) and not a.resume:
        raise SystemExit(f"{out} 에 저장된 실행이 있다 — 이어 쓰려면 --resume (덮어쓰지 않는다)")

    tables = gold = None
    if a.condition == "gold":
        gold, tables = gold_cells_by_query(a.split, a.header_rule, a.label_rule)
        bad = [r["query_id"] for r in scored if len(gold.get(r["query_id"], [])) != r["m"]]
        if bad:
            raise SystemExit(f"gold 셀 수가 검색 레코드와 다르다: {len(bad)}건 (예: {bad[:3]})")

    # 검색 레코드의 answer 는 bevaya 판의 잘린 값일 수 있다 — 채점 정답은 공식 파일에서 읽는다
    answers = official_answers(a.split)
    for r in scored:
        r["answer"] = answers[r["query_id"]]
    check = self_check([r["answer"] for r in scored], [r["kind"] == "arith" for r in scored])
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
    details = llm.metadata() if hasattr(llm, "metadata") else {"name": llm.name}
    run = run_config(a, llm, details, ctxs)
    if run_path.exists():
        saved = json.loads(run_path.read_text(encoding="utf-8"))
        bad = sorted(k for k in set(run) | set(saved) if run.get(k) != saved.get(k))
        if bad:
            raise SystemExit(f"재개 조건이 저장된 실행과 다르다: {bad}")
    elif out.exists():
        raise SystemExit(f"{out} 은 있는데 {run_path.name} 이 없다 — 이어 쓸 근거가 없다")
    else:
        with run_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(run, stream, ensure_ascii=False, indent=2, allow_nan=False)
    order = [r["query_id"] for r in scored]
    done = load_saved_rows(out, order, ctxs) if out.exists() else {}

    rows, t0 = list(done.values()), time.time()
    # 문항마다 즉시 쓴다 — 중단돼도 끝난 문항은 디스크에 남는다
    with out.open("a", encoding="utf-8", newline="\n") as stream:
        for k, r in enumerate(scored, 1):
            if r["query_id"] in done:
                continue
            t = time.time()
            ctx = ctxs[r["query_id"]]
            user = "Context:\n" + "\n".join(ctx) + f"\n\nQuestion: {r['question']}\nAnswer:"
            n_tok = (llm.n_prompt_tokens(PROMPTS[a.prompt], user)
                     if hasattr(llm, "n_prompt_tokens") else None)
            check_context_limit(n_tok, a.max_tokens, limit)
            raw = llm.complete(PROMPTS[a.prompt], user, max_tokens=a.max_tokens,
                               temperature=0.0)
            pred, marked = extract(raw) if a.prompt == "cot" else (raw, True)
            row = {"query_id": r["query_id"], "layer": r["layer"], "kind": r["kind"],
                   "m": r["m"], "question": r["question"],
                   "retrieval_correct": r[a.scope]["correct"],
                   "answer_correct": int(mh_exact_match(pred, r["answer"], r["kind"] == "arith")),
                   "answer_correct_docmath": int(docmath_match(pred, r["answer"], r["kind"] == "arith")),
                   "n_ctx": len(ctx), "n_tok": n_tok,
                   "cells_in_context": r[a.scope]["cells_in_context"],
                   "context_sha256": digest(ctx),
                   "source_context_sha256": r[a.scope].get("context_sha256"),
                   "context_condition": a.condition,
                   "pred": pred, "answer": r["answer"]}
            if a.prompt == "cot":
                row.update(raw=raw, marker_found=marked)
            row["seconds"] = round(time.time() - t, 3)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            rows.append(row)
            if k % 100 == 0:
                print(f"  {k}/{len(scored)}  {time.time() - t0:.0f}s  "
                      f"em={sum(x['answer_correct'] for x in rows) / len(rows):.4f}", flush=True)
    if [x["query_id"] for x in rows] != order:
        raise SystemExit("저장된 행이 실행 계획과 다르다 — 누락 또는 중복")

    summary = {"records": a.records, "scope": a.scope, "condition": a.condition,
               "context_version": 2, "provenance": provenance(ROOT),
               "retrieval_records_sha256": file_digest(a.records),
               "retrieval_unit": meta.get("unit"), "retrieval_template": meta.get("template"),
               "query_ids_sha256": digest(sorted(r["query_id"] for r in scored)),
               "reader_details": details,
               "prompt_sha256": digest(PROMPTS[a.prompt]), "prompt_text": PROMPTS[a.prompt],
               "scorer": "multihiertt_em.mh_exact_match (공식 포팅, 문항 유형 갈래)",
               "scorer_secondary": "multihiertt_em.docmath_match (DocMath-Eval compare_two_numbers)",
               "answer_source": f"{OFFICIAL_REPO}@{OFFICIAL_REV}",
               "scorer_self_check": check, "reader": llm.name, "prompt": a.prompt,
               "seed": a.seed, "max_new_tokens": a.max_tokens, "batch_size": 1,
               "stratum_cap": a.stratum_cap, "sample_seed": a.sample_seed,
               "header_rule": a.header_rule, "label_rule": a.label_rule,
               "same_queries_as": a.same_queries_as or None,
               "same_contexts_as": a.same_contexts_as or None,
               "n_missing_from_same_queries": n_missing_same,
               "generation_seconds": round(sum(x.get("seconds", 0) for x in rows), 1),
               "resumed_rows": len(done), "run_config_sha256": file_digest(run_path),
               "marker_missing": (sum(not x["marker_found"] for x in rows)
                                  if a.prompt == "cot" else None),
               **summarize(rows, limit)}
    summary["records_sha256"] = file_digest(out)
    with out.with_suffix(".json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "provenance"},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
