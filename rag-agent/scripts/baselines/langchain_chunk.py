#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""`langchain_chunk` — LangChain 공식 RAG 튜토리얼 파이프라인을 MultiHiertt 비교군으로.

튜토리얼(https://docs.langchain.com/oss/python/langchain/rag) 그대로:
  문서 -> RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
     -> 임베딩 -> InMemoryVectorStore -> similarity_search(question) (기본 k=4)
     -> rlm/rag-prompt -> LLM.
바꾼 것은 임베딩(BAAI/bge-base-en-v1.5, 본 방법과 같은 인코더·질의 접두어)과 LLM(Qwen3-8B 4bit 비생각 greedy)뿐이며,
숫자 EM 채점을 위해 rag-prompt 끝에 `answer_accuracy_mh.py` 의 'Final answer:' 표시 문장을 붙였다(README 참고).

같은 조건: 검색 범위 = 질문의 문서(train)의 표만, `chunk`·`fulltable` 과 같은 markdown(`markdown_source`, 머리글 v3.3u),
셀 배달 판정 = `serialization/chunks.table_chunks` 와 같은 span 규칙(값이 통째로 청크 안에 있어야 배달).

  PYTHONPATH=. .venv/bin/python scripts/baselines/langchain_chunk.py --stage retrieve [--limit 3]
  PYTHONPATH=. .venv/bin/python scripts/baselines/langchain_chunk.py --stage answer   [--limit 3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from answer_accuracy import check_context_limit                        # noqa: E402
from answer_accuracy_mh import COT, extract                             # noqa: E402
from mh_arms import (NO_TITLE, OFFICIAL_REPO, OFFICIAL_REV, build_tables,  # noqa: E402
                     cell_id, layer, load_population, official_answers, resolve_gold)
from rag_agent.eval.artifacts import digest, file_digest, provenance   # noqa: E402
from rag_agent.eval.multihiertt_em import mh_exact_match, self_check   # noqa: E402
from rag_agent.eval.multihiertt_f1 import mh_f1                        # noqa: E402
from rag_agent.llm.factory import _split_spec                          # noqa: E402
from rag_agent.serialization.chunks import markdown_source             # noqa: E402

# 튜토리얼 rlm/rag-prompt 원문. 마지막 문장(Final answer 표시)만 이 저장소의 채점 규약을 위해 덧붙였다.
RAG_PROMPT = ("You are an assistant for question-answering tasks. Use the following pieces of retrieved "
              "context to answer the question. If you don't know the answer, just say that you don't know. "
              "Use three sentences maximum and keep the answer concise.")
MARKER = " On the last line write 'Final answer: ' followed by only the final value, with no units or explanation."
PROMPT = "{instr}\nQuestion: {question} \nContext: {context} \nAnswer:"
PROMPT_CF = "{instr}\nContext: {context} \nQuestion: {question} \nAnswer:"   # original_cf: 검색 결과를 질문 앞에(PREREG-2026-10-15-prompt-order.md)
# 표 없이 질문만(PREREG-2026-10-13-gpt4o.md): 원문에서 문맥 문장("Use the following pieces of retrieved context to answer the question.")만 뺐다.
CLOSED_PROMPT = ("You are an assistant for question-answering tasks. Answer the question. If you don't know the answer, "
                 "just say that you don't know. Use three sentences maximum and keep the answer concise.")
CLOSED_TEMPLATE = "{instr}\nQuestion: {question} \nAnswer:"
TOP_K, DUMP, CHUNK, OVERLAP = 4, 30, 1000, 200
EMBED, READER, MAX_NEW, SEED, BATCH = "BAAI/bge-base-en-v1.5", "Qwen/Qwen3-8B", 384, 42, 128
S3C = ROOT / "results/mh_only_20260928/retrieval/mh_train_s3c_records.jsonl"
SUBSET = ROOT / "results/mh_arms/cap300_20260924/cell_uniq.jsonl"
OUT = ROOT / "results/mh_only_20260928/baselines/langchain_chunk"


def doc_text(uid, tables, hdr):
    """문서의 표를 markdown 으로 이어 붙인 한 문자열과 [(start, end, 짧은 셀 id)] — fulltable 조건과 같은 렌더링
    (`markdown_source`, 제목이 없으므로 'Table name: ' 줄 제거), 표 사이는 빈 줄."""
    parts, spans, off, head = [], [], 0, "Table name: \n"
    for tid in sorted((t for t in tables if t.split("::")[0] == uid), key=lambda t: int(t.split("::")[1])):
        tab = tables[tid]
        text, sp = markdown_source(tab, tab.table, NO_TITLE)
        assert text.startswith(head)
        text = text[len(head):]
        spans += [(off + s - len(head), off + e - len(head), cell_id((tid, i, j), hdr).split("::", 1)[1])
                  for s, e, (i, j) in sp]
        parts.append(text)
        off += len(text) + 2                                            # "\n\n"
    return "\n\n".join(parts), spans


def retrieve(a):
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from transformers import AutoTokenizer
    from rag_agent.retrieve.encoders import default_encoder

    enc = default_encoder(model_name=a.embed_model)

    class BGE(Embeddings):                                              # 본 방법과 같은 인코더·접두어
        def embed_documents(self, texts):
            return enc.encode(texts).tolist()

        def embed_query(self, text):
            return enc.encode_query([text])[0].tolist()

    qtok = AutoTokenizer.from_pretrained(READER)
    n_tok = lambda t: len(qtok(t, add_special_tokens=False)["input_ids"])
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK, chunk_overlap=OVERLAP, add_start_index=True)

    queries, docs, skipped = load_population(a.split)
    tables, hdr = build_tables(docs, "v3.3u", "none")
    live = {(tid, i, j) for tid, tab in tables.items() for i, row in enumerate(tab.table.data)
            for j, v in enumerate(row) if str(v).strip()}
    queries = [q for q in resolve_gold(queries, tables, hdr, live) if q["gold"] and not q["excluded"]]
    if a.split == "train":
        ref = {json.loads(l)["query_id"] for l in open(S3C, encoding="utf-8") if '"doc"' in l}
        assert {q["uid"] for q in queries} == ref, "질의 집합이 본 방법 레코드(2,885)와 다르다"
    else:                                   # dev 등: --ids 의 문항만(그 문항이 모두 있어야 한다)
        keep = {json.loads(l)["query_id"] for l in open(a.ids, encoding="utf-8") if l.strip()}
        queries = [q for q in queries if q["uid"] in keep]
        assert {q["uid"] for q in queries} == keep, "--ids 문항이 이 분할의 채점 가능 문항에 없다"
    want = [json.loads(l)["query_id"] for l in open(SUBSET, encoding="utf-8")][:a.limit] if a.limit else None
    if want:
        queries = sorted((q for q in queries if q["uid"] in set(want)), key=lambda q: q["uid"])
    print(f"[pop] {len(queries)} queries (제외 {dict(skipped)})", flush=True)

    out_dir = Path(a.out_dir)
    out = out_dir / "langchain_chunk_records.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f"{out} 있음 — 덮어쓰지 않는다")
    recs, t0, n_chunks = [], time.time(), 0
    with out.open("x", encoding="utf-8", newline="\n") as stream:
        for n, q in enumerate(queries, 1):
            text, spans = doc_text(q["uid"], tables, hdr)
            splits = splitter.split_documents([Document(page_content=text)])
            for d in splits:
                b, e = d.metadata["start_index"], d.metadata["start_index"] + len(d.page_content)
                assert text[b:e] == d.page_content
                d.metadata["cells"] = sorted(c for s, t, c in spans if b <= s and t <= e)
            store = InMemoryVectorStore(embedding=BGE())
            store.add_documents(splits)
            hits = store.similarity_search(q["question"], k=DUMP)      # 상위 4 = 튜토리얼 similarity_search(question)
            top = hits[:TOP_K]
            got = set().union(*(set(d.metadata["cells"]) for d in top))
            gold = sorted(cell_id(c, hdr).split("::", 1)[1] for c in q["gold"])
            n_chunks += len(splits)
            r = {"query_id": q["uid"], "kind": q["kind"], "m": len(q["gold"]), "layer": layer(q),
                 "question": q["question"], "gold_ids": gold, "n_chunks_in_doc": len(splits),
                 "ranked_units": [{"cells": d.metadata["cells"], "text": d.page_content,
                                   "n_tok": n_tok(d.page_content)} for d in hits],
                 "context": "\n\n".join(d.page_content for d in top),
                 "correct": int(set(gold) <= got), "any_DIAGNOSTIC": int(bool(set(gold) & got)),
                 "cells_in_context": len(got)}
            stream.write(json.dumps(r, ensure_ascii=False) + "\n")
            recs.append(r)
            if n % 200 == 0:
                print(f"  {n}/{len(queries)}  {time.time() - t0:.0f}s", flush=True)
    by = defaultdict(list)
    for r in recs:
        by[r["layer"]].append(r); by["ALL"].append(r)
    rate = lambda v: round(sum(v) / len(v), 4) if v else None
    summary = {"arm": "langchain_chunk", "split": a.split, "header_rule": "v3.3u", "label_rule": "none",
               "renderer": "serialization.chunks.markdown_source (fulltable 조건과 같음), 표 사이 빈 줄",
               "splitter": f"RecursiveCharacterTextSplitter(chunk_size={CHUNK}, chunk_overlap={OVERLAP}, add_start_index=True)",
               "vector_store": "langchain_core InMemoryVectorStore (cosine), similarity_search k=4",
               "encoder": enc.name, "query_prefix": enc.query_prefix, "passage_prefix": enc.passage_prefix,
               "cell_credit": "값이 통째로 청크 안에 있는 셀만 (serialization.chunks.table_chunks 규칙)",
               "n_queries": len(recs), "n_chunks_total": n_chunks, "limit": a.limit,
               "seconds": round(time.time() - t0, 1), "provenance": provenance(ROOT),
               "by_layer": {k: {"n": len(v), "accuracy_all": rate([x["correct"] for x in v]),
                                "accuracy_any_DIAGNOSTIC": rate([x["any_DIAGNOSTIC"] for x in v]),
                                "cells_delivered_mean": round(sum(x["cells_in_context"] for x in v) / len(v), 1),
                                "context_tokens_mean": round(sum(sum(u["n_tok"] for u in x["ranked_units"][:TOP_K])
                                                                 for x in v) / len(v), 1)}
                            for k, v in sorted(by.items())},
               "records_sha256": file_digest(out)}
    (out_dir / "langchain_chunk.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "provenance"}, ensure_ascii=False, indent=2))


def answer(a):
    import torch
    from rag_agent.llm.local_qwen import LocalQwenLLM

    class UserOnly(LocalQwenLLM):
        """튜토리얼의 프롬프트는 human 메시지 하나다 — system 메시지를 넣지 않는다."""
        def _prompt(self, system, user, thinking=False):
            return self.tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                                      add_generation_prompt=True, enable_thinking=False)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = {r["query_id"]: r for r in map(json.loads, open(a.records or out_dir / "langchain_chunk_records.jsonl", encoding="utf-8"))}
    want = [json.loads(l)["query_id"] for l in open(a.ids or SUBSET, encoding="utf-8") if l.strip()]
    want = want[:a.limit] if a.limit else want
    missing = [q for q in want if q not in recs]
    if missing:
        raise SystemExit(f"검색 레코드에 없는 질의 {len(missing)}건 (예: {missing[:3]}) — retrieve 먼저")
    answers = official_answers(a.split)
    rows_in = [recs[q] for q in want]
    check = self_check([answers[r["query_id"]] for r in rows_in], [r["kind"] == "arith" for r in rows_in])
    print(f"[scorer] gold 를 예측으로 넣었을 때 EM={check['em']} (n={check['n']})", flush=True)

    out = out_dir / "langchain_chunk_answer.jsonl"
    done = {}
    if out.exists():
        if not a.resume:
            raise SystemExit(f"{out} 있음 — 덮어쓰지 않는다(--resume)")
        done = {r["query_id"]: r for r in map(json.loads, open(out, encoding="utf-8"))}
        print(f"[resume] 저장된 {len(done)}건 건너뜀", flush=True)
    cot = a.prompt == "cot"                                        # 본 방법 지시문·틀(system=COT, user=Context/Question) — 지시문×단위 2×2
    api = a.reader.startswith("openai:")                           # API 리더(PREREG-2026-10-13-gpt4o.md): 한 문항씩, 토큰 수는 응답의 usage
    if api:
        if a.batch_size != 1:
            raise SystemExit("API 리더는 --batch-size 1 로만 돌린다")
        from rag_agent.llm.factory import _load_dotenv
        from rag_agent.llm.openai_llm import _DEFAULT_BASE, OpenAILLM
        _load_dotenv()
        name, kw = _split_spec(a.reader.removeprefix("openai:"))  # "gpt-4o?key_env=OPENAI_API_KEY_OLD" = .env 의 어느 키를 쓸지
        llm = OpenAILLM(model_name=name, base_url=_DEFAULT_BASE, api_key=os.environ[kw.get("key_env", "OPENAI_API_KEY")])
    elif a.reader.startswith("vllm:"):                             # 2026-10-23: 로컬 vLLM 서버(같은 프롬프트 렌더링, /v1/completions)
        from rag_agent.llm.vllm_completions import VLLMCompletionsLLM

        class UserOnlyV(VLLMCompletionsLLM):
            def _prompt(self, system, user, thinking=False):
                return self.tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                                          add_generation_prompt=True, enable_thinking=False)
        name, kw = _split_spec(a.reader.removeprefix("vllm:"))
        llm = (VLLMCompletionsLLM if cot else UserOnlyV)(model_name=name, **kw)
    else:
        name, kw = _split_spec(a.reader.removeprefix("local:"))  # "Qwen/Qwen3-1.7B?quantization=none" 도 받는다
        llm = (LocalQwenLLM if cot else UserOnly)(model_name=name, **{"quantization": "4bit", **kw})
        torch.manual_seed(SEED)
    system = COT if cot else ""
    closed = a.prompt == "closed"                                  # 표 없이 질문만
    INSTR = CLOSED_PROMPT if closed else RAG_PROMPT if a.prompt in ("original", "original_cf") else RAG_PROMPT + MARKER   # original = rlm/rag-prompt 원문, 표시 문장 없음
    TPL = PROMPT_CF if a.prompt == "original_cf" else PROMPT
    build = ((lambda r: "Context:\n" + r["context"] + f"\n\nQuestion: {r['question']}\nAnswer:") if cot else
             (lambda r: CLOSED_TEMPLATE.format(instr=INSTR, question=r["question"])) if closed else
             (lambda r: TPL.format(instr=INSTR, question=r["question"], context=r["context"])))
    limit = None if api else llm.context_limit
    rows, t0 = [done[r["query_id"]] for r in rows_in if r["query_id"] in done], time.time()
    pending = [r for r in rows_in if r["query_id"] not in done]
    with out.open("a", encoding="utf-8", newline="\n") as stream:
        bs = a.batch_size
        for s in range(0, len(pending), bs):
            batch, t = pending[s:s + bs], time.time()
            users = [build(r) for r in batch]
            if api:
                raws = [llm.complete(system, users[0], max_tokens=MAX_NEW)]
                n_toks = [llm.last_usage.get("prompt_tokens")]
            else:
                n_toks = [llm.n_prompt_tokens(system, u) for u in users]
                for n in n_toks:
                    check_context_limit(n, MAX_NEW, limit)
                raws = ([llm.complete(system, users[0], max_tokens=MAX_NEW, temperature=0.0)] if bs == 1 else
                        llm.complete_batch(system, users, max_tokens=MAX_NEW))      # 1 = 한 문항씩(긴 입력, 묶음 캐시 부족 때)
                if bs == 1:
                    torch.cuda.empty_cache()                               # 한 문항씩일 때 캐시 비우기(WSL 넘침 완화, Yu 882 전례)
            sec = round((time.time() - t) / len(batch), 3)
            for r, n, raw in zip(batch, n_toks, raws):
                gold, prog = answers[r["query_id"]], r["kind"] == "arith"
                pred, marked = extract(raw)
                row = {"query_id": r["query_id"], "layer": r["layer"], "kind": r["kind"], "m": r["m"],
                       "question": r["question"], "retrieval_correct": r["correct"],
                       "pred": pred, "answer": gold, "answer_correct": int(mh_exact_match(pred, gold, prog)),
                       "f1": mh_f1(pred, gold, prog), "raw": raw, "marker_found": marked,
                       "n_tok": n, "n_llm_calls": 1, "cells_in_context": r["cells_in_context"],
                       "context_sha256": digest(r["context"]), "seconds": sec,
                       **({"finish_reason": llm.last_finish_reason,
                           "n_tok_out": llm.last_usage.get("completion_tokens")} if api else {})}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
            stream.flush(); os.fsync(stream.fileno())
            print(f"  {len(rows)}/{len(rows_in)}  {time.time() - t0:.0f}s", flush=True)
    by = defaultdict(list)
    for r in rows:
        by[r["layer"]].append(r); by["ALL"].append(r)
    rate = lambda v: round(sum(v) / len(v), 4) if v else None
    summary = {"arm": "langchain_chunk", "reader": llm.name, "reader_details": {k: v for k, v in llm.metadata().items()
                                                                            if k != "chat_template"},
               "prompt": a.prompt, "system_message": COT if cot else None,
               "prompt_text": (COT + "\n---\nContext:\n{context}\n\nQuestion: {question}\nAnswer:" if cot else
                               CLOSED_TEMPLATE.format(instr=INSTR, question="{question}") if closed else
                               TPL.format(instr=INSTR, question="{question}", context="{context}")),
               "prompt_sha256": digest(COT if cot else INSTR),
               "max_new_tokens": MAX_NEW, "seed": None if api else SEED, "batch_size": a.batch_size, "temperature": 0.0,
               "scorer": "multihiertt_em.mh_exact_match + multihiertt_f1.mh_f1 (공식 포팅)",
               "answer_source": f"{OFFICIAL_REPO}@{OFFICIAL_REV}", "scorer_self_check": check,
               "limit": a.limit, "n": len(rows), "marker_missing": sum(not x["marker_found"] for x in rows),
               "n_over_context_limit": 0, "generation_seconds": round(sum(x["seconds"] for x in rows), 1),
               "provenance": provenance(ROOT),
               "by_layer": {k: {"n": len(v), "retrieval_accuracy_here": rate([x["retrieval_correct"] for x in v]),
                                "answer_em": rate([x["answer_correct"] for x in v]),
                                "answer_f1": rate([x["f1"] for x in v]),
                                "input_tokens_mean": round(sum(x["n_tok"] for x in v) / len(v), 1)}
                            for k, v in sorted(by.items())},
               "records_sha256": file_digest(out)}
    (out_dir / "langchain_chunk_answer.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("provenance", "reader_details")},
                     ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["retrieve", "answer"])
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = 검색 2,885 전부·답변 882 부분집합. N = 882 부분집합(query_id 순)의 앞 N 건만 두 단계 모두")
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--records", default="", help="답변 단계가 읽을 검색 레코드(기본 <out-dir>/langchain_chunk_records.jsonl). "
                                                "리더만 바꾼 재실행은 기존 레코드를 가리킨다")
    ap.add_argument("--reader", default="local:Qwen/Qwen3-8B?quantization=4bit", help="답변 LLM 사양(local:<hf id>?quantization=4bit|none)")
    ap.add_argument("--embed-model", default=EMBED)
    ap.add_argument("--resume", action="store_true", help="답변 단계: 저장된 행은 건너뛰고 이어 쓴다")
    ap.add_argument("--split", default="train", help="검색 단계 분할(기본 train). train 이 아니면 --ids 의 문항만 검색")
    ap.add_argument("--batch-size", type=int, default=BATCH, help="답변 묶음 크기(기본 128). 1 = 한 문항씩 complete")
    ap.add_argument("--ids", default="", help="답변 단계 문항 목록·순서(jsonl, query_id). 기본 882 부분집합 순서")
    ap.add_argument("--prompt", default="tutorial", choices=["tutorial", "cot", "original", "closed", "original_cf"],
                    help="답변 단계 지시문. tutorial = rlm/rag-prompt(원방법). cot = 본 방법 지시문·틀(answer_accuracy_mh.COT) — "
                         "PREREG-2026-10-07-claims.md 1(지시문×단위 2×2). original = rlm/rag-prompt 원문 그대로(Final answer 표시 문장 없음, "
                         "PREREG-2026-10-12-cellform.md). closed = 표 없이 질문만(원문에서 문맥 문장만 뺌, PREREG-2026-10-13-gpt4o.md). "
                         "original_cf = original 과 같은 문장, Context 를 Question 앞에(PREREG-2026-10-15-prompt-order.md)")
    a = ap.parse_args()
    (retrieve if a.stage == "retrieve" else answer)(a)


if __name__ == "__main__":
    main()
