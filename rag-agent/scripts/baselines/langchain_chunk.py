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
from answer_accuracy_mh import extract                                  # noqa: E402
from mh_arms import (NO_TITLE, OFFICIAL_REPO, OFFICIAL_REV, build_tables,  # noqa: E402
                     cell_id, layer, load_population, official_answers, resolve_gold)
from rag_agent.eval.artifacts import digest, file_digest, provenance   # noqa: E402
from rag_agent.eval.multihiertt_em import mh_exact_match, self_check   # noqa: E402
from rag_agent.eval.multihiertt_f1 import mh_f1                        # noqa: E402
from rag_agent.serialization.chunks import markdown_source             # noqa: E402

# 튜토리얼 rlm/rag-prompt 원문. 마지막 문장(Final answer 표시)만 이 저장소의 채점 규약을 위해 덧붙였다.
RAG_PROMPT = ("You are an assistant for question-answering tasks. Use the following pieces of retrieved "
              "context to answer the question. If you don't know the answer, just say that you don't know. "
              "Use three sentences maximum and keep the answer concise.")
MARKER = " On the last line write 'Final answer: ' followed by only the final value, with no units or explanation."
PROMPT = "{instr}\nQuestion: {question} \nContext: {context} \nAnswer:"
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

    enc = default_encoder(model_name=EMBED)

    class BGE(Embeddings):                                              # 본 방법과 같은 인코더·접두어
        def embed_documents(self, texts):
            return enc.encode(texts).tolist()

        def embed_query(self, text):
            return enc.encode_query([text])[0].tolist()

    qtok = AutoTokenizer.from_pretrained(READER)
    n_tok = lambda t: len(qtok(t, add_special_tokens=False)["input_ids"])
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK, chunk_overlap=OVERLAP, add_start_index=True)

    queries, docs, skipped = load_population("train")
    tables, hdr = build_tables(docs, "v3.3u", "none")
    live = {(tid, i, j) for tid, tab in tables.items() for i, row in enumerate(tab.table.data)
            for j, v in enumerate(row) if str(v).strip()}
    queries = [q for q in resolve_gold(queries, tables, hdr, live) if q["gold"] and not q["excluded"]]
    ref = {json.loads(l)["query_id"] for l in open(S3C, encoding="utf-8") if '"doc"' in l}
    assert {q["uid"] for q in queries} == ref, "질의 집합이 본 방법 레코드(2,885)와 다르다"
    want = [json.loads(l)["query_id"] for l in open(SUBSET, encoding="utf-8")][:a.limit] if a.limit else None
    if want:
        queries = sorted((q for q in queries if q["uid"] in set(want)), key=lambda q: q["uid"])
    print(f"[pop] {len(queries)} queries (제외 {dict(skipped)})", flush=True)

    out = OUT / "langchain_chunk_records.jsonl"
    OUT.mkdir(parents=True, exist_ok=True)
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
    summary = {"arm": "langchain_chunk", "split": "train", "header_rule": "v3.3u", "label_rule": "none",
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
    (OUT / "langchain_chunk.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "provenance"}, ensure_ascii=False, indent=2))


def answer(a):
    import torch
    from rag_agent.llm.local_qwen import LocalQwenLLM

    class UserOnly(LocalQwenLLM):
        """튜토리얼의 프롬프트는 human 메시지 하나다 — system 메시지를 넣지 않는다."""
        def _prompt(self, system, user, thinking=False):
            return self.tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                                      add_generation_prompt=True, enable_thinking=False)

    recs = {r["query_id"]: r for r in map(json.loads, open(OUT / "langchain_chunk_records.jsonl", encoding="utf-8"))}
    want = [json.loads(l)["query_id"] for l in open(SUBSET, encoding="utf-8")]
    want = want[:a.limit] if a.limit else want
    missing = [q for q in want if q not in recs]
    if missing:
        raise SystemExit(f"검색 레코드에 없는 질의 {len(missing)}건 (예: {missing[:3]}) — retrieve 먼저")
    answers = official_answers("train")
    rows_in = [recs[q] for q in want]
    check = self_check([answers[r["query_id"]] for r in rows_in], [r["kind"] == "arith" for r in rows_in])
    print(f"[scorer] gold 를 예측으로 넣었을 때 EM={check['em']} (n={check['n']})", flush=True)

    out = OUT / "langchain_chunk_answer.jsonl"
    if out.exists():
        raise SystemExit(f"{out} 있음 — 덮어쓰지 않는다")
    llm = UserOnly(model_name=READER, quantization="4bit")
    torch.manual_seed(SEED)
    limit = llm.context_limit
    rows, t0 = [], time.time()
    with out.open("x", encoding="utf-8", newline="\n") as stream:
        for s in range(0, len(rows_in), BATCH):
            batch, t = rows_in[s:s + BATCH], time.time()
            users = [PROMPT.format(instr=RAG_PROMPT + MARKER, question=r["question"], context=r["context"]) for r in batch]
            n_toks = [llm.n_prompt_tokens("", u) for u in users]
            for n in n_toks:
                check_context_limit(n, MAX_NEW, limit)
            raws = llm.complete_batch("", users, max_tokens=MAX_NEW)
            sec = round((time.time() - t) / len(batch), 3)
            for r, n, raw in zip(batch, n_toks, raws):
                gold, prog = answers[r["query_id"]], r["kind"] == "arith"
                pred, marked = extract(raw)
                row = {"query_id": r["query_id"], "layer": r["layer"], "kind": r["kind"], "m": r["m"],
                       "question": r["question"], "retrieval_correct": r["correct"],
                       "pred": pred, "answer": gold, "answer_correct": int(mh_exact_match(pred, gold, prog)),
                       "f1": mh_f1(pred, gold, prog), "raw": raw, "marker_found": marked,
                       "n_tok": n, "n_llm_calls": 1, "cells_in_context": r["cells_in_context"],
                       "context_sha256": digest(r["context"]), "seconds": sec}
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
               "prompt_text": PROMPT.format(instr=RAG_PROMPT + MARKER, question="{question}", context="{context}"),
               "prompt_sha256": digest(RAG_PROMPT + MARKER), "system_message": None,
               "max_new_tokens": MAX_NEW, "seed": SEED, "batch_size": BATCH, "temperature": 0.0,
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
    (OUT / "langchain_chunk_answer.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("provenance", "reader_details")},
                     ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["retrieve", "answer"])
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = 검색 2,885 전부·답변 882 부분집합. N = 882 부분집합(query_id 순)의 앞 N 건만 두 단계 모두")
    a = ap.parse_args()
    (retrieve if a.stage == "retrieve" else answer)(a)


if __name__ == "__main__":
    main()
