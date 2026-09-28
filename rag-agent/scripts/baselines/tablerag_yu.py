#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""TableRAG (Yu, Jian & Chen, EMNLP 2025) 원 방법 비교군 `tablerag_yu` — 공식 코드의 online 파이프라인 전체.

공식 코드: https://github.com/yxh-y/TableRAG  (vendor/tablerag_yu, 커밋 f8d798fc60fb44736d7285ca3ca1dbcb7b3e984d)
  offline: excel_to_markdown → RecursiveCharacterTextSplitter(1000, 200) → "File name: {key}\\n" 접두
           (online_inference/tools/retriever.py), 표 → DataFrame → DB + {table_name, column_list} 스키마
           (offline_data_ingestion_and_query_interface/src/data_persistent.py)
  online : SYSTEM_EXPLORE_PROMPT + solve_subquery 도구로 질의 분해 → 하위 질의마다 청크 회수 30·재정렬 5·상위 3
           + NL2SQL(NL2SQL_SYSTEM_PROMPT/NL2SQL_USER_PROMPT) → SQL 실행 → COMBINE_PROMPT 로 중간 답
           → 최대 5회 반복, "<Answer>" 로 최종 답 (online_inference/main.py: TableRAG._run)

같은 조건 치환·불가피한 적응은 README.md 에 논문 값과 나란히 적는다. 기존 스크립트·결과는 건드리지 않는다.

  PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 flock .gpu.lock \\
      .venv/bin/python scripts/baselines/tablerag_yu.py --stage retrieve            # 2,885 검색 → records
  ... --stage answer [--limit 3]                                                  # 882 전체 파이프라인 → answer
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                                                       # noqa: E402

from answer_accuracy_mh import COT, extract                              # noqa: E402
from mh_arms import build_tables, load_population, official_answers, resolve_gold  # noqa: E402
from rag_agent.eval.artifacts import digest                              # noqa: E402
from rag_agent.eval.multihiertt_em import mh_exact_match                 # noqa: E402
from rag_agent.eval.multihiertt_f1 import mh_f1                          # noqa: E402
from rag_agent.llm.factory import build_llm                              # noqa: E402
from rag_agent.serialization.base import join_path                       # noqa: E402
from rag_agent.serialization.chunks import markdown_source, table_chunks  # noqa: E402
from rag_agent.serialization.tablerag_unit import official_frame         # noqa: E402

VENDOR = ROOT / "vendor" / "tablerag_yu"
VENDOR_COMMIT = "f8d798fc60fb44736d7285ca3ca1dbcb7b3e984d"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, VENDOR / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 공식 프롬프트·도구·이름 규칙을 vendored 파일에서 그대로 읽는다 (복사하지 않는다)
_online_prompt = _load("trag_yu_online_prompt", "online_inference/prompt.py")
_offline_prompt = _load("trag_yu_offline_prompt", "offline_data_ingestion_and_query_interface/src/prompt.py")
_tool_utils = _load("trag_yu_tool_utils", "online_inference/utils/tool_utils.py")
SYSTEM_EXPLORE_PROMPT, COMBINE_PROMPT = _online_prompt.SYSTEM_EXPLORE_PROMPT, _online_prompt.COMBINE_PROMPT
NL2SQL_SYSTEM_PROMPT = _offline_prompt.NL2SQL_SYSTEM_PROMPT
# 적응 (i) MySQL → SQLite: 프롬프트의 방언 이름만 바꾼다 (README "패치 1")
NL2SQL_USER_PROMPT = _offline_prompt.NL2SQL_USER_PROMPT.replace("MySQL syntax", "SQLite syntax")
assert NL2SQL_USER_PROMPT != _offline_prompt.NL2SQL_USER_PROMPT
Embedder, Reranker = _tool_utils.Embedder, _tool_utils.Reranker
# 같은 추출 규칙(모든 비교군 공통): 최종 <Answer> 를 내는 프롬프트 끝에 answer_accuracy_mh 의 마커 문장을 붙이고
# 같은 `extract` 로 'Final answer:' 뒤를 읽는다 (README 이탈 "답 추출"). 공식 문구는 그대로, 한 문장만 뒤에 추가.
FINAL_MARKER = COT[COT.index("On the last line write 'Final answer: '"):]
EXPLORE_PROMPT = SYSTEM_EXPLORE_PROMPT.rstrip("\n") + "\n" + FINAL_MARKER + "\n"

MAX_ITER = 5                    # main.py MAX_ITER
RECALL_NUM, RERANK_NUM, N_DOCS = 30, 5, 3   # main.py: retrieve(sub_query, 30, 5) → unique[:3]
SQL_RESULT_CHARS = 1000         # sql_alchemy_helper.fetchall: json 문자열 1000자 절단
TOOLS = [{"type": "function", "function": {           # main.py create_tools 그대로
    "name": "solve_subquery", "description": "Return answer for the decomposed subquery.",
    "parameters": {"type": "object", "properties": {"subquery": {
        "type": "string", "description": "The subquery to be solved, only take natural language as input."}},
        "required": ["subquery"], "additionalProperties": False}, "strict": True}}]
_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_SQL_BLOCK = re.compile(r"```sql([\s\S]*?)```", re.DOTALL)   # service.py extract_sql_statement


# ---------------------------------------------------------------- offline: transfer_name / schema (data_persistent.py)
def transfer_name(original_name: str) -> str:
    """common_utils.transfer_name 그대로 (빈 이름은 원본이 name[0] 에서 죽는다 — 호출 쪽에서 막는다)."""
    import hashlib
    name = original_name.split('.')[0]
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    name = re.sub(r'_+', '_', name)
    if len(name) > 2:
        name = name.strip('_')
    name = name.lower()
    if name[0].isdigit():
        name = 't_' + name
    if len(name) > 64:
        prefix = name[:20].rstrip('_')
        name = f"{prefix}_{hashlib.md5(name.encode('utf-8')).hexdigest()[:8]}"
    return name


def pandas_to_mysql_dtype(dtype) -> str:
    """data_persistent.pandas_to_mysql_dtype (dtype_mapping.py 값) — 스키마에 보이는 형 이름."""
    import pandas as pd
    if pd.api.types.is_integer_dtype(dtype):
        return {"int8": "TINYINT", "int16": "SMALLINT", "int32": "INT", "int64": "BIGINT",
                "Int64": "BIGINT", "UInt64": "BIGINT UNSIGNED"}.get(str(dtype), "INT")
    if pd.api.types.is_float_dtype(dtype):
        return {"float16": "FLOAT", "float32": "FLOAT", "float64": "DOUBLE"}.get(str(dtype), "FLOAT")
    if pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "DATETIME"
    if pd.api.types.is_timedelta64_dtype(dtype):
        return "TIME"
    if pd.api.types.is_string_dtype(dtype):
        return "VARCHAR(255)"
    return "TEXT"


def sql_frame(t, sql_name: str, seed: int):
    """적응 (ii): 평면 프레임 = official_frame(table, mode="path") (열 이름 = 결합 머리글 경로)
    + 첫 열 `row_header` = 결합 행 머리글 경로 (원본 표는 첫 열이 행 이름 데이터 열인 평면 표라는 가정).
    열 이름은 transfer_df_columns 규칙(transfer_name, 중복 _1/_2)으로 정리하고 원 경로 매핑을 남긴다.
    스키마는 generate_schema_info 형식 [열, 형, 'sample values:[...]'] + 4번째 'original header: 경로'."""
    import pandas as pd
    df = official_frame(t, mode="path")
    original = ["row header path", *df.columns]
    df.insert(0, "row_header", [join_path(t.row_path(i)) for i in range(t.n_rows)])
    seen, cols = {}, []
    for c in df.columns:
        name = transfer_name(str(c)) if re.sub(r'[^a-zA-Z0-9_]', '', str(c)) else f"col{len(cols)}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        cols.append(name)
    df.columns = cols
    rng = random.Random(seed)
    column_list = []
    for c, orig in zip(cols, original):
        valid = [str(x) for x in df[c].dropna().unique() if pd.notnull(x) and len(str(x)) < 64]
        sample = rng.sample(valid, min(3, len(valid))) or ['no sample values available']
        column_list.append([c, pandas_to_mysql_dtype(df[c].dtype), 'sample values:' + str(sample),
                            f"original header: {orig}"])
    return df, {"table_name": sql_name, "column_list": column_list}


def run_sql(conn, sql: str) -> str:
    """sql_alchemy_helper.fetchall: 행을 dict 로 json 화, 1000자 절단; 실패는 service.py 문구."""
    try:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        s = json.dumps(rows, ensure_ascii=False, default=str)
        return s[:SQL_RESULT_CHARS]
    except Exception as e:                                   # noqa: BLE001
        return f"SQL execution failed: {e}"


def extract_sql_statement(resp: str) -> str:
    m = _SQL_BLOCK.search(resp or "")
    return re.sub(r'\s+', ' ', m.group(1).strip()) if m else (resp or "")


# ---------------------------------------------------------------- 문서 하나의 표·청크·DB
class Doc:
    def __init__(self, uid: str, tables: dict, hdr: dict, seed: int):
        self.uid = uid
        self.tids = sorted((t for t in tables if t.startswith(uid + "::")), key=lambda t: int(t.split("::")[1]))
        self.chunks, self.chunk_cells, self.chunk_tid = [], [], []
        self.markdown, self.frames, self.schemas = {}, {}, {}
        for tid in self.tids:
            tab, (nhr, nhc) = tables[tid], hdr[tid]
            k = int(tid.split("::")[1])
            # 청크 = 기존 trag_hetero arm 과 같은 렌더러·분할기·"File name:" 접두 (제목 없음 → 빈 이름)
            for txt, cells in table_chunks(tab, tab.table, tab.title, 1000, 200):
                self.chunks.append(txt)
                self.chunk_cells.append(sorted(f"{k}-{i + nhr}-{j + nhc}" for i, j in cells))
                self.chunk_tid.append(tid)
            self.markdown[tid] = markdown_source(tab, tab.table, tab.title)[0].removeprefix("Table name: \n")
            self.frames[tid], self.schemas[tid] = sql_frame(tab.table, f"table_{k + 1}", seed)

    def db(self):
        conn = sqlite3.connect(":memory:")
        for tid, df in self.frames.items():
            df.to_sql(self.schemas[tid]["table_name"], conn, index=False, if_exists="replace")
        return conn


# ---------------------------------------------------------------- 검색 (SemanticRetriever.recall / rerank)
class Retriever:
    def __init__(self, embed_model: str, reranker_model: str, reranker_device: str, cache: Path):
        self.embedder = Embedder(embed_model, device_id=0)          # CLS, 정규화 없음 (tool_utils.Embedder)
        self.reranker = Reranker(reranker_model, use_fp16=False, device=reranker_device)   # 공식 기본 fp32
        self.cache = cache
        self.emb = {}

    def index(self, docs: dict):
        texts = [c for d in docs.values() for c in d.chunks]
        f = self.cache / f"emb_{len(texts)}_{digest(texts)[:16]}.npy"
        if f.exists():
            mat = np.load(f)
        else:
            t = time.time()
            mat = np.concatenate([self.embedder.encode(texts[s:s + 128]) for s in range(0, len(texts), 128)])
            np.save(f, mat)
            print(f"[embed] {len(texts)} chunks in {time.time() - t:.0f}s -> {f}", flush=True)
        pos = 0
        for uid, d in docs.items():
            self.emb[uid] = mat[pos:pos + len(d.chunks)]
            pos += len(d.chunks)
        assert pos == len(mat)

    def retrieve(self, doc: Doc, query: str):
        """검색 범위 = 질문의 문서(같은 조건). recall: 내적 상위 30, rerank: 재정렬 점수 상위 5.
        반환 (재정렬 순 인덱스, 회수 순 인덱스, 재정렬 점수)."""
        q = self.embedder.encode([query])[0]
        scores = self.emb[doc.uid] @ q
        recall = list(np.argsort(-scores, kind="stable")[:RECALL_NUM])
        rr = self.reranker.compute_score([[query, doc.chunks[i]] for i in recall])
        ranked = [i for _s, i in sorted(zip(rr, recall), key=lambda x: -x[0])][:RERANK_NUM]
        return ranked, recall, dict(zip(map(int, recall), map(float, rr)))


# ---------------------------------------------------------------- LLM (Qwen3-8B 4bit, 비생각, greedy)
class Chat:
    """공식 코드의 OpenAI tool-calling 호출을 같은 리더로: Qwen3 채팅 템플릿의 tools 지원(<tool_call>)을 쓴다."""

    def __init__(self, llm, max_tokens: int):
        self.llm, self.tok, self.max_tokens = llm, llm.tokenizer, max_tokens
        self.n_calls = self.n_prompt_tokens = 0

    def __call__(self, messages, tools=None) -> str:
        import torch
        prompt = self.tok.apply_chat_template(messages, tools=tools, tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)
        inputs = self.tok(prompt, return_tensors="pt").to(self.llm.device)
        self.n_calls += 1
        self.n_prompt_tokens += int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            out = self.llm.model.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=False,
                                          pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def parse_tool_calls(text: str):
    """main.py extract_subquery 대응: (reasoning, [subquery, ...] | None)."""
    subs = []
    for m in _TOOL_CALL.finditer(text):
        try:
            call = json.loads(m.group(1))
            if call.get("name") == "solve_subquery" and isinstance(call.get("arguments"), dict):
                subs.append(str(call["arguments"]["subquery"]))
        except (ValueError, KeyError):
            continue
    return _TOOL_CALL.sub("", text).strip(), (subs or None)


def is_final(text: str) -> bool:
    """공식: '<Answer>' 가 있으면 최종 답. 마커 문장을 붙였으므로 'Final answer:' 만 쓴 출력도 최종 답으로 받는다."""
    return "<Answer>" in text or bool(extract(text)[1])


def extract_answer(text: str):
    """main.py extract_answer: '<Answer>' 뒤 전부(없으면 출력 전체) → answer_accuracy_mh.extract 로 'Final answer:' 뒤를
    읽는다(같은 추출 규칙). 마커가 없으면 <Answer> 내용 전체에서 앞의 ':'·감싼 [ ] 를 벗기고 끝의 '.' 을 뗀다.
    반환 (원문, 채점용 답, 마커 있었나)."""
    raw = (text[text.index("<Answer>") + len("<Answer>"):] if "<Answer>" in text else text).strip()
    pred, marked = extract(raw)
    if not marked:
        pred = raw.lstrip(":").strip()
        if pred.startswith("[") and pred.endswith("]"):
            pred = pred[1:-1].strip()
        pred = pred.rstrip(".").strip()
    return raw, pred, marked


# ---------------------------------------------------------------- online: TableRAG._run 이식
def run_case(question: str, doc: Doc, ret: Retriever, chat: Chat, n_tok):
    trace = {"init_retrieval": None, "rounds": [], "sql": []}
    # main.py: 첫 검색(질문)으로 상위 1 청크의 표를 고른다 (HybridQA 의 "The given table is in {table_id}" 접미는
    # 표 id 가 주어지는 설정이라 붙이지 않는다 — README). 그 표 markdown 전체가 초기 프롬프트에 들어간다.
    ranked, recall, rr = ret.retrieve(doc, question)
    top1_tid = doc.chunk_tid[ranked[0]]
    trace["init_retrieval"] = {"top1_table": doc.schemas[top1_tid]["table_name"],
                               "reranked": [int(i) for i in ranked], "recall": [int(i) for i in recall]}
    text_messages = [{"role": "user", "content": EXPLORE_PROMPT.format(
        query=question, table_content=doc.markdown[top1_tid])}]
    conn = doc.db()
    round1 = None                     # 1회차 하위 질의의 검색 결과 (records 의 ranked_units·context)
    current_iter = MAX_ITER
    answer = None
    while current_iter:
        current_iter -= 1
        response = chat(text_messages, TOOLS)
        reasoning, sub_queries = parse_tool_calls(response)
        step = {"step": MAX_ITER - current_iter, "response": response, "sub_queries": sub_queries}
        trace["rounds"].append(step)
        # 원본: 1회차에는 답을 받지 않는다 (current_iter != max_iter - 1)
        if not sub_queries and is_final(reasoning) and current_iter != MAX_ITER - 1:
            step["answer_raw"], answer, step["marker_found"] = extract_answer(reasoning)
            break
        if not sub_queries:
            text_messages.append({"role": "user", "content": "ERROR: Did not call tool with a suquery!"})
            continue
        text_messages.append({"role": "assistant", "content": reasoning, "tool_calls": [
            {"type": "function", "function": {"name": "solve_subquery", "arguments": {"subquery": s}}}
            for s in sub_queries]})
        for sub_query in sub_queries:
            ranked, recall, rr = ret.retrieve(doc, sub_query)
            # 원본 list(set(reranked_docs))[:3] 은 순서가 비결정적 — 재정렬 순서를 지키며 중복만 뺀다
            uniq = list(dict.fromkeys(ranked))[:N_DOCS]
            doc_content = "\n".join(doc.chunks[i] for i in uniq)
            # service.process_tablerag_request: 스키마 = 상위 1 표 하나 (related_table_name_list = [top1])
            schema = json.dumps([doc.schemas[top1_tid]], ensure_ascii=False)
            nl2sql_prompt = NL2SQL_USER_PROMPT.format(schema_list=schema, user_query=sub_query)
            nl2sql_resp = chat([{"role": "system", "content": NL2SQL_SYSTEM_PROMPT},
                                {"role": "user", "content": nl2sql_prompt}])
            sql_str = extract_sql_statement(nl2sql_resp)
            sql_result = run_sql(conn, sql_str)
            trace["sql"].append({"sub_query": sub_query, "nl2sql_response": nl2sql_resp,
                                 "sql": sql_str, "result": sql_result})
            # main.py 의 schema 는 nl2sql_prompt 에서 'Based on the schemas above' 앞부분 (= 스키마 JSON)
            combine = COMBINE_PROMPT.format(docs=doc_content, schema=schema, nl2sql_model_response=sql_str,
                                            sql_execute_result=sql_result, query=sub_query)
            sub_answer = chat([{"role": "user", "content": combine}]) or ""
            text_messages.append({"role": "tool", "content": "Subquery Answer: " + sub_answer})
            step.setdefault("tool_runs", []).append({
                "sub_query": sub_query, "reranked_top3": [int(i) for i in uniq],
                "recall": [int(i) for i in recall], "sub_answer": sub_answer})
            if round1 is None:
                rest = [i for i in recall if i not in uniq]
                round1 = {"sub_query": sub_query, "context": doc_content,
                          "ranked_units": [{"cells": doc.chunk_cells[i], "text": doc.chunks[i],
                                            "n_tok": n_tok(doc.chunks[i]), "rerank_score": rr[int(i)]}
                                           for i in [*uniq, *rest]]}
    conn.close()
    if round1 is None:                # 도구를 한 번도 안 불렀다 — 첫 검색(질문) 목록을 남긴다
        round1 = {"sub_query": None, **retrieve_only(doc, question, ret, n_tok)}
    trace["messages"] = text_messages
    return answer, round1, trace


def retrieve_only(doc: Doc, query: str, ret: Retriever, n_tok):
    """1회차 텍스트 검색만(LLM·SQL 없음): 재정렬 상위 3 먼저, 이어 나머지 회수 청크(회수 순), 최대 30.
    `context` = 상위 3 청크를 "\\n" 으로 이은, LLM 에 들어갈 원문."""
    ranked, recall, rr = ret.retrieve(doc, query)
    uniq = list(dict.fromkeys(ranked))[:N_DOCS]
    order = [*uniq, *[i for i in recall if i not in uniq]]
    return {"context": "\n".join(doc.chunks[i] for i in uniq),
            "ranked_units": [{"cells": doc.chunk_cells[i], "text": doc.chunks[i], "n_tok": n_tok(doc.chunks[i]),
                              "rerank_score": rr[int(i)]} for i in order]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", default="results/mh_only_20260928/retrieval/mh_train_s3c_records.jsonl")
    ap.add_argument("--answer-ids", default="results/mh_arms/cap300_20260924/cell_uniq.jsonl")
    ap.add_argument("--out-dir", default="results/mh_only_20260928/baselines/tablerag_yu")
    ap.add_argument("--reader", default="local:Qwen/Qwen3-8B?quantization=4bit")
    ap.add_argument("--embed-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--stage", required=True, choices=["retrieve", "answer"],
                    help="retrieve: 2,885 문항 1회차 텍스트 검색만(질문으로 회수 30 → 재정렬; LLM·SQL 없음) → records. "
                         "answer: 882 답변 집합 전체 파이프라인 → answer")
    ap.add_argument("--reranker-device", default=None,
                    help="기본 retrieve=cuda:0, answer=cpu (fp32). answer 에서 리더와 GPU 를 같이 쓰면 메모리 7.9/8 GB 로 "
                         "생성이 3배 느렸다(README 점검)")
    ap.add_argument("--max-tokens", type=int, default=512, help="모든 LLM 호출의 생성 상한")
    ap.add_argument("--limit", type=int, default=0, help="앞 N 문항만 (query_id 정렬, retrieve 는 882 답변 집합이 먼저)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-dir", default=".cache/tablerag_yu")
    a = ap.parse_args()
    a.reranker_device = a.reranker_device or ("cuda:0" if a.stage == "retrieve" else "cpu")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / ("tablerag_yu_records.jsonl" if a.stage == "retrieve" else "tablerag_yu_answer.jsonl")
    if out_path.exists() and not a.resume:
        raise SystemExit(f"{out_path} 가 있다 — 이어 쓰려면 --resume")

    recs = [json.loads(l) for l in open(a.records, encoding="utf-8")]
    recs = {r["query_id"]: r for r in recs if "doc" in r}
    answer_ids = sorted(json.loads(l)["query_id"] for l in open(a.answer_ids, encoding="utf-8"))
    assert set(answer_ids) <= set(recs), "답변 집합이 검색 레코드 밖"
    order = answer_ids + (sorted(set(recs) - set(answer_ids)) if a.stage == "retrieve" else [])
    if a.limit:
        order = order[:a.limit]
    done = set()
    if a.resume and out_path.exists():
        done = {json.loads(l)["query_id"] for l in open(out_path, encoding="utf-8")}
        print(f"[resume] {len(done)} 저장됨", flush=True)
    todo = [q for q in order if q not in done]
    print(f"[plan] stage={a.stage} {len(order)} 문항, 남은 {len(todo)}", flush=True)

    # 표: 본 방법과 같은 build_tables(v3.3u, unique_doc), 표만
    _queries, docs_raw, _ = load_population("train")
    tables, hdr = build_tables({u: docs_raw[u] for u in todo}, "v3.3u", "none")
    # gold 좌표 규약 검증: 레코드의 gold_ids ("표-행-열", 펼친 격자 좌표) 를 resolve_gold 로 다시 풀어 대조
    live = {(tid, i, j) for tid, tab in tables.items() for i, row in enumerate(tab.table.data)
            for j, v in enumerate(row) if str(v).strip()}
    qs = resolve_gold([q for q in _queries if q["uid"] in set(todo)], tables, hdr, live)
    for q in qs:
        got = sorted(f"{tid.split('::')[1]}-{i + hdr[tid][0]}-{j + hdr[tid][1]}" for tid, i, j in q["gold"])
        assert got == recs[q["uid"]]["gold_ids"], (q["uid"], got, recs[q["uid"]]["gold_ids"])
    answers = official_answers("train")
    docs = {u: Doc(u, tables, hdr, a.seed) for u in todo}
    print(f"[corpus] {len(docs)} docs / {sum(len(d.chunks) for d in docs.values())} chunks", flush=True)

    cache = Path(a.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    ret = Retriever(a.embed_model, a.reranker, a.reranker_device, cache)
    ret.index(docs)
    from transformers import AutoTokenizer
    qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    n_tok = lambda t: len(qwen_tok(t, add_special_tokens=False)["input_ids"])

    if a.stage == "retrieve":
        t_all = time.time()
        with out_path.open("a", encoding="utf-8") as f:
            for n, qid in enumerate(todo, 1):
                r, t0 = recs[qid], time.time()
                got = retrieve_only(docs[qid], r["question"], ret, n_tok)
                f.write(json.dumps({"query_id": qid, "kind": r["kind"], "m": r["m"], "layer": r["layer"],
                                    "question": r["question"], "gold_ids": r["gold_ids"], "stage": "retrieve",
                                    **got, "seconds": round(time.time() - t0, 3)}, ensure_ascii=False) + "\n")
                if n % 100 == 0 or n == len(todo):
                    print(f"  {n}/{len(todo)} {time.time() - t_all:.0f}s", flush=True)
        print(f"[retrieve] {len(todo)} 문항 {time.time() - t_all:.1f}s (문항당 {(time.time() - t_all) / max(len(todo), 1):.2f}s)", flush=True)
        return 0

    llm = build_llm(a.reader)
    import torch
    torch.manual_seed(a.seed)
    print(f"[gpu] allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB, reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB", flush=True)
    chat = Chat(llm, a.max_tokens)
    with out_path.open("a", encoding="utf-8") as fans:
        for n, qid in enumerate(todo, 1):
            r, t0 = recs[qid], time.time()
            calls0, toks0 = chat.n_calls, chat.n_prompt_tokens
            answer, round1, trace = run_case(r["question"], docs[qid], ret, chat, n_tok)
            seconds = round(time.time() - t0, 3)
            trace["round1"] = round1
            gold, program = answers[qid], r["kind"] == "arith"
            n_err = sum(s["result"].startswith("SQL execution failed") for s in trace["sql"])
            fans.write(json.dumps({
                "query_id": qid, "kind": r["kind"], "m": r["m"], "layer": r["layer"], "question": r["question"],
                "gold_ids": r["gold_ids"], "pred": answer, "answer": gold,
                "answer_correct": int(mh_exact_match(answer, gold, program)),
                "f1": mh_f1(answer, gold, program),
                "sub_queries": [s for st in trace["rounds"] for s in (st["sub_queries"] or [])],
                "sql": trace["sql"], "raw": trace,
                "n_tok": chat.n_prompt_tokens - toks0, "n_llm_calls": chat.n_calls - calls0,
                "n_rounds": len(trace["rounds"]), "n_sql_errors": n_err,
                "seconds": seconds}, ensure_ascii=False) + "\n")
            fans.flush()
            print(f"  {n}/{len(todo)} {qid[:8]} rounds={len(trace['rounds'])} calls={chat.n_calls - calls0} "
                  f"sql_err={n_err} pred={answer!r} {seconds}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
