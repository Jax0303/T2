#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""`tablerag_chen` — Google TableRAG(Chen et al., NeurIPS 2024) 공식 코드의 전체 파이프라인을 MultiHiertt 비교군으로.

공식 코드 `vendor/table_rag` (google-research @08a8d67) 의 `TableRAGAgent` 를 그대로 부른다:
  질의 확장(extract_column_prompt / extract_cell_prompt, LLM 이 JSON 목록 생성)
  → 스키마 검색 + 셀 검색(build_schema_corpus / build_cell_corpus, 임베딩 top_k=5 질의당, B=max_encode_cell=10000)
  → PyReAct 풀이(solver_loop: Thought/Action/Observation ≤ max_depth=5, python_repl_ast 로 pandas 실행, 호출당 128토큰).
프롬프트는 wtq(표 QA) 것. 바꾼 것(모델·임베딩 교체, 계층 머리글, 문서당 표 여러 개)은 README.md 의 목록에 있다.

같은 조건: 검색 범위 = 질문의 문서(train)의 표만(머리글 v3.3u, `mh_arms.build_tables`), 모든 LLM 호출 = Qwen3-8B 4bit 비생각
greedy(seed 42), 임베딩 = BAAI/bge-base-en-v1.5. 채점 = 공식 EM(`multihiertt_em.mh_exact_match`) + 공식 F1(`multihiertt_f1.mh_f1`).

  PYTHONPATH=. PYTHONHASHSEED=0 .venv/bin/python scripts/baselines/tablerag_chen.py [--limit 3 --out-dir ...]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor/table_rag"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(VENDOR))

from mh_arms import (OFFICIAL_REPO, OFFICIAL_REV, build_tables, cell_id,  # noqa: E402
                     layer, load_population, official_answers, resolve_gold)
from rag_agent.eval.artifacts import digest, file_digest, provenance    # noqa: E402
from rag_agent.eval.multihiertt_em import mh_exact_match, self_check    # noqa: E402
from rag_agent.eval.multihiertt_f1 import mh_f1                         # noqa: E402
from rag_agent.serialization.base import join_path                      # noqa: E402
from rag_agent.serialization.tablerag_unit import official_frame        # noqa: E402
import agent.agent as agent_mod                                         # noqa: E402  (vendored)
from agent.rag_agent import TableRAGAgent                               # noqa: E402
from agent.retriever import Retriever                                   # noqa: E402
from utils.utils import infer_dtype                                     # noqa: E402  (vendored)

# 원본 기본값 (agent/agent.py TableAgent.__init__, run.py main): top_k=5, max_encode_cell=10000, max_depth=5,
# max_tokens=128, stop=['Observation:']. temperature 0.8→0.0(greedy, 같은 조건 규칙).
TOP_K, MAX_ENCODE_CELL, MAX_DEPTH, MAX_TOKENS, TASK = 5, 10000, 5, 128, "wtq"
READER, EMBED, SEED, DUMP = "local:Qwen/Qwen3-8B", "BAAI/bge-base-en-v1.5", 42, 30
S3C = ROOT / "results/mh_only_20260928/retrieval/mh_train_s3c_records.jsonl"
SUBSET = ROOT / "results/mh_arms/cap300_20260924/cell_uniq.jsonl"
OUT = ROOT / "results/mh_only_20260928/baselines/tablerag_chen"
DB = ROOT / ".cache/tablerag_chen/db"
ONE_DF = "The name of the dataframe is `df`."
MANY_DF = ("The document has {n} tables, loaded as pandas dataframes {names} (one per table). Column names in the "
           "retrieved schemas and cells below are prefixed with the dataframe name as `df_k.<column>`; "
           "index a column as `df_k[\"<column>\"]`.")
_CELL = re.compile(r'^\{"column_name": "(.*)", "cell_value": "(.*)"\}$', re.S)
_SUMM = re.compile(r'^\{"column_name": "(.*)", "dtype": "[^"]*", "min": .*, "max": .*\}$', re.S)
_ERR = re.compile(r"^\s*\w*(Error|Exception)\b")


def doc_frames(uid, tables):
    """{표 번호 k: DataFrame} — 표 k 의 프레임 = `official_frame(mode="path")`(경로 결합 열 이름, 원본 infer_dtype) 앞에
    행 라벨 열 하나(이름 = 격자 모서리 글, 없으면 'row'; 값 = 행 머리글 경로 결합). 원본은 첫 행을 머리글로 읽은 평면 표라
    행 라벨이 보통 첫 열이다 — 우리 파싱은 행 머리글을 data 밖에 두므로 여기서 되돌려 넣는다(README (i)).
    그 열도 원본 infer_dtype 를 거친다(연도 행 라벨은 원본처럼 숫자 열이 된다)."""
    out = {}
    for tid in sorted((t for t in tables if t.split("::")[0] == uid), key=lambda t: int(t.split("::")[1])):
        t = tables[tid].table
        df = official_frame(t, "path")
        name = " ".join(x.strip() for row in t.grid[:t.nhr] for x in row[:t.nhc] if x and x.strip()) or "row"
        if name in df.columns:
            name += " (row)"
        df.insert(0, name, [join_path(t.row_path(r)) for r in range(t.n_rows)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = infer_dtype(df)
        out[int(tid.split("::")[1])] = df
    return out


class DocRetriever(Retriever):
    """문서의 표 여러 개(README (ii)): 표마다 원본 build_*_corpus 를 부르고 문서를 합친다. 열 이름은 `df_k.<열>`."""

    def get_docs(self, data_type, frames):
        docs = []
        for k, df in frames.items():
            docs += super().get_docs(data_type, df.rename(columns=lambda c, k=k: f"df_{k}.{c}"))
        return docs


class DocAgent(TableRAGAgent):
    """풀이 프롬프트의 '`df` 하나' 문장만 표 여러 개 문장으로 바꾼다(README (ii)). 나머지 문구는 원본."""
    frames = {}

    def get_prompt(self, prompt_type, **kw):
        text = super().get_prompt(prompt_type, **kw)
        if prompt_type == "solve_table_prompt":
            assert ONE_DF in text
            names = ", ".join(f"`df_{k}`" for k in self.frames)
            text = text.replace(ONE_DF, MANY_DF.format(n=len(self.frames), names=names))
        return text


def doc_cells(text, frames, uid, hdr):
    """검색된 셀 문서 하나 → 배달하는 데이터 셀 id("표-행-열", 펼친 격자 좌표). `retrieval_accuracy.tablerag_units` 규칙:
    숫자·날짜 열 요약 문서 → 그 열의 min·max 행(같은 값은 마지막 행), 범주 문서 → 그 열에서 값이 같은 셀 전부.
    행 라벨 열(프레임 0열)은 데이터 셀이 아니므로 배달 없음. 빈 값 문서도 배달 없음."""
    m = _CELL.match(text) or _SUMM.match(text)
    if not m:
        return []
    k, col = m.group(1).split(".", 1)
    k, value = int(k[3:]), (m.group(2) if m.re is _CELL else None)
    df, (nhr, nhc) = frames[k], hdr[f"{uid}::{k}"]
    cells = set()
    for c in [i for i, n in enumerate(df.columns) if n == col and i > 0]:
        s = df.iloc[:, c]
        if value is not None:
            rows = [r for r, v in enumerate(s.astype(str)) if v == value and v.strip()]
        else:
            vals = {v: r for r, v in enumerate(s) if v == v}
            rows = sorted({vals[min(vals)], vals[max(vals)]}) if vals else []
        cells |= {f"{k}-{r + nhr}-{c - 1 + nhc}" for r in rows}
    return sorted(cells)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = 검색 2,885 전부 + 답변 882 부분집합. N = 882 부분집합(query_id 순)의 앞 N 건만(검색+답변)")
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--resume", action="store_true", help="저장된 records 의 query_id 는 건너뛰고 이어 쓴다")
    ap.add_argument("--embed-device", default=None, help="임베딩 모델 장치(기본 자동=cuda). GPU 가 모자라면 cpu")
    a = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("PYTHONHASHSEED=0 로 실행한다 — 원본이 검색 결과를 set() 으로 합쳐 순서가 해시에 달렸다")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path, ans_path = out_dir / "tablerag_chen_records.jsonl", out_dir / "tablerag_chen_answer.jsonl"
    if (rec_path.exists() or ans_path.exists()) and not a.resume:
        raise SystemExit(f"{out_dir} 에 결과가 있다 — 덮어쓰지 않는다(--resume 또는 새 --out-dir)")

    queries, docs, skipped = load_population("train")
    tables, hdr = build_tables(docs, "v3.3u", "none")
    live = {(tid, i, j) for tid, tab in tables.items() for i, row in enumerate(tab.table.data)
            for j, v in enumerate(row) if str(v).strip()}
    queries = {q["uid"]: q for q in resolve_gold(queries, tables, hdr, live) if q["gold"] and not q["excluded"]}
    ref = {json.loads(l)["query_id"] for l in open(S3C, encoding="utf-8") if '"doc"' in l}
    assert set(queries) == ref, "질의 집합이 본 방법 레코드(2,885)와 다르다"
    subset = [json.loads(l)["query_id"] for l in open(SUBSET, encoding="utf-8")]
    assert subset == sorted(subset) and set(subset) <= set(queries)
    order = subset[:a.limit] if a.limit else subset + sorted(set(queries) - set(subset))
    answer_set = set(subset)
    answers = official_answers("train")
    check = self_check([answers[q] for q in order if q in answer_set], [queries[q]["kind"] == "arith" for q in order if q in answer_set])
    print(f"[pop] {len(order)} queries (답변 {sum(q in answer_set for q in order)}, 제외 {dict(skipped)}); "
          f"[scorer] gold 를 예측으로 넣었을 때 EM={check['em']}", flush=True)
    done = set()
    if a.resume and rec_path.exists():
        done = {json.loads(l)["query_id"] for l in open(rec_path, encoding="utf-8")}
        print(f"[resume] 저장된 {len(done)}건 건너뜀", flush=True)
    done_ans = {json.loads(l)["query_id"] for l in open(ans_path, encoding="utf-8")} if ans_path.exists() else set()

    import torch
    agent_mod.Retriever = DocRetriever                      # TableAgent.__init__ 이 만드는 Retriever 를 문서용으로
    agent = DocAgent(model_name=READER, retrieve_mode="embed", embed_model_name=EMBED, task=TASK, agent_type="TableRAG",
                     top_k=TOP_K, max_encode_cell=MAX_ENCODE_CELL, temperature=0.0, top_p=0.95,
                     stop_tokens=["Observation:"], max_tokens=MAX_TOKENS, max_depth=MAX_DEPTH, db_dir=str(DB))
    if a.embed_device:
        from agent.local_backend import HuggingFaceEmbeddings
        agent.retriever.embedder = HuggingFaceEmbeddings(EMBED, device=a.embed_device)
    torch.manual_seed(SEED)
    llm, tok = agent.model.client, agent.model.tokenizer
    n_tok = lambda t: len(tok(t, add_special_tokens=False)["input_ids"])

    recs, rows, t0 = [], [], time.time()
    with rec_path.open("a", encoding="utf-8", newline="\n") as rs, ans_path.open("a", encoding="utf-8", newline="\n") as ans:
        for n, uid in enumerate(order, 1):
            if uid in done:
                continue
            q, t1 = queries[uid], time.time()
            frames = doc_frames(uid, tables)
            agent.frames = frames
            agent.retriever.init_retriever(uid, frames)
            calls0, in0 = llm.n_calls, agent.total_input_token_count
            # ---- 원본 TableRAGAgent.run 의 단계 그대로 (표 여러 개라 run() 대신 여기서 부른다) ----
            schema_text, col_queries, schema_docs = agent.retrieve_schema_by_prompt(
                agent.get_prompt("extract_column_prompt", table_caption="", query=q["question"]))
            cell_text, cell_queries, cell_docs = agent.retrieve_cell_by_prompt(
                agent.get_prompt("extract_cell_prompt", table_caption="", query=q["question"]))
            # 합쳐진 셀 문서를 가장 가까운 질의와의 거리(L2, 원본 FAISS 기본) 순으로
            best = {}
            for cq in cell_queries:
                for d, dist in agent.retriever.cell_retriever.search(cq):
                    best[d.page_content] = min(dist, best.get(d.page_content, float("inf")))
            assert set(best) == set(cell_docs)
            ranked = sorted(best, key=best.get)[:DUMP]
            units = [{"cells": doc_cells(d, frames, uid, hdr), "text": d, "n_tok": n_tok(d), "dist": round(best[d], 5)} for d in ranked]
            gold = sorted(cell_id(c, hdr).split("::", 1)[1] for c in q["gold"])
            got = set().union(*(set(u["cells"]) for u in units)) if units else set()
            context = schema_text + "\n\n" + cell_text
            r = {"query_id": uid, "kind": q["kind"], "m": len(q["gold"]), "layer": layer(q), "question": q["question"],
                 "gold_ids": gold, "queries": {"schema": col_queries, "cell": cell_queries},
                 "schema_hits": schema_docs, "ranked_units": units, "context": context,
                 "n_docs_in_corpus": {"schema": agent.retriever.schema_retriever.store.index.ntotal,
                                      "cell": agent.retriever.cell_retriever.store.index.ntotal},
                 "correct": int(set(gold) <= got), "any_DIAGNOSTIC": int(bool(set(gold) & got)),
                 "cells_in_context": len(got), "n_llm_calls": llm.n_calls - calls0,
                 "n_tok": agent.total_input_token_count - in0}
            if uid in answer_set and uid not in done_ans:
                prompt = agent.get_prompt("solve_table_prompt", table_caption="", query=q["question"],
                                          schema_retrieval_result=schema_text, cell_retrieval_result=cell_text)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pred, n_iter, solution = agent.solver_loop({f"df_{k}": df for k, df in frames.items()}, prompt)
                gold_ans, prog = answers[uid], q["kind"] == "arith"
                obs = re.findall(r"Observation: ?\n?([^\n]*)", solution)
                row = {"query_id": uid, "layer": r["layer"], "kind": q["kind"], "m": r["m"], "question": q["question"],
                       "retrieval_correct": r["correct"], "pred": pred, "answer": gold_ans,
                       "answer_correct": int(mh_exact_match(pred, gold_ans, prog)), "f1": mh_f1(pred, gold_ans, prog),
                       "raw": prompt + solution, "n_iter": n_iter, "terminal": agent.is_terminal(solution.split("Thought:")[-1]),
                       "n_tok": agent.total_input_token_count - in0, "n_llm_calls": llm.n_calls - calls0,
                       "n_code_errors": sum(bool(_ERR.match(o)) for o in obs), "cells_in_context": len(got),
                       "context_sha256": digest(context), "seconds": round(time.time() - t1, 3)}
                ans.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
            r["seconds"] = round(time.time() - t1, 3)
            rs.write(json.dumps(r, ensure_ascii=False) + "\n")
            rs.flush(); ans.flush(); os.fsync(rs.fileno()); os.fsync(ans.fileno())
            recs.append(r)
            if n % 20 == 0 or n == len(order):
                print(f"  {n}/{len(order)}  {time.time() - t0:.0f}s  답변 {len(rows)}"
                      + (f"  em={sum(x['answer_correct'] for x in rows) / len(rows):.4f}" if rows else ""), flush=True)

    rate = lambda v: round(sum(v) / len(v), 4) if v else None
    by = defaultdict(list)
    for r in recs:
        by[r["layer"]].append(r); by["ALL"].append(r)
    common = {"arm": "tablerag_chen", "split": "train", "header_rule": "v3.3u", "label_rule": "none",
              "official_code": (VENDOR / "VENDOR_COMMIT").read_text(encoding="utf-8").strip().splitlines(),
              "task_prompts": TASK, "top_k": TOP_K, "max_encode_cell": MAX_ENCODE_CELL, "max_depth": MAX_DEPTH,
              "max_tokens_per_call": MAX_TOKENS, "retrieve_mode": "embed (IndexFlatL2, sentence-transformers, 질의 접두어 없음)",
              "encoder": EMBED, "reader": agent.model.client.name, "seed": SEED, "temperature": 0.0,
              "limit": a.limit, "resumed": len(done), "provenance": provenance(ROOT)}
    summary = {**common, "n_queries": len(recs), "seconds": round(sum(r["seconds"] for r in recs), 1),
               "cell_credit": "요약 문서→min·max 행, 범주 문서→같은 값 셀 전부 (retrieval_accuracy.tablerag_units 규칙)",
               "by_layer": {k: {"n": len(v), "accuracy_all": rate([x["correct"] for x in v]),
                                "accuracy_any_DIAGNOSTIC": rate([x["any_DIAGNOSTIC"] for x in v]),
                                "cells_delivered_mean": round(sum(x["cells_in_context"] for x in v) / len(v), 1),
                                "units_mean": round(sum(len(x["ranked_units"]) for x in v) / len(v), 1),
                                "context_tokens_mean": round(sum(n_tok(x["context"]) for x in v) / len(v), 1),
                                "expansion_llm_calls_mean": round(sum(x["n_llm_calls"] for x in v) / len(v), 2)}
                            for k, v in sorted(by.items())},
               "records_sha256": file_digest(rec_path)}
    (out_dir / "tablerag_chen.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        bya = defaultdict(list)
        for x in rows:
            bya[x["layer"]].append(x); bya["ALL"].append(x)
        asum = {**common, "n": len(rows), "scorer": "multihiertt_em.mh_exact_match + multihiertt_f1.mh_f1 (공식 포팅)",
                "answer_source": f"{OFFICIAL_REPO}@{OFFICIAL_REV}", "scorer_self_check": check,
                "pred_rule": "원본 solver_loop 의 answer (마지막 'Answer:' 뒤 첫 줄) 그대로, 추가 추출 없음",
                "generation_seconds": round(sum(x["seconds"] for x in rows), 1),
                "by_layer": {k: {"n": len(v), "retrieval_accuracy_here": rate([x["retrieval_correct"] for x in v]),
                                 "answer_em": rate([x["answer_correct"] for x in v]), "answer_f1": rate([x["f1"] for x in v]),
                                 "input_tokens_mean": round(sum(x["n_tok"] for x in v) / len(v), 1),
                                 "llm_calls_mean": round(sum(x["n_llm_calls"] for x in v) / len(v), 2),
                                 "code_errors_mean": round(sum(x["n_code_errors"] for x in v) / len(v), 2),
                                 "terminal_rate": rate([int(x["terminal"]) for x in v]),
                                 "seconds_mean": round(sum(x["seconds"] for x in v) / len(v), 1)}
                             for k, v in sorted(bya.items())},
                "records_sha256": file_digest(ans_path)}
        (out_dir / "tablerag_chen_answer.json").write_text(json.dumps(asum, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in asum.items() if k not in ("provenance", "official_code")}, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("provenance", "official_code")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
