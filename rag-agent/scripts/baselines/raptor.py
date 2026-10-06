#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""RAPTOR (Sarthi et al., ICLR 2024) 원본 코드(vendor/raptor)로 만든 MultiHiertt 비교군 `raptor`.

문서 = 질문이 속한 MultiHiertt 문서의 표 전부를 markdown 으로 이어 붙인 글(청킹 arm 과 같은 줄 렌더링,
`rag_agent/serialization/chunks.py: markdown_source`). 문서마다 트리 하나를 만들어 내용 해시로 캐시한다.
트리 만들기·재귀 군집·요약 프롬프트·collapsed-tree 검색은 원본 코드 그대로이고, 모델만 확장점
(BaseEmbeddingModel / BaseSummarizationModel / BaseQAModel)으로 바꿔 끼운다 — README.md 의 이탈 목록.

셀 배달 규칙: 잎 청크는 셀 값 글 전체가 청크 안에 온전히 들어 있는 셀을 배달하고, 요약 노드는 자식 노드가
배달하는 셀의 합집합을 배달한다.

  PYTHONPATH=. .venv/bin/python scripts/baselines/raptor.py --limit 3 --out-dir results/mh_only_20260928/baselines/raptor/smoke3
  PYTHONPATH=. .venv/bin/python scripts/baselines/raptor.py --dry-run     # LLM 없이 문서·청크 통계만
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# vendor 를 맨 앞에 — 이 파일 이름도 raptor.py 라 scripts/baselines 가 앞서면 자기 자신을 import 한다
sys.path.insert(0, str(ROOT / "vendor" / "raptor"))
sys.path.insert(1, str(ROOT))
sys.path.insert(2, str(ROOT / "scripts"))

import numpy as np                                                       # noqa: E402
import tiktoken                                                          # noqa: E402

from raptor import (BaseEmbeddingModel, BaseQAModel, BaseSummarizationModel,  # noqa: E402
                    ClusterTreeBuilder, ClusterTreeConfig)
from raptor.tree_structures import Tree                                # noqa: E402
from raptor.utils import (distances_from_embeddings, get_node_list, get_text,  # noqa: E402
                          reverse_mapping, split_text)

from answer_accuracy import check_context_limit                          # noqa: E402
from answer_accuracy_mh import extract                                   # noqa: E402
from mh_arms import OFFICIAL_REPO, OFFICIAL_REV, build_tables, load_population, official_answers  # noqa: E402
from rag_agent.eval.artifacts import digest, provenance                  # noqa: E402
from rag_agent.eval.multihiertt_em import mh_exact_match                # noqa: E402
from rag_agent.eval.multihiertt_f1 import mh_f1                         # noqa: E402
from rag_agent.serialization.chunks import markdown_source               # noqa: E402

RAPTOR_COMMIT = "7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767"
RECORDS = ROOT / "results/mh_only_20260928/retrieval/mh_train_s3c_records.jsonl"
ANSWER_SET = ROOT / "results/mh_arms/cap300_20260924/cell_uniq.jsonl"
BUDGETS = (2000, 575)          # (a) 논문 collapsed tree 2000, (b) 본 방법 20셀 문맥 평균(882건, 574.49)
CHUNK_TOKENS = 100             # 논문·코드 기본 잎 청크 크기 (cl100k)
SUMMARY_TOKENS = 100           # 코드 기본 summarization_length
QA_MAX_TOKENS = 384            # 본 방법 리더와 같은 생성 상한 (원본 QA 모델 기본 150)

# 원본 프롬프트 글자 그대로 (SummarizationModels.GPT3TurboSummarizationModel, QAModels.GPT3TurboQAModel)
SUM_SYSTEM = "You are a helpful assistant."
SUM_USER = "Write a summary of the following, including as many key details as possible: {context}:"
QA_SYSTEM = "You are Question Answering Portal"
QA_USER = "Given Context: {context} Give the best full answer amongst the option to question {question}"
# 이탈: EM 채점을 위해 answer_accuracy_mh.py 의 최종 답 표시 규약을 뒤에 붙인다
QA_MARKER = " On the last line write 'Final answer: ' followed by only the final value, with no units or explanation."


# ---------------------------------------------------------------- 확장점 구현
class BgeEmbedding(BaseEmbeddingModel):
    """SBERT multi-qa-mpnet → BAAI/bge-base-en-v1.5 (본 방법과 같은 인코더). 노드는 passage 접두어(없음)."""

    def __init__(self, enc):
        self.enc = enc

    def create_embedding(self, text):
        return self.enc.encode([text])[0].astype(np.float32)


class QwenSummarizer(BaseSummarizationModel):
    """gpt-3.5-turbo → Qwen3-8B 4bit 비생각 greedy. 프롬프트·max_tokens 는 원본 그대로."""

    def __init__(self, llm):
        self.llm, self.calls = llm, 0

    def summarize(self, context, max_tokens=150):
        self.calls += 1
        return self.llm.complete(SUM_SYSTEM, SUM_USER.format(context=context),
                                 max_tokens=max_tokens, temperature=0.0)


class QwenQA(BaseQAModel):
    def __init__(self, llm, batch_size=1):
        self.llm, self.batch_size = llm, batch_size

    def prompt(self, context, question):
        return QA_USER.format(context=context, question=question) + QA_MARKER

    def answer_question(self, context, question):
        return self.llm.complete(QA_SYSTEM, self.prompt(context, question),
                                 max_tokens=QA_MAX_TOKENS, temperature=0.0)

    def answer_batch(self, pairs):
        users = [self.prompt(c, q) for c, q in pairs]
        if self.batch_size == 1:
            return [self.llm.complete(QA_SYSTEM, u, max_tokens=QA_MAX_TOKENS, temperature=0.0) for u in users]
        return self.llm.complete_batch(QA_SYSTEM, users, max_tokens=QA_MAX_TOKENS)


# ---------------------------------------------------------------- 문서 렌더링과 잎 청크의 셀 배달
def render_doc(uid, tables, hdr):
    """(문서 글, [(start, end, "표-행-열")]) — 표마다 청킹 arm 과 같은 markdown 줄, 표 사이 빈 줄.
    셀 id 는 table_evidence 와 같은 펼친 격자 좌표(mh_arms.cell_id 의 '::' 뒤)."""
    parts, cells, off = [], [], 0
    prefix = "Table name: \n"
    for tid in sorted((t for t in tables if t.startswith(uid + "::")), key=lambda t: int(t.split("::")[1])):
        tab = tables[tid]
        text, spans = markdown_source(tab, tab.table, "")
        assert text.startswith(prefix)
        t_idx, (nhr, nhc) = int(tid.split("::")[1]), hdr[tid]
        for s, e, (i, j) in spans:
            cells.append((off + s - len(prefix), off + e - len(prefix), f"{t_idx}-{i + nhr}-{j + nhc}"))
        parts.append(text[len(prefix):])
        off += len(text) - len(prefix) + 2
    return "\n\n".join(parts), cells


def split_text_spans(text, tokenizer, max_tokens):
    """raptor.utils.split_text 를 한 줄씩 그대로 따라가며 조각의 원문 위치를 함께 돌려준다.
    [(청크 글, [(start, end), ...])]. 호출한 쪽이 원본 split_text 결과와 글이 같은지 확인한다."""
    delim = re.compile("|".join(map(re.escape, [".", "!", "?", "\n"])))
    pieces, start = [], 0
    for m in delim.finditer(text):
        pieces.append((start, m.start()))
        start = m.end()
    pieces.append((start, len(text)))
    n_tokens = [len(tokenizer.encode(" " + text[s:e])) for s, e in pieces]
    chunks, cur, cur_len = [], [], 0
    for (s, e), cnt in zip(pieces, n_tokens):
        sent = text[s:e]
        if not sent.strip():
            continue
        if cnt > max_tokens:
            subs, sub_start = [], s
            for m in re.finditer(r"[,;:]", sent):
                subs.append((sub_start, s + m.start()))
                sub_start = s + m.end()
            subs.append((sub_start, e))
            filtered = []
            for a, b in subs:
                raw = text[a:b]
                if raw.strip():
                    lead = len(raw) - len(raw.lstrip())
                    filtered.append((a + lead, a + lead + len(raw.strip())))
            sub_counts = [len(tokenizer.encode(" " + text[a:b])) for a, b in filtered]
            sub_chunk, sub_len = [], 0
            for span, c in zip(filtered, sub_counts):
                if sub_len + c > max_tokens and sub_chunk:
                    chunks.append(sub_chunk)
                    sub_chunk, sub_len = [], 0
                sub_chunk.append(span)
                sub_len += c
            if sub_chunk:
                chunks.append(sub_chunk)
        elif cur_len + cnt > max_tokens:
            chunks.append(cur)
            cur, cur_len = [(s, e)], cnt
        else:
            cur.append((s, e))
            cur_len += cnt
    if cur:
        chunks.append(cur)
    return [(" ".join(text[a:b] for a, b in spans), spans) for spans in chunks]


def leaf_cells(chunk_spans, cells):
    """청크의 원문 조각 중 하나가 셀 값 글 전체를 담는 셀만 배달한다."""
    return sorted({cid for s, e, cid in cells if any(a <= s and e <= b for a, b in chunk_spans)})


# ---------------------------------------------------------------- 트리
def node_cells(tree, leaf_map):
    """{노드 index: 배달 셀} — 잎은 leaf_map, 요약 노드는 자식의 합집합."""
    out = {}

    def rec(idx):
        if idx not in out:
            node = tree.all_nodes[idx]
            out[idx] = sorted(leaf_map[idx]) if not node.children else sorted(
                set().union(*(set(rec(c)) for c in node.children)))
        return out[idx]

    for idx in tree.all_nodes:
        rec(idx)
    return out


def build_tree(doc_text, cells, config, summarizer, tok_cl, cache_dir, key):
    path = cache_dir / f"{key}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    # 문서마다 새 빌더 — 원본 construct_tree 는 층을 멈출 때 self.num_layers 를 그 층 번호로 덮어써서
    # 빌더를 재사용하면 다음 문서부터 군집·요약이 사라진다(연기 시험에서 확인, vendor 패치 대신 여기서 피함)
    builder = ClusterTreeBuilder(config)
    chunks = split_text_spans(doc_text, tok_cl, CHUNK_TOKENS)
    if [c for c, _ in chunks] != split_text(doc_text, tok_cl, CHUNK_TOKENS):
        raise RuntimeError(f"split_text_spans 가 원본 split_text 와 다르다 ({key})")
    calls0, t0 = summarizer.calls, time.time()
    cluster_error = None
    try:
        tree = builder.build_from_text(doc_text, use_multithreading=False)
    except Exception as e:                       # 원본 군집(UMAP+GMM)이 작은 문서에서 죽는다 (sklearn ill-defined covariance 등)
        cluster_error = f"{type(e).__name__}: {e}"[:200]
        print(f"  [cluster-fail → leaves only] {key} {cluster_error}", flush=True)
        leaves = {i: builder.create_node(i, c)[1] for i, (c, _) in enumerate(chunks)}
        tree = Tree(dict(leaves), list(leaves.values()), leaves, 0, {0: list(leaves.values())})
    if len(tree.leaf_nodes) != len(chunks):
        raise RuntimeError("잎 수가 청크 수와 다르다")
    leaf_map = {}
    for idx, node in tree.leaf_nodes.items():
        if node.text != chunks[idx][0]:
            raise RuntimeError("잎 글이 청크 글과 다르다")
        leaf_map[idx] = leaf_cells(chunks[idx][1], cells)
    entry = {"tree": tree, "cells": node_cells(tree, leaf_map),
             "layer_of": reverse_mapping(tree.layer_to_nodes),
             "stats": {"doc_key": key, "n_chars": len(doc_text),
                       "n_doc_tok_cl100k": len(tok_cl.encode(doc_text)),
                       "n_leaves": len(tree.leaf_nodes), "n_nodes": len(tree.all_nodes),
                       "n_layers": tree.num_layers,
                       "nodes_per_layer": {str(k): len(v) for k, v in tree.layer_to_nodes.items()},
                       "n_summary_calls": summarizer.calls - calls0,
                       "cluster_failed_leaves_only": cluster_error,
                       "n_cells": len(cells),
                       "n_cells_deliverable_by_a_leaf": len(set().union(*map(set, leaf_map.values()))),
                       "seconds": round(time.time() - t0, 1)}}
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(entry))
    with (cache_dir / "stats.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry["stats"]) + "\n")
    return entry


def collapsed_rank(entry, qv):
    """원본 retrieve_information_collapse_tree 의 순위: 모든 노드를 코사인 거리 오름차순."""
    nodes = get_node_list(entry["tree"].all_nodes)
    dist = distances_from_embeddings(qv, [n.embeddings["EMB"] for n in nodes])
    return [nodes[i] for i in np.argsort(dist)]


def select(ranked, n_tok, budget):
    """원본과 같은 그리디: 상한을 처음 넘기는 노드에서 멈춘다(건너뛰지 않는다)."""
    out, total = [], 0
    for node in ranked:
        if total + n_tok[node.index] > budget:
            break
        out.append(node)
        total += n_tok[node.index]
    return out


# ---------------------------------------------------------------- 실행
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="882 답변 집합의 앞 N 건만(검색·답변 둘 다). 0 = 검색 2,885 + 답변 882")
    ap.add_argument("--out-dir", default="results/mh_only_20260928/baselines/raptor")
    ap.add_argument("--tree-dir", default="", help="트리 캐시. 기본 <out-dir>/trees")
    ap.add_argument("--reader", default="local:Qwen/Qwen3-8B?quantization=4bit")
    ap.add_argument("--embed-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--batch-size", type=int, default=1, help="답변 생성 묶음(>1 이면 LocalQwenLLM.complete_batch)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="LLM 없이 문서·잎 청크·배달 가능 셀 통계만")
    ap.add_argument("--resume", action="store_true", help="저장된 레코드의 query_id 는 건너뛰고 이어 쓴다(트리는 <tree-dir> 캐시, 답변은 원래부터 이어 씀)")
    ap.add_argument("--embed-device", default=None, help="인코더 장치(기본 자동=cuda). 리더와 같이 올려 GPU 가 모자라면 cpu")
    a = ap.parse_args()
    out = Path(a.out_dir)
    tree_dir = Path(a.tree_dir) if a.tree_dir else out / "trees"
    rec_path = out / "raptor_records.jsonl"
    if rec_path.exists() and not (a.dry_run or a.resume):
        raise SystemExit(f"{rec_path} 가 있다 — 새 --out-dir")
    logging.getLogger().setLevel(logging.WARNING)      # vendor 코드의 INFO 로그를 줄인다

    recs = {r["query_id"]: r for r in map(json.loads, RECORDS.open(encoding="utf-8")) if "doc" in r}
    answer_ids = [json.loads(l)["query_id"] for l in ANSWER_SET.open(encoding="utf-8")]
    assert len(recs) == 2885 and len(answer_ids) == 882 and set(answer_ids) <= set(recs)
    if a.limit:
        answer_ids = answer_ids[:a.limit]
        recs = {q: recs[q] for q in answer_ids}
    _q, docs, _ = load_population("train")
    answers = official_answers("train")
    docs = {u: docs[u] for u in recs}
    tables, hdr = build_tables(docs, "v3.3u")
    tok_cl = tiktoken.get_encoding("cl100k_base")
    from transformers import AutoTokenizer
    tok_q = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    n_qtok = lambda t: len(tok_q(t, add_special_tokens=False)["input_ids"])

    rendered, excluded = {}, Counter()
    for q, r in recs.items():
        text, cells = render_doc(q, tables, hdr)
        key = hashlib.sha256(text.encode()).hexdigest()[:16]
        rendered[q] = (key, text, cells)
        missing = set(r["gold_ids"]) - {c for _, _, c in cells}
        if missing:
            excluded[q] = f"gold_cell_not_rendered:{sorted(missing)}"
    print(f"[docs] {len(recs)} queries, {len({v[0] for v in rendered.values()})} distinct documents, "
          f"excluded {len(excluded)}", flush=True)

    if a.dry_run:
        leaves, ceiling, dtok = [], [], []
        for q, (key, text, cells) in rendered.items():
            chunks = split_text_spans(text, tok_cl, CHUNK_TOKENS)
            assert [c for c, _ in chunks] == split_text(text, tok_cl, CHUNK_TOKENS)
            leaves.append(len(chunks)); dtok.append(n_qtok(text))
            ok = set().union(*(set(leaf_cells(sp, cells)) for _, sp in chunks))
            ceiling.append(set(recs[q]["gold_ids"]) <= ok)
        print(json.dumps({"n_queries": len(recs), "n_distinct_docs": len({v[0] for v in rendered.values()}),
                          "doc_tokens_qwen3_mean": round(float(np.mean(dtok)), 1),
                          "leaves_mean": round(float(np.mean(leaves)), 2), "leaves_min": min(leaves),
                          "leaves_max": max(leaves), "docs_with_>11_leaves": sum(x > 11 for x in leaves),
                          "gold_all_deliverable_by_leaves": round(float(np.mean(ceiling)), 4)}, indent=1))
        return 0

    from rag_agent.llm.factory import build_llm
    from rag_agent.retrieve.encoders import default_encoder
    import torch
    torch.manual_seed(a.seed)
    enc = default_encoder(model_name=a.embed_model, **({"device": a.embed_device} if a.embed_device else {}))
    llm = build_llm(a.reader)
    summarizer = QwenSummarizer(llm)
    config = ClusterTreeConfig(
        tokenizer=tok_cl, max_tokens=CHUNK_TOKENS, summarization_length=SUMMARY_TOKENS,
        summarization_model=summarizer, embedding_models={"EMB": BgeEmbedding(enc)},
        cluster_embedding_model="EMB")
    qa = QwenQA(llm, a.batch_size)

    # 1) 트리 + collapsed-tree 검색 레코드
    out.mkdir(parents=True, exist_ok=True)
    entries, rows, t0 = {}, [], time.time()
    done = set()
    if a.resume and rec_path.exists():
        saved = [json.loads(l) for l in rec_path.open(encoding="utf-8")]
        done = {x["query_id"] for x in saved}
        rows = [x for x in saved if "excluded" not in x]
        print(f"[resume] 저장된 레코드 {len(done)}건 건너뜀(트리 통계는 이번 실행에서 만든 트리만)", flush=True)
    with rec_path.open("a" if a.resume else "w", encoding="utf-8") as stream:
        for n, (q, r) in enumerate(recs.items(), 1):
            if q in done:
                continue
            key, text, cells = rendered[q]
            if q in excluded:
                stream.write(json.dumps({"query_id": q, "excluded": excluded[q]}) + "\n")
                continue
            if key not in entries:
                entries[key] = build_tree(text, cells, config, summarizer, tok_cl, tree_dir, key)
                print(f"  [tree] {key} {entries[key]['stats']}", flush=True)
            e = entries[key]
            n_tok = {i: n_qtok(nd.text) for i, nd in e["tree"].all_nodes.items()}
            ranked = collapsed_rank(e, enc.encode_query([r["question"]])[0].astype(np.float32))
            row = {"query_id": q, "kind": r["kind"], "layer": r["layer"], "question": r["question"],
                   "gold_ids": r["gold_ids"], "doc_key": key, "n_nodes": len(ranked),
                   "n_layers": e["tree"].num_layers,
                   "ranked_units": [{"cells": e["cells"][nd.index], "text": nd.text, "n_tok": n_tok[nd.index],
                                     "layer": e["layer_of"][nd.index]} for nd in ranked[:30]]}
            for b in BUDGETS:
                sel = select(ranked, n_tok, b)
                got = set().union(*(set(e["cells"][nd.index]) for nd in sel)) if sel else set()
                row[f"context_{b}"] = get_text(sel)
                row[f"n_units_{b}"] = len(sel)
                row[f"n_tok_{b}"] = sum(n_tok[nd.index] for nd in sel)
                row[f"cells_{b}"] = len(got)
                row[f"correct_{b}"] = int(set(r["gold_ids"]) <= got)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            if n % 100 == 0:
                print(f"  {n}/{len(recs)} trees {len(entries)} {time.time() - t0:.0f}s", flush=True)
    by = defaultdict(list)
    for x in rows:
        by[x["layer"]].append(x); by["ALL"].append(x)
    acc = lambda v: round(sum(v) / len(v), 4) if v else None
    stats = [e["stats"] for e in entries.values()]
    summary = {"baseline": "raptor", "raptor_repo": "https://github.com/parthsarthi03/raptor", "raptor_commit": RAPTOR_COMMIT,
               "records_source": str(RECORDS.relative_to(ROOT)), "answer_set": str(ANSWER_SET.relative_to(ROOT)),
               "limit": a.limit, "n_queries": len(recs), "n_scored": len(rows), "n_excluded": len(excluded),
               "excluded": dict(excluded), "n_documents": len(entries),
               "chunk_tokens_cl100k": CHUNK_TOKENS, "summary_max_tokens": SUMMARY_TOKENS,
               "budgets_qwen3_tokens": list(BUDGETS), "encoder": enc.name, "query_prefix": enc.query_prefix,
               "summarizer": llm.name, "n_summary_calls_total": summarizer.calls,
               "tree_seconds_total": round(sum(s["seconds"] for s in stats), 1),
               "cluster_failed_leaves_only": {"n": sum(1 for s in stats if s.get("cluster_failed_leaves_only")),
                                              "doc_keys": [s["doc_key"] for s in stats if s.get("cluster_failed_leaves_only")]},
               "tree_stats_mean": {k: round(float(np.mean([s[k] for s in stats])), 2)
                                   for k in ("n_leaves", "n_nodes", "n_layers", "n_summary_calls", "seconds", "n_doc_tok_cl100k")} if stats else None,
               "by_layer": {k: {"n": len(v), **{f"accuracy_all_{b}": acc([x[f"correct_{b}"] for x in v]) for b in BUDGETS},
                                **{f"n_tok_{b}_mean": round(float(np.mean([x[f"n_tok_{b}"] for x in v])), 1) for b in BUDGETS}}
                            for k, v in sorted(by.items())},
               "provenance": provenance(ROOT), "arguments": vars(a)}
    (out / "raptor_records.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n_scored", "n_documents", "tree_stats_mean", "cluster_failed_leaves_only", "by_layer")}, ensure_ascii=False, indent=1), flush=True)

    # 2) 답변 — 882 × 두 설정
    row_of = {x["query_id"]: x for x in rows}
    todo = [row_of[q] for q in answer_ids if q in row_of]
    limit_ctx = getattr(llm, "context_limit", 0)
    for b in BUDGETS:
        path = out / f"raptor_answer_{b}.jsonl"
        done, arows = set(), []
        if path.exists():
            arows = [json.loads(l) for l in path.open(encoding="utf-8")]
            done = {x["query_id"] for x in arows}
        pending = [x for x in todo if x["query_id"] not in done]
        t0 = time.time()
        with path.open("a", encoding="utf-8") as stream:
            for s in range(0, len(pending), a.batch_size):
                batch = pending[s:s + a.batch_size]
                t1 = time.time()
                users = [qa.prompt(x[f"context_{b}"], x["question"]) for x in batch]
                n_toks = [llm.n_prompt_tokens(QA_SYSTEM, u) for u in users]
                for nt in n_toks:
                    check_context_limit(nt, QA_MAX_TOKENS, limit_ctx)
                raws = qa.answer_batch([(x[f"context_{b}"], x["question"]) for x in batch])
                sec = round((time.time() - t1) / len(batch), 3)
                for x, nt, raw in zip(batch, n_toks, raws):
                    pred, marked = extract(raw)
                    gold, prog = answers[x["query_id"]], x["kind"] == "arith"
                    arow = {"query_id": x["query_id"], "layer": x["layer"], "kind": x["kind"],
                            "question": x["question"], "budget": b, "retrieval_correct": x[f"correct_{b}"],
                            "n_units": x[f"n_units_{b}"], "cells_in_context": x[f"cells_{b}"],
                            "pred": pred, "answer": gold,
                            "answer_correct": int(mh_exact_match(pred, gold, prog)),
                            "f1": mh_f1(pred, gold, prog), "raw": raw, "marker_found": marked,
                            "n_tok": nt, "n_llm_calls": 1, "seconds": sec,
                            "context_sha256": digest(x[f"context_{b}"])}
                    stream.write(json.dumps(arow, ensure_ascii=False) + "\n")
                    stream.flush()
                    arows.append(arow)
                if len(arows) % 50 == 0:
                    print(f"  [answer {b}] {len(arows)}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
        by = defaultdict(list)
        for x in arows:
            by[x["layer"]].append(x); by["ALL"].append(x)
        asum = {"budget_qwen3_tokens": b, "n": len(arows), "reader": llm.name, "seed": a.seed,
                "max_new_tokens": QA_MAX_TOKENS, "batch_size": a.batch_size,
                "qa_system": QA_SYSTEM, "qa_user": QA_USER + QA_MARKER,
                "scorer": "multihiertt_em.mh_exact_match (공식 EM) + multihiertt_f1.mh_f1 (공식 F1 포팅)",
                "answer_source": f"{OFFICIAL_REPO}@{OFFICIAL_REV}",
                "marker_missing": sum(not x["marker_found"] for x in arows),
                "generation_seconds": round(sum(x["seconds"] for x in arows), 1),
                "by_layer": {k: {"n": len(v), "answer_em": acc([x["answer_correct"] for x in v]),
                                 "f1": round(float(np.mean([x["f1"] for x in v])), 4),
                                 "retrieval_accuracy_here": acc([x["retrieval_correct"] for x in v]),
                                 "input_tokens_mean": round(float(np.mean([x["n_tok"] for x in v])), 1)}
                             for k, v in sorted(by.items())},
                "provenance": provenance(ROOT)}
        path.with_suffix(".json").write_text(json.dumps(asum, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: asum[k] for k in ("budget_qwen3_tokens", "n", "marker_missing", "generation_seconds", "by_layer")},
                         ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
