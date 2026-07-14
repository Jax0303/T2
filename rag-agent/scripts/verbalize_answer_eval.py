#!/usr/bin/env python3
"""End-to-end 답변 비교 — "쿼리가 오면 어떤 답이 나오는가" (3번 실험 후반부).

리더 LLM·프롬프트·평가는 전 조건 동일. 다른 것은 "무엇을 검색해 무엇을 읽히는가"뿐:

  oracle         gold 표를 그냥 줌 (리더 상한선; 검색 오류 0 가정)
  original_sent  [우리 베이스] 셀 문장(sent_long) 임베딩으로 표 top-1 식별
                 → 컨텍스트 = 통짜 원본 표 (누락 0 구조)
  rag_1t1c       1테이블=1청크 RAG: 전체표 임베딩(top-1) → 컨텍스트 = 그 청크
  rag_rowchunk   행 청킹 RAG: 행 청크 top-10 → 컨텍스트 = 그 청크들 (셀 누락 가능)

컨텍스트 직렬화는 전 조건 S2(구조 보존)로 통일 — 차이가 검색·커버리지에서만 나오게.
임베딩은 verbalize/cell_omission 실험 캐시 재사용. 평가: evaluate_answer(EM/NM).

사용: /usr/bin/python3 scripts/verbalize_answer_eval.py --n 120 --model llama-3.1-8b-instant
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_agent.bench.hitab import load_queries  # noqa: E402
from rag_agent.serialize.serializers import serialize_table, fulltable_chunk, S1, S2  # noqa: E402
from rag_agent.serialize.verbalize import verbalize_table  # noqa: E402
from rag_agent.retrieve.encoders import SentenceTransformerEncoder  # noqa: E402
from rag_agent.generate.answerer import answer, evaluate_answer  # noqa: E402

SEED = 42
MODEL = "BAAI/bge-small-en-v1.5"
QUERY_INSTR = "Represent this sentence for searching relevant passages: "
CACHE_DIR = ROOT / "data" / "verbalize_cache"
ARMS = ("oracle", "original_sent", "rag_1t1c", "rag_rowchunk")


def load_env():
    envf = ROOT / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def encode_cached(enc, tag: str, texts):
    path = CACHE_DIR / f"{tag}.npy"
    if path.exists():
        vecs = np.load(path)
        if len(vecs) == len(texts):
            return vecs
    vecs = enc.encode(texts)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, vecs)
    return vecs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--model", default="llama-3.1-8b-instant")
    ap.add_argument("--rowchunk-k", type=int, default=10)
    ap.add_argument("--mode", default="direct", choices=["direct", "codegen"])
    # Groq free tier: TPM 6000는 요청당 상한이기도 함 → 컨텍스트 예산을 그 아래로.
    ap.add_argument("--max-context-tokens", type=int, default=1200)
    ap.add_argument("--tpm", type=int, default=6000, help="token/min throttle budget")
    args = ap.parse_args()

    load_env()
    from rag_agent.llm.groq_llm import GroqLLM
    llm = GroqLLM(model_name=args.model)

    queries, tables = load_queries("data/hitab", "dev")
    rng = random.Random(SEED)
    rng.shuffle(queries)
    sample = queries[:args.n]
    table_order = sorted(tables)
    tid2col = {t: i for i, t in enumerate(table_order)}
    print(f"n={len(sample)} pool={len(table_order)} reader={llm.name} mode={args.mode}")

    enc = SentenceTransformerEncoder(model_name=MODEL, batch_size=128)
    qvecs = enc.encode([QUERY_INSTR + q.question for q in sample])
    mtag = MODEL.replace("/", "_")

    # --- 검색 인덱스 3종 (전부 캐시 재사용) -------------------------------
    # original_sent: 셀 문장 → 표 max-pool top-1
    sent_texts, sent_cols = [], []
    for tid in table_order:
        for ch in verbalize_table(tables[tid], "long"):
            sent_texts.append(ch.text)
            sent_cols.append(tid2col[tid])
    sent_vecs = encode_cached(enc, f"dev_sent_long_{mtag}", sent_texts)
    sent_cols = np.array(sent_cols)

    # 컨텍스트는 전 조건 S1 전체표(모든 셀 포함, 콤팩트)로 통일.
    # rag_1t1c: 그 동일 청크를 임베딩·검색·읽기 (정통 1테이블=1청크 RAG).
    full_chunks = {tid: fulltable_chunk(tables[tid], S1) for tid in table_order}
    ft_vecs = encode_cached(enc, f"dev_fulltable_s1_{mtag}",
                            [full_chunks[t].text for t in table_order])

    # rag_rowchunk: 행 청크 (s2)
    row_chunks = []
    for tid in table_order:
        row_chunks.extend(serialize_table(tables[tid], S2))
    rc_vecs = encode_cached(enc, f"dev_rowchunk_{S2}_{mtag}",
                            [c.text for c in row_chunks])

    def top1_table_sent(qi):
        sims = qvecs[qi] @ sent_vecs.T
        ts = np.full(len(table_order), -1e9, dtype=np.float32)
        np.maximum.at(ts, sent_cols, sims)
        return table_order[int(np.argmax(ts))]

    def context_for(arm, qi, q):
        if arm == "oracle":
            return [full_chunks[q.gold_table_id]], q.gold_table_id
        if arm == "original_sent":
            tid = top1_table_sent(qi)
            return [full_chunks[tid]], tid
        if arm == "rag_1t1c":
            tid = table_order[int(np.argmax(qvecs[qi] @ ft_vecs.T))]
            return [full_chunks[tid]], tid
        if arm == "rag_rowchunk":
            sims = qvecs[qi] @ rc_vecs.T
            top = np.argsort(-sims)[:args.rowchunk_k]
            sel = [row_chunks[i] for i in top]
            return sel, None
        raise ValueError(arm)

    # --- 실행 --------------------------------------------------------------
    out = ROOT / "results" / f"verbalize_answer_{args.mode}_{args.model.replace('/','_')}_n{args.n}.json"
    sent_log = []  # (timestamp, est_tokens) — TPM 스로틀용

    def throttle(est_tokens: int):
        now = time.time()
        while True:
            recent = sum(t for ts, t in sent_log if now - ts < 60)
            if recent + est_tokens <= args.tpm * 0.9:
                break
            time.sleep(3)
            now = time.time()
        sent_log.append((now, est_tokens))

    records = []
    acc = {a: [] for a in ARMS}
    t0 = time.time()
    for qi, q in enumerate(sample):
        rec = {"query_id": q.query_id, "question": q.question,
               "gold": q.answer, "gold_table": q.gold_table_id, "arms": {}}
        for arm in ARMS:
            chunks, tid = context_for(arm, qi, q)
            est = min(sum(len(c.text) for c in chunks),
                      args.max_context_tokens * 4) // 4 + 400
            throttle(est)
            try:
                res = answer(q.question, chunks, llm, mode=args.mode,
                             max_context_tokens=args.max_context_tokens)
            except Exception as e:
                rec["arms"][arm] = {"error": str(e)[:200], "correct": False,
                                    "table_hit": False, "operands_covered": None,
                                    "truncated": None, "pred": None}
                acc[arm].append(0)
                continue
            ok = evaluate_answer(res.answer, q.answer)
            acc[arm].append(1 if ok else 0)
            gold_chunks = [c for c in chunks if c.table_id == q.gold_table_id]
            covered = (bool(gold_chunks) and all(
                any(c.covers(op.row, op.col) for c in gold_chunks)
                for op in q.gold_operands)) if q.gold_operands else None
            rec["arms"][arm] = {
                "pred": res.answer if not isinstance(res.answer, float) else round(res.answer, 6),
                "correct": ok,
                "table_hit": (tid == q.gold_table_id) if tid else bool(gold_chunks),
                "operands_covered": covered,
                "truncated": res.context_truncated,
            }
        records.append(rec)
        if (qi + 1) % 10 == 0:
            line = " ".join(f"{a}={np.mean(acc[a]):.3f}" for a in ARMS)
            print(f"  {qi+1}/{len(sample)} ({time.time()-t0:.0f}s) {line}", flush=True)
            out.write_text(json.dumps({"partial": qi + 1, "records": records},
                                      ensure_ascii=False))

    # --- 집계 --------------------------------------------------------------
    summary = {}
    for arm in ARMS:
        v = np.array(acc[arm], dtype=float)
        s = {"accuracy": round(float(v.mean()), 4), "n": len(v)}
        hits = np.array([r["arms"][arm]["table_hit"] for r in records])
        s["table_hit_rate"] = round(float(hits.mean()), 4)
        if hits.any():
            s["acc_given_table_hit"] = round(float(v[hits].mean()), 4)
        cov = [r["arms"][arm]["operands_covered"] for r in records]
        cmask = np.array([c is True for c in cov])
        mmask = np.array([c is False for c in cov])
        if cmask.any():
            s["acc_operands_covered"] = round(float(v[cmask].mean()), 4)
        if mmask.any():
            s["acc_operands_missing"] = round(float(v[mmask].mean()), 4)
            s["n_operands_missing"] = int(mmask.sum())
        summary[arm] = s
        print(arm, s)

    # paired bootstrap: ours − 각 RAG
    rng2 = np.random.default_rng(SEED)
    idx = rng2.integers(0, len(sample), size=(2000, len(sample)))
    for base in ("rag_1t1c", "rag_rowchunk"):
        a, b = np.array(acc["original_sent"], float), np.array(acc[base], float)
        d = a[idx].mean(1) - b[idx].mean(1)
        lo, hi = np.percentile(d, [2.5, 97.5])
        summary[f"delta_original_vs_{base}"] = {
            "delta": round(float(a.mean() - b.mean()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
        }
        print(f"Δ original_sent − {base} = {a.mean()-b.mean():+.4f} [{lo:+.4f},{hi:+.4f}]")

    out.write_text(json.dumps({
        "config": vars(args) | {"reader": llm.name, "embedder": MODEL, "seed": SEED,
                                "context_serialization": "S1 fulltable (oracle/original/1t1c), S2 rowchunks"},
        "summary": summary, "records": records,
    }, indent=2, ensure_ascii=False))
    print("saved →", out)


if __name__ == "__main__":
    main()
