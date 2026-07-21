#!/usr/bin/env python3
"""셀 누락률(cell omission) 측정 — "표를 맞게 찾았으면 누락 셀도 없어야 한다" 검증.

정의:
  누락률 = 1 − (gold operand 셀이 리더 컨텍스트에 포함된 비율)
           단, gold 표가 top-k 검색 결과에 든 쿼리 조건부(표는 맞게 찾은 경우만).

조건:
  * rowchunk_s1 / rowchunk_s2 — 일반 VDB RAG: 표를 행 단위 청크로 쪼개 전체 코퍼스
    (540표의 모든 행 청크)에서 top-k 청크 검색. 컨텍스트 = top-k 청크.
    operand (r,c)가 gold 표의 검색된 청크에 덮이지 않으면 누락.
  * original_sent_long — 원본-보관 베이스: 셀 문장(sent_long, 캐시)으로 표를 top-k
    식별 → 컨텍스트 = 통짜 원본 표. gold 표가 top-k에 들면 모든 셀 포함 = 누락 0
    (by construction — 그 0을 실측으로 찍는 것이 목적).

지표 (k ∈ {1,5,10,20}):
  table_found@k   gold 표가 top-k(청크 조건은 "gold 표의 청크 ≥1개")에 든 비율
  miss_rate@k     조건부 평균 operand 누락률 (macro, per-query)
  any_miss@k      조건부, operand가 하나라도 누락된 쿼리 비율
  ctx_chars@k     조건부 평균 컨텍스트 크기(문자) — 공정성 캐비앳용

사용: /usr/bin/python3 scripts/cell_omission_eval.py
"""
from __future__ import annotations

import json
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

SEED = 42
KS = (1, 5, 10, 20)
MODEL = "BAAI/bge-small-en-v1.5"
QUERY_INSTR = "Represent this sentence for searching relevant passages: "
CACHE_DIR = ROOT / "data" / "verbalize_cache"


def encode_cached(enc, tag: str, texts):
    path = CACHE_DIR / f"{tag}.npy"
    if path.exists():
        vecs = np.load(path)
        if len(vecs) == len(texts):
            return vecs
    t0 = time.time()
    vecs = enc.encode(texts)
    print(f"    encoded {len(texts)} in {time.time()-t0:.1f}s")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, vecs)
    return vecs


def topk_indices(qvecs, cvecs, k, block=256):
    out = np.zeros((len(qvecs), k), dtype=np.int64)
    for s in range(0, len(qvecs), block):
        sims = qvecs[s:s + block] @ cvecs.T
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        row = np.take_along_axis(sims, part, axis=1)
        out[s:s + block] = np.take_along_axis(part, np.argsort(-row, axis=1), axis=1)
    return out


def eval_rowchunk(scheme_name, scheme, tables, queries, qvecs, enc):
    chunks = []
    for tid in sorted(tables):
        chunks.extend(serialize_table(tables[tid], scheme))
    texts = [c.text for c in chunks]
    print(f"[{scheme_name}] chunks={len(chunks)}")
    cvecs = encode_cached(enc, f"dev_rowchunk_{scheme}_{MODEL.replace('/','_')}", texts)
    top = topk_indices(qvecs, cvecs, max(KS))

    res = {}
    for k in KS:
        found, miss_rates, any_miss, ctx = [], [], [], []
        for qi, q in enumerate(queries):
            if not q.gold_operands:
                continue
            sel = [chunks[i] for i in top[qi, :k]]
            gold_chunks = [c for c in sel if c.table_id == q.gold_table_id]
            if not gold_chunks:
                found.append(0)
                continue
            found.append(1)
            missed = sum(1 for op in q.gold_operands
                         if not any(c.covers(op.row, op.col) for c in gold_chunks))
            miss_rates.append(missed / len(q.gold_operands))
            any_miss.append(1 if missed else 0)
            ctx.append(sum(len(c.text) for c in sel))
        res[k] = {
            "table_found": round(float(np.mean(found)), 4),
            "n_cond": len(miss_rates),
            "miss_rate": round(float(np.mean(miss_rates)), 4),
            "any_miss": round(float(np.mean(any_miss)), 4),
            "ctx_chars": int(np.mean(ctx)),
        }
        print(f"  k={k:>2} found={res[k]['table_found']:.3f} miss_rate={res[k]['miss_rate']:.4f} "
              f"any_miss={res[k]['any_miss']:.4f} ctx={res[k]['ctx_chars']}")
    return res


def eval_original(tables, queries, qvecs, enc):
    """sent_long 포인터로 표 top-k 식별 → 통짜 원본 컨텍스트."""
    texts, tids = [], []
    for tid in sorted(tables):
        for ch in verbalize_table(tables[tid], "long"):
            texts.append(ch.text)
            tids.append(tid)
    cvecs = encode_cached(enc, f"dev_sent_long_{MODEL.replace('/','_')}", texts)
    table_order = sorted(tables)
    tid2col = {t: i for i, t in enumerate(table_order)}
    ccol = np.array([tid2col[t] for t in tids])
    full_len = {tid: len(fulltable_chunk(tables[tid], S1).text) for tid in table_order}

    # 표 점수 = 소속 문장 max → 표 랭킹
    res = {}
    n_t = len(table_order)
    tscore_top = np.zeros((len(qvecs), max(KS)), dtype=np.int64)
    for s in range(0, len(qvecs), 256):
        sims = qvecs[s:s + 256] @ cvecs.T
        ts = np.full((sims.shape[0], n_t), -1e9, dtype=np.float32)
        np.maximum.at(ts.T, ccol, sims.T)
        tscore_top[s:s + 256] = np.argsort(-ts, axis=1)[:, :max(KS)]

    print("[original_sent_long] pointer=cell sentences, context=full original table")
    for k in KS:
        found, miss_rates, any_miss, ctx = [], [], [], []
        for qi, q in enumerate(queries):
            if not q.gold_operands:
                continue
            sel_tids = [table_order[j] for j in tscore_top[qi, :k]]
            if q.gold_table_id not in sel_tids:
                found.append(0)
                continue
            found.append(1)
            # 원본 표 전체가 컨텍스트: 모든 (r,c)가 표 안에 있으므로 누락 불가능.
            # 정의대로 실측: 표 좌표 범위 검사.
            t = tables[q.gold_table_id]
            missed = sum(1 for op in q.gold_operands
                         if not (0 <= op.row < t.n_rows and 0 <= op.col < t.n_cols))
            miss_rates.append(missed / len(q.gold_operands))
            any_miss.append(1 if missed else 0)
            ctx.append(sum(full_len[t2] for t2 in sel_tids))
        res[k] = {
            "table_found": round(float(np.mean(found)), 4),
            "n_cond": len(miss_rates),
            "miss_rate": round(float(np.mean(miss_rates)), 4),
            "any_miss": round(float(np.mean(any_miss)), 4),
            "ctx_chars": int(np.mean(ctx)),
        }
        print(f"  k={k:>2} found={res[k]['table_found']:.3f} miss_rate={res[k]['miss_rate']:.4f} "
              f"any_miss={res[k]['any_miss']:.4f} ctx={res[k]['ctx_chars']}")
    return res


def main():
    queries, tables = load_queries("data/hitab", "dev")
    n_op = sum(1 for q in queries if q.gold_operands)
    print(f"queries={len(queries)} (operand 있는 쿼리 {n_op}) tables={len(tables)}")
    enc = SentenceTransformerEncoder(model_name=MODEL, batch_size=128)
    qvecs = enc.encode([QUERY_INSTR + q.question for q in queries])

    out = {
        "config": {"model": MODEL, "split": "dev", "n_queries": len(queries),
                   "n_operand_queries": n_op, "table_pool": len(tables), "seed": SEED,
                   "definition": "miss_rate = mean fraction of gold operand cells absent "
                                 "from reader context, conditional on gold table in top-k"},
        "original_sent_long": eval_original(tables, queries, qvecs, enc),
        "rowchunk_s1": eval_rowchunk("rowchunk_s1", S1, tables, queries, qvecs, enc),
        "rowchunk_s2": eval_rowchunk("rowchunk_s2", S2, tables, queries, qvecs, enc),
    }
    path = ROOT / "results" / "cell_omission_dev_bge-small.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("saved →", path)


def _no_args() -> None:
    """This script takes no options. Without a parser, argparse-style flags are
    silently ignored and the full experiment runs anyway — which is how a bare
    ``--help`` sweep silently regenerated committed artifacts."""
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()


if __name__ == "__main__":
    _no_args()
    main()
