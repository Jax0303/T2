#!/usr/bin/env python3
"""원본-표-보관 베이스 + 셀 문장화(캡션+헤더) 임베딩 검색 vs 1테이블-1청크 RAG.

베이스 아키텍처: 원본 표는 벡터DB에 넣지 않고 그대로 보관한다(원본 스토어).
임베딩되는 것은 표 자체가 아니라 표를 "가리키는" 텍스트이고, 검색 결과는
table_id → 원본 표 반환이다. 조건별로 그 텍스트만 달라진다:

  fulltable_s1 / fulltable_s2 — 1테이블=1청크=1벡터 (전형적 RAG 대조군; 긴 표는
                                 인코더 max_seq에서 잘려나감 = 고전 RAG의 약점)
  sent_short / sent_medium / sent_long — 셀당 1문장 (캡션+헤더 문장화, 길이 ablation);
                                 표 점수 = 소속 문장 max cosine

지표:
  * 표 단위 R@1/5/10, MRR@10 (전 조건 공통; gold table이 top-k에 드는가)
  * 문장 조건 한정 cell-hit@1/5/10: top-k 문장 중 gold operand 셀에서 나온
    문장이 있는가 (검색이 답 셀까지 바로 짚는지)
  * paired bootstrap: 각 조건 R@1 − fulltable_s1 R@1 의 95% CI

사용:
  /usr/bin/python3 scripts/verbalize_retrieval_eval.py --encoder hashing --n 100   # smoke
  /usr/bin/python3 scripts/verbalize_retrieval_eval.py                              # full dev
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_agent.bench.hitab import load_queries  # noqa: E402
from rag_agent.serialize.serializers import fulltable_chunk, S1, S2  # noqa: E402
from rag_agent.serialize.verbalize import verbalize_table, STYLES  # noqa: E402

SEED = 42
KS = (1, 5, 10)
MODEL = "BAAI/bge-small-en-v1.5"
QUERY_INSTR = "Represent this sentence for searching relevant passages: "
CACHE_DIR = ROOT / "data" / "verbalize_cache"


def build_corpus(tables: dict, cond: str):
    """조건 → (chunk 텍스트 리스트, table_id 리스트, (row,col) 리스트)."""
    texts, tids, cells = [], [], []
    for tid in sorted(tables):
        t = tables[tid]
        if cond in ("fulltable_s1", "fulltable_s2"):
            ch = fulltable_chunk(t, S1 if cond.endswith("s1") else S2)
            texts.append(ch.text)
            tids.append(tid)
            cells.append(None)
        else:
            style = cond.removeprefix("sent_")
            for ch in verbalize_table(t, style):
                texts.append(ch.text)
                tids.append(tid)
                cells.append((ch.rows[0], ch.cols[0]))
    return texts, tids, cells


def get_encoder(kind: str):
    if kind == "hashing":
        from rag_agent.retrieve.encoders import HashingEncoder
        enc = HashingEncoder()
        return enc, ""
    from rag_agent.retrieve.encoders import SentenceTransformerEncoder
    return SentenceTransformerEncoder(model_name=kind, batch_size=128), QUERY_INSTR


def encode_cached(enc, tag: str, texts, use_cache: bool):
    safe = tag.replace("/", "_")
    path = CACHE_DIR / f"{safe}.npy"
    if use_cache and path.exists():
        vecs = np.load(path)
        if len(vecs) == len(texts):
            return vecs
    t0 = time.time()
    vecs = enc.encode(texts)
    print(f"    encoded {len(texts)} texts in {time.time()-t0:.1f}s")
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(path, vecs)
    return vecs


def eval_condition(qvecs, cvecs, ctids, ccells, table_order, queries, block=256):
    """블록 단위 Q·C^T → 표 점수(max-pool) 랭킹 + top-10 문장 인덱스."""
    n_q = len(qvecs)
    tid2col = {t: i for i, t in enumerate(table_order)}
    cchunk_tcol = np.array([tid2col[t] for t in ctids])
    hit = {k: np.zeros(n_q, dtype=bool) for k in KS}
    rr = np.zeros(n_q)
    cell_hit = {k: np.zeros(n_q, dtype=bool) for k in KS}
    has_cells = ccells[0] is not None

    for s in range(0, n_q, block):
        sims = qvecs[s:s + block] @ cvecs.T          # (b, n_chunks)
        # 표 점수 = 소속 청크 max
        tscores = np.full((sims.shape[0], len(table_order)), -1e9, dtype=np.float32)
        np.maximum.at(tscores.T, cchunk_tcol, sims.T)
        order = np.argsort(-tscores, axis=1)[:, :max(KS)]
        top_sent = np.argsort(-sims, axis=1)[:, :max(KS)] if has_cells else None

        for bi in range(sims.shape[0]):
            qi = s + bi
            q = queries[qi]
            gold_col = tid2col[q.gold_table_id]
            ranked = order[bi]
            pos = np.where(ranked == gold_col)[0]
            rank = pos[0] + 1 if len(pos) else None
            for k in KS:
                hit[k][qi] = rank is not None and rank <= k
            if rank is not None:
                rr[qi] = 1.0 / rank
            if has_cells and q.gold_operands:
                gold_cells = {(q.gold_table_id, op.row, op.col) for op in q.gold_operands}
                for j, si in enumerate(top_sent[bi]):
                    if (ctids[si], *ccells[si]) in gold_cells:
                        for k in KS:
                            if j < k:
                                cell_hit[k][qi] = True
                        break
    return hit, rr, cell_hit


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot=2000):
    """평균차 a−b 의 95% CI (paired, seed 고정)."""
    rng = np.random.default_rng(SEED)
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = (a[idx].mean(axis=1) - b[idx].mean(axis=1))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default=MODEL, help="ST model name or 'hashing'")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=0, help="query subsample (0=all)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    queries, tables = load_queries("data/hitab", args.split)
    if args.n:
        import random
        rng = random.Random(SEED)
        rng.shuffle(queries)
        queries = queries[:args.n]
        # 표 풀은 항상 split 전체 gold pool 유지 (distractor 축소 금지)
    table_order = sorted(tables)
    print(f"queries={len(queries)} table_pool={len(table_order)} encoder={args.encoder}")

    enc, qinstr = get_encoder(args.encoder)
    use_cache = not args.no_cache and args.encoder != "hashing"
    enc_tag = args.encoder.replace("/", "_")

    qvecs = enc.encode([qinstr + q.question for q in queries])

    conds = ["fulltable_s1", "fulltable_s2"] + [f"sent_{s}" for s in STYLES]
    results = {}
    per_query_r1 = {}
    for cond in conds:
        texts, ctids, ccells = build_corpus(tables, cond)
        lens = [len(t) for t in texts]
        print(f"[{cond}] chunks={len(texts)} char len p50={int(np.median(lens))} p95={int(np.percentile(lens,95))}")
        cvecs = encode_cached(enc, f"{args.split}_{cond}_{enc_tag}", texts, use_cache)
        hit, rr, cell_hit = eval_condition(qvecs, cvecs, ctids, ccells, table_order, queries)
        n_op = sum(1 for q in queries if q.gold_operands)
        res = {
            "chunks": len(texts),
            "chunk_char_p50": int(np.median(lens)),
            **{f"table_R@{k}": round(float(hit[k].mean()), 4) for k in KS},
            "MRR@10": round(float(rr.mean()), 4),
        }
        if ccells[0] is not None and n_op:
            op_mask = np.array([bool(q.gold_operands) for q in queries])
            res.update({f"cell_hit@{k}": round(float(cell_hit[k][op_mask].mean()), 4) for k in KS})
            res["n_operand_queries"] = n_op
        results[cond] = res
        per_query_r1[cond] = hit[1].astype(float)
        print("   ", {k: v for k, v in res.items() if k != "chunks"})

    for base in ("fulltable_s1", "fulltable_s2"):
        for cond in conds:
            if cond == base:
                continue
            lo, hi = paired_bootstrap(per_query_r1[cond], per_query_r1[base])
            d = float(per_query_r1[cond].mean() - per_query_r1[base].mean())
            results[cond][f"delta_R@1_vs_{base}"] = round(d, 4)
            results[cond][f"delta_ci95_vs_{base}"] = [round(lo, 4), round(hi, 4)]
            sig = "*" if lo > 0 or hi < 0 else " "
            print(f"Δ R@1 {cond:>12} − {base} = {d:+.4f}  CI95[{lo:+.4f},{hi:+.4f}] {sig}")

    out = Path(args.out) if args.out else ROOT / "results" / f"verbalize_retrieval_{args.split}_{enc_tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"encoder": args.encoder, "split": args.split,
                   "n_queries": len(queries), "table_pool": len(table_order),
                   "seed": SEED, "query_instr": bool(qinstr)},
        "results": results,
    }, indent=2, ensure_ascii=False))
    print("saved →", out)


if __name__ == "__main__":
    main()
