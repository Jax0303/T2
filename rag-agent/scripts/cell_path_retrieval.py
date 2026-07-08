#!/usr/bin/env python3
"""저장된 cell_paths를 코퍼스로 한 *셀 단위* 검색(BM25/dense/hybrid)의 operand recall@k.

기존 operand_rag_eval 은 행(row) 청크 단위, 기존 cell_retrieval_eval 은 dense-only +
값매칭 gold 였다. 여기선 교수님 메타(data/table_meta 의 cell_paths = 셀별 헤더경로+값)를
*검색 인덱스*로 쓰고, bench 의 gold_operands 를 정답 셀로 써서 recall@k 를 잰다.

방법: BM25(lexical), dense(임베딩 cosine), hybrid(RRF k=60) — HybridRetriever 재사용.
지표: recall@{1,3,5,10} = gold operand 셀이 top-k 안에 들어온 비율. LLM 미사용 → 빠름.

사용: .venv/bin/python scripts/cell_path_retrieval.py --benches hitab,finqa,wikisql --n 50 --device cuda
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_agent.bench import registry                                   # noqa: E402
from rag_agent.bench.schema import Chunk                               # noqa: E402
from rag_agent.retrieve.operand_retriever import HybridRetriever, _tok, operand_recall  # noqa: E402
from rag_agent.query.operand_decomposer import Embedder                # noqa: E402
from scripts.build_table_meta import safe_name                         # noqa: E402

SEED = 42
KS = (1, 3, 5, 10, 20, 50)


def cell_chunks(meta: dict, tid: str):
    """저장된 cell_paths → 셀별 Chunk (text=헤더경로, rows=[r], cols=[c])."""
    chunks = []
    for cp in meta.get("cell_paths", []):
        text = cp.get("path") or str(cp.get("value", ""))
        chunks.append(Chunk(table_id=tid, chunk_id=f"{cp['row']}_{cp['col']}",
                            text=text, rows=[cp["row"]], cols=[cp["col"]]))
    return chunks


def rank(scores):
    return sorted(range(len(scores)), key=lambda i: -scores[i])


def rrf(ranks_a, ranks_b, k=60):
    score = {}
    for r, i in enumerate(ranks_a):
        score[i] = score.get(i, 0.0) + 1.0 / (k + r + 1)
    for r, i in enumerate(ranks_b):
        score[i] = score.get(i, 0.0) + 1.0 / (k + r + 1)
    return sorted(score, key=lambda i: -score[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", default="hitab,finqa,wikisql")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--meta-store", default="data/table_meta")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/cell_retrieval/cell_path_summary.json")
    args = ap.parse_args()

    embedder = Embedder(args.embed_model, device=args.device)
    summary = {"config": {"embed_model": args.embed_model, "n": args.n, "seed": SEED,
                          "unit": "cell", "methods": ["bm25", "dense", "hybrid"],
                          "gold": "bench.gold_operands"}, "results": {}}

    for bench in [b.strip() for b in args.benches.split(",") if b.strip()]:
        queries, tables = registry.load(bench, max_samples=None)
        rng = random.Random(SEED)
        pool = [q for q in queries if q.gold_operands and q.gold_table_id in tables]
        rng.shuffle(pool)
        chosen = pool[:args.n]
        metadir = ROOT / args.meta_store / bench

        methods = ("bm25", "dense", "hybrid")
        agg = {m: {k: [] for k in KS} for m in methods}        # recall@k (부분)
        full = {m: {k: [] for k in KS} for m in methods}       # full-set recall@k (전부)
        mrr = {m: [] for m in methods}                          # 첫 정답 셀의 1/rank
        retr_cache = {}
        n_used = 0
        for q in chosen:
            tid = q.gold_table_id
            mp = metadir / f"{safe_name(tid)}.json"
            if not mp.exists():
                continue
            meta = json.loads(mp.read_text())
            if tid not in retr_cache:
                ch = cell_chunks(meta, tid)
                retr_cache[tid] = (ch, HybridRetriever(ch, embedder)) if ch else None
            if retr_cache[tid] is None:
                continue
            chunks, retr = retr_cache[tid]
            gold = {(o.row, o.col) for o in q.gold_operands}
            if not gold:
                continue
            n_used += 1

            bm25_rank = rank(retr._bm25.get_scores(_tok(q.question)))
            qv = embedder.encode([q.question])[0]
            dense_rank = rank((retr._emb @ qv).tolist())
            hyb_rank = rrf(bm25_rank, dense_rank)

            for m, rk in (("bm25", bm25_rank), ("dense", dense_rank), ("hybrid", hyb_rank)):
                hit_cells = [(chunks[i].rows[0], chunks[i].cols[0]) for i in rk]
                # recall@k (부분) + full-set recall@k (전부)
                for k in KS:
                    found = gold & set(hit_cells[:k])
                    agg[m][k].append(len(found) / len(gold))
                    full[m][k].append(1.0 if len(found) == len(gold) else 0.0)
                # MRR: 첫 정답 셀이 등장하는 순위
                rr = 0.0
                for pos, cell in enumerate(hit_cells, 1):
                    if cell in gold:
                        rr = 1.0 / pos
                        break
                mrr[m].append(rr)

        def avg(xs):
            return round(sum(xs) / len(xs), 4) if xs else None
        res = {m: {"recall@k": {str(k): avg(agg[m][k]) for k in KS},
                   "fullset_recall@k": {str(k): avg(full[m][k]) for k in KS},
                   "MRR": avg(mrr[m])} for m in methods}
        res["n_eval"] = n_used
        summary["results"][bench] = res
        print(f"[{bench}] n={n_used}  " +
              "  ".join(f"{m}: R@5={res[m]['recall@k']['5']} "
                       f"full@5={res[m]['fullset_recall@k']['5']} MRR={res[m]['MRR']}"
                       for m in methods), flush=True)

    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n=== recall@k (cell-level retrieval over stored cell_paths) ===", flush=True)
    print(json.dumps(summary["results"], ensure_ascii=False, indent=2), flush=True)
    print(f"→ {outp}", flush=True)


if __name__ == "__main__":
    main()
