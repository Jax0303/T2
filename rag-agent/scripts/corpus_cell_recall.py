#!/usr/bin/env python3
"""교수님 가설(표+셀 단위, 전체 코퍼스): 라벨링한 셀 메타데이터(헤더경로)에 쿼리를 매핑하면
*전체 표 코퍼스에서* 맞는 표와 셀을 오류 없이 찾는가? — LLM 없이 측정.

cell_path_retrieval.py 는 *정답 표 안에서만* 셀을 찾았다(표 선택은 gold로 가정). 여기선 그
가정을 제거한다: data/table_meta 의 모든 표·모든 셀을 하나의 코퍼스로 인덱싱하고, 쿼리를
그 위에서 검색해 (a) 맞는 표를 찾는지(table recall) (b) 맞는 셀을 찾는지(cell recall)를 동시에 본다.

이 스크립트는 원본 벤치 데이터 + build_table_meta 산출물(data/table_meta/<bench>/*.json)이
있는 환경(랩 머신)에서 실행한다. 먼저:
    python3 scripts/build_table_meta.py --benches hitab,finqa,wikisql
지표:
  table_recall@k  gold 표의 어떤 셀이라도 top-k 안에 든 비율
  cell_recall@k   gold operand 셀이 top-k 안에 든 비율(부분)
  cell_full@k     gold operand 셀 *전부* 가 top-k 안에 든 비율
  MRR             첫 정답 셀의 1/rank
주의(규모): 전체 코퍼스 셀 수가 매우 크다(hier ~수십만). dense 인덱스는 메모리/시간이 크므로
기본 retriever 는 BM25, dense 는 --dense 로 옵트인. 표당 셀 상한 --max-cells-per-table 로 통제.
사용: python3 scripts/corpus_cell_recall.py --bench hitab --n 200 --device cuda
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_agent.bench import registry                       # noqa: E402
from scripts.build_table_meta import safe_name             # noqa: E402

SEED = 42
KS = (1, 3, 5, 10, 20, 50, 100)
_TOK_RE = re.compile(r"[a-z0-9]+")


def tok(text: str):
    return _TOK_RE.findall(text.lower())


def load_corpus_cells(metadir: Path, max_cells: int):
    """모든 표의 셀 메타 → 평탄한 셀 리스트. 반환: cells[i]=(table_id,row,col,text)."""
    cells = []
    for mp in sorted(metadir.glob("*.json")):
        meta = json.loads(mp.read_text())
        tid = meta["table_id"]
        cps = meta.get("cell_paths", [])
        if max_cells and len(cps) > max_cells:
            cps = cps[:max_cells]
        for cp in cps:
            text = cp.get("path") or str(cp.get("value", ""))
            cells.append((tid, cp["row"], cp["col"], text))
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="hitab")
    ap.add_argument("--meta-store", default="data/table_meta")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-cells-per-table", type=int, default=0)
    ap.add_argument("--dense", action="store_true", help="BM25 외에 dense(임베딩) 추가")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/corpus_cell_recall.json")
    args = ap.parse_args()

    metadir = ROOT / args.meta_store / args.bench
    if not metadir.exists():
        sys.exit(f"메타 없음: {metadir} — 먼저 build_table_meta.py 실행")

    cells = load_corpus_cells(metadir, args.max_cells_per_table)
    cell_tid = [c[0] for c in cells]
    cell_rc = [(c[1], c[2]) for c in cells]
    print(f"[corpus_cell_recall:{args.bench}] corpus cells={len(cells)} "
          f"from {len(set(cell_tid))} tables", flush=True)

    from rank_bm25 import BM25Okapi
    t0 = time.time()
    bm25 = BM25Okapi([tok(c[3]) for c in cells])
    print(f"  BM25 built in {time.time()-t0:.0f}s", flush=True)

    emb_mat = embedder = None
    if args.dense:
        from rag_agent.query.operand_decomposer import Embedder
        import numpy as np  # noqa: F401
        embedder = Embedder(args.embed_model, device=args.device)
        t0 = time.time()
        emb_mat = embedder.encode([c[3] for c in cells])
        print(f"  dense matrix {emb_mat.shape} in {time.time()-t0:.0f}s", flush=True)

    queries, tables = registry.load(args.bench, max_samples=None)
    rng = random.Random(SEED)
    pool = [q for q in queries if q.gold_operands and q.gold_table_id in tables]
    rng.shuffle(pool)
    chosen = pool[:args.n]

    methods = ["bm25"] + (["dense", "hybrid"] if args.dense else [])

    def fresh():
        return {"table@k": {k: 0 for k in KS}, "cell@k": {k: 0 for k in KS},
                "cellfull@k": {k: 0 for k in KS}, "mrr": 0.0, "n": 0}
    stats = {m: fresh() for m in methods}

    def rank_desc(scores):
        return sorted(range(len(scores)), key=lambda i: -scores[i])

    def rrf(ra, rb, k=60):
        sc = {}
        for r, i in enumerate(ra):
            sc[i] = sc.get(i, 0.0) + 1.0 / (k + r + 1)
        for r, i in enumerate(rb):
            sc[i] = sc.get(i, 0.0) + 1.0 / (k + r + 1)
        return sorted(sc, key=lambda i: -sc[i])

    import numpy as np
    t0 = time.time()
    for qi, q in enumerate(chosen, 1):
        gold_tid = q.gold_table_id
        gold_cells = {(gold_tid, o.row, o.col) for o in q.gold_operands}
        if not gold_cells:
            continue
        bm25_rank = rank_desc(bm25.get_scores(tok(q.question)))
        ranks = {"bm25": bm25_rank}
        if args.dense:
            qv = embedder.encode([q.question])[0]
            dense_rank = rank_desc((emb_mat @ qv).tolist())
            ranks["dense"] = dense_rank
            ranks["hybrid"] = rrf(bm25_rank, dense_rank)

        for m in methods:
            rk = ranks[m]
            st = stats[m]
            st["n"] += 1
            # 상위 max(KS)만 펼쳐서 (table_id,row,col) 비교
            topcells = [(cell_tid[i], cell_rc[i][0], cell_rc[i][1]) for i in rk[:max(KS)]]
            toptables = [cell_tid[i] for i in rk[:max(KS)]]
            for k in KS:
                # table recall: gold 표가 top-k 셀들의 표 집합에 있나
                if gold_tid in set(toptables[:k]):
                    st["table@k"][k] += 1
                found = gold_cells & set(topcells[:k])
                if found:
                    st["cell@k"][k] += 1
                if len(found) == len(gold_cells):
                    st["cellfull@k"][k] += 1
            # MRR over first gold cell
            rr = 0.0
            for pos, c in enumerate(topcells, 1):
                if c in gold_cells:
                    rr = 1.0 / pos
                    break
            st["mrr"] += rr
        if qi % 25 == 0 or qi == len(chosen):
            print(f"  {qi}/{len(chosen)} ({time.time()-t0:.0f}s)", flush=True)

    results = {}
    for m in methods:
        st = stats[m]
        n = st["n"] or 1
        results[m] = {
            "n_eval": st["n"],
            "table_recall@k": {str(k): round(st["table@k"][k] / n, 4) for k in KS},
            "cell_recall@k": {str(k): round(st["cell@k"][k] / n, 4) for k in KS},
            "cell_full@k": {str(k): round(st["cellfull@k"][k] / n, 4) for k in KS},
            "MRR": round(st["mrr"] / n, 4),
        }
        print(f"[{m}] table@10={results[m]['table_recall@k']['10']} "
              f"cell@10={results[m]['cell_recall@k']['10']} "
              f"cellfull@10={results[m]['cell_full@k']['10']} MRR={results[m]['MRR']}",
              flush=True)

    summary = {"config": {"bench": args.bench, "retriever": methods, "llm": None,
                          "n": args.n, "seed": SEED, "corpus_cells": len(cells),
                          "n_tables": len(set(cell_tid))}, "results": results}
    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n→ {outp}", flush=True)


if __name__ == "__main__":
    main()
