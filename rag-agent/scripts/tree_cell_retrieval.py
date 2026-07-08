#!/usr/bin/env python3
"""Task 2/3: 트리 구조(분해형) 셀 검색 vs flatten 문자열 검색 비교 (LLM 없음).

flat        : 셀을 "행경로 > 열경로" 한 문자열로 잇고 BM25 한 번 매칭(기존).
compositional: 질의를 행 헤더축(left_path)과 열 헤더축(top_path)에 *따로* 매칭하고
              점수를 합산 → 트리에서 (행 서브트리 × 열 서브트리) 교차로 셀을 좁힘.
              교수님의 "헤더 계층 트리를 제대로 써라"의 구현.

코퍼스 = 전체 표의 모든 데이터 셀. 표는 registry 로더에서 직접(메타 파일 불필요).
gold = bench.gold_operands (offset 좌표 수정 반영). 지표: cell R@k / full@k / oracle.
사용: python3 scripts/tree_cell_retrieval.py --bench hitab --n 300
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_agent.bench import registry                       # noqa: E402

SEED = 42
KS = (1, 3, 5, 10, 20, 50)
_TOK = re.compile(r"[a-z0-9]+")
tok = lambda s: _TOK.findall(s.lower())                    # noqa: E731


def build_corpus(tables):
    """모든 표의 데이터 셀 → (tid,r,c, row_str, col_str, flat_str)."""
    cells = []
    for tid, tb in tables.items():
        if not tb.data:
            continue
        for r in range(tb.n_rows):
            rp = " > ".join(p for p in tb.left_paths[r] if p) if r < len(tb.left_paths) else ""
            for c in range(tb.n_cols):
                cp = " > ".join(p for p in tb.top_paths[c] if p) if c < len(tb.top_paths) else ""
                flat = (rp + " " + cp).strip()
                cells.append((tid, r, c, rp, cp, flat))
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="hitab")
    ap.add_argument("--data-dir", default="/mnt/d/hart_data/hitab/HiTab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="results/tree_cell_retrieval.json")
    args = ap.parse_args()

    from rank_bm25 import BM25Okapi
    queries, tables = registry.load(args.bench, split=args.split,
                                    data_dir=args.data_dir, max_samples=None)
    cells = build_corpus(tables)
    keys = [(c[0], c[1], c[2]) for c in cells]
    print(f"[tree_cell:{args.bench}] cells={len(cells)} tables={len(tables)}", flush=True)

    bm_flat = BM25Okapi([tok(c[5]) for c in cells])
    bm_row = BM25Okapi([tok(c[3]) for c in cells])
    bm_col = BM25Okapi([tok(c[4]) for c in cells])

    pool = [q for q in queries if q.gold_operands and q.gold_table_id in tables]
    random.Random(SEED).shuffle(pool)
    chosen = pool[:args.n]

    def rank_flat(qtext):
        sc = bm_flat.get_scores(tok(qtext))
        return sorted(range(len(cells)), key=lambda i: -sc[i])

    def rank_comp(qtext):
        qt = tok(qtext)
        sr = bm_row.get_scores(qt)
        sc = bm_col.get_scores(qt)
        comb = sr + sc                       # numpy elementwise: 행축 + 열축
        return sorted(range(len(cells)), key=lambda i: -comb[i])

    def evaluate(ranker, use_oracle=False):
        f = {k: 0 for k in KS}; full = {k: 0 for k in KS}; orc = 0; n = 0
        for q in chosen:
            gold = {(q.gold_table_id, o.row, o.col) for o in q.gold_operands}
            if not gold:
                continue
            n += 1
            rk = ranker(q.question)
            top = [keys[i] for i in rk[:max(KS)]]
            for k in KS:
                if gold & set(top[:k]):
                    f[k] += 1
                if gold <= set(top[:k]):
                    full[k] += 1
            if use_oracle:
                # 정답 셀의 자기 경로를 질의로 → 1등인가 (식별력 천장)
                o = q.gold_operands[0]
                cell = next(c for c in cells if (c[0], c[1], c[2]) == (q.gold_table_id, o.row, o.col))
                rk2 = ranker(cell[5])
                if keys[rk2[0]] == (q.gold_table_id, o.row, o.col):
                    orc += 1
        out = {"n": n,
               "cell_recall@k": {str(k): round(f[k] / n, 4) for k in KS},
               "cell_full@k": {str(k): round(full[k] / n, 4) for k in KS}}
        if use_oracle:
            out["oracle_R@1"] = round(orc / n, 4)
        return out

    results = {}
    for name, ranker in (("flat", rank_flat), ("compositional", rank_comp)):
        print(f"  evaluating {name} ...", flush=True)
        results[name] = evaluate(ranker, use_oracle=True)
        r = results[name]
        print(f"  [{name}] cell R@1={r['cell_recall@k']['1']} "
              f"R@10={r['cell_recall@k']['10']} full@10={r['cell_full@k']['10']} "
              f"oracle@1={r['oracle_R@1']}", flush=True)

    summary = {"config": {"bench": args.bench, "split": args.split, "n": args.n,
                          "seed": SEED, "retriever": "bm25", "llm": None,
                          "corpus_cells": len(cells)}, "results": results}
    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n→ {outp}", flush=True)


if __name__ == "__main__":
    main()
