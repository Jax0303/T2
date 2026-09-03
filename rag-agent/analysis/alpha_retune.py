#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PREREG-2026-09-02-alpha-retune.md — 색인 한 번, α 전부.

`cell_rank_dump.py`를 α마다 돌리면 같은 색인을 8번 짓는다. 질의당 dense/BM25 점수는
α와 무관하므로 한 번 계산하고 정렬만 다시 한다 (`--table-prior`가 이미 쓰는 방식).

산술이 `cell_rank_dump`와 같아야 하므로 α=0.7 칸을 그 덤프와 대조한다: 다르면
`[CONTROL] FAIL`을 찍고 종료 코드 1로 죽는다.

  PYTHONPATH=. .venv/bin/python analysis/alpha_retune.py \
      --split dev --population hitab_dev_lookup_all \
      --embed-model models/bge-cell-ft-x0 --out-dir results/alpha_ft
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "analysis"))

import numpy as np                                                   # noqa: E402
import corpus_dump_vs_cell as cdv                                    # noqa: E402
from cell_rank_dump import cell_texts                                # noqa: E402
from header_path_coverage import load_corpus                         # noqa: E402
from rag_agent.retrieve.encoders import default_encoder              # noqa: E402
from rag_agent.retrieve.hybrid_index import HybridIndex, _minmax     # noqa: E402
from rag_agent.serialization.base import Chunk                       # noqa: E402

ALPHAS = (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
KS = (1, 5, 10, 20, 50)


class A:
    pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--population", default="hitab_dev_lookup_all")
    ap.add_argument("--embed-model", default="models/bge-cell-ft-x0")
    ap.add_argument("--cell-scheme", default="S3c")
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--cache-dir", default=".cache/corpus_dump_vs_cell")
    ap.add_argument("--out-dir", default="results/alpha_ft")
    ap.add_argument("--alphas", type=float, nargs="*", default=None,
                    help="잴 α. 기본은 스윕 격자 전부. test 확정은 사전등록 규칙 3에 "
                         "따라 선택된 α 하나만 줘서 나머지를 아예 계산하지 않는다")
    ap.add_argument("--control", default="",
                    help="α=0.7이 재현해야 하는 cell_rank_dump ranks.jsonl")
    a = ap.parse_args()
    a.dataset, a.seed, a.rhb_question_types, a.rhb_em_only = "hitab", 42, [], False
    global ALPHAS
    if a.alphas:
        ALPHAS = tuple(a.alphas)

    C = load_corpus(a)
    pop = C.queries
    print(f"[corpus] {len(C.tids)} tables / {len(C.cell_owner)} cells | "
          f"[pop] {len(pop)} queries", flush=True)
    chunks = [Chunk(table_id=t, chunk_id=f"c::{t}::{i}:{j}", text=x,
                    scheme=a.cell_scheme, kind="cell")
              for x, (t, i, j) in zip(cell_texts(C, a.cell_scheme), C.cell_owner)]
    enc = cdv._CachedEncoder(default_encoder(model_name=a.embed_model), a.cache_dir,
                             f"hitab_{a.split}_{a.embed_model}")
    t0 = time.time()
    ix = HybridIndex(chunks, encoder=enc, alpha=0.5)
    print(f"[index] built in {time.time() - t0:.0f}s", flush=True)

    pos_of = {c: n for n, c in enumerate(C.cell_owner)}
    by_table = defaultdict(list)
    for n, (t, _i, _j) in enumerate(C.cell_owner):
        by_table[t].append(n)

    recs = {al: [] for al in ALPHAS}
    t0 = time.time()
    for k, q in enumerate(pop, 1):
        bm = ix._bm25_scores(q["question"])
        dn = ix._dense_scores(q["question"])
        nb, nd = _minmax(bm), _minmax(dn)
        gold = [pos_of[g] for g in sorted(q["gold_cells"]) if g in pos_of]
        if not gold:
            continue
        for al in ALPHAS:
            cs = dn if al == 1.0 else bm if al == 0.0 else al * nd + (1 - al) * nb
            order = np.argsort(-cs)
            at = np.empty(len(order), dtype=np.int64)
            at[order] = np.arange(len(order))
            ranks = [int(at[g]) for g in gold]
            # 표 순위 = 셀 투표 (그 표의 최상위 셀 순위), cell_rank_dump와 동일
            seen, trank = {}, None
            for r_, p_ in enumerate(order):
                t_ = C.cell_owner[p_][0]
                if t_ not in seen:
                    seen[t_] = len(seen) + 1
                    if t_ == q["gold_table"]:
                        trank = seen[t_]
                        break
            recs[al].append({"query_id": q["query_id"], "m": len(gold),
                             "gold_table": q["gold_table"], "ranks": ranks,
                             "table_rank_cellvote": trank})
        if k % 100 == 0:
            print(f"  {k}/{len(pop)}  {time.time() - t0:.0f}s", flush=True)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.population}_{a.cell_scheme}_{Path(a.embed_model).name}"
    summary = {}
    for al in ALPHAS:
        rs = recs[al]
        n = len(rs)
        row = {"n": n}
        for kk in KS:
            row[f"setEM@{kk}"] = round(sum(1 for r in rs if max(r["ranks"]) < kk) / n, 4)
            row[f"R@{kk}"] = round(sum(sum(1 for x in r["ranks"] if x < kk)
                                       / len(r["ranks"]) for r in rs) / n, 4)
        row["MRR"] = round(sum(1.0 / (min(r["ranks"]) + 1) for r in rs) / n, 4)
        row["표 recall@1"] = round(sum(1 for r in rs
                                       if r["table_rank_cellvote"] == 1) / n, 4)
        row["표 recall@10"] = round(sum(1 for r in rs if r["table_rank_cellvote"]
                                        and r["table_rank_cellvote"] <= 10) / n, 4)
        summary[str(al)] = row
        with open(out / f"{tag}_a{al}_ranks.jsonl", "w") as fh:
            for r in rs:
                fh.write(json.dumps(r) + "\n")

    ok = None
    if a.control and Path(a.control).exists():
        ref = {r["query_id"]: r for r in map(json.loads, open(a.control))}
        mine = {r["query_id"]: r for r in recs[0.7]}
        ids = sorted(set(ref) & set(mine))
        rs = sum(1 for i in ids if max(ref[i]["ranks"]) < 10) / len(ids)
        ms = sum(1 for i in ids if max(mine[i]["ranks"]) < 10) / len(ids)
        ok = abs(rs - ms) < 1e-9
        print(f"[CONTROL] α=0.7 setEM@10 mine={ms:.4f} ref={rs:.4f} "
              f"-> {'OK' if ok else 'FAIL'}", flush=True)
        summary["_control"] = {"ref": round(rs, 4), "mine": round(ms, 4), "ok": ok}

    json.dump(summary, open(out / f"{tag}_summary.json", "w"), indent=1,
              ensure_ascii=False)
    for al in ALPHAS:
        r = summary[str(al)]
        print(f"  α={al}  setEM@10={r['setEM@10']:.4f}  R@1={r['R@1']:.4f}  "
              f"MRR={r['MRR']:.4f}  표R@1={r['표 recall@1']:.4f}")
    print(f"-> {out}/{tag}_summary.json")
    return 0 if ok is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
