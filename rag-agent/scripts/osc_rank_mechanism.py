#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Why all-or-nothing OSC falls at m>=2 -- measured from a fresh retrieval run.

Reads no committed result. Builds the corpus, builds the index, ranks every
cell for every query, and records where each gold cell actually landed.

The claim under test is mechanical. A budget admits the top C cells, so

    OSC(q) = 1  iff  max(rank of gold cell) < C

-- the single worst gold cell decides the query, and the other m-1 are
irrelevant once it is placed. Two different things can then push OSC down as m
grows, and they call for opposite fixes:

  STRUCTURAL  the per-cell rank distribution is unchanged, but taking a max over
              m draws instead of 1 reaches further into its tail. Nothing about
              retrieval got worse. Fixed by budget or by a metric that is not
              all-or-nothing.
  RETRIEVAL   the per-cell ranks themselves are worse at higher m -- multi-cell
              questions are genuinely harder to match. Fixed by the retriever.

Separating them needs a counterfactual: resample m ranks from the m=1 pool and
ask what OSC that alone would produce. The gap between that and the measured
OSC is the part structure does not explain.

  PYTHONPATH=.:scripts .venv/bin/python scripts/osc_rank_mechanism.py
"""
from __future__ import annotations

import json
import random
import sys
from statistics import median

BUDGET, ALPHA, SEED, N_QUERIES = 512, 0.7, 42, 400
TRIALS = 20000


def main() -> int:
    import numpy as np

    from baseline_comparison_llm import Budget
    from corpus_dump_vs_cell import (HybridIndex, _CachedEncoder, cell_text,
                                     multihiertt_corpus, positions, rank_of)
    from rag_agent.retrieve.encoders import default_encoder
    from rag_agent.serialization.base import Chunk

    EMBED = "BAAI/bge-small-en-v1.5"
    C = multihiertt_corpus(N_QUERIES, SEED)
    print(f"[corpus] 표 {len(C.md_lines)} / 셀 {len(C.cell_text)}", flush=True)

    # S3c == S2 here: MultiHiertt ships no table titles, so the scheme collapses
    text = [cell_text(rp, cp, v, "S2") for rp, cp, v in C.cell_paths]
    chunks = [Chunk(table_id=t, chunk_id=f"c::{t}::{i}:{j}", text=x,
                    scheme="S3c", kind="cell")
              for x, (t, i, j) in zip(text, C.cell_owner)]
    enc = _CachedEncoder(default_encoder(model_name=EMBED), ".cache",
                         f"multihiertt_dev_{EMBED}")
    idx = HybridIndex(chunks, encoder=enc, alpha=0.5)
    bud = Budget(EMBED)
    tok = [bud.count(x) for x in text]
    where = {o: i for i, o in enumerate(C.cell_owner)}

    rows = []
    for n, q in enumerate(C.queries, 1):
        order = rank_of(idx, q["question"], ALPHA)
        pos = positions(order)
        gold = [where[g] for g in q["gold_cells"] if g in where]
        if len(gold) != len(q["gold_cells"]):
            continue                       # never happens on this corpus; guard anyway
        # how many cells the budget actually admits for THIS query
        used = fits = 0
        for i in order:
            if used + tok[i] > BUDGET:
                break
            used += tok[i]
            fits += 1
        ranks = sorted(int(pos[g]) for g in gold)
        rows.append({"m": len(gold), "ranks": ranks, "cutoff": fits,
                     "osc": int(ranks[-1] < fits)})
        if n % 100 == 0:
            print(f"  {n}/{len(C.queries)}", flush=True)

    json.dump(rows, open("results/osc_rank_mechanism.json", "w"))
    cut = int(median(r["cutoff"] for r in rows))
    print(f"\n예산 {BUDGET}토큰이 받는 셀 수: 중앙 {cut}개 "
          f"(코퍼스 {len(C.cell_text)}셀 중 {cut / len(C.cell_text):.2%})\n")

    pool = [x for r in rows if r["m"] == 1 for x in r["ranks"]]   # m=1 랭크 분포
    rng = random.Random(SEED)

    print("=== 실측 랭크와 구조만의 예측 ===")
    print(f"{'m':>3} {'n':>5} {'OSC':>7} {'구조만':>7} {'차이':>7} "
          f"{'최선셀랭크':>10} {'최악셀랭크':>10} {'셀당recall':>10}")
    print("-" * 70)
    for m in (1, 2, 3, 4):
        grp = [r for r in rows if (r["m"] == m if m < 4 else r["m"] >= 4)]
        if not grp:
            continue
        osc = sum(r["osc"] for r in grp) / len(grp)
        mm = median(r["m"] for r in grp)
        # 구조만: m=1 의 랭크 분포에서 m개를 뽑아 전부 cutoff 안에 드는 비율
        hit = 0
        for _ in range(TRIALS):
            c = rng.choice(grp)["cutoff"]
            hit += all(rng.choice(pool) < c for _ in range(int(mm)))
        struct = hit / TRIALS
        pc = sum(sum(x < r["cutoff"] for x in r["ranks"]) / r["m"] for r in grp) / len(grp)
        best = median(r["ranks"][0] for r in grp)
        worst = median(r["ranks"][-1] for r in grp)
        lab = f"{m}" if m < 4 else "4+"
        print(f"{lab:>3} {len(grp):>5} {osc:>7.3f} {struct:>7.3f} "
              f"{osc - struct:>+7.3f} {best:>10.0f} {worst:>10.0f} {pc:>10.3f}")

    print("\n구조만 = m=1 의 랭크 분포에서 m개를 독립으로 뽑아 전부 예산 안에 드는 비율.")
    print("차이가 0 근처면 OSC 하락은 전부 all-or-nothing 의 산수다.")
    print("차이가 음수면 m 이 큰 질의의 랭크 자체가 나쁘다 = 검색이 실제로 못 찾는다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
