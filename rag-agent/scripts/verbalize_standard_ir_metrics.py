#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Literature-standard IR metrics for the verbalized cell-sentence retriever.

Replaces the ad-hoc ``cell_hit@k`` in verbalize_retrieval_eval.py with the
exact metric set the rest of the paper reports (prof's rule: metrics must
match the comparison literature) — Recall@k, Hit@k, MRR, nDCG@k, set-EM@k —
computed from each gold operand cell's 1-based rank in the full sentence
ranking, reusing the metric/significance functions of
standard_ir_metrics_from_records.py verbatim.

Conditions are the sentence-length ablation (sent_short/medium/long); the
fulltable baselines have no cell granularity, so cell-level metrics do not
exist for them (table-level R@k/MRR for all five conditions live in
verbalize_retrieval_eval.py results). Paired tests: sent_long vs the other
two styles, Wilcoxon for continuous metrics, exact binomial flip for binary.

Usage:
  .venv/bin/python scripts/verbalize_standard_ir_metrics.py                # full dev
  .venv/bin/python scripts/verbalize_standard_ir_metrics.py --n 50        # smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rag_agent.bench.hitab import load_queries  # noqa: E402
from rag_agent.serialize.verbalize import verbalize_table, STYLES  # noqa: E402
from standard_ir_metrics_from_records import (  # noqa: E402
    summarize, paired_tests)
import standard_ir_metrics_from_records as sim  # noqa: E402

SEED = 42
MODEL = "BAAI/bge-small-en-v1.5"
QUERY_INSTR = "Represent this sentence for searching relevant passages: "
CACHE_DIR = ROOT / "data" / "verbalize_cache"
KS = (10, 20, 50)
sim.KS = KS  # summarize/paired_tests read the module-level constant


def build_sentence_corpus(tables: dict, style: str):
    texts, keys = [], []  # keys: (table_id, row, col)
    for tid in sorted(tables):
        for ch in verbalize_table(tables[tid], style):
            texts.append(ch.text)
            keys.append((tid, ch.rows[0], ch.cols[0]))
    return texts, keys


def encode_cached(enc, tag: str, texts):
    path = CACHE_DIR / f"{tag.replace('/', '_')}.npy"
    if path.exists():
        vecs = np.load(path)
        if len(vecs) == len(texts):
            return vecs
    t0 = time.time()
    vecs = enc.encode(texts)
    print(f"    encoded {len(texts)} in {time.time()-t0:.0f}s")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, vecs)
    return vecs


def gold_cell_ranks(qvecs, cvecs, keys, queries, block=64):
    """Per query: 1-based full-ranking rank of each gold operand cell's
    sentence (None if the cell has no sentence, e.g. empty cell)."""
    key2idx = {k: i for i, k in enumerate(keys)}
    per_query = {}
    for s in range(0, len(qvecs), block):
        sims = qvecs[s:s + block] @ cvecs.T
        # rank of chunk i = 1 + count of strictly higher scores
        for bi in range(sims.shape[0]):
            q = queries[s + bi]
            row = sims[bi]
            ranks = []
            for op in q.gold_operands:
                idx = key2idx.get((q.gold_table_id, op.row, op.col))
                if idx is None:
                    ranks.append(None)
                else:
                    ranks.append(int((row > row[idx]).sum()) + 1)
            per_query[q.query_id] = ranks
    return per_query


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default=MODEL)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    queries, tables = load_queries("data/hitab", args.split)
    queries = [q for q in queries if q.gold_operands]
    if args.n:
        import random
        random.Random(SEED).shuffle(queries)
        queries = queries[:args.n]
    print(f"operand queries={len(queries)} table_pool={len(tables)}")

    from rag_agent.retrieve.encoders import SentenceTransformerEncoder
    enc = SentenceTransformerEncoder(model_name=args.encoder, batch_size=256)
    qvecs = enc.encode([QUERY_INSTR + q.question for q in queries])

    enc_tag = args.encoder.replace("/", "_")
    report = {"config": {"encoder": args.encoder, "split": args.split,
                         "n_operand_queries": len(queries),
                         "table_pool": len(tables), "ks": list(KS),
                         "metric_defs": "standard_ir_metrics_from_records.py"},
              "by_condition": {}, "paired_vs_sent_long": {}}

    per_cond = {}
    for style in STYLES:
        texts, keys = build_sentence_corpus(tables, style)
        cvecs = encode_cached(enc, f"{args.split}_sent_{style}_{enc_tag}", texts)
        pq = gold_cell_ranks(qvecs, cvecs, keys, queries)
        per_cond[style] = pq
        report["by_condition"][f"sent_{style}"] = summarize(pq)
        print(f"[sent_{style}]", report["by_condition"][f"sent_{style}"])

    for base in [s for s in STYLES if s != "long"]:
        report["paired_vs_sent_long"][f"sent_{base}->sent_long"] = paired_tests(
            per_cond[base], per_cond["long"])

    out = Path(args.out) if args.out else \
        ROOT / "results" / f"verbalize_standard_ir_{args.split}_{enc_tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print("saved →", out)

    hdr = (f"{'condition':<12} {'MRR':>6} " +
           " ".join(f"{m}@{k:<3}" for k in KS
                    for m in ("hit", "R", "nDCG", "EM")))
    print(hdr)
    for cond, m in report["by_condition"].items():
        cells = " ".join(
            f"{m[f'{name}@{k}']:>6.3f}" for k in KS
            for name in ("hit", "recall", "ndcg", "set_em"))
        print(f"{cond:<12} {m['mrr']:>6.3f} {cells}")


if __name__ == "__main__":
    main()
