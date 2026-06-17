#!/usr/bin/env python3
"""Evaluate operand-targeted retrieval: operand_recall@k on HiTab.

Produces the paper result-table metric — the fraction of gold operands the
retriever surfaces in the top-k cells — at k = 1, 3, 5, 10, and writes a JSON
log (per-sample + aggregate) for error analysis and ablation.

By default the dense backend is the NumPy HashingEncoder fallback, so this runs
anywhere. Pass ``--encoder bge`` on the GPU box to use the real BGE model.

Usage
-----
    python scripts/operand_recall_eval.py --split dev --max-samples 200
    python scripts/operand_recall_eval.py --encoder bge --alpha 0.5
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_agent.data.loader import load_hitab  # noqa: E402
from rag_agent.serialization import from_hitab_raw  # noqa: E402
from rag_agent.retrieve.encoders import HashingEncoder  # noqa: E402
from rag_agent.retrieve.operand_retrieval import (  # noqa: E402
    OperandTargetedRetriever,
    operand_recall_at_k,
    gold_operands_from_hitab,
)

KS = (1, 3, 5, 10)


def _build_encoder(name: str):
    if name == "bge":
        from rag_agent.retrieve.encoders import SentenceTransformerEncoder
        return SentenceTransformerEncoder("BAAI/bge-base-en-v1.5")
    return HashingEncoder(dim=512)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.5, help="dense weight in hybrid score")
    ap.add_argument("--encoder", choices=["hashing", "bge"], default="hashing")
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--out", default="results/operand_recall_eval.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    random.seed(args.seed)

    samples = load_hitab(args.data_dir, split=args.split, max_samples=args.max_samples)
    retriever = OperandTargetedRetriever(encoder=_build_encoder(args.encoder), alpha=args.alpha)

    per_sample = []
    sums = {k: 0.0 for k in KS}
    n = 0
    for s in samples:
        gold = gold_operands_from_hitab(s)
        if not gold:
            continue
        table = from_hitab_raw(s["table"])
        res = retriever.retrieve(s["question"], table, k=max(KS))
        rec = {k: operand_recall_at_k(gold, res.retrieved, k=k) for k in KS}
        for k in KS:
            sums[k] += rec[k]
        n += 1
        per_sample.append(
            {
                "id": s.get("id"),
                "table_id": table.table_id,
                "n_gold_operands": len(gold),
                "n_decomposed": len(res.operands),
                "recall": rec,
            }
        )

    agg = {f"operand_recall@{k}": (sums[k] / n if n else 0.0) for k in KS}
    out = {
        "config": {
            "split": args.split,
            "n_eval": n,
            "alpha": args.alpha,
            "encoder": args.encoder,
            "seed": args.seed,
        },
        "aggregate": agg,
        "per_sample": per_sample,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"n_eval={n}  encoder={args.encoder}  alpha={args.alpha}")
    for k in KS:
        print(f"  operand_recall@{k} = {agg[f'operand_recall@{k}']:.3f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
