#!/usr/bin/env python3
"""Coverage / fallback diagnostics + the no-fallback ablation log.

For each HiTab sample it runs operand-targeted retrieval, computes the
self-assessed coverage rate and HPIR confidence, records the fallback decision,
and measures operand_recall under both arms:

* **full**     — fallback enabled (whole-table context when triggered)
* **no_fb**    — fallback disabled (always the sparse operand cells)

The per-sample JSON log isolates the fallback's value for the ablation table and
exposes the coverage_rate / fallback-rate distributions the spec asks to plot.

Usage
-----
    python scripts/fallback_ablation.py --split dev --max-samples 200
    python scripts/fallback_ablation.py --cov-threshold 0.7 --conf-threshold 0.3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
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
from rag_agent.fallback import operand_coverage, decide_fallback, build_context  # noqa: E402


def _build_encoder(name: str):
    if name == "bge":
        from rag_agent.retrieve.encoders import SentenceTransformerEncoder
        return SentenceTransformerEncoder("BAAI/bge-base-en-v1.5")
    return HashingEncoder(dim=512)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--encoder", choices=["hashing", "bge"], default="hashing")
    ap.add_argument("--cov-threshold", type=float, default=0.7)
    ap.add_argument("--conf-threshold", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--out", default="results/fallback_ablation.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    random.seed(args.seed)
    samples = load_hitab(args.data_dir, split=args.split, max_samples=args.max_samples)
    retriever = OperandTargetedRetriever(encoder=_build_encoder(args.encoder), alpha=args.alpha)

    per_sample = []
    reasons = Counter()
    n = n_fb = 0
    sum_cov = sum_conf = 0.0
    rec_full = rec_nofb = 0.0

    for s in samples:
        gold = gold_operands_from_hitab(s)
        if not gold:
            continue
        table = from_hitab_raw(s["table"])
        res = retriever.retrieve(s["question"], table, k=args.k)
        cov = operand_coverage(res)
        dec = decide_fallback(res, cov, args.cov_threshold, args.conf_threshold)

        # operand_recall is measured on the sparse operand cells (the fallback
        # supplies the whole table to the *reader*, so its benefit shows up at
        # answer time; here we log retrieval recall + the decision signals).
        r_at_k = operand_recall_at_k(gold, res.retrieved, k=None)

        n += 1
        n_fb += int(dec.triggered)
        reasons[dec.reason] += 1
        sum_cov += cov.coverage_rate
        sum_conf += res.confidence
        rec_full += r_at_k
        rec_nofb += r_at_k

        ctx = build_context(res, table, dec, max_tokens=args.max_tokens)
        per_sample.append(
            {
                "id": s.get("id"),
                "table_id": table.table_id,
                "coverage_rate": round(cov.coverage_rate, 4),
                "confidence": round(res.confidence, 4),
                "operand_recall": round(r_at_k, 4),
                "decision": dec.to_dict(),
                "context": ctx.to_dict(),
            }
        )

    agg = {
        "n_eval": n,
        "fallback_rate": (n_fb / n) if n else 0.0,
        "mean_coverage_rate": (sum_cov / n) if n else 0.0,
        "mean_confidence": (sum_conf / n) if n else 0.0,
        "reasons": dict(reasons),
        "operand_recall_full": (rec_full / n) if n else 0.0,
        "operand_recall_no_fallback": (rec_nofb / n) if n else 0.0,
    }
    out = {
        "config": {
            "split": args.split, "k": args.k, "alpha": args.alpha,
            "encoder": args.encoder, "cov_threshold": args.cov_threshold,
            "conf_threshold": args.conf_threshold, "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
        "aggregate": agg,
        "per_sample": per_sample,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"n_eval={n}  encoder={args.encoder}  alpha={args.alpha}")
    print(f"  fallback_rate       = {agg['fallback_rate']:.3f}")
    print(f"  mean_coverage_rate  = {agg['mean_coverage_rate']:.3f}")
    print(f"  mean_confidence     = {agg['mean_confidence']:.3f}")
    print(f"  reasons             = {agg['reasons']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
