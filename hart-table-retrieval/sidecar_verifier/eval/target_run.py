"""Run TARGET benchmark on baseline vs verified retriever.

Usage:
  python sidecar_verifier/eval/target_run.py --dataset fetaqa --top-k 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "target_bench"))

from target_benchmark.evaluators import TARGET

from sidecar_verifier.target_adapter import (
    VectorBaselineRetriever,
    VerifierRerankRetriever,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="fetaqa",
                   choices=["fetaqa", "tabfact", "ottqa", "spider-validation",
                            "bird-validation", "dummy-dataset"])
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--split", default="test")
    p.add_argument("--chroma-dir", default="/home/user/T2/hart-table-retrieval/data/target_chroma")
    p.add_argument("--out-dir", default="results/target")
    p.add_argument("--methods", nargs="+", default=["baseline", "verified"],
                   choices=["baseline", "verified"])
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target = TARGET(("Table Retrieval Task", args.dataset))

    summary = {}
    for method in args.methods:
        print(f"\n=== Method: {method} ===")
        if method == "baseline":
            retr = VectorBaselineRetriever(chroma_dir=args.chroma_dir)
        else:
            retr = VerifierRerankRetriever(chroma_dir=args.chroma_dir)

        out_jsonl = out_dir / f"{args.dataset}_{method}_top{args.top_k}.jsonl"
        perf = target.run(
            retriever=retr,
            split=args.split,
            top_k=args.top_k,
            retrieval_results_file=str(out_jsonl),
        )
        print(f"  Performance: {perf}")
        summary[method] = perf if isinstance(perf, dict) else str(perf)

    out_summary = out_dir / f"{args.dataset}_summary.json"
    with open(out_summary, "w") as f:
        json.dump({"dataset": args.dataset, "top_k": args.top_k, "results": summary},
                  f, indent=2, default=str)
    print(f"\nSaved summary to {out_summary}")


if __name__ == "__main__":
    main()
