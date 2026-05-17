"""Evaluate v2 sidecar verifier on HiTab dev. Compares modes head-to-head."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.loader import (
    get_query_from_sample,
    get_table_from_sample,
    get_table_id,
    load_hitab,
)
from sidecar_verifier.agent.reconciler import (
    filter_only,
    filter_then_rerank,
    rerank,
)
from sidecar_verifier.agent.retriever import VectorRetriever
from sidecar_verifier.agent.verifier import verify_hits
from sidecar_verifier.store.table_store import TableStore


def _hit_recall(ranked, gold_id, k):
    return int(gold_id in [h["table_id"] for h in ranked[:k]])


def _mrr(ranked, gold_id, kmax=10):
    for i, h in enumerate(ranked[:kmax]):
        if h["table_id"] == gold_id:
            return 1.0 / (i + 1)
    return 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/home/user/T2/hart-table-retrieval/data/hitab")
    p.add_argument("--chroma-dir", default="/home/user/T2/hart-table-retrieval/data/chroma_db")
    p.add_argument("--serializer", default="plain_markdown")
    p.add_argument("--max-queries", type=int, default=300)
    p.add_argument("--top-k-vectors", type=int, default=20)
    p.add_argument("--top-k-tables", type=int, default=10)
    p.add_argument("--out", default="results/verifier_eval.json")
    args = p.parse_args()

    full = load_hitab(data_dir=args.data_dir)
    samples = full[: args.max_queries] if args.max_queries else full

    store = TableStore()
    seen = set()
    for s in full:
        tid = get_table_id(s)
        if tid in seen:
            continue
        seen.add(tid)
        t = get_table_from_sample(s)
        t["table_id"] = tid
        store.add(t)
    print(f"TableStore: {len(store)} tables")

    retriever = VectorRetriever(chroma_dir=args.chroma_dir, serializer=args.serializer)

    # All configurations to compare
    configs = [
        ("vector",        lambda h: h),
        ("filter@0.10",   lambda h: filter_only(h, 0.10)),
        ("filter@0.20",   lambda h: filter_only(h, 0.20)),
        ("filter@0.30",   lambda h: filter_only(h, 0.30)),
        ("filter@0.50",   lambda h: filter_only(h, 0.50)),
        ("rerank w=0.1",  lambda h: rerank(h, 0.9, 0.1)),
        ("rerank w=0.2",  lambda h: rerank(h, 0.8, 0.2)),
        ("rerank w=0.3",  lambda h: rerank(h, 0.7, 0.3)),
        ("rerank w=0.5",  lambda h: rerank(h, 0.5, 0.5)),
        ("f@0.2 + r=0.3", lambda h: filter_then_rerank(h, 0.2, 0.7, 0.3)),
        ("f@0.3 + r=0.5", lambda h: filter_then_rerank(h, 0.3, 0.5, 0.5)),
    ]

    ks = [1, 5, 10]
    sums = defaultdict(lambda: defaultdict(float))
    n = 0

    for s in samples:
        q = get_query_from_sample(s)
        gold = get_table_id(s)
        if not q or not gold:
            continue

        vector_hits = retriever.retrieve(
            q, top_k_vectors=args.top_k_vectors, top_k_tables=args.top_k_tables
        )
        verified = verify_hits(q, store, vector_hits)

        for name, fn in configs:
            ranked = fn(verified) if name != "vector" else vector_hits
            for k in ks:
                sums[name][f"R@{k}"] += _hit_recall(ranked, gold, k)
            sums[name]["MRR"] += _mrr(ranked, gold)
            sums[name]["kept_mean"] += len(ranked)
        n += 1

    print(f"\nEvaluated {n} queries on serializer={args.serializer}")
    header = f"{'config':18s} | " + " | ".join(f"{m:>6s}" for m in ["R@1", "R@5", "R@10", "MRR", "kept"])
    print(header)
    print("-" * len(header))
    rows = []
    for name, _ in configs:
        row = {"config": name}
        for k in ks:
            row[f"R@{k}"] = sums[name][f"R@{k}"] / n
        row["MRR"] = sums[name]["MRR"] / n
        row["kept_mean"] = sums[name]["kept_mean"] / n
        rows.append(row)
        print(
            f"{name:18s} | "
            f"{row['R@1']:6.3f} | {row['R@5']:6.3f} | {row['R@10']:6.3f} | "
            f"{row['MRR']:6.3f} | {row['kept_mean']:6.2f}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"n": n, "serializer": args.serializer, "results": rows}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
