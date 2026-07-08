"""Full HiTab dev comparison of every retrieval method.

Runs BM25 / dense vector / verifier-rerank / cross-encoder-rerank on the full
HiTab dev split (or a capped subset) using the same serializer and the same
table corpus. Saves per-query hits for downstream bootstrap CI / failure mode
analysis.

Methods
-------
- bm25                  : sparse, plain_markdown serialization
- vector                : dense (BGE-large) baseline
- verifier_rerank w=0.2 : vector + our keyword/numeric verifier rerank
- ce_rerank             : vector + BGE-reranker-base cross-encoder (heavy)

Usage
-----
  python -m sidecar_verifier.eval.full_baseline_compare \\
      --max-queries 0   # 0 = full dev
      --include-ce      # opt-in: CE adds ~8min model load + ~3min rerank
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.loader import (
    get_query_from_sample,
    get_table_from_sample,
    get_table_id,
    load_hitab,
)
from src.serializers.plain_markdown import PlainMarkdownSerializer

from sidecar_verifier.agent.bm25_retriever import build_bm25_from_samples
from sidecar_verifier.agent.reconciler import rerank
from sidecar_verifier.agent.retriever import VectorRetriever
from sidecar_verifier.agent.verifier import verify_hits
from sidecar_verifier.store.table_store import TableStore


KS = [1, 5, 10]


def _recall(ranked_ids: List[str], gold: str, k: int) -> int:
    return int(gold in ranked_ids[:k])


def _mrr(ranked_ids: List[str], gold: str, kmax: int = 10) -> float:
    for i, t in enumerate(ranked_ids[:kmax]):
        if t == gold:
            return 1.0 / (i + 1)
    return 0.0


def bootstrap_ci(values: List[int], iters: int = 1000, alpha: float = 0.05,
                 seed: int = 42) -> Tuple[float, float, float]:
    """Returns (mean, lo, hi) for the 1-alpha CI using bootstrap resampling."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iters):
        s = sum(values[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int(iters * alpha / 2)]
    hi = means[int(iters * (1 - alpha / 2))]
    return sum(values) / n, lo, hi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/mnt/d/hart_data/hitab/HiTab")
    p.add_argument("--chroma-dir", default="/mnt/d/hart_data/chroma_db")
    p.add_argument("--serializer", default="plain_markdown")
    p.add_argument("--max-queries", type=int, default=0,
                   help="0 = full split")
    p.add_argument("--top-k-vectors", type=int, default=20)
    p.add_argument("--top-k-tables", type=int, default=10)
    p.add_argument("--w-verify", type=float, default=0.2)
    p.add_argument("--include-bm25", action="store_true", default=True)
    p.add_argument("--include-ce", action="store_true",
                   help="Add cross-encoder rerank. ~8min model load on WSL DrvFs.")
    p.add_argument("--ce-model", default="BAAI/bge-reranker-base")
    p.add_argument("--out", default="results/full_baseline_compare.json")
    args = p.parse_args()

    # ---- Load data ----
    print("Loading HiTab dev ...")
    full = load_hitab(data_dir=args.data_dir, split="dev")
    samples = full[: args.max_queries] if args.max_queries else full
    print(f"  {len(samples)} queries / {len(full)} total")

    # ---- TableStore (verifier needs it) ----
    store = TableStore()
    seen = set()
    for s in full:
        if "table" not in s:
            continue
        tid = get_table_id(s)
        if tid in seen:
            continue
        seen.add(tid)
        t = get_table_from_sample(s)
        t["table_id"] = tid
        store.add(t)
    print(f"  TableStore: {len(store)} tables")

    # ---- Retrievers ----
    print("Building VectorRetriever ...")
    vec_retriever = VectorRetriever(
        chroma_dir=args.chroma_dir, serializer=args.serializer,
    )

    bm25 = None
    if args.include_bm25:
        print("Building BM25 index ...")
        ser = PlainMarkdownSerializer()
        bm25 = build_bm25_from_samples(full, ser)
        print(f"  BM25 over {len(bm25.table_ids)} tables")

    ce_reranker = None
    if args.include_ce:
        print(f"Loading cross-encoder {args.ce_model} (this is slow on DrvFs) ...")
        from sidecar_verifier.agent.cross_encoder import CrossEncoderReranker
        t0 = time.time()
        ce_reranker = CrossEncoderReranker(model_name=args.ce_model)
        print(f"  CE loaded in {time.time()-t0:.1f}s")

    # ---- Per-method per-query state ----
    method_names = ["vector", f"verifier_rerank_w{args.w_verify}"]
    if args.include_bm25:
        method_names.insert(0, "bm25")
    if args.include_ce:
        method_names.append("ce_rerank")

    # per-query records: {method: {"R@1":[0,1,...], "R@5":[...], "R@10":[...], "MRR":[...]}}
    per_q = {m: defaultdict(list) for m in method_names}
    # latency total per method (seconds)
    elapsed = {m: 0.0 for m in method_names}
    n = 0

    t_eval = time.time()
    for i, s in enumerate(samples):
        q = get_query_from_sample(s)
        gold = get_table_id(s)
        if not q or not gold:
            continue
        n += 1

        # ---- vector (compute once, reused by verifier and CE) ----
        t = time.time()
        vec_hits = vec_retriever.retrieve(
            q, top_k_vectors=args.top_k_vectors, top_k_tables=args.top_k_tables,
        )
        elapsed["vector"] += time.time() - t

        vec_ids = [h["table_id"] for h in vec_hits]

        # ---- verifier_rerank ----
        t = time.time()
        verified = verify_hits(q, store, vec_hits)
        v_ranked = rerank(verified, w_vector=1.0 - args.w_verify, w_verify=args.w_verify)
        elapsed[f"verifier_rerank_w{args.w_verify}"] += time.time() - t
        verif_ids = [h["table_id"] for h in v_ranked]

        # ---- bm25 ----
        if bm25 is not None:
            t = time.time()
            bm25_ids = [h["table_id"] for h in bm25.retrieve(q, top_k_tables=max(KS))]
            elapsed["bm25"] += time.time() - t
        else:
            bm25_ids = None

        # ---- ce_rerank ----
        if ce_reranker is not None:
            t = time.time()
            ce_ranked = ce_reranker.rerank(q, list(vec_hits))
            elapsed["ce_rerank"] += time.time() - t
            ce_ids = [h["table_id"] for h in ce_ranked]
        else:
            ce_ids = None

        # ---- record per-query hits ----
        for name, ids in [
            ("bm25", bm25_ids),
            ("vector", vec_ids),
            (f"verifier_rerank_w{args.w_verify}", verif_ids),
            ("ce_rerank", ce_ids),
        ]:
            if ids is None:
                continue
            for k in KS:
                per_q[name][f"R@{k}"].append(_recall(ids, gold, k))
            per_q[name]["MRR"].append(_mrr(ids, gold))

        if (i + 1) % 200 == 0:
            print(f"  ... processed {i+1}/{len(samples)} queries ({time.time()-t_eval:.1f}s)")

    total_t = time.time() - t_eval
    print(f"\nEvaluated n={n} in {total_t:.1f}s")

    # ---- Summarize with bootstrap CI ----
    summary = {
        "n": n,
        "serializer": args.serializer,
        "methods": {},
    }
    print(f"\n{'method':<28} " + " ".join(f"{m:>14s}" for m in ["R@1[CI]", "R@5[CI]", "R@10[CI]", "MRR", "lat(ms/q)"]))
    print("-" * 110)
    for name in method_names:
        row = {"name": name, "metrics": {}}
        cells = [f"{name:<28}"]
        for k in KS:
            vals = per_q[name][f"R@{k}"]
            m, lo, hi = bootstrap_ci(vals)
            row["metrics"][f"R@{k}"] = {"mean": m, "lo": lo, "hi": hi}
            cells.append(f"{m:.3f}[{lo:.3f},{hi:.3f}]")
        mrr_vals = per_q[name]["MRR"]
        mrr_mean = sum(mrr_vals) / max(len(mrr_vals), 1)
        row["metrics"]["MRR"] = mrr_mean
        cells.append(f"{mrr_mean:>13.3f}")
        lat_ms = elapsed[name] / n * 1000 if n else 0.0
        row["metrics"]["latency_ms_per_q"] = lat_ms
        cells.append(f"{lat_ms:>13.1f}")
        summary["methods"][name] = row
        print(" ".join(cells))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
