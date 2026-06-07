#!/usr/bin/env python3
"""HPIR retrieval evaluation — does header-path query expansion improve search?

Isolates the *query-processing* effect: the SAME retriever and the SAME candidate
pool are run on (a) the raw narrative query and (b) the HPIR-expanded query
(``rag_agent.query.expand_for_retrieval``). Any delta is attributable to query
processing alone, not to the index or the embedder.

Retrievers compared (each raw vs +HPIR):
  dense    : Chroma BGE-large (plain_markdown collection)   — DPR/DTR lineage
  keyword  : original-store header/cell token overlap

Metrics (DTR R@k protocol; HiTab dev; single-gold binary nDCG):
  R@1, R@5, MRR, nDCG@10  + paired bootstrap 95% CI + exact McNemar on R@1.

Run (on the machine that holds the HiTab data + chroma index):

  HITAB_DIR=/path/to/hitab CHROMA_DIR=/path/to/chroma_db \
  python scripts/hpir_retrieval_eval.py --limit 1671 \
      --out results/hpir_retrieval.json

Leave ``--limit`` off for the full dev set. ``--device cuda`` for GPU embedding.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import codegen_eval as ce          # noqa: E402  (OriginalDB / VectorDB / loaders)
import retrieval_eval as re_eval   # noqa: E402  (paired_bootstrap / mcnemar / metrics)
from rag_agent.query import expand_for_retrieval  # noqa: E402
from rag_agent.eval.metrics import difficulty_class  # noqa: E402


def _metrics(ids, gold):
    return {
        "r1": re_eval.r_at_k(ids, gold, 1),
        "r5": re_eval.r_at_k(ids, gold, 5),
        "mrr": re_eval.mrr(ids, gold),
        "ndcg": re_eval.ndcg_at_k(ids, gold, 10),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap eval queries")
    ap.add_argument("--topk", type=int, default=540, help="rank depth (>= pool for full ranking)")
    ap.add_argument("--device", default="cpu", help="embedder device: cpu|cuda")
    ap.add_argument("--out", default="results/hpir_retrieval.json")
    args = ap.parse_args()

    # honour env overrides for portability across machines
    ce.HITAB_DIR = os.environ.get("HITAB_DIR", ce.HITAB_DIR)
    ce.CHROMA_DIR = os.environ.get("CHROMA_DIR", ce.CHROMA_DIR)

    samples = ce.load_samples("dev")
    if args.limit:
        samples = samples[: args.limit]

    print("Building OriginalDB pool (dev unique gold tables)...")
    orig = ce.OriginalDB()
    seen = set()
    for s in samples:
        tid = ce.get_table_id(s)
        if tid and tid not in seen:
            raw = ce.load_table(tid)
            if raw:
                orig.add(raw)
                seen.add(tid)
    pool = set(orig._tables)
    print(f"  pool: {len(pool)} tables")

    print(f"Loading VectorDB (BGE-large, {args.device})...")
    vdb = ce.VectorDB(ce.CHROMA_DIR, device=args.device)

    # retriever variants: (name, fn(query)->ranked_ids)
    def dense(q):
        return [t for t, _ in vdb.search(q, top_k=args.topk, allowed_ids=pool)]

    def keyword(q):
        return [t for t, _ in orig.keyword_search(q, top_k=args.topk)]

    variants = {
        "dense": lambda q: dense(q),
        "dense_hpir": lambda q: dense(expand_for_retrieval(q)),
        "keyword": lambda q: keyword(q),
        "keyword_hpir": lambda q: keyword(expand_for_retrieval(q)),
    }

    rows = []
    miss = 0
    t0 = time.time()
    for i, s in enumerate(samples):
        q = ce.get_query(s)
        gold = ce.get_table_id(s)
        if gold not in pool:
            miss += 1
            continue
        row = {"cls": difficulty_class(s), "gold": gold}
        for name, fn in variants.items():
            row[name] = _metrics(fn(q), gold)
        rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(samples)}  ({time.time()-t0:.0f}s)")

    names = list(variants)
    mets = ["r1", "r5", "mrr", "ndcg"]

    def overall(name, m):
        return sum(r[name][m] for r in rows) / len(rows) if rows else 0.0

    print("\n" + "=" * 70)
    print(f"pool {len(pool)} | eval {len(rows)} (out-of-pool gold {miss} skipped)")
    print("=" * 70)
    print(f"{'retriever':<16}" + "".join(f"{x:>9}" for x in ["R@1", "R@5", "MRR", "nDCG@10"]))
    for name in names:
        print(f"{name:<16}" + "".join(f"{overall(name, m):>9.3f}" for m in mets))

    # paired comparisons: each retriever raw vs +HPIR
    print("\npaired (HPIR - raw; 95% bootstrap CI; * excludes 0)")
    comparisons = {}
    for base, hpir in (("dense", "dense_hpir"), ("keyword", "keyword_hpir")):
        print(f"\n  {base}  →  {hpir}")
        comp = {}
        for m in mets:
            diffs = [r[hpir][m] - r[base][m] for r in rows]
            mean, lo, hi = re_eval.paired_bootstrap(diffs)
            sig = "*" if (lo > 0 or hi < 0) else " "
            print(f"    Δ{m.upper():<8} {mean:+.4f}  [{lo:+.4f}, {hi:+.4f}] {sig}")
            comp[m] = {"delta": mean, "ci": [lo, hi], "sig": sig.strip() == "*"}
        b01, b10, p = re_eval.mcnemar([r[base]["r1"] for r in rows],
                                      [r[hpir]["r1"] for r in rows])
        print(f"    McNemar R@1: HPIR>raw={b01} raw>HPIR={b10} p={p:.4g}")
        comp["mcnemar_r1"] = {"hpir_better": b01, "raw_better": b10, "p": p}
        comparisons[f"{base}_vs_{hpir}"] = comp

    by_cls = defaultdict(list)
    for r in rows:
        by_cls[r["cls"]].append(r)
    cls_out = {}
    print("\nper-class R@1 (dense raw vs dense+HPIR)")
    for cls, rs in sorted(by_cls.items()):
        a = sum(r["dense"]["r1"] for r in rs) / len(rs)
        b = sum(r["dense_hpir"]["r1"] for r in rs) / len(rs)
        print(f"  {cls:<22} raw={a:.3f}  hpir={b:.3f}  (n={len(rs)})")
        cls_out[cls] = {"dense_r1": a, "dense_hpir_r1": b, "n": len(rs)}

    out = {
        "config": {
            "pool_size": len(pool), "n_eval": len(rows), "gold_out_of_pool": miss,
            "topk": args.topk, "device": args.device,
            "embedder": "BAAI/bge-large-en-v1.5",
            "collection": "plain_markdown_bge_large_en_v1_5",
            "protocol": "DTR R@k (Herzig et al. NAACL 2021); HiTab dev; binary single-gold nDCG",
            "method": "HPIR expand_for_retrieval (header-path query expansion)",
        },
        "overall": {name: {m: overall(name, m) for m in mets} for name in names},
        "paired": comparisons,
        "by_class": cls_out,
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nsaved: {outp}")


if __name__ == "__main__":
    main()
