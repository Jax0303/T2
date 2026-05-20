#!/usr/bin/env python3
"""End-to-end RAG-agent evaluation on a HiTab stratified hard subset.

Reports paper-aligned metrics:
  - Retrieval: R@1, R@5, MRR, nDCG@10
  - Answer:    Exact Match, Numeric Match (±rel-tol)
  - Symbolic execution accuracy (arithmetic classes only)
  - Per-difficulty-class breakdown matching HiTab paper's appendix.

Reuses the existing HiTab loader from ``hart-table-retrieval/src/data/loader.py``.

Usage examples:
  python scripts/run_eval.py --llm groq:llama-3.3-70b-versatile --per-class 8
  python scripts/run_eval.py --llm local:Qwen/Qwen2.5-7B-Instruct --per-class 8
  python scripts/run_eval.py --llm groq:llama-3.3-70b-versatile \\
      --symbolic-llm local:Qwen/Qwen2.5-7B-Instruct --per-class 8

By default points at /home/user/T2/hart-table-retrieval/{data/hitab, data/chroma_db}.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Make the hart-table-retrieval loader importable (used for parsing HiTab JSON).
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
HART_DIR = REPO_ROOT / "hart-table-retrieval"
sys.path.insert(0, str(REPO_ROOT / "rag-agent"))
sys.path.insert(0, str(HART_DIR))

from src.data.loader import (  # noqa: E402 (relies on sys.path mutation above)
    get_answer, get_query_from_sample, get_table_from_sample, get_table_id, load_hitab,
)
from rag_agent.agent import RAGAgent  # noqa: E402
from rag_agent.eval.metrics import (  # noqa: E402
    HARD_CLASSES, difficulty_class, exact_match, mrr, ndcg_at_k, numeric_match, recall_at_k,
)
from rag_agent.llm.factory import build_llm  # noqa: E402
from rag_agent.stores.original_store import OriginalStore  # noqa: E402
from rag_agent.stores.vector_store import VectorStore  # noqa: E402


def stratified_hard_subset(samples, per_class: int, seed: int = 0):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for s in samples:
        buckets[difficulty_class(s)].append(s)

    chosen, populations = [], {}
    for cls in HARD_CLASSES:
        bucket = buckets.get(cls, [])
        populations[cls] = len(bucket)
        rng.shuffle(bucket)
        for s in bucket[:per_class]:
            chosen.append((cls, s))
    return chosen, populations


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/home/user/T2/hart-table-retrieval/data/hitab")
    p.add_argument("--chroma-dir", default="/home/user/T2/hart-table-retrieval/data/chroma_db")
    p.add_argument("--embedder", default="BAAI/bge-large-en-v1.5")
    p.add_argument("--serializer", default="plain_markdown")
    p.add_argument("--retriever-device", default=None,
                   help="cpu | cuda. Auto if omitted; pass cpu when LLM needs all VRAM.")
    p.add_argument("--llm", default="groq:llama-3.3-70b-versatile",
                   help="LLM spec for the reader + symbolic extractor (unless --symbolic-llm given). "
                        "Examples: 'local:Qwen/Qwen2.5-7B-Instruct', 'groq:llama-3.3-70b-versatile'.")
    p.add_argument("--symbolic-llm", default=None,
                   help="Optional override for the cell-extractor LLM.")
    p.add_argument("--per-class", type=int, default=8)
    p.add_argument("--top-k-vectors", type=int, default=20)
    p.add_argument("--top-k-tables", type=int, default=5)
    p.add_argument("--w-vector", type=float, default=0.7)
    p.add_argument("--w-verify", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/rag_agent_eval.json")
    p.add_argument("--limit", type=int, default=0, help="Cap total queries (0 = all chosen).")
    return p


def main():
    args = build_arg_parser().parse_args()

    print("[1/5] Loading HiTab dev samples ...")
    samples = load_hitab(data_dir=args.data_dir, split="dev")
    print(f"  loaded {len(samples)} samples")

    print(f"[2/5] Stratifying hard subset (per_class={args.per_class}) ...")
    chosen, populations = stratified_hard_subset(samples, args.per_class, args.seed)
    if args.limit:
        chosen = chosen[: args.limit]
    print(f"  picked {len(chosen)} queries")
    for cls in HARD_CLASSES:
        print(f"    {cls:22s}: pool={populations.get(cls, 0):4d}")

    print("[3/5] Building OriginalStore over unique tables in dev split ...")
    original = OriginalStore()
    seen = set()
    for s in samples:
        tid = get_table_id(s)
        if tid in seen:
            continue
        seen.add(tid)
        t = get_table_from_sample(s)
        if not isinstance(t, dict) or "data" not in t:
            continue
        t = dict(t)
        t["table_id"] = tid
        original.add(t)
    print(f"  OriginalStore size = {len(original)}")

    print(f"[4/5] Loading VectorStore ({args.embedder}, serializer={args.serializer}) ...")
    vstore = VectorStore(
        chroma_dir=args.chroma_dir,
        embedder_model=args.embedder,
        serializer=args.serializer,
        device=args.retriever_device,
    )
    print(f"  VectorStore size = {len(vstore)}")

    print(f"[5/5] Loading LLM ({args.llm}) ...")
    t_load = time.time()
    llm = build_llm(args.llm)
    sym_llm = build_llm(args.symbolic_llm) if args.symbolic_llm else llm
    print(f"  Reader loaded in {time.time() - t_load:.1f}s")

    agent = RAGAgent(
        original_store=original,
        vector_store=vstore,
        llm=llm,
        symbolic_llm=sym_llm,
        top_k_vectors=args.top_k_vectors,
        top_k_tables=args.top_k_tables,
        w_vector=args.w_vector,
        w_verify=args.w_verify,
    )

    per_class = {cls: Counter() for cls in HARD_CLASSES}
    rows = []

    def _save_partial(out_path, populations, per_class, rows, args, error=None):
        """Save whatever we have so far, even if the run aborts mid-stream."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        overall = Counter()
        for cls in HARD_CLASSES:
            for k, v in per_class[cls].items():
                overall[k] += v
        with open(out_path, "w") as f:
            json.dump({
                "config": vars(args),
                "class_populations": populations,
                "per_class": {cls: dict(per_class[cls]) for cls in HARD_CLASSES if per_class[cls]["n"]},
                "overall": dict(overall),
                "rows": rows,
                "error": error,
            }, f, indent=2, default=str)

    try:
      for i, (cls, s) in enumerate(chosen, 1):
        q = get_query_from_sample(s)
        gold_tid = get_table_id(s)
        gold_ans = get_answer(s)
        if not q or not gold_tid:
            continue
        t0 = time.time()
        try:
            res = agent.run(q)
        except Exception as e:
            print(f"  ✗ run failed on query {i} ({cls}): {type(e).__name__}: {e}")
            _save_partial(args.out, populations, per_class, rows, args,
                          error=f"{type(e).__name__}: {e}")
            raise
        elapsed = time.time() - t0

        vec_ids = [h["table_id"] for h in res.vector_ranked]
        final_ids = [h["table_id"] for h in res.final_ranked]

        r1_vec = recall_at_k(vec_ids, gold_tid, 1)
        r1_fin = recall_at_k(final_ids, gold_tid, 1)
        r5_fin = recall_at_k(final_ids, gold_tid, 5)
        mrr_fin = mrr(final_ids, gold_tid)
        ndcg_fin = ndcg_at_k(final_ids, gold_tid, 10)

        pred = res.answer or ""
        ans_num = numeric_match(pred, gold_ans)
        ans_em = exact_match(pred, gold_ans)
        sym_ok = bool(res.symbolic and res.symbolic.get("ok"))
        sym_correct = bool(sym_ok and numeric_match(res.symbolic.get("value"), gold_ans))

        st = per_class[cls]
        st["n"] += 1
        st["R@1_vec"] += r1_vec
        st["R@1_final"] += r1_fin
        st["R@5_final"] += r5_fin
        st["MRR_final"] += mrr_fin
        st["nDCG@10"] += ndcg_fin
        st["EM"] += int(ans_em)
        st["NM"] += int(ans_num)
        st["symbolic_attempted"] += int(sym_ok)
        st["symbolic_correct"] += int(sym_correct)
        st["from_symbolic"] += int(res.source == "symbolic")
        st["elapsed"] += elapsed

        rows.append({
            "class": cls, "query": q, "gold_table": gold_tid, "gold_answer": gold_ans,
            "gold_formula": (s.get("answer_formulas") or [""])[0],
            "intent": res.intent, "plan": res.plan, "source": res.source,
            "vector_top": vec_ids[:5], "final_top": final_ids[:5],
            "answer": pred, "answer_em": ans_em, "answer_numeric_match": ans_num,
            "symbolic_attempted": sym_ok, "symbolic_correct": sym_correct,
            "symbolic_value": res.symbolic.get("value") if res.symbolic else None,
            "symbolic_expr": res.symbolic.get("expression") if res.symbolic else None,
            "symbolic_resolved_cells": (res.symbolic or {}).get("resolved_cells"),
            "symbolic_error": (res.symbolic or {}).get("error"),
            "reader_raw": (res.reader or {}).get("raw_output"),
            "elapsed_s": round(elapsed, 2),
        })
        print(f"[{i:3d}/{len(chosen)}] {cls:22s} src={res.source or '-':<10s} "
              f"R1f={r1_fin} NM={int(ans_num)} sym={int(sym_correct)} "
              f"({elapsed:.1f}s) pred={pred!r}")
    except KeyboardInterrupt:
        print("\n[interrupted] saving partial ...")
        _save_partial(args.out, populations, per_class, rows, args, error="KeyboardInterrupt")
        raise

    print("\n=== Per-class summary ===")
    hdr = f"{'class':24s} {'n':>3s}  {'R@1_v':>6s} {'R@1':>6s} {'R@5':>6s} {'MRR':>6s} {'nDCG':>6s}  {'EM':>5s} {'NM':>5s}  {'sym_atm':>8s} {'sym_corr':>9s}"
    print(hdr)
    overall = Counter()
    for cls in HARD_CLASSES:
        st = per_class[cls]
        n = st["n"]
        if n == 0:
            continue
        print(f"{cls:24s} {n:3d}  "
              f"{st['R@1_vec']/n:6.3f} {st['R@1_final']/n:6.3f} {st['R@5_final']/n:6.3f} "
              f"{st['MRR_final']/n:6.3f} {st['nDCG@10']/n:6.3f}  "
              f"{st['EM']/n:5.3f} {st['NM']/n:5.3f}  "
              f"{st['symbolic_attempted']/n:8.3f} {st['symbolic_correct']/n:9.3f}")
        for k, v in st.items():
            overall[k] += v

    n = overall["n"]
    if n:
        print(f"{'OVERALL':24s} {n:3d}  "
              f"{overall['R@1_vec']/n:6.3f} {overall['R@1_final']/n:6.3f} {overall['R@5_final']/n:6.3f} "
              f"{overall['MRR_final']/n:6.3f} {overall['nDCG@10']/n:6.3f}  "
              f"{overall['EM']/n:5.3f} {overall['NM']/n:5.3f}  "
              f"{overall['symbolic_attempted']/n:8.3f} {overall['symbolic_correct']/n:9.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "config": vars(args),
            "class_populations": populations,
            "per_class": {cls: dict(per_class[cls]) for cls in HARD_CLASSES if per_class[cls]["n"]},
            "overall": dict(overall),
            "rows": rows,
        }, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
