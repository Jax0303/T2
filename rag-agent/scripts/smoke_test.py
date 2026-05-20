#!/usr/bin/env python3
"""Offline integration smoke test — no LLM, no API key needed.

Verifies the retrieve + verify + skip-symbolic flow on real HiTab samples by
stubbing the LLM with one that returns a deterministic empty extraction (so
symbolic-eval marks the plan unparsed and the reader path also returns "").
We are NOT measuring answer quality here — only that the pipeline runs end-
to-end and the per-stage trace is populated for the right classes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "rag-agent"))

from rag_agent.data.loader import (  # noqa: E402
    get_answer, get_query_from_sample, get_table_from_sample, get_table_id, load_hitab,
)
from rag_agent.agent import RAGAgent  # noqa: E402
from rag_agent.llm.base import BaseLLM  # noqa: E402
from rag_agent.stores.original_store import OriginalStore  # noqa: E402
from rag_agent.stores.vector_store import VectorStore  # noqa: E402
from rag_agent.eval.metrics import HARD_CLASSES, difficulty_class, recall_at_k  # noqa: E402


class StubLLM(BaseLLM):
    name = "stub"
    def complete(self, system, user, max_tokens=256):
        # Empty cell-extractor plan + empty reader answer.
        return '{"cells": [], "expression": ""}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--chroma-dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-per-class", type=int, default=2)
    args = p.parse_args()

    samples = load_hitab(data_dir=args.data_dir, split="dev")
    store = OriginalStore()
    seen = set()
    for s in samples:
        tid = get_table_id(s)
        if tid in seen: continue
        seen.add(tid)
        t = get_table_from_sample(s)
        if isinstance(t, dict) and "data" in t:
            t = dict(t); t["table_id"] = tid
            store.add(t)
    print(f"OriginalStore: {len(store)} tables")

    vs = VectorStore(chroma_dir=args.chroma_dir, device=args.device)
    print(f"VectorStore: {len(vs)} vectors\n")

    agent = RAGAgent(store, vs, llm=StubLLM())

    # 2 per class
    picked = {}
    for s in samples:
        cls = difficulty_class(s)
        picked.setdefault(cls, [])
        if len(picked[cls]) < args.n_per_class:
            picked[cls].append(s)

    total = 0
    r1_vec = 0; r1_final = 0
    per_class_stages: dict = {}
    for cls in HARD_CLASSES:
        for s in picked.get(cls, []):
            q = get_query_from_sample(s); gold = get_table_id(s)
            if not q or not gold: continue
            t0 = time.time()
            res = agent.run(q)
            dt = time.time() - t0
            vec_ids = [h["table_id"] for h in res.vector_ranked]
            fin_ids = [h["table_id"] for h in res.final_ranked]
            r1v = recall_at_k(vec_ids, gold, 1)
            r1f = recall_at_k(fin_ids, gold, 1)
            r1_vec += r1v; r1_final += r1f; total += 1
            per_class_stages.setdefault(cls, []).append(res.plan["stages"])
            print(f"  [{cls:22s}] R1v={r1v} R1f={r1f} src={res.source} stages={res.plan['stages']}  ({dt:.1f}s)")
    print(f"\n total={total}  R@1(vector)={r1_vec/total:.3f}  R@1(final)={r1_final/total:.3f}")
    print("per-class plan stages (sanity — symbolic should only appear in arith/multi_op classes):")
    for cls, planss in per_class_stages.items():
        # Are stages consistent within a class?
        unique = set(tuple(p) for p in planss)
        print(f"  {cls:22s} → {unique}")


if __name__ == "__main__":
    main()
