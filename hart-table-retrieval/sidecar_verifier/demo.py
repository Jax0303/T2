"""Quick interactive demo of the sidecar verifier agent (v2) on HiTab dev samples."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import (
    get_answer,
    get_query_from_sample,
    get_table_from_sample,
    get_table_id,
    load_hitab,
)
from sidecar_verifier.agent.pipeline import VerifierAgent
from sidecar_verifier.agent.retriever import VectorRetriever
from sidecar_verifier.store.table_store import TableStore


def _build_full_store(samples) -> TableStore:
    store = TableStore()
    seen = set()
    for s in samples:
        tid = get_table_id(s)
        if tid in seen:
            continue
        seen.add(tid)
        t = get_table_from_sample(s)
        t["table_id"] = tid
        store.add(t)
    return store


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/home/user/T2/hart-table-retrieval/data/hitab")
    p.add_argument("--chroma-dir", default="/home/user/T2/hart-table-retrieval/data/chroma_db")
    p.add_argument("--serializer", default="plain_markdown")
    p.add_argument("--n-queries", type=int, default=5)
    p.add_argument("--top-k-tables", type=int, default=5)
    p.add_argument("--mode", default="rerank",
                   choices=["rerank", "filter", "filter+rerank"])
    p.add_argument("--w-vector", type=float, default=0.8)
    p.add_argument("--w-verify", type=float, default=0.2)
    p.add_argument("--filter-threshold", type=float, default=0.2)
    p.add_argument("--llm", action="store_true",
                   help="Use local LLM (Qwen2.5-3B-Instruct) to generate answer.")
    p.add_argument("--llm-model", default="Qwen/Qwen2.5-3B-Instruct")
    args = p.parse_args()

    print(f"Loading HiTab samples from {args.data_dir} ...")
    samples = load_hitab(data_dir=args.data_dir)
    store = _build_full_store(samples)
    print(f"  TableStore built with {len(store)} tables")

    print(f"Loading retriever (serializer={args.serializer}) ...")
    retriever = VectorRetriever(chroma_dir=args.chroma_dir, serializer=args.serializer)

    answerer = None
    if args.llm:
        from sidecar_verifier.agent.answerer import LocalLLMAnswerer
        print(f"Loading LLM {args.llm_model} ...")
        answerer = LocalLLMAnswerer(model_name=args.llm_model)

    agent = VerifierAgent(
        retriever, store,
        mode=args.mode, w_vector=args.w_vector, w_verify=args.w_verify,
        filter_threshold=args.filter_threshold,
        answerer=answerer,
    )

    chosen = samples[: args.n_queries]
    for i, s in enumerate(chosen):
        query = get_query_from_sample(s)
        gold_table = get_table_id(s)
        gold_answer = get_answer(s)
        # If LLM is on, let agent produce its own answer; else pretend a candidate was given.
        cand = None if args.llm else ", ".join(str(a) for a in gold_answer)

        print("\n" + "=" * 80)
        print(f"[{i + 1}/{args.n_queries}] {query}")
        print(f"  Gold table : {gold_table}")
        print(f"  Gold answer: {gold_answer}")

        result = agent.run(
            query, top_k_tables=args.top_k_tables, candidate_answer=cand
        )

        print("\n  Vector ranking (top-k):")
        for j, h in enumerate(result.vector_ranked):
            mark = " <- gold" if h["table_id"] == gold_table else ""
            print(f"    {j + 1}. {h['table_id']:40s} score={h['score']:.3f}{mark}")

        print(f"\n  Final ranking ({args.mode}) — {len(result.final_ranked)} kept:")
        for j, h in enumerate(result.final_ranked):
            mark = " <- gold" if h["table_id"] == gold_table else ""
            v = h.get("verification") or {}
            fused = h.get("fused_score")
            score_str = f"fused={fused:.3f}" if fused is not None else f"kept (vec={h['score']:.3f})"
            print(
                f"    {j + 1}. {h['table_id']:40s} "
                f"{score_str} "
                f"kw={v.get('keyword_overlap'):.2f} num={v.get('numeric_overlap'):.2f}{mark}"
            )

        d = result.disagreement_signal
        print(f"\n  Disagreement vs vector top-3:  overlap={d['overlap@top']} "
              f"vector_only={d['vector_only']} verified_only={d['verified_only']}")

        if result.answer is not None:
            print(f"\n  LLM answer: {result.answer.answer!r}")
            if result.answer.raw_output != result.answer.answer:
                print(f"  (raw)     : {result.answer.raw_output!r}")

        if result.trace is not None:
            tr = result.trace
            print(f"\n  Answer trace ({tr.answer!r}):")
            print(f"    grounded_fraction = {tr.grounded_fraction:.2f}")
            print(f"    grounded_cells    = {tr.grounded_cells}")
            print(f"    ungrounded        = {tr.ungrounded_spans}")


if __name__ == "__main__":
    main()
