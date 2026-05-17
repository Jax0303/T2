"""End-to-end answer-accuracy evaluation: retrieve → rerank → LLM answer → score vs gold."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.loader import (
    get_answer,
    get_query_from_sample,
    get_table_from_sample,
    get_table_id,
    load_hitab,
)
from sidecar_verifier.agent.answerer import LocalLLMAnswerer
from sidecar_verifier.agent.pipeline import VerifierAgent
from sidecar_verifier.agent.retriever import VectorRetriever
from sidecar_verifier.store.table_store import TableStore


_NUM_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+\.?\d*")


def _to_nums(s):
    if isinstance(s, (int, float)):
        return [float(s)]
    if isinstance(s, list):
        out = []
        for x in s:
            out.extend(_to_nums(x))
        return out
    if s is None:
        return []
    out = []
    for m in _NUM_RE.findall(str(s)):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _numeric_match(pred: str, gold) -> bool:
    pred_nums = _to_nums(pred)
    gold_nums = _to_nums(gold)
    if not gold_nums:
        return str(pred).strip().lower() == str(gold).strip().lower()
    # All gold numbers must appear in pred (order-insensitive).
    pred_set = set(round(x, 3) for x in pred_nums)
    return all(round(g, 3) in pred_set for g in gold_nums)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/home/user/T2/hart-table-retrieval/data/hitab")
    p.add_argument("--chroma-dir", default="/home/user/T2/hart-table-retrieval/data/chroma_db")
    p.add_argument("--serializer", default="plain_markdown")
    p.add_argument("--max-queries", type=int, default=30)
    p.add_argument("--top-k-tables", type=int, default=5)
    p.add_argument("--mode", default="rerank")
    p.add_argument("--w-vector", type=float, default=0.8)
    p.add_argument("--w-verify", type=float, default=0.2)
    p.add_argument("--llm-model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--quantization", default="4bit")
    p.add_argument("--out", default="results/answer_accuracy.json")
    p.add_argument("--also-gold", action="store_true",
                   help="Also run LLM on gold-table directly (oracle retrieval) for upper bound.")
    args = p.parse_args()

    full = load_hitab(data_dir=args.data_dir)
    samples = full[: args.max_queries]

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
    print(f"TableStore: {len(store)}")

    retriever = VectorRetriever(chroma_dir=args.chroma_dir, serializer=args.serializer)
    print(f"Loading LLM {args.llm_model} (quant={args.quantization}) ...")
    answerer = LocalLLMAnswerer(model_name=args.llm_model, quantization=args.quantization)

    agent_v = VerifierAgent(
        retriever, store, mode=args.mode,
        w_vector=args.w_vector, w_verify=args.w_verify,
        answerer=answerer,
    )

    n = 0
    correct_oracle = correct_vector = correct_verified = 0
    retrieved_gold_vector = retrieved_gold_verified = 0
    grounded_sum = 0.0
    rows = []

    for s in samples:
        q = get_query_from_sample(s)
        gold = get_table_id(s)
        gold_ans = get_answer(s)
        if not q or not gold:
            continue
        n += 1

        # Vector-only top-1
        vec_hits = retriever.retrieve(q, top_k_vectors=20, top_k_tables=args.top_k_tables)
        v_top = vec_hits[0]["table_id"] if vec_hits else None
        retrieved_gold_vector += int(v_top == gold)
        if v_top and store.get(v_top) is not None:
            v_ans = answerer.answer(q, store.get(v_top)).answer
        else:
            v_ans = ""
        v_ok = _numeric_match(v_ans, gold_ans)
        correct_vector += int(v_ok)

        # Verified rerank top-1 (via agent.run with no candidate so it generates)
        res = agent_v.run(q, top_k_tables=args.top_k_tables)
        vf_top = res.final_ranked[0]["table_id"] if res.final_ranked else None
        retrieved_gold_verified += int(vf_top == gold)
        vf_ans = res.answer.answer if res.answer is not None else ""
        vf_ok = _numeric_match(vf_ans, gold_ans)
        correct_verified += int(vf_ok)
        if res.trace is not None:
            grounded_sum += res.trace.grounded_fraction

        # Oracle: LLM on gold table directly
        o_ans = ""
        o_ok = None
        if args.also_gold:
            rec = store.get(gold)
            if rec is not None:
                o_ans = answerer.answer(q, rec).answer
                o_ok = _numeric_match(o_ans, gold_ans)
                correct_oracle += int(o_ok)

        rows.append({
            "query": q,
            "gold_table": gold,
            "gold_answer": gold_ans,
            "vector_top": v_top,
            "vector_answer": v_ans,
            "vector_correct": v_ok,
            "verified_top": vf_top,
            "verified_answer": vf_ans,
            "verified_correct": vf_ok,
            "oracle_answer": o_ans if args.also_gold else None,
            "oracle_correct": o_ok,
        })

        print(
            f"[{n}] gold={gold} | v_top={v_top}({'OK' if v_top==gold else 'X'}) "
            f"v_ans={v_ans!r:<25}{'✓' if v_ok else '✗'} | "
            f"verified_top={vf_top}({'OK' if vf_top==gold else 'X'}) "
            f"verified_ans={vf_ans!r:<25}{'✓' if vf_ok else '✗'}"
            + (f" | oracle={'✓' if o_ok else '✗'} ({o_ans!r})" if args.also_gold else "")
        )

    print("\n=== Summary ===")
    print(f"queries: {n}")
    print(f"retrieval R@1: vector={retrieved_gold_vector/n:.3f}  verified={retrieved_gold_verified/n:.3f}")
    print(f"answer  acc : vector={correct_vector/n:.3f}  verified={correct_verified/n:.3f}"
          + (f"  oracle={correct_oracle/n:.3f}" if args.also_gold else ""))
    print(f"grounded_fraction (verified top): mean={grounded_sum/n:.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "n": n,
            "retrieval_R@1_vector": retrieved_gold_vector / n,
            "retrieval_R@1_verified": retrieved_gold_verified / n,
            "answer_acc_vector": correct_vector / n,
            "answer_acc_verified": correct_verified / n,
            "answer_acc_oracle": correct_oracle / n if args.also_gold else None,
            "grounded_fraction_mean": grounded_sum / n,
            "rows": rows,
        }, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
