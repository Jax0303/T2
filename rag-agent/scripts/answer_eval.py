#!/usr/bin/env python3
"""End-to-end answer evaluation over the full pipeline.

Runs, per HiTab sample: operand-targeted retrieval -> coverage/fallback context
-> answer generation (direct and/or codegen) -> EM / numeric-match scoring, with
per-sample JSON logging. This is the result table that ties retrieval
completeness to answer accuracy.

The LLM is pluggable: ``--llm mock`` (default; deterministic plumbing baseline,
no API), ``--llm groq:llama-3.3-70b-versatile``, or ``--llm local:Qwen/...``.

Usage
-----
    python scripts/answer_eval.py --llm mock --max-samples 100
    python scripts/answer_eval.py --llm groq:llama-3.1-8b-instant --modes direct codegen
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_agent.data.loader import load_hitab, get_answer  # noqa: E402
from rag_agent.serialization import from_hitab_raw  # noqa: E402
from rag_agent.retrieve.encoders import HashingEncoder  # noqa: E402
from rag_agent.retrieve.operand_retrieval import OperandTargetedRetriever  # noqa: E402
from rag_agent.fallback import assemble_context  # noqa: E402
from rag_agent.generation import Answerer, MockLLM, score_answer  # noqa: E402


def _build_llm(spec: str):
    if spec == "mock":
        return MockLLM()
    from rag_agent.llm.factory import build_llm
    return build_llm(spec)


def _build_encoder(name: str):
    if name == "bge":
        from rag_agent.retrieve.encoders import SentenceTransformerEncoder
        return SentenceTransformerEncoder("BAAI/bge-base-en-v1.5")
    return HashingEncoder(dim=512)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max-samples", type=int, default=100)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--encoder", choices=["hashing", "bge"], default="hashing")
    ap.add_argument("--llm", default="mock", help="mock | groq:<model> | local:<model>")
    ap.add_argument("--modes", nargs="+", default=["direct", "codegen"],
                    choices=["direct", "codegen"])
    ap.add_argument("--no-fallback", action="store_true", help="ablation: disable fallback")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--out", default="results/answer_eval.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    random.seed(args.seed)
    samples = load_hitab(args.data_dir, split=args.split, max_samples=args.max_samples)
    retriever = OperandTargetedRetriever(encoder=_build_encoder(args.encoder), alpha=args.alpha)
    answerer = Answerer(_build_llm(args.llm))

    per_sample = []
    acc = {m: {"em": 0.0, "nm": 0.0, "exec": 0.0, "fb": 0.0} for m in args.modes}
    n = 0
    for s in samples:
        table = from_hitab_raw(s["table"])
        res = retriever.retrieve(s["question"], table, k=args.k)
        bundle = assemble_context(
            res, table, max_tokens=args.max_tokens, enable_fallback=not args.no_fallback
        )
        gold = get_answer(s)
        n += 1
        rec = {"id": s.get("id"), "table_id": table.table_id, "gold": gold,
               "used_fallback": bundle.used_fallback, "modes": {}}
        for m in args.modes:
            try:
                a = answerer.answer(s["question"], bundle, mode=m)
                sc = score_answer(a.answer, gold)
                rec["modes"][m] = {"answer": a.answer, **sc, "exec_ok": a.exec_ok}
            except Exception as e:  # e.g. LLM rate limit — log and keep going
                sc = {"em": False, "nm": False}
                a = None
                rec["modes"][m] = {"answer": None, **sc, "exec_ok": False,
                                   "error": str(e)[:200]}
            acc[m]["em"] += sc["em"]
            acc[m]["nm"] += sc["nm"]
            acc[m]["fb"] += int(bundle.used_fallback)
            if m == "codegen":
                acc[m]["exec"] += int(a.exec_ok if a else False)
        per_sample.append(rec)

    agg = {}
    for m in args.modes:
        agg[m] = {
            "exact_match": acc[m]["em"] / n if n else 0.0,
            "numeric_match": acc[m]["nm"] / n if n else 0.0,
            "fallback_rate": acc[m]["fb"] / n if n else 0.0,
        }
        if m == "codegen":
            agg[m]["exec_accuracy"] = acc[m]["exec"] / n if n else 0.0

    out = {
        "config": {
            "split": args.split, "n_eval": n, "k": args.k, "alpha": args.alpha,
            "encoder": args.encoder, "llm": args.llm, "modes": args.modes,
            "no_fallback": args.no_fallback, "seed": args.seed,
        },
        "aggregate": agg,
        "per_sample": per_sample,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"n_eval={n}  llm={args.llm}  encoder={args.encoder}  no_fallback={args.no_fallback}")
    for m in args.modes:
        line = f"  {m:8s} EM={agg[m]['exact_match']:.3f} NM={agg[m]['numeric_match']:.3f}"
        if m == "codegen":
            line += f" exec_acc={agg[m]['exec_accuracy']:.3f}"
        print(line)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
