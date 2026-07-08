#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""E4 (W6) — retrieval→generation context format effect (H3).

Retrieval is held fixed at the **oracle** operand set (the gold cells), so any
difference is purely the context *format*. Two arms carry the same numbers and the
same header words, differing only in whether the (header-path → value) binding is
explicit:

  * flat   — a naive flattened dump: "<leaf header> <value>" tokens in one blob,
             the binding between a header and its value is not made explicit.
  * struct — one "<full header path> = <value>" line per operand cell.

H3: the structured (header-path, value) context reduces "silent grounding errors"
(the model emits a number with no exception, but the wrong one). We bucket each
answer as correct / silent-wrong (a number, but wrong) / non-number.

Population: HiTab dev arithmetic m>=2 (n≈158). Needs an LLM (codegen mode).
Run:
    PYTHONPATH=. python scripts/e4_format.py --split dev --llm groq:llama-3.1-8b-instant
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.bench.hitab import load_queries
from rag_agent.bench.schema import Chunk
from rag_agent.generate import answer as gen_answer, evaluate_answer
from rag_agent.eval.metrics import _to_nums

SEED = 42
ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def flat_chunk(ops, table_id) -> Chunk:
    """Naive flattened dump: leaf header + value tokens, binding not explicit."""
    toks = []
    for o in ops:
        leaf = o.header_path[-1] if o.header_path else ""
        toks.append(f"{leaf} {o.value}")
    return Chunk(table_id=table_id, chunk_id="flat", text="  ".join(toks))


def struct_chunk(ops, table_id) -> Chunk:
    """Structured: one explicit (header path = value) line per operand cell."""
    lines = [f"{' > '.join(o.header_path)} = {o.value}" for o in ops]
    return Chunk(table_id=table_id, chunk_id="struct", text="\n".join(lines))


def bucket(res, gold):
    ok = evaluate_answer(res.answer, gold)
    if ok:
        return "correct"
    # numeric_match(x, x) was a self-comparison tautology (true for ANY non-empty
    # answer, text included, via its string-substring branch) — check directly
    # whether the raw answer parses to a number instead.
    if _is_number(res.answer) or bool(_to_nums(res.answer)):
        return "silent_wrong"   # produced a number, but wrong
    return "non_number"          # text / refusal / no number parsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--llm", default="groq:llama-3.1-8b-instant")
    ap.add_argument("--mode", default="codegen", choices=["codegen", "direct"])
    ap.add_argument("--out", default="results/e4_format.json")
    args = ap.parse_args()

    queries, _ = load_queries(args.data_dir, args.split, args.max)
    pop = [q for q in queries
           if (q.aggregation or "none") in ARITH and len(q.gold_operands) >= 2]
    print(f"[pop] arithmetic m>=2: {len(pop)}")

    from rag_agent.llm.factory import build_llm
    llm = build_llm(args.llm)
    print(f"[llm] {args.llm} mode={args.mode}")

    arms = {"flat": flat_chunk, "struct": struct_chunk}
    recs = []
    for i, q in enumerate(pop):
        rec = {"query_id": q.query_id, "m": len({(o.row, o.col) for o in q.gold_operands})}
        for arm, mk in arms.items():
            res = gen_answer(q.question, [mk(q.gold_operands, q.gold_table_id)],
                             llm, mode=args.mode)
            rec[f"{arm}_bucket"] = bucket(res, q.answer)
            rec[f"{arm}_correct"] = int(rec[f"{arm}_bucket"] == "correct")
        recs.append(rec)
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(pop)}")

    n = len(recs)

    def summary(arm):
        b = {k: sum(1 for r in recs if r[f"{arm}_bucket"] == k)
             for k in ("correct", "silent_wrong", "non_number")}
        return {"n": n, "nm_accuracy": round(b["correct"] / n, 4),
                "silent_wrong_rate": round(b["silent_wrong"] / n, 4),
                "non_number_rate": round(b["non_number"] / n, 4), "buckets": b}

    # paired McNemar on correctness (struct vs flat)
    s_only = sum(1 for r in recs if r["struct_correct"] and not r["flat_correct"])
    f_only = sum(1 for r in recs if r["flat_correct"] and not r["struct_correct"])
    # paired bootstrap CI of (struct - flat) accuracy: resample the query index
    # once per term so the pairing is preserved.
    rng = random.Random(SEED)
    diffs = []
    for _ in range(2000):
        acc = 0
        for _ in range(n):
            r = recs[rng.randrange(n)]
            acc += r["struct_correct"] - r["flat_correct"]
        diffs.append(acc / n)
    diffs.sort()

    out = {
        "experiment": "E4_format", "hypothesis": "H3: structured (header-path,value) context cuts silent grounding errors",
        "split": args.split, "seed": SEED, "llm": args.llm, "mode": args.mode,
        "retrieval": "oracle (gold operands fixed)",
        "flat": summary("flat"), "struct": summary("struct"),
        "delta_nm_struct_minus_flat": round(summary("struct")["nm_accuracy"] - summary("flat")["nm_accuracy"], 4),
        "delta_nm_ci95": [round(diffs[50], 4), round(diffs[1949], 4)],
        "mcnemar": {"struct_only": s_only, "flat_only": f_only, "n_discordant": s_only + f_only},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: out[k] for k in ("flat", "struct", "delta_nm_struct_minus_flat",
                                          "delta_nm_ci95", "mcnemar")}, indent=2))
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
