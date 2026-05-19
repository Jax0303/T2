#!/usr/bin/env python3
"""End-to-end answer eval on a HiTab hard-query subset.

Reuses sidecar_verifier pipeline (retriever -> verifier -> reconciler -> answerer).
Difficulty classes derived from HiTab's `aggregation` and `answer_formulas` fields,
which are the supervision used in the HiTab paper (Cheng et al., ACL 2022) appendix.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
_OP_RE = re.compile(r"[+\-*/]")


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


def _flatten_strs(g):
    if isinstance(g, list):
        return [str(x) for x in g]
    return [str(g)] if g is not None else []


def _numeric_match(pred, gold, rel_tol: float = 0.02) -> bool:
    """Tolerant match.

    - For numeric gold: accept exact, ±rel_tol%, ×100 (% form), ÷100 (fraction form),
      and abs() (covers HiTab "opposite" aggregation where gold is the absolute change).
    - For string gold: case-insensitive substring either direction. Handles list-style
      gold (e.g. ['quebec']) which the original str()==str() comparison did not.
    """
    if pred is None:
        return False
    pred_s = str(pred).strip().lower()

    g_nums = _to_nums(gold)
    p_nums = _to_nums(pred)
    if g_nums:
        p_variants = [
            {round(x, 2) for x in p_nums},
            {round(x * 100, 2) for x in p_nums},
            {round(x / 100, 4) for x in p_nums},
            {round(abs(x), 2) for x in p_nums},
        ]
        for g in g_nums:
            g_cands = [round(g, 2), round(g * 100, 2), round(g / 100, 4), round(abs(g), 2)]
            ok = False
            for gc in g_cands:
                for pv in p_variants:
                    if gc in pv:
                        ok = True
                        break
                    for pn in pv:
                        if abs(pn - gc) / max(abs(gc), 1e-9) < rel_tol:
                            ok = True
                            break
                    if ok:
                        break
                if ok:
                    break
            if not ok:
                return False
        return True

    for gs in (s.strip().lower() for s in _flatten_strs(gold) if s.strip()):
        if gs in pred_s or pred_s in gs:
            return True
    return False


def _formula_ops(sample) -> int:
    """Max operator count across the sample's formulas."""
    fs = sample.get("answer_formulas") or []
    if not fs:
        return 0
    return max(len(_OP_RE.findall(f.lstrip("="))) for f in fs)


def _difficulty_class(sample) -> str:
    """Map HiTab supervision to a coarse difficulty label.

    The order matches the HiTab paper's appendix categories, from easiest to hardest.
    """
    agg = tuple(sorted(set(sample.get("aggregation") or ["none"])))
    ops = _formula_ops(sample)

    if ops >= 2:
        return "multi_op_formula"           # e.g. =(B+C+D)/E
    if "div" in agg or "sum" in agg or "diff" in agg or "average" in agg or "range" in agg:
        return "arithmetic_agg"             # sum / diff / div / avg / range
    if "pair-argmax" in agg or "pair-argmin" in agg or "topk-argmax" in agg or "topk-argmin" in agg or "kth-argmax" in agg:
        return "pair_or_topk_arg"           # compare entities / pick k-th
    if "argmax" in agg or "argmin" in agg or "max" in agg or "min" in agg:
        return "single_arg"
    if "greater_than" in agg or "less_than" in agg or "opposite" in agg or "counta" in agg:
        return "comparison_or_count"
    if ops == 1:
        return "single_op_formula"
    return "simple_lookup"


HARD_CLASSES = [
    "multi_op_formula",
    "arithmetic_agg",
    "pair_or_topk_arg",
    "single_arg",
    "comparison_or_count",
    "single_op_formula",
]


def stratified_hard_subset(samples, per_class: int, seed: int = 0):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for s in samples:
        cls = _difficulty_class(s)
        buckets[cls].append(s)

    chosen = []
    for cls in HARD_CLASSES:
        bucket = buckets.get(cls, [])
        rng.shuffle(bucket)
        chosen.extend([(cls, s) for s in bucket[:per_class]])
    return chosen, {cls: len(buckets.get(cls, [])) for cls in HARD_CLASSES}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/home/user/T2/hart-table-retrieval/data/hitab")
    p.add_argument("--chroma-dir", default="/home/user/T2/hart-table-retrieval/data/chroma_db")
    p.add_argument("--serializer", default="plain_markdown")
    p.add_argument("--per-class", type=int, default=8)
    p.add_argument("--top-k-tables", type=int, default=5)
    p.add_argument("--mode", default="rerank")
    p.add_argument("--w-vector", type=float, default=0.8)
    p.add_argument("--w-verify", type=float, default=0.2)
    p.add_argument("--llm-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--quantization", default="4bit")
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--retriever-device", default="cpu",
                   help="Device for the retriever's embedder. Default cpu — saves VRAM "
                        "for the LLM. Set to 'cuda' if VRAM is plentiful.")
    p.add_argument("--out", default="results/hard_query_eval.json")
    p.add_argument("--also-gold", action="store_true",
                   help="Also run LLM on gold table (oracle retrieval) for upper bound.")
    args = p.parse_args()

    full = load_hitab(data_dir=args.data_dir)

    chosen, class_population = stratified_hard_subset(full, args.per_class, args.seed)
    print(f"Hard subset: {len(chosen)} queries over {len(HARD_CLASSES)} classes")
    print("Class population in dev set:")
    for cls in HARD_CLASSES:
        print(f"  {cls:22s}: pool={class_population[cls]:4d}  picked={min(args.per_class, class_population[cls])}")

    print("\nBuilding original-table store ...")
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

    print(f"Loading retriever (serializer={args.serializer}, device={args.retriever_device}) ...")
    retriever = VectorRetriever(
        chroma_dir=args.chroma_dir,
        serializer=args.serializer,
        device=args.retriever_device,
    )

    print(f"Loading reader {args.llm_model} (quant={args.quantization}) ...")
    t_load = time.time()
    answerer = LocalLLMAnswerer(
        model_name=args.llm_model,
        quantization=args.quantization,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"Reader loaded in {time.time() - t_load:.1f}s")

    agent = VerifierAgent(
        retriever, store, mode=args.mode,
        w_vector=args.w_vector, w_verify=args.w_verify,
        answerer=answerer,
    )

    per_class_stats = {cls: Counter() for cls in HARD_CLASSES}
    rows = []

    for i, (cls, s) in enumerate(chosen, 1):
        q = get_query_from_sample(s)
        gold = get_table_id(s)
        gold_ans = get_answer(s)
        if not q or not gold:
            continue
        gold_formula = (s.get("answer_formulas") or [""])[0]

        t0 = time.time()
        res = agent.run(q, top_k_tables=args.top_k_tables)
        elapsed = time.time() - t0

        vec_top = res.vector_ranked[0]["table_id"] if res.vector_ranked else None
        final_top = res.final_ranked[0]["table_id"] if res.final_ranked else None
        pred = res.answer.answer if res.answer is not None else ""
        retr_ok_vec = vec_top == gold
        retr_ok_final = final_top == gold
        ans_ok = _numeric_match(pred, gold_ans)
        grounded = res.trace.grounded_fraction if res.trace is not None else None

        oracle_ans = None
        oracle_ok = None
        if args.also_gold:
            rec = store.get(gold)
            if rec is not None:
                oracle_ans = answerer.answer(q, rec).answer
                oracle_ok = _numeric_match(oracle_ans, gold_ans)

        st = per_class_stats[cls]
        st["n"] += 1
        st["retr_vec_R@1"] += int(retr_ok_vec)
        st["retr_final_R@1"] += int(retr_ok_final)
        st["ans_correct"] += int(ans_ok)
        if oracle_ok is not None:
            st["oracle_correct"] += int(oracle_ok)

        rows.append({
            "class": cls,
            "query": q,
            "gold_table": gold,
            "gold_answer": gold_ans,
            "gold_formula": gold_formula,
            "vector_top": vec_top,
            "final_top": final_top,
            "predicted_answer": pred,
            "retr_vec_R@1": retr_ok_vec,
            "retr_final_R@1": retr_ok_final,
            "answer_correct": ans_ok,
            "grounded_fraction": grounded,
            "oracle_answer": oracle_ans,
            "oracle_correct": oracle_ok,
            "elapsed_s": round(elapsed, 2),
        })

        print(
            f"[{i:3d}/{len(chosen)}] {cls:22s} gold={gold} "
            f"final={'OK' if retr_ok_final else 'X '} ans={'✓' if ans_ok else '✗'} "
            f"({elapsed:.1f}s)  pred={pred!r}"
        )

    print("\n=== Per-class summary ===")
    print(f"{'class':24s} {'n':>3s}  {'R@1_vec':>8s} {'R@1_final':>10s} {'ans_acc':>8s}" + ("  oracle_acc" if args.also_gold else ""))
    overall = Counter()
    for cls in HARD_CLASSES:
        st = per_class_stats[cls]
        n = st["n"]
        if n == 0:
            continue
        line = (
            f"{cls:24s} {n:3d}  "
            f"{st['retr_vec_R@1']/n:8.3f} "
            f"{st['retr_final_R@1']/n:10.3f} "
            f"{st['ans_correct']/n:8.3f}"
        )
        if args.also_gold:
            line += f"  {st['oracle_correct']/n:10.3f}"
        print(line)
        for k, v in st.items():
            overall[k] += v

    n = overall["n"]
    if n:
        line = (
            f"{'OVERALL':24s} {n:3d}  "
            f"{overall['retr_vec_R@1']/n:8.3f} "
            f"{overall['retr_final_R@1']/n:10.3f} "
            f"{overall['ans_correct']/n:8.3f}"
        )
        if args.also_gold:
            line += f"  {overall['oracle_correct']/n:10.3f}"
        print(line)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "config": vars(args),
            "class_population_in_dev": class_population,
            "per_class_summary": {
                cls: {k: v for k, v in per_class_stats[cls].items()}
                for cls in HARD_CLASSES if per_class_stats[cls]["n"]
            },
            "overall": dict(overall),
            "rows": rows,
        }, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
