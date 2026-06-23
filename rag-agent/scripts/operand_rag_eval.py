#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Operand-targeted RAG — end-to-end evaluation runner.

Produces the result tables the method needs, with baselines and ablations, over
any of {hitab, finqa, wikisql}. Retrieval-side metrics need no LLM; the
answer-accuracy table is computed only when ``--llm`` is given.

Tables emitted (``results/operand_rag/<bench>/``):
  1. ceiling.*          header_path_match_accuracy × {fuzzy, embedding, hybrid}
  2. operand_recall.*   R@{1,3,5,10} × {plain(no-HPIR), operand, oracle}
                        × {S1(flat ablation), S2}
  3. coverage.*         coverage_rate histogram + fallback rate
  4. answers.*          answer accuracy × {full, no-fallback, no-HPIR, S1-only}
                        × {direct, codegen}   (only with --llm)

Baselines / ablations mapped to the spec:
  * BM25-only            → mode=plain, S2 (and the S1 row is the flat control)
  * Oracle               → mode=oracle (gold operand header paths)
  * no-HPIR ablation     → plain vs operand
  * S1 (no header path)  → S1 vs S2
  * no-fallback ablation → answers with/without coverage fallback

Determinism: ``seed=42`` everywhere; every per-query record is logged to
``records.jsonl`` for error analysis.

Usage:
  python scripts/operand_rag_eval.py --bench hitab --max-samples 300 --device cuda
  python scripts/operand_rag_eval.py --bench hitab --llm groq:llama-3.1-8b-instant \
         --answer-samples 100
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.bench import registry
from rag_agent.serialize import serialize_table, S1, S2
from rag_agent.query.operand_decomposer import header_path_match_accuracy, Embedder
from rag_agent.retrieve.operand_retriever import HybridRetriever, retrieve, operand_recall
from rag_agent.retrieve.coverage import assess, apply_fallback

SEED = 42
K_LIST = (1, 3, 5, 10)
MATCHERS = ("fuzzy", "embedding", "hybrid")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def _retrievers(tables, schemes, embedder):
    """Pre-build one HybridRetriever per (table, scheme)."""
    cache = {}
    for tid, tab in tables.items():
        for s in schemes:
            cache[(tid, s)] = HybridRetriever(serialize_table(tab, s), embedder)
    return cache


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def section_ceiling(queries, tables, embedder):
    out = {}
    for m in MATCHERS:
        emb = embedder if m in ("embedding", "hybrid") else None
        accs = [header_path_match_accuracy(q.question, tables[q.gold_table_id],
                                           q.gold_operands, matcher=m, embedder=emb)
                for q in queries]
        out[m] = round(_mean(accs), 4)
    return out


def section_operand_recall(queries, tables, retr, embedder, records):
    use_dense = embedder is not None
    table = {}  # (scheme, mode) -> {k: recall}
    for scheme in (S1, S2):
        for mode in ("plain", "operand", "oracle"):
            per_k = {}
            for k in K_LIST:
                recs = []
                for q in queries:
                    r = retrieve(q.question, tables[q.gold_table_id], q.gold_operands,
                                 mode=mode, k=k, scheme=scheme, embedder=embedder,
                                 retriever=retr[(q.gold_table_id, scheme)])
                    rec = operand_recall(r.retrieved, q.gold_operands)
                    recs.append(rec)
                    if scheme == S2 and mode == "operand" and k == 5:
                        records.append({"query_id": q.query_id, "operand_recall@5": rec,
                                        "n_gold_operands": len(q.gold_operands),
                                        "operands": [o.path_str() for o in r.operands]})
                per_k[k] = round(_mean(recs), 4)
            table[f"{scheme}|{mode}"] = per_k
    return table


def section_coverage(queries, tables, retr, tau_cov, score_floor):
    covs, fb = [], 0
    base_r, full_r = [], []
    for q in queries:
        tab = tables[q.gold_table_id]
        r = retrieve(q.question, tab, q.gold_operands, mode="operand", k=5,
                     scheme=S2, retriever=retr[(q.gold_table_id, S2)])
        rep = assess(r.operands, r.retrieved, tab, tau_cov=tau_cov, score_floor=score_floor)
        covs.append(rep.coverage_rate)
        fb += rep.fallback
        ctx = apply_fallback(r.retrieved, tab, rep, S2)
        base_r.append(operand_recall(r.retrieved, q.gold_operands))
        full_r.append(operand_recall(ctx, q.gold_operands))
    hist = Counter(round(c, 1) for c in covs)
    n = len(queries)
    return {
        "mean_coverage_rate": round(_mean(covs), 4),
        "fallback_rate": round(fb / n, 4) if n else 0.0,
        "histogram": {str(k): hist[k] for k in sorted(hist)},
        "operand_recall@5_no_fallback": round(_mean(base_r), 4),
        "operand_recall@5_with_fallback": round(_mean(full_r), 4),
    }


def section_answers(queries, tables, retr, embedder, llm, answer_modes, tau_cov, score_floor, records):
    """answer accuracy for the answer-side ablations (needs an LLM)."""
    from rag_agent.generate import answer as gen_answer, evaluate_answer
    configs = {
        "full":       dict(mode="operand", scheme=S2, fallback=True),
        "no-fallback": dict(mode="operand", scheme=S2, fallback=False),
        "no-HPIR":    dict(mode="plain",   scheme=S2, fallback=False),
        "S1-only":    dict(mode="operand", scheme=S1, fallback=True),
    }
    out = {}
    for cname, cfg in configs.items():
        for amode in answer_modes:
            correct, used_cg, n = 0, 0, 0
            for q in queries:
                tab = tables[q.gold_table_id]
                r = retrieve(q.question, tab, q.gold_operands, mode=cfg["mode"], k=5,
                             scheme=cfg["scheme"], embedder=embedder,
                             retriever=retr[(q.gold_table_id, cfg["scheme"])])
                ctx = r.retrieved
                if cfg["fallback"]:
                    rep = assess(r.operands, r.retrieved, tab, tau_cov=tau_cov, score_floor=score_floor)
                    ctx = apply_fallback(r.retrieved, tab, rep, cfg["scheme"])
                a = gen_answer(q.question, ctx, llm, mode=amode)
                ok = evaluate_answer(a.answer, q.answer)
                correct += ok; used_cg += a.used_codegen; n += 1
                records.append({"query_id": q.query_id, "config": cname, "answer_mode": amode,
                                "pred": a.answer, "gold": q.answer, "correct": bool(ok),
                                "used_codegen": a.used_codegen})
            out[f"{cname}|{amode}"] = {"accuracy": round(correct / n, 4) if n else 0.0,
                                       "n": n, "used_codegen": used_cg}
    return out


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def to_markdown(bench, summary):
    L = [f"# Operand-targeted RAG — {bench}", "",
         f"seed={SEED} · n={summary['n_queries']} (operand-bearing={summary['n_operand_queries']}) · "
         f"split={summary['split']} · dense={summary['dense']}", ""]
    L += ["## 1. Decomposition ceiling (header_path_match_accuracy)", "",
          "| matcher | accuracy |", "|---|---|"]
    for m, v in summary["ceiling"].items():
        L.append(f"| {m} | {v} |")
    L += ["", "## 2. operand_recall@k", "",
          "| serialization | mode | " + " | ".join(f"R@{k}" for k in K_LIST) + " |",
          "|---|---|" + "|".join("---" for _ in K_LIST) + "|"]
    for key, perk in summary["operand_recall"].items():
        scheme, mode = key.split("|")
        sname = "S1(flat)" if scheme == S1 else "S2(header-path)"
        L.append(f"| {sname} | {mode} | " + " | ".join(str(perk[k]) for k in K_LIST) + " |")
    cov = summary["coverage"]
    L += ["", "## 3. Coverage + fallback", "",
          f"- mean coverage_rate: **{cov['mean_coverage_rate']}**",
          f"- fallback rate: **{cov['fallback_rate']}**",
          f"- operand_recall@5: no-fallback {cov['operand_recall@5_no_fallback']} "
          f"→ +fallback **{cov['operand_recall@5_with_fallback']}**",
          f"- histogram: {cov['histogram']}"]
    if summary.get("answers"):
        L += ["", "## 4. Answer accuracy (ablations)", "",
              "| config | answer_mode | accuracy | n | used_codegen |",
              "|---|---|---|---|---|"]
        for key, v in summary["answers"].items():
            cfg, amode = key.split("|")
            L.append(f"| {cfg} | {amode} | {v['accuracy']} | {v['n']} | {v['used_codegen']} |")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=registry.BENCHMARKS, default="hitab")
    ap.add_argument("--split", default=None)
    ap.add_argument("--max-samples", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-dense", action="store_true", help="BM25-only (skip embedder)")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--tau-cov", type=float, default=0.7)
    ap.add_argument("--score-floor", type=float, default=0.3)
    ap.add_argument("--llm", default=None, help="e.g. groq:llama-3.1-8b-instant or local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit")
    ap.add_argument("--answer-samples", type=int, default=100)
    ap.add_argument("--answer-modes", default="direct,codegen")
    ap.add_argument("--out-dir", default="results/operand_rag")
    args = ap.parse_args()

    random.seed(SEED)
    out_dir = Path(args.out_dir) / args.bench
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    print(f"[load] {args.bench} (max={args.max_samples}) ...")
    queries, tables = registry.load(args.bench, split=args.split, max_samples=args.max_samples)
    operand_queries = [q for q in queries if q.gold_operands]
    print(f"[load] {len(queries)} queries, {len(tables)} tables, "
          f"{len(operand_queries)} operand-bearing")

    embedder = None if args.no_dense else Embedder(args.embed_model, device=args.device)
    print(f"[index] building retrievers (dense={embedder is not None}) ...")
    retr = _retrievers(tables, (S1, S2), embedder)

    print("[1/4] ceiling ...")
    ceiling = section_ceiling(operand_queries, tables, embedder or Embedder(args.embed_model, args.device))
    print("[2/4] operand_recall ...")
    op_recall = section_operand_recall(operand_queries, tables, retr, embedder, records)
    print("[3/4] coverage + fallback ...")
    coverage = section_coverage(operand_queries, tables, retr, args.tau_cov, args.score_floor)

    summary = {
        "bench": args.bench,
        "split": args.split or registry._DEFAULT_SPLIT.get(args.bench),
        "seed": SEED,
        "dense": embedder is not None,
        "n_queries": len(queries),
        "n_operand_queries": len(operand_queries),
        "ceiling": ceiling,
        "operand_recall": op_recall,
        "coverage": coverage,
    }

    if args.llm:
        print(f"[4/4] answers via {args.llm} (n={args.answer_samples}) ...")
        from rag_agent.llm.factory import build_llm
        llm = build_llm(args.llm)
        sub = operand_queries[:args.answer_samples]
        try:
            summary["answers"] = section_answers(
                sub, tables, retr, embedder, llm,
                tuple(args.answer_modes.split(",")), args.tau_cov, args.score_floor, records)
        except Exception as e:  # noqa: BLE001 — keep retrieval results even if LLM fails
            summary["answers_error"] = f"{type(e).__name__}: {e}"
            print(f"[warn] answer section failed: {summary['answers_error']}")
    else:
        print("[4/4] answers skipped (no --llm)")

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (out_dir / "summary.md").write_text(to_markdown(args.bench, summary))
    with open(out_dir / "records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[done] wrote {out_dir}/summary.{{json,md}} + records.jsonl")
    print(to_markdown(args.bench, summary))


if __name__ == "__main__":
    main()
